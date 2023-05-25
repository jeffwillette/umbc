import torch
import torch.nn as nn
import torch.nn.functional as F

from base import HashableModule
from umbc.models.diem.layers import DIEM
from umbc.models.layers.fspool import FSPool
from umbc.models.layers.set_xformer import PMA, SAB
from umbc.models.layers.sse import SlotSetEncoder


class CNPEncoder(HashableModule):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = []
        for i in range(num_layers - 1):
            if i == 0:
                in_dim = x_dim + y_dim
            else:
                in_dim = hidden_dim
            self.encoder.append(nn.Linear(in_dim, hidden_dim))
            self.encoder.append(nn.ReLU(inplace=True))
        self.encoder.append(nn.Linear(hidden_dim, hidden_dim))
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, context):
        return self.encoder(context)


class CNPDecoder(nn.Module):
    def __init__(self,
                 x_dim: int = 2,
                 y_dim: int = 3,
                 hidden_dim: int = 64,
                 num_layers: int = 4):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.decoder = []
        for i in range(num_layers):
            if i == 0:
                in_dim = x_dim + hidden_dim
            else:
                in_dim = hidden_dim
            self.decoder.append(nn.Linear(in_dim, hidden_dim))
            self.decoder.append(nn.ReLU(inplace=True))
        self.decoder.append(nn.Linear(hidden_dim, 2 * y_dim))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, rep, target_x):
        rep = rep.repeat(1, target_x.size(1), 1)
        x = torch.cat([rep, target_x], dim=-1)
        h = self.decoder(x)
        mu, log_sigma = torch.split(h, self.y_dim, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        if hasattr(self, "hook"):
            self.hook(self, [rep, target_x], x)
            self.hook(self, [log_sigma], sigma)
        return dist


class MBC(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 3,
        d_dim: int = 64,        # hidden dim for encoder
        h_dim: int = 128,       # slot dim
        d_hat: int = 64,        # dim after q,k,v
        K: int = 1,
        e_depth: int = 4,
        d_depth: int = 4,
        heads: int = 1,
        ln_slots: bool = False,
        ln_after: bool = True,
        slot_type: str = "learned",
        slot_drop: float = 0.0,
        attn_act: str = "softmax",
        slot_residual: bool = False,
    ) -> None:
        super().__init__()

        self.pooler = SlotSetEncoder(
            K=K, h=h_dim, d=d_dim, d_hat=d_hat, slot_type=slot_type,
            ln_slots=ln_slots, heads=heads, slot_drop=slot_drop,
            attn_act=attn_act, slot_residual=slot_residual, ln_after=ln_after
        )

        self.name = f"MBC/{self.pooler.name}_K_{K}_heads_{heads}_{attn_act}"

        self.encoder = CNPEncoder(x_dim, y_dim, d_dim, e_depth)
        self.decoder = CNPDecoder(x_dim, y_dim, d_dim, d_depth)

    def forward(self, context, target_x, target_y):
        h = self.encoder(context)
        rep = self.pooler(h)
        dist = self.decoder(rep, target_x)
        # https://github.com/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb
        if target_y is not None:
            log_prob = dist.log_prob(target_y)
            if hasattr(self, "hook"):
                self.hook(self, [target_y], log_prob)
        else:
            log_prob = None

        return log_prob

    def partitioned_forward(
            self,
            context,
            context_nograd,
            target_x,
            target_y):
        h = self.encoder(context)
        if context_nograd is not None:
            with torch.no_grad():
                h_nograd = self.encoder(context_nograd)
        else:
            h_nograd = None
        rep = self.pooler.partitioned_forward(h, h_nograd)
        dist = self.decoder(rep, target_x)

        if target_y is not None:
            log_prob = dist.log_prob(target_y)
            if hasattr(self, "hook"):
                self.hook(self, [target_y], log_prob)
        else:
            log_prob = None

        return log_prob

    def sse_grad_correct(self, c: int = 1):
        for p in self.encoder.parameters():
            p.grad.data.mul_(c)
        self.pooler.grad_correct(c)


class DeepSet(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 3,
        d_dim: int = 64,
        h_dim: int = 128,
        d_hat: int = 64,
        K: int = 1,
        e_depth: int = 4,
        d_depth: int = 4,
        heads: int = 1,
        ln_slots: bool = False,
        ln_after: bool = True,
        slot_type: str = "learned",
        slot_drop: float = 0.0,
        attn_act: str = "sigmoid-slot",
        slot_residual: bool = False,
    ) -> None:
        super().__init__()

        self.name = f"Deepset-avg"
        self.encoder = CNPEncoder(x_dim, y_dim, d_dim, e_depth)
        self.decoder = CNPDecoder(x_dim, y_dim, d_dim, d_depth)

    def forward(self, context, target_x, target_y):
        h = self.encoder(context)
        rep = torch.mean(h, dim=1, keepdim=True)
        dist = self.decoder(rep, target_x)
        if target_y is not None:
            log_prob = dist.log_prob(target_y)
            if hasattr(self, "hook"):
                self.hook(self, [target_y], log_prob)
        else:
            log_prob = None

        return log_prob

    def partitioned_forward(
            self,
            context,
            context_nograd,
            target_x,
            target_y):
        h = self.encoder(context)
        if context_nograd is not None:
            with torch.no_grad():
                h_nograd = self.encoder(context_nograd)
            h = torch.cat([h, h_nograd], dim=1)
            c = context.size(1) + context_nograd.size(1)
            rep = torch.sum(h, dim=1, keepdim=True) / c
        else:
            rep = torch.sum(h, dim=1, keepdim=True) / context.size(1)
        dist = self.decoder(rep, target_x)
        if target_y is not None:
            log_prob = dist.log_prob(target_y)
            if hasattr(self, "hook"):
                self.hook(self, [target_y], log_prob)
                self.hook(self, [h], rep)
        else:
            log_prob = None

        return log_prob

    def sse_grad_correct(self, c: int = 1):
        for p in self.encoder.parameters():
            p.grad.data.mul_(c)


class UMBC(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 3,
        d_dim: int = 64,        # hidden dim for encoder
        h_dim: int = 128,       # slot dim
        d_hat: int = 64,        # dim after q,k,v
        K: int = 128,
        e_depth: int = 4,
        d_depth: int = 4,
        heads: int = 1,
        ln_slots: bool = False,
        ln_after: bool = True,
        slot_type: str = "random",
        slot_drop: float = 0.0,
        attn_act: str = "softmax",
        slot_residual: bool = False,
    ) -> None:
        super().__init__()
        assert d_hat == d_dim
        self.pooler = SlotSetEncoder(
            K=K, h=h_dim, d=d_dim, d_hat=d_hat, slot_type=slot_type,
            ln_slots=ln_slots, heads=heads, slot_drop=slot_drop,
            attn_act=attn_act, slot_residual=slot_residual, ln_after=ln_after
        )

        self.set_trans = nn.Sequential(SAB(dim_in=d_hat, dim_out=d_hat,
                                           num_heads=1, ln=True),
                                       SAB(dim_in=d_hat, dim_out=d_hat,
                                           num_heads=1, ln=True),
                                       PMA(dim=d_hat, num_heads=1,
                                           num_seeds=1, ln=True))
        self.name = f"UMBC/{self.pooler.name}_K_{K}_heads_{heads}_{attn_act}_ST"

        self.encoder = CNPEncoder(x_dim, y_dim, d_dim, e_depth)
        self.decoder = CNPDecoder(x_dim, y_dim, d_dim, d_depth)

    def forward(self, context, target_x, target_y):
        h = self.encoder(context)
        rep = self.pooler(h)
        rep = self.set_trans(rep)

        dist = self.decoder(rep, target_x)
        # https://github.com/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb
        if target_y is not None:
            log_prob = dist.log_prob(target_y)
            if hasattr(self, "hook"):
                self.hook(self, [target_y], log_prob)
        else:
            log_prob = None

        return log_prob

    def partitioned_forward(
            self,
            context,
            context_nograd,
            target_x,
            target_y):
        h = self.encoder(context)
        if context_nograd is not None:
            with torch.no_grad():
                h_nograd = self.encoder(context_nograd)
        else:
            h_nograd = None
        rep = self.pooler.partitioned_forward(h, h_nograd)
        rep = self.set_trans(rep)
        dist = self.decoder(rep, target_x)

        if target_y is not None:
            log_prob = dist.log_prob(target_y)
            if hasattr(self, "hook"):
                self.hook(self, [target_y], log_prob)
        else:
            log_prob = None

        return log_prob

    def sse_grad_correct(self, c: int = 1):
        for p in self.encoder.parameters():
            p.grad.data.mul_(c)
        self.pooler.grad_correct(c)


class SetTransformer(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 3,
        d_dim: int = 128,        # hidden dim for encoder
        h_dim: int = 128,        # do not use
        d_hat: int = 128,        # do not use
        K: int = 1,
        e_depth: int = 4,
        d_depth: int = 4,
        heads: int = 4,
        ln_slots: bool = False,        # do not use
        ln_after: bool = True,         # do not use
        slot_type: str = "random",
        slot_drop: float = 0.0,        # do not use
        attn_act: str = "softmax",
        slot_residual: bool = False,   # do not use
    ) -> None:
        super().__init__()
        self.name = "Set-Transformer"

        self.pooler = nn.Sequential(
            SAB(dim_in=d_dim, dim_out=d_dim, num_heads=heads, ln=True),
            SAB(dim_in=d_dim, dim_out=d_dim, num_heads=heads, ln=True),
            PMA(dim=d_dim, num_heads=heads, num_seeds=K, ln=True))

        self.encoder = CNPEncoder(x_dim, y_dim, d_dim, e_depth)
        self.decoder = CNPDecoder(x_dim, y_dim, d_dim, d_depth)

    def forward(self, context, target_x, target_y, split_size=None):
        if split_size is None:
            h = self.encoder(context)
            rep = self.pooler(h)
        else:
            c_xs = torch.split(context, split_size, dim=1)
            all_rep = []
            for c_x in c_xs:
                h = self.encoder(c_x)
                rep = self.pooler(h)
                all_rep.append(rep)
            rep = sum(all_rep) / len(all_rep)
        dist = self.decoder(rep, target_x)
        # https://github.com/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb
        if target_y is not None:
            log_prob = dist.log_prob(target_y)
            if hasattr(self, "hook"):
                self.hook(self, [target_y], log_prob)
        else:
            log_prob = None

        return log_prob


class DiEM(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 3,
        d_dim: int = 128,        # hidden dim for encoder
        h_dim: int = 128,        # do not use
        d_hat: int = 128,        # do not use
        K: int = 1,
        e_depth: int = 4,
        d_depth: int = 4,
        heads: int = 4,
        ln_slots: bool = False,        # do not use
        ln_after: bool = True,         # do not use
        slot_type: str = "random",
        slot_drop: float = 0.0,        # do not use
        attn_act: str = "softmax",
        slot_residual: bool = False,   # do not use
    ) -> None:
        super().__init__()
        self.name = "DiEM"
        set_enc = DIEM(d_dim, H=4, p=64, L=3, tau=0.01, out="select_best2")
        self.pooler = nn.Sequential(
            set_enc,
            nn.Linear(set_enc.outdim, d_dim)
        )

        self.encoder = CNPEncoder(x_dim, y_dim, d_dim, e_depth)
        self.decoder = CNPDecoder(x_dim, y_dim, d_dim, d_depth)

    def forward(self, context, target_x, target_y, split_size=None):
        if split_size is not None:
            c_xs = torch.split(context, 100, dim=1)
            all_rep = []
            for c_x in c_xs:
                h = self.encoder(c_x)
                rep = self.pooler(h)
                all_rep.append(rep)
            rep = sum(all_rep) / len(all_rep)
        else:
            h = self.encoder(context)
            rep = self.pooler(h)
        rep = rep.unsqueeze(1)
        dist = self.decoder(rep, target_x)
        # https://github.com/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb
        if target_y is not None:
            log_prob = dist.log_prob(target_y)
            if hasattr(self, "hook"):
                self.hook(self, [target_y], log_prob)
        else:
            log_prob = None

        return log_prob


class CNPFSPool(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 3,
        d_dim: int = 128,        # hidden dim for encoder
        h_dim: int = 128,        # do not use
        d_hat: int = 128,        # do not use
        K: int = 1,
        e_depth: int = 4,
        d_depth: int = 4,
        heads: int = 4,
        ln_slots: bool = False,        # do not use
        ln_after: bool = True,         # do not use
        slot_type: str = "random",
        slot_drop: float = 0.0,        # do not use
        attn_act: str = "softmax",
        slot_residual: bool = False,   # do not use
    ) -> None:
        super().__init__()
        self.name = "fspool"
        self.pooler = FSPool(d_dim, n_pieces=20)

        self.encoder = CNPEncoder(x_dim, y_dim, d_dim, e_depth)
        self.decoder = CNPDecoder(x_dim, y_dim, d_dim, d_depth)

    def forward(self, context, target_x, target_y, split_size=None):
        if split_size is not None:
            c_xs = torch.split(context, 100, dim=1)
            all_rep = []
            for c_x in c_xs:
                h = self.encoder(c_x)
                rep, _ = self.pooler(h)
                all_rep.append(rep)
            rep = sum(all_rep) / len(all_rep)
        else:
            h = self.encoder(context)
            rep  = self.pooler(h)
        rep = rep.unsqueeze(1)
        dist = self.decoder(rep, target_x)
        # https://github.com/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb
        if target_y is not None:
            log_prob = dist.log_prob(target_y)
            if hasattr(self, "hook"):
                self.hook(self, [target_y], log_prob)
        else:
            log_prob = None

        return log_prob
