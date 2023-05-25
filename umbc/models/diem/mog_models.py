from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from umbc.models.diem import layers

T = torch.Tensor


class EmbedderMoG(nn.Module):
    def __init__(
        self,
        dim_input,
        num_outputs,
        out_type='param_cat',
        num_proto=5,
        num_ems=2,
        dim_feat=128,
        num_heads=3,
        tau=10.0,
        distr_emb_args=None,
        set_out_size: int = None,
        net_type: str = "mog"
    ):
        super().__init__()

        self.name = "DiffEM"
        self.rff = RFFNet(dim_input, dim_feat,
                          n_layers=3 if net_type == "imagenet" else 0)
        self.diem = layers.DIEM(dim_feat, H=num_heads, p=num_proto,
                                L=num_ems, tau=tau, out=out_type, distr_emb_args=distr_emb_args)
        outsize = dim_input if set_out_size is None else set_out_size

        self.outnet: nn.Module
        if net_type == "mog" or net_type == "imagenet":
            self.outnet = OutputNet(self.diem.outdim, num_outputs, outsize)
        elif net_type == "modelnet":
            self.outnet = OutputNetClass(self.diem.outdim, dim_feat, outsize)
        else:
            raise NotImplementedError(f"{net_type=} not implemented")

        self.temp = torch.Tensor()
        self.temp_count = 0

    def forward(self, X, cards=None):
        B, N_max, d0 = X.shape
        S = self.rff(X)
        mask = torch.ones(B, N_max).to(S)
        if cards is not None:
            for ii in range(B):
                mask[ii][cards[ii]:] = 0.0

        FS = self.diem(S, mask)
        return self.outnet(FS)

    def reset(self) -> None:
        self.temp = torch.Tensor()
        self.temp_count = 0

    def process_minibatch(self, X: T, cards: Any = None) -> T:
        B, N_max, d0 = X.shape
        S = self.rff(X)
        mask = torch.ones(B, N_max).to(S)
        if cards is not None:
            for ii in range(B):
                mask[ii][cards[ii]:] = 0.0

        FS = self.diem(S, mask)

        self.temp = FS if (self.temp.numel() == 0) else self.temp + FS
        self.temp_count += 1

        return self.outnet(self.temp / self.temp_count)


class RFFNet(nn.Module):
    def __init__(self, d0, d, n_layers: int = 0):
        super(RFFNet, self).__init__()

        layers: List[nn.Module] = []
        for i in range(n_layers):
            layers.extend([nn.Linear(d0 if i == 0 else d, d), nn.ReLU()])

        layers.append(nn.Linear(d0 if n_layers == 0 else d, d))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class OutputNet(nn.Module):
    def __init__(self, din, K, d0):
        super(OutputNet, self).__init__()
        # dout = 2 * K + K * d0
        dout = K + 2 * K * d0
        dhid = (din + dout) // 2

        self.net = nn.Sequential(
            nn.Linear(din, dhid),
            nn.ReLU(),
            nn.Linear(dhid, dout),
        )

        self.K = K
        self.d0 = d0

    def forward_original(self, FS):
        B = FS.shape[0]
        K, d0 = self.K, self.d0
        MoGvecs = self.net(FS)
        pi = torch.softmax(MoGvecs[:, :K], -1)
        mu = MoGvecs[:, K:(K + K * d0)].reshape(B, K, d0)
        sigma = F.softplus(MoGvecs[:, (K + K * d0):]).reshape(B, K, 1)
        return pi, (mu, sigma)

    def forward(self, FS):
        B = FS.shape[0]
        K, d0 = self.K, self.d0
        MoGvecs = self.net(FS)

        pi = MoGvecs[:, :K].unsqueeze(-1)
        mu = MoGvecs[:, K:(K + K * d0)].reshape(B, K, d0)
        sigma = MoGvecs[:, (K + K * d0):].reshape(B, K, d0)

        out = torch.cat((pi, mu, sigma), dim=-1)
        return out


class OutputNetClass(nn.Module):
    def __init__(self, dim: int, h_dim: int, dout: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dout),
        )

    def forward(self, FS):
        return self.net(FS)
