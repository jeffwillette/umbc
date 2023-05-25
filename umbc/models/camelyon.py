from typing import Any, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from utils import seed

from umbc.models.layers.ab_mil import GatedAttention
from umbc.models.layers.bigset import OT, IdentityExtractor, MBCFunction, T
from umbc.models.layers.deepsets import DeepSetsPooler
from umbc.models.layers.ds_mil import BClassifier, IClassifier, MILNet
from umbc.models.layers.fspool import FSPool
from umbc.models.layers.linear import get_activation, get_linear_layer
from umbc.models.layers.resnet import ResNetSimCLR
from umbc.models.layers.set_xformer import ISAB, PMA, SAB
from umbc.models.layers.sse import SlotSetEncoder

FEAT_DIM = 256


class CamelyonMBC:
    """
    for the DDP wrapper, we need to be able to call the forward method
    because it is a special method which causes a syncrhonization
    """

    decoder: Any
    lin: Any
    extractor: Any
    mbc_pooler: Any

    def pre_forward_mbc(self) -> None:
        self.extractor.pre_forward_mbc()
        self.mbc_pooler.pre_forward_mbc()

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        mask: OT = None,
        grad: bool = True,
        augmentation: Any = None,
    ) -> Tuple[T, T, OT]:
        if augmentation is not None:
            X = augmentation(X)

        x, _ = self.extractor.forward_mbc(X, grad=grad, mask=mask)
        x = F.dropout(x, training=self.extractor.training)
        x_t, c_t = self.mbc_pooler.forward_mbc(
            x,
            X_prev=X_prev,
            c_prev=c_prev,
            grad=grad,
            mask=mask,
        )

        return x, x_t, c_t

    def post_forward_mbc(
        self,
        X: T,
        c: OT = None,
        mask: OT = None,
    ) -> T:
        return self.mbc_pooler.post_forward_mbc(X=X, c=c, mask=mask)

    def forward(  # type: ignore
        self,
        x_loader: DataLoader,
        grad_size: int,
        test: bool = False,
        augmentation: Any = None,
    ) -> Tuple[T, T]:

        # pre sampling seeds
        n_seeds = len(x_loader) + 10
        self.pre_sampled_seeds = np.random.randint(2**32 - 1, size=(n_seeds,))
        seed(self.pre_sampled_seeds[0])
        self.pre_forward_mbc()

        x_ft, x_grad, x_t, c_t = [], None, None, None
        t = tqdm(x_loader, ncols=75, leave=False)
        for i, x in enumerate(t):
            # save a chunk of the first batch for the grad chunk
            if x_grad is None:
                x_grad, x = x[:grad_size], x[grad_size:]
                if x.numel() == 0:
                    continue

            # if the grad chunk was bigger than the first set size then this
            # will be zero so skip it to avoid an error
            if x.size(0) > 0:
                seed(self.pre_sampled_seeds[i])
                x_features, x_t, c_t = self.forward_mbc(
                    x.cuda(),
                    X_prev=x_t,
                    c_prev=c_t,
                    grad=False,
                    augmentation=augmentation,
                )
                x_ft.append(x_features)

        assert isinstance(x_grad, T)
        seed(self.pre_sampled_seeds[0])

        x_features, x_t, c_t = self.forward_mbc(
            x_grad.cuda(),
            X_prev=x_t,
            c_prev=c_t,
            grad=not test,
            augmentation=augmentation,
        )

        x_ft.append(x_features)

        seed(self.pre_sampled_seeds[0])
        x_out = self.post_forward_mbc(x_t, c=c_t)  # type: ignore

        xft = torch.cat(x_ft, dim=1)
        seed(self.pre_sampled_seeds[0])
        inst_out, bag_out = self.decoder(xft, x_out)
        return inst_out.view(-1), bag_out.view(-1)  # type: ignore


def get_extractor(resnet: bool) -> nn.Module:
    if resnet:
        return ResNetSimCLR()
    return IdentityExtractor()


class DeepSetsMILDecoder(nn.Module):
    def __init__(
        self, inst_in: int, bag_in: int, h_dim: int, out_dim: int, n_layers: int
    ) -> None:
        super().__init__()
        self.inst = nn.Linear(inst_in, out_dim)
        self.squeezer = Squeezer()

        bag = []
        for i in range(n_layers):
            d = bag_in if i == 0 else h_dim
            bag.extend([nn.Linear(d, h_dim), nn.ReLU()])
        bag.append(nn.Linear(h_dim, out_dim))
        self.bag = nn.Sequential(*bag)

    def forward(self, x_ft: T, p_ft: T) -> Tuple[T, T]:
        x_ft, p_ft = self.squeezer(x_ft), self.squeezer(p_ft)
        return self.inst(x_ft), self.bag(p_ft)


class CamelyonDeepSets(CamelyonMBC, MBCFunction):  # type: ignore
    def __init__(
        self,
        out_dim: int,
        n_decoder_layers: int,
        h_dim: int = 128,
        pool: str = "max",
        use_resnet: bool = True,
    ) -> None:
        super().__init__()
        self.name = "DeepSets"
        self.is_mbc = True
        self.n_decoder_layers = n_decoder_layers

        self.set_hashable_attrs(["n_decoder_layers"])

        self.extractor = get_extractor(resnet=use_resnet)
        self.mbc_pooler = DeepSetsPooler(h_dim=FEAT_DIM, pool=pool)

        self.decoder = DeepSetsMILDecoder(
            FEAT_DIM, FEAT_DIM, h_dim, out_dim, n_decoder_layers
        )


class Squeezer(nn.Module):
    def forward(self, x: T) -> T:
        if len(x.size()) == 2:
            return x
        if len(x.size()) == 3 and x.size(0) == 1:
            return x.squeeze(0)

        raise ValueError(f"got unknown x size: {x.size()}")


class CamelyonSSE(CamelyonMBC, MBCFunction):  # type: ignore
    def __init__(
        self,
        h: int,
        d: int,
        d_hat: int,
        out_dim: int,
        n_decoder_layers: int,
        slot_type: str = "random",
        ln_slots: bool = False,
        heads: int = 4,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        n_parallel: int = 1,
        ln_after: bool = True,
        use_resnet: bool = True,
    ) -> None:
        super().__init__()
        self.name = "SSE"
        self.is_mbc = True

        self.n_decoder_layers = n_decoder_layers
        self.set_hashable_attrs(["n_decoder_layers"])

        self.extractor = get_extractor(resnet=use_resnet)
        self.mbc_pooler = SlotSetEncoder(
            K=1,
            h=h,
            d=d,
            d_hat=d_hat,
            slot_type=slot_type,
            ln_slots=ln_slots,
            heads=heads,
            slot_drop=slot_drop,
            attn_act=attn_act,
            slot_residual=slot_residual,
            ln_after=ln_after,
        )

        self.decoder = DeepSetsMILDecoder(
            FEAT_DIM, d_hat, d_hat, out_dim, n_decoder_layers
        )


class UMBCMILDecoder(nn.Module):
    def __init__(
        self,
        inst_in: int,
        bag_in: int,
        h_dim: int,
        out_dim: int = 1,
        heads: int = 4,
        n_layers: int = 1,
        isab_ind: int = -1,
    ) -> None:
        super().__init__()
        self.inst = nn.Linear(inst_in, out_dim)
        self.squeezer = Squeezer()

        sab: List[nn.Module] = []
        for i in range(n_layers - 1):
            if isab_ind > 0:
                sab.append(
                    ISAB(
                        bag_in if i == 0 else h_dim,
                        h_dim,
                        num_heads=heads,
                        num_inds=1,
                        ln=True,
                    )
                )
                continue

            sab.append(
                SAB(bag_in if i == 0 else h_dim, h_dim, num_heads=heads, ln=True)
            )

        self.bag = nn.Sequential(
            *sab,
            nn.Dropout(),
            PMA(h_dim, num_heads=heads, num_seeds=1, ln=True),
            nn.Dropout(),
            nn.Linear(h_dim, out_dim),
        )

    def forward(self, x_ft: T, p_ft: T) -> Tuple[T, T]:
        x_ft = self.squeezer(x_ft)
        return self.inst(x_ft), self.squeezer(self.bag(p_ft))


class CamelyonSSEUMBC(CamelyonMBC, MBCFunction):  # type: ignore
    def __init__(
        self,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        out_dim: int,
        n_decoder_layers: int,
        slot_type: str = "random",
        ln_slots: bool = False,
        heads: int = 4,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        n_parallel: int = 1,
        ln_after: bool = True,
        use_resnet: bool = True,
    ) -> None:
        super().__init__()
        self.name = "SSEUMBC"
        self.is_mbc = True

        self.n_decoder_layers = n_decoder_layers
        self.set_hashable_attrs(["n_decoder_layers"])

        self.extractor = get_extractor(resnet=use_resnet)
        self.mbc_pooler = SlotSetEncoder(
            K=K,
            h=h,
            d=d,
            d_hat=d_hat,
            slot_type=slot_type,
            ln_slots=ln_slots,
            heads=heads,
            slot_drop=slot_drop,
            attn_act=attn_act,
            slot_residual=slot_residual,
            ln_after=ln_after,
        )

        self.decoder = UMBCMILDecoder(
            FEAT_DIM,
            d_hat,
            d_hat,
            out_dim,
            heads,
            n_decoder_layers,
        )


class FSPoolMILDecoder(nn.Module):
    def __init__(
        self,
        inst_in: int,
        bag_in: int,
        h_dim: int,
        out_dim: int = 1,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.inst = nn.Linear(inst_in, out_dim)
        self.squeezer = Squeezer()

        dec: List[nn.Module] = [FSPool(h_dim, n_pieces=20)]
        for i in range(n_layers):
            dec.extend([nn.Linear(h_dim, h_dim), nn.ReLU()])

        self.bag = nn.Sequential(*dec, nn.Linear(h_dim, out_dim))

    def forward(self, x_ft: T, p_ft: T) -> Tuple[T, T]:
        x_ft = self.squeezer(x_ft)
        return self.inst(x_ft), self.squeezer(self.bag(p_ft))


class CamelyonFSPoolUMBC(CamelyonMBC, MBCFunction):  # type: ignore
    def __init__(
        self,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        out_dim: int,
        n_decoder_layers: int,
        slot_type: str = "random",
        ln_slots: bool = False,
        heads: int = 4,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        n_parallel: int = 1,
        ln_after: bool = True,
        use_resnet: bool = True,
    ) -> None:
        super().__init__()
        self.name = "FSPoolUMBC"
        self.is_mbc = True

        self.n_decoder_layers = n_decoder_layers
        self.set_hashable_attrs(["n_decoder_layers"])

        self.extractor = get_extractor(resnet=use_resnet)
        self.mbc_pooler = SlotSetEncoder(
            K=K,
            h=h,
            d=d,
            d_hat=d_hat,
            slot_type=slot_type,
            ln_slots=ln_slots,
            heads=heads,
            slot_drop=slot_drop,
            attn_act=attn_act,
            slot_residual=slot_residual,
            ln_after=ln_after,
        )

        self.decoder = FSPoolMILDecoder(
            FEAT_DIM,
            d_hat,
            d_hat,
            out_dim,
            n_decoder_layers,
        )


class CamelyonABMIL(nn.Module):
    def __init__(
        self,
        out_dim: int,
        h_dim: int = 128,
        use_resnet: bool = True,
    ) -> None:
        super().__init__()
        self.name = "ABMIL"
        self.h_dim = h_dim
        self.is_mbc = False

        self.extractor = get_extractor(resnet=use_resnet)
        self.decoder = nn.Sequential(
            Squeezer(), nn.Dropout(), GatedAttention(FEAT_DIM, h_dim, 1)
        )

    def forward(self, x: T) -> Tuple[T, T]:
        x = self.extractor(x)
        inst, out = self.decoder(x)
        return inst, out


class CamelyonDSMIL(nn.Module):
    def __init__(
        self,
        out_dim: int,
        h_dim: int = 128,
        use_resnet: bool = True,
    ) -> None:
        super().__init__()
        self.name = "DSMIL"
        self.h_dim = h_dim
        self.is_mbc = False

        self.extractor = get_extractor(resnet=use_resnet)
        self.decoder = nn.Sequential(
            nn.Dropout(),
            MILNet(
                IClassifier(Squeezer(), FEAT_DIM, out_dim),
                BClassifier(FEAT_DIM, out_dim),
            ),
        )

    def forward(self, x: T) -> Tuple[T, T]:
        x = self.extractor(x)
        inst, out = self.decoder(x)
        return inst, out
