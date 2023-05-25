from typing import Any, List, Optional, Tuple

import torch
from base import HashableModule
from torch import nn

from umbc.models.diem import layers
from umbc.models.diem.mog_models import RFFNet
from umbc.models.layers.bigset import (OT, MBCExtractAndPool,
                                             MBCExtractor, MBCFunction, T)
from umbc.models.layers.deepsets import DeepSetsPooler
from umbc.models.layers.fspool import FSPool
from umbc.models.layers.linear import (Resizer, get_activation,
                                             get_linear_layer)
from umbc.models.layers.set_xformer import ISAB, PMA, SAB
from umbc.models.layers.sse import SlotSetEncoder


class MVNMBCSetFunction(MBCFunction):
    encoder: MBCExtractAndPool
    decoder: nn.Module

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        self.encoder.pre_forward_mbc()

        x_t, c_t = self.encoder.forward_mbc(X, grad=True)
        x_t, c_t = self.encoder.forward_mbc(
            X_nograd, X_prev=x_t, c_prev=c_t, grad=False)

        x = self.encoder.post_forward_mbc(x_t, c=c_t)

        # x = self.encoder.partitioned_forward(X, X_nograd)
        return self.decoder(x)  # type: ignore

    def forward(self, X: T) -> T:
        x = self.encoder(X)
        return self.decoder(x)  # type: ignore

    def grad_correct(self, c: float) -> None:
        self.encoder.grad_correct(c)

    def pre_forward_mbc(self) -> None:
        self.encoder.pre_forward_mbc()

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None,
    ) -> Tuple[T, OT]:
        x_t, c_t = self.encoder.forward_mbc(
            X, X_prev=X_prev,
            c_prev=c_prev, grad=grad, mask=mask
        )
        return x_t, c_t

    def post_forward_mbc(
        self,
        X: T,
        c: OT = None,
        mask: OT = None,
    ) -> T:
        return self.encoder.post_forward_mbc(X=X, c=c, mask=mask)


class MVNDeepSets(MVNMBCSetFunction):
    def __init__(
        self,
        K: int,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_extractor_layers: int,
        n_decoder_layers: int,
        linear_type: str = "Linear",
        activation: str = "relu",
        pool: str = "mean"
    ) -> None:
        super().__init__()
        self.name = "DeepSets"
        self.n_extractor_layers = n_extractor_layers
        self.n_decoder_layers = n_decoder_layers
        self.is_mbc = True
        self.K = K
        self.activation = activation

        self.set_hashable_attrs(
            ["n_extractor_layers", "n_decoder_layers", "K", "activation"])

        self.encoder = MBCExtractAndPool(
            mbc_extractor=MBCExtractor(
                n_layers=n_extractor_layers,
                in_dim=in_dim,
                h_dim=h_dim,
                out_dim=h_dim,
                activation=activation
            ),
            mbc_pooler=DeepSetsPooler(h_dim=h_dim, pool=pool)
        )

        linear = get_linear_layer(linear_type)
        act = get_activation(activation)

        dec = []
        for i in range(n_decoder_layers):
            dec.extend([linear(h_dim, h_dim), act()])
        dec.extend([linear(h_dim, out_dim * K), Resizer(K, out_dim)])
        self.decoder = nn.Sequential(*dec)


class MVNSSEUMBC(MVNMBCSetFunction):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_extractor_layers: int,
        n_decoder_layers: int,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        slot_type: str = "random",
        ln_slots: bool = False,
        heads: int = 4,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        extractor_lin_act: str = "relu"
    ) -> None:
        super().__init__()
        self.name = "SSEUMBC"
        self.is_mbc = True
        self.n_extractor_layers = n_extractor_layers
        self.n_decoder_layers = n_decoder_layers
        self.set_hashable_attrs(["n_extractor_layers", "n_decoder_layers"])

        self.encoder = MBCExtractAndPool(
            mbc_extractor=MBCExtractor(
                n_layers=n_extractor_layers,
                in_dim=in_dim,
                h_dim=h_dim,
                out_dim=h_dim,
                activation=extractor_lin_act
            ),
            mbc_pooler=SlotSetEncoder(
                K=K, h=h, d=d, d_hat=d_hat,
                slot_type=slot_type, ln_slots=ln_slots,
                heads=heads, slot_drop=slot_drop,
                attn_act=attn_act, slot_residual=slot_residual,
                eps=1e-8
            )
        )

        linear = get_linear_layer("Linear")
        self.decoder = nn.Sequential(
            *[SAB(h_dim, h_dim, num_heads=heads, ln=False)
              for _ in range(n_decoder_layers)],
            linear(h_dim, out_dim)
        )


class MVNDiffEMUMBC(MVNMBCSetFunction):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_extractor_layers: int,
        n_decoder_layers: int,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        slot_type: str = "random",
        ln_slots: bool = False,
        heads: int = 4,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        extractor_lin_act: str = "relu",
        # everything below here will be used for diffem
        diffem_heads: int = 3,
        out_type: str = 'select_best2',
        num_proto: int = 5,
        num_ems: int = 2,
        tau: float = 10.0,
    ) -> None:
        super().__init__()
        self.name = "DiffEMUMBC"
        self.is_mbc = True
        self.n_extractor_layers = n_extractor_layers
        self.n_decoder_layers = n_decoder_layers
        self.set_hashable_attrs(["n_extractor_layers", "n_decoder_layers"])

        self.encoder = MBCExtractAndPool(
            mbc_extractor=MBCExtractor(
                n_layers=n_extractor_layers,
                in_dim=in_dim,
                h_dim=h_dim,
                out_dim=h_dim,
                activation=extractor_lin_act
            ),
            mbc_pooler=SlotSetEncoder(
                K=K, h=h, d=d, d_hat=d_hat,
                slot_type=slot_type, ln_slots=ln_slots,
                heads=heads, slot_drop=slot_drop,
                attn_act=attn_act, slot_residual=slot_residual,
                eps=1e-8
            )
        )

        linear = get_linear_layer("Linear")
        act = get_activation(extractor_lin_act)

        diem = layers.DIEM(
            h_dim, H=diffem_heads, p=num_proto, L=num_ems,
            tau=tau, out=out_type, distr_emb_args=None
        )

        dec = []
        for i in range(n_decoder_layers - 1):
            dim = diem.outdim if i == 0 else h_dim
            dec.extend([linear(dim, h_dim), act()])

        self.decoder = nn.Sequential(
            diem,
            *dec,
            linear(h_dim, out_dim * K),
            Resizer(K, out_dim)
        )


class MVNFSPoolUMBC(MVNMBCSetFunction):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_extractor_layers: int,
        n_decoder_layers: int,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        slot_type: str = "random",
        ln_slots: bool = False,
        heads: int = 4,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        extractor_lin_act: str = "relu"
    ) -> None:
        super().__init__()
        self.name = "FSPoolUMBC"
        self.is_mbc = True
        self.n_extractor_layers = n_extractor_layers
        self.n_decoder_layers = n_decoder_layers
        self.set_hashable_attrs(["n_extractor_layers", "n_decoder_layers"])

        self.encoder = MBCExtractAndPool(
            mbc_extractor=MBCExtractor(
                n_layers=n_extractor_layers,
                in_dim=in_dim,
                h_dim=h_dim,
                out_dim=h_dim,
                activation=extractor_lin_act
            ),
            mbc_pooler=SlotSetEncoder(
                K=K, h=h, d=d, d_hat=d_hat,
                slot_type=slot_type, ln_slots=ln_slots,
                heads=heads, slot_drop=slot_drop,
                attn_act=attn_act, slot_residual=slot_residual,
                eps=1e-8
            )
        )

        linear = get_linear_layer("Linear")
        act = get_activation(extractor_lin_act)

        dec = []
        for i in range(n_decoder_layers - 1):
            dec.extend([linear(h_dim, h_dim), act()])

        self.decoder = nn.Sequential(
            FSPool(h_dim, n_pieces=20),
            *dec,
            linear(h_dim, out_dim * K),
            Resizer(K, out_dim)
        )


class MVNSSE(MVNMBCSetFunction):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_extractor_layers: int,
        n_decoder_layers: int,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        slot_type: str = "random",
        ln_slots: bool = False,
        heads: int = 4,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        linear_type: str = "Linear",
        activation: str = "relu"
    ) -> None:
        super().__init__()
        self.name = "SSE"
        self.is_mbc = True
        self.n_extractor_layers = n_extractor_layers
        self.n_decoder_layers = n_decoder_layers
        self.set_hashable_attrs(["n_extractor_layers", "n_decoder_layers"])

        self.encoder = MBCExtractAndPool(
            mbc_extractor=MBCExtractor(
                n_layers=n_extractor_layers,
                in_dim=in_dim,
                h_dim=h_dim,
                out_dim=h_dim,
                activation=activation,
            ),
            mbc_pooler=SlotSetEncoder(
                K=K, h=h, d=d, d_hat=d_hat,
                slot_type=slot_type, ln_slots=ln_slots,
                heads=heads, slot_drop=slot_drop,
                attn_act=attn_act, slot_residual=slot_residual,
                eps=1e-8
            )
        )

        linear = get_linear_layer(linear_type)
        act = get_activation(activation)

        dec = []
        for i in range(n_decoder_layers):
            dec.extend([linear(h_dim, h_dim), act()])
        dec.extend([linear(h_dim, out_dim)])
        self.decoder = nn.Sequential(*dec)


class MVNSSEHierarchical(MVNMBCSetFunction):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_extractor_layers: int,
        n_decoder_layers: int,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        slot_type: str = "random",
        ln_slots: bool = False,
        heads: int = 4,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        linear_type: str = "Linear",
        activation: str = "relu"
    ) -> None:
        super().__init__()
        self.name = "SSEHierarchical"
        self.is_mbc = True
        self.n_extractor_layers = n_extractor_layers
        self.n_decoder_layers = n_decoder_layers
        self.set_hashable_attrs(["n_extractor_layers", "n_decoder_layers"])

        self.encoder = MBCExtractAndPool(
            mbc_extractor=MBCExtractor(
                n_layers=n_extractor_layers,
                in_dim=in_dim,
                h_dim=h_dim,
                out_dim=h_dim,
                activation=activation,
            ),
            mbc_pooler=SlotSetEncoder(
                K=K, h=h, d=d, d_hat=d_hat,
                slot_type=slot_type, ln_slots=ln_slots,
                heads=heads, slot_drop=slot_drop,
                attn_act=attn_act, slot_residual=slot_residual,
                eps=1e-8
            )
        )

        linear = get_linear_layer(linear_type)

        dec: List[Any] = []
        for i in range(n_decoder_layers):
            dec.extend([
                SlotSetEncoder(
                    K=K, h=h, d=d, d_hat=d_hat,
                    slot_type=slot_type, ln_slots=ln_slots,
                    heads=heads, slot_drop=slot_drop,
                    attn_act=attn_act, slot_residual=slot_residual,
                    eps=1e-8
                )
            ])

        dec.extend([linear(h_dim, out_dim)])
        self.decoder = nn.Sequential(*dec)


class MVNSetTransformer(HashableModule):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_extractor_layers: int,
        n_decoder_layers: int,
        K: int,
        num_heads: int = 4,
        num_inds: int = 16,
        ln: bool = True,
        isab_enc: bool = True,
    ) -> None:
        super().__init__()
        # the original repo did not use layernorm in their experiments,
        # but we could not reproduce their results without it so we included
        # it here

        self.name = "SetTransformer"
        self.is_mbc = False
        self.K = K
        self.num_heads = num_heads
        self.temp = torch.Tensor()
        self.temp_count = 0

        x: List[nn.Module] = []
        for i in range(n_extractor_layers):
            if isab_enc:
                isab = ISAB(
                    dim_in=in_dim if i == 0 else h_dim,
                    dim_out=h_dim,
                    num_heads=num_heads,
                    num_inds=num_inds,
                    ln=ln
                )

                x.append(isab)
                continue

            x.extend([
                nn.Linear(in_dim if i == 0 else h_dim, h_dim),
                nn.ReLU()
            ])

        e = PMA(dim=h_dim, num_heads=num_heads, num_seeds=K, ln=ln)

        d: List[nn.Module] = []
        for i in range(n_decoder_layers):
            d.append(SAB(
                dim_in=h_dim,
                dim_out=h_dim,
                num_heads=num_heads,
                ln=ln)
            )
        d.append(nn.Linear(h_dim, out_dim))

        self.encoder = nn.Sequential(*x, e)
        self.decoder = nn.Sequential(*d)

    def forward(self, x: T) -> T:
        x = self.encoder(x)  # type: ignore
        x = self.decoder(x)  # type: ignore
        return x

    def pre_forward_mbc(self) -> None:
        pass

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None,
    ) -> Tuple[T, OT]:
        with torch.set_grad_enabled(grad):
            x_t = self.encoder(X)

        return (
            X_prev + x_t if X_prev is not None else x_t,
            c_prev + 1 if c_prev is not None else torch.tensor(1)
        )

    def post_forward_mbc(
        self,
        X: T,
        c: OT = None,
        mask: OT = None,
    ) -> T:
        if c is None:
            raise ValueError("c is required for Set Transformer post forward")
        return X / (c + 1e-8)


class MVNDiffEM(HashableModule):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_extractor_layers: int,
        n_decoder_layers: int,
        K: int,
        num_heads: int = 3,
        out_type: str = 'select_best2',
        num_proto: int = 5,
        num_ems: int = 2,
        tau: float = 10.0,
        activation: str = "relu",
        linear_type: str = "Linear",
    ):
        super().__init__()
        self.name = "DiffEM"
        self.n_extractor_layers = n_extractor_layers
        self.n_decoder_layers = n_decoder_layers
        self.is_mbc = False
        self.K = K
        self.activation = activation

        self.set_hashable_attrs(
            ["n_extractor_layers", "n_decoder_layers", "K", "activation"])

        self.rff = RFFNet(in_dim, h_dim, n_layers=n_extractor_layers)
        self.diem = layers.DIEM(
            h_dim, H=num_heads, p=num_proto, L=num_ems,
            tau=tau, out=out_type, distr_emb_args=None
        )

        linear = get_linear_layer(linear_type)
        act = get_activation(activation)

        dec = []
        for i in range(n_decoder_layers):
            d = self.diem.outdim if i == 0 else h_dim
            dec.extend([linear(d, h_dim), act()])
        dec.extend([linear(h_dim, out_dim * K), Resizer(K, out_dim)])
        self.decoder = nn.Sequential(*dec)

    def forward(self, X: T, cards: Any = None) -> T:
        B, N_max, d0 = X.shape
        S = self.rff(X)
        mask = torch.ones(B, N_max).to(S)
        if cards is not None:
            for ii in range(B):
                mask[ii][cards[ii]:] = 0.0

        FS = self.diem(S, mask)
        return self.decoder(FS)  # type: ignore

    def pre_forward_mbc(self) -> None:
        pass

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None,
    ) -> Tuple[T, OT]:
        with torch.set_grad_enabled(grad):
            x = self.rff(X)
            x_t = self.diem(x)

        return (
            X_prev + x_t if X_prev is not None else x_t,
            c_prev + 1 if c_prev is not None else torch.tensor(1)
        )

    def post_forward_mbc(
        self,
        X: T,
        c: OT = None,
        mask: OT = None,
    ) -> T:
        if c is None:
            raise ValueError("c is required for Diff EM. post forward")
        return X / (c + 1e-8)


class MVNFSPool(HashableModule):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_extractor_layers: int,
        n_decoder_layers: int,
        K: int,
        activation: str = "relu",
        linear_type: str = "Linear",
    ):
        super().__init__()
        self.name = "FSPool"
        self.n_extractor_layers = n_extractor_layers
        self.n_decoder_layers = n_decoder_layers
        self.is_mbc = False
        self.K = K
        self.activation = activation

        self.set_hashable_attrs(
            ["n_extractor_layers", "n_decoder_layers", "K", "activation"])

        self.rff = RFFNet(in_dim, h_dim, n_layers=n_extractor_layers)
        self.fspool = FSPool(h_dim, n_pieces=20)

        linear = get_linear_layer(linear_type)
        act = get_activation(activation)

        dec = []
        for i in range(n_decoder_layers):
            dec.extend([linear(h_dim, h_dim), act()])
        dec.extend([linear(h_dim, out_dim * K), Resizer(K, out_dim)])
        self.decoder = nn.Sequential(*dec)

    def forward(self, X: T, cards: Any = None) -> T:
        S = self.rff(X)
        FS = self.fspool(S)
        return self.decoder(FS)  # type: ignore

    def pre_forward_mbc(self) -> None:
        pass

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None,
    ) -> Tuple[T, OT]:
        with torch.set_grad_enabled(grad):
            x = self.rff(X)
            x_t = self.fspool(x)

        return (
            X_prev + x_t if X_prev is not None else x_t,
            c_prev + 1 if c_prev is not None else torch.tensor(1)
        )

    def post_forward_mbc(
        self,
        X: T,
        c: OT = None,
        mask: OT = None,
    ) -> T:
        if c is None:
            raise ValueError("c is required for Diff EM. post forward")
        return X / (c + 1e-8)
