from typing import Optional, Type

import torch
import torch.nn as nn
from base import HashableModule

T = torch.Tensor


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name = "id"

    def forward(self, x: T) -> T:
        return x


class BMM(nn.Module):
    def __init__(self) -> None:
        super(BMM, self).__init__()
        self.A_m = 0
        self.A_n = 0
        self.B_p = 0

    def forward(self, A: T, B: T) -> T:
        if not self.training:
            if self.A_m == 0:
                _, self.A_m, self.A_n = A.size()
                _, _, self.B_p = B.size()

        return torch.bmm(A, B)

    def flops(self) -> float:
        return 2 * self.A_m * self.A_n * self.B_p


def clip_grad(model: nn.Module, max_norm: int) -> float:
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2

    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)

    return total_norm


class ResFF(HashableModule):
    def __init__(self, d: int, p: float = 0.0, activation: str = "relu"):
        super().__init__()
        self.name = "ResFF"
        self.p = p
        self.activation = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU
        }[activation]

        self.layer = nn.Sequential(
            nn.Linear(d, d),
            self.activation(),
            nn.Dropout(p=p),
        )

        self.set_hashable_attrs(["p"])

    def forward(self, x: T) -> T:
        return x + self.layer(x)  # type: ignore


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.name = "Linear"
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features, bias=bias)

    def forward(self, x: T) -> T:
        return self.linear(x)  # type: ignore

    def flops(self) -> float:
        # NOTE: We ignore activation funcitons.
        MAC = self.out_features * self.in_features
        ADD = 0
        if self.bias:
            ADD = self.out_features
        flops = 2 * MAC + ADD
        return flops


class PermEquiMax(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.Gamma = Linear(in_dim, out_dim)

    def forward(self, x: T) -> T:
        xm, _ = x.max(1, keepdim=True)
        x = self.Gamma(x - xm)
        return x


class PermEquiMean(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.Gamma = Linear(in_dim, out_dim)

    def forward(self, x: T) -> T:
        xm = x.mean(1, keepdim=True)
        x = self.Gamma(x - xm)
        return x


class PermEquiMax2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.Gamma = Linear(in_dim, out_dim)
        self.Lambda = Linear(in_dim, out_dim, bias=False)

    def forward(self, x: T) -> T:
        xm, _ = x.max(1, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x


class PermEquiMean2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.Gamma = Linear(in_dim, out_dim)
        self.Lambda = Linear(in_dim, out_dim, bias=False)

    def forward(self, x: T) -> T:
        xm = x.mean(1, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x


class Unsqueezer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: T) -> T:
        return x.unsqueeze(self.dim)


class Resizer(nn.Module):
    def __init__(self, dim_one: int, dim_two: int) -> None:
        super().__init__()
        self.one = dim_one
        self.two = dim_two

    def forward(self, x: T) -> T:
        return x.view(-1, self.one, self.two)


_layers = {
    "PermEquiMax": PermEquiMax,
    "PermEquiMean": PermEquiMean,
    "PermEquiMax2": PermEquiMax2,
    "PermEquiMean2": PermEquiMean2,
    "Linear": Linear,
}


def get_linear_layer(typ: str) -> Type[nn.Linear]:
    return _layers[typ]


_acts = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swich": nn.SiLU,
}


def get_activation(typ: str) -> Type[nn.Module]:
    return _acts[typ]
