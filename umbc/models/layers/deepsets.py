from typing import Any, Optional, Tuple

import torch
from torch import nn

from umbc.models.layers.bigset import MBCFunction

T = torch.Tensor
OT = Optional[T]

pool_funcs = {
    "min": torch.amin,
    "max": torch.amax,
    "mean": torch.sum,
    "sum": torch.sum
}

# mask values gives the values which will negate the masked instances during
# the pooling operation
mask_values = {
    "min": float("inf"),
    "max": float("-inf"),
    "mean": 0.0,
    "sum": 0.0
}


class DeepSetsPooler(MBCFunction):
    def __init__(self, h_dim: int, pool: str = "max"):
        super().__init__()
        self.name = f"DeepSetsPooler-{pool}"
        self.pool = pool
        self.h_dim = h_dim
        self.set_hashable_attrs(["pool", "h_dim"])

        # this descriptive name is needed to catch a special ignore
        # case in testing. We ignore the weight for gradient
        # correction on this becasue it comes after pooling
        self.norm_after = nn.LayerNorm(h_dim)

        self.pool_func: Any = pool_funcs[pool]
        self.mask_value = mask_values[pool]
        self.hook: Any

    def pre_forward_mbc(self) -> None:
        pass

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None,
        # detach_c: bool = False
    ) -> Tuple[T, T]:
        b, c, d = X.size()
        if mask is not None:
            # unsqueeze to let the last dimension project to whatever
            # the data dimension is. after masked_fill, the tensor
            # should be (Batch, Set, 1) in [0, 1]
            mask = mask.unsqueeze(-1)
            c = c - (mask == 0).sum(dim=1, keepdim=True)
            X = X.masked_fill(mask == 0, self.mask_value)

        # call the pooling on this chunk with grad conitionally enabled
        with torch.set_grad_enabled(grad):
            # this is for calculating the memory usage of stored activations
            # during training. Every non-module operation should store the
            # input and output
            if hasattr(self, "hook"):
                pooled = self.pool_func(X, dim=1, keepdim=True)
                self.hook(self, [X], pooled)

            x = self.pool_func(X, dim=1, keepdim=True)

        # if given a previous chunk, then concatenate and pool.
        # this should maintain any previously existing gradient
        if X_prev is not None:
            if hasattr(self, "hook"):
                catted = torch.cat((x, X_prev), dim=1)
                pooled = self.pool_func(catted, dim=1, keepdim=True)
                self.hook(self, [catted], pooled)

            x = self.pool_func(torch.cat((x, X_prev), dim=1),
                               dim=1, keepdim=True)

        if self.pool == "mean":
            # there is no need to detach the c here becuase c is
            # just an integer
            c_prev = c + c_prev if c_prev is not None else c  # type: ignore

        return x, c_prev  # type: ignore

    def post_forward_mbc(
        self,
        X: T,
        c: OT = None,
        mask: OT = None
    ) -> T:
        if c is not None:
            # this is for calculating the memory usage of stored activations
            # during training. Every non-module operation should store the
            # input and output
            if hasattr(self, "hook"):
                self.hook(self, [X], X / c)

            return self.norm_after(X / c)  # type: ignore
        return self.norm_after(X)  # type: ignore

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """
        for training the model where x_nograd has no gradient
        in the base layer.
        """
        b, s, d = X.size()
        b_nograd, s_nograd, d_nograd = X_nograd.size()

        x = self.pool_func(X, dim=1, keepdim=True)
        with torch.no_grad():
            x_nograd = self.pool_func(X_nograd, dim=1, keepdim=True)

        x = self.pool_func(torch.cat((x, x_nograd), dim=1),
                           dim=1, keepdim=True)

        if self.pool == "mean":
            x = x / (s + s_nograd)  # type: ignore

        return self.norm_after(x)  # type: ignore

    def forward(self, X: T) -> T:
        b, s, d = X.size()

        x = self.pool_func(X, dim=1, keepdim=True)
        if self.pool == "mean":
            x = x / s  # type: ignore
        return self.norm_after(x)  # type: ignore

    def grad_correct(self, c: float) -> None:
        # there are no parameters to correct here. layernorm parameters come
        # after pooling, so they can be ignored
        pass
