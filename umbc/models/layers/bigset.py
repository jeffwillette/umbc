from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from base import HashableModule
from torch import nn

from umbc.models.layers.linear import get_activation, get_linear_layer

T = torch.Tensor
OT = Optional[T]

LT = List[T]
OLT = Optional[LT]


class GradCorrecter(nn.Module):
    def register_grad_correct_hooks(
        self,
        grad_size: int,
        set_size: int
    ) -> Any:

        def backward_hook(g: T) -> T:
            return (set_size / grad_size) * g

        handles = []
        for n, p in self.named_parameters():
            if "norm_after" not in n:
                h = p.register_hook(backward_hook)
                handles.append(h)

        def remove() -> None:
            [h.remove() for h in handles]

        return remove


class MBCFunction(HashableModule, GradCorrecter):
    norm_after: nn.Module

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """for training the model where x_nograd has no gradient."""
        raise NotImplementedError()

    def pre_forward_mbc(self) -> None:
        """
        for pre processing the minibatched forward pooled
        vector and normalization constant
        """
        raise NotImplementedError()

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None
    ) -> Tuple[T, OT]:
        """
        for training the model with minibatches.
        X_prev: is the previous state at timestep (t-1)
        grad: whether the gradient is computed on this chunk.
        Returns: The current state at timestep (t)
        """
        raise NotImplementedError()

    def post_forward_mbc(self, X: T, c: OT = None, mask: OT = None) -> T:
        """
        for post processing the minibatched forward pooled
        vector and normalization constant
        """
        raise NotImplementedError()

    def forward(self, X: T) -> T:
        raise NotImplementedError()

    def grad_correct(self, c: float) -> None:
        raise NotImplementedError()


class MBCExtractAndPool(MBCFunction, GradCorrecter):
    def __init__(
        self,
        mbc_extractor: MBCFunction,
        mbc_pooler: MBCFunction
    ):
        super().__init__()
        self.mbc_extractor = mbc_extractor
        self.mbc_pooler = mbc_pooler
        self.name = f"bigset-{mbc_pooler.name}"

    def pre_forward_mbc(self) -> None:
        self.mbc_extractor.pre_forward_mbc()
        self.mbc_pooler.pre_forward_mbc()

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None
    ) -> Tuple[T, OT]:
        x, _ = self.mbc_extractor.forward_mbc(X, grad=grad, mask=mask)
        x, c = self.mbc_pooler.forward_mbc(
            X=x, X_prev=X_prev, c_prev=c_prev, grad=grad, mask=mask)

        return x, c

    def post_forward_mbc(
        self,
        X: T,
        c: OT = None,
        mask: OT = None
    ) -> T:
        return self.mbc_pooler.post_forward_mbc(
            X=X, c=c, mask=mask)

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """for training the model where x_nograd has no gradient."""
        b, s, d = X.size()

        x = self.mbc_extractor.partitioned_forward(X, X_nograd)
        x = self.mbc_pooler.partitioned_forward(x[:, :s], x[:, s:])
        return x

    def forward(self, X: T) -> T:
        x = self.mbc_extractor(X)
        x = self.mbc_pooler(x)
        return x  # type: ignore

    def grad_correct(self, c: float) -> None:
        self.mbc_extractor.grad_correct(c)
        self.mbc_pooler.grad_correct(c)


class ParallelMBCFunction(HashableModule):
    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """for training the model where x_nograd has no gradient."""
        raise NotImplementedError()

    def pre_forward_mbc(self) -> None:
        """
        for pre processing the minibatched forward pooled
        vector and normalization constant
        """
        raise NotImplementedError()

    def forward_mbc(
        self,
        X: T,
        X_prev: OLT = None,
        c_prev: OLT = None,
        grad: bool = True,
        mask: OT = None
    ) -> Tuple[LT, OLT]:
        """
        for training the model with minibatches.
        X_prev: is the previous state at timestep (t-1)
        grad: whether the gradient is computed on this chunk.
        Returns: The current state at timestep (t)
        """
        raise NotImplementedError()

    def post_forward_mbc(self, X: LT, c: OLT = None, mask: OT = None) -> T:
        """
        for post processing the minibatched forward pooled
        vector and normalization constant
        """
        raise NotImplementedError()

    def forward(self, X: T) -> T:
        raise NotImplementedError()

    def grad_correct(self, c: float) -> None:
        raise NotImplementedError()


class ParallelMBCExtractAndPool(ParallelMBCFunction, GradCorrecter):
    def __init__(
        self,
        mbc_extractor: MBCFunction,
        mbc_pooler: ParallelMBCFunction
    ):
        super().__init__()
        self.mbc_extractor = mbc_extractor
        self.mbc_pooler = mbc_pooler
        self.name = f"bigset-{mbc_pooler.name}"

    def pre_forward_mbc(self) -> None:
        self.mbc_extractor.pre_forward_mbc()
        self.mbc_pooler.pre_forward_mbc()

    def forward_mbc(
        self,
        X: T,
        X_prev: OLT = None,
        c_prev: OLT = None,
        grad: bool = True,
        mask: OT = None
    ) -> Tuple[LT, OLT]:
        x, _ = self.mbc_extractor.forward_mbc(X, grad=grad, mask=mask)
        xx, c = self.mbc_pooler.forward_mbc(
            X=x, X_prev=X_prev, c_prev=c_prev, grad=grad, mask=mask)

        return xx, c

    def post_forward_mbc(
        self,
        X: LT,
        c: OLT = None,
        mask: OT = None
    ) -> T:
        return self.mbc_pooler.post_forward_mbc(
            X=X, c=c, mask=mask)

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """for training the model where x_nograd has no gradient."""
        b, s, d = X.size()

        x = self.mbc_extractor.partitioned_forward(X, X_nograd)
        x = self.mbc_pooler.partitioned_forward(x[:, :s], x[:, s:])
        return x

    def forward(self, X: T) -> T:
        x = self.mbc_extractor(X)
        x = self.mbc_pooler(x)
        return x  # type: ignore

    def grad_correct(self, c: float) -> None:
        self.mbc_extractor.grad_correct(c)
        self.mbc_pooler.grad_correct(c)


class MBCExtractor(MBCFunction):
    def __init__(
        self,
        n_layers: int,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        activation: str = "relu"
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.name = "BigSetExtractor"
        self.activation = activation
        self.set_hashable_attrs(
            ["n_layers", "in_dim", "h_dim", "out_dim", "activation"])

        act, layer = get_activation(activation), get_linear_layer("Linear")

        lyr = []
        for i in range(n_layers):
            a = in_dim if i == 0 else h_dim
            b = h_dim
            lyr.extend([layer(a, b), act()])
        lyr.append(layer(h_dim, out_dim))

        self.layer = nn.Sequential(*lyr)

    def pre_forward_mbc(self) -> None:
        pass

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None
    ) -> Tuple[T, T]:
        """extractor should be a row-wise operation which has no aggregation"""
        with torch.set_grad_enabled(grad):
            return self.layer(X), T()  # type: ignore

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """for training the model where x_nograd has no gradient."""
        x = self.layer(X)
        with torch.no_grad():
            x_nograd = self.layer(X_nograd)

        return torch.cat((x, x_nograd), dim=1)

    def forward(self, X: T) -> T:
        return self.layer(X)  # type: ignore

    def grad_correct(self, c: float) -> None:
        for n, p in self.named_parameters():
            if p.requires_grad:
                p.grad.data.mul_(c)


class IdentityExtractor(MBCFunction):
    """
    This isn't really used for anythng. It was made for debugging tests
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "IdentityExtractor"

    def pre_forward_mbc(self) -> None:
        pass

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None
    ) -> Tuple[T, T]:
        """extractor should be a row-wise operation which has no aggregation"""
        with torch.set_grad_enabled(grad):
            return X, T()  # type: ignore

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """for training the model where x_nograd has no gradient."""
        return torch.cat((X, X_nograd), dim=1)

    def forward(self, X: T) -> T:
        return X  # type: ignore

    def grad_correct(self, c: float) -> None:
        pass
