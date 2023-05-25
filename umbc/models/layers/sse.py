import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from base import HashableModule
from torch.nn import functional as F

from umbc.models.layers.bigset import (LT, OLT, OT, MBCFunction,
                                             ParallelMBCFunction, T)
from umbc.models.layers.linear import Linear  # type: ignore

EPS = 1e-20


def set_preact_value(W: T, mask: OT = None, val: float = -1e20) -> T:
    if mask is None:
        return W

    # mask is (batch * heads, 1, set size). The middle dimension
    # get projected to all slots in the that dimension
    return W.masked_fill(mask == 0, val)


def apply_slot_normalization_with_mask(
    W: T,
    mask: OT = None,
    eps: float = EPS
) -> T:
    if mask is None:
        return W / (W.sum(dim=-2, keepdim=True) + eps)

    W = W.masked_fill(mask == 0, 0.0)
    return W / (W.sum(dim=-2, keepdim=True) + eps)


def update_set_normalization_constant(
    W: T,
    mask: OT = None,
    eps: float = EPS
) -> T:
    if mask is None:
        return W.sum(dim=-1, keepdim=True)

    # mask is (batch * heads, 1, set size).
    W = W.masked_fill(mask == 0, 0.0)
    return W.sum(dim=-1, keepdim=True)


def softmax_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    # normalization over the set dimension happens within the SSE
    W = set_preact_value(W, mask=mask).clamp(max=10)
    return torch.exp(W)


def sigmoid_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    W = set_preact_value(W, mask=mask)
    return torch.sigmoid(W)


def slot_sigmoid_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    W = set_preact_value(W, mask=mask)
    W = torch.sigmoid(W)
    return apply_slot_normalization_with_mask(W, mask, eps=eps)


def slot_softmax_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    W = set_preact_value(W, mask=mask)
    W = W.softmax(dim=-2)
    # its possible that every value in the set was set to the
    # null value so we have to reset them all to zero to be safe
    if mask is not None:
        W = W.masked_fill(mask == 0, 0.0)
    return W


def slot_exp_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    W = set_preact_value(W, mask=mask)
    W = torch.exp(W - W.amax(dim=-2, keepdim=True))
    # its possible that every value in the set was set to the
    # null value so we have to reset them all to zero to be safe
    if mask is not None:
        W = W.masked_fill(mask == 0, 0.0)
    return W


attn_act_funcs = {
    "softmax": softmax_attn_act,
    "sigmoid": sigmoid_attn_act,
    "slot-sigmoid": slot_sigmoid_attn_act,
    "slot-softmax": slot_softmax_attn_act,
    "slot-exp": slot_exp_attn_act
}


class Slots(HashableModule):
    def __init__(self, K: int, h: int, slot_type: str) -> None:
        super().__init__()
        self.name = "Slots"
        self.K = K                      # Number of Slots
        self.h = h                      # Slot size
        self.slot_type = slot_type      # Deterministic or Random
        self.set_hashable_attrs(["slot_type", "K", "h"])

        if slot_type not in ["random", "deterministic"]:
            raise ValueError(
                "{} not implemented for slots".format(self.slot_type))

        if slot_type == "random":
            # same initialization as "Weight Uncertainty in Neural Networks"
            self.mu = nn.Parameter(torch.zeros(
                1, self.K, self.h).uniform_(-0.2, 0.2), requires_grad=True)
            self.sigma = nn.Parameter(torch.zeros(
                1, self.K, self.h).uniform_(-5.0, -4.0), requires_grad=True)
            return

        self.S = nn.Parameter(torch.zeros(
            1, self.K, self.h), requires_grad=True)
        nn.init.xavier_uniform_(self.S)  # type: ignore

    def sample_s(self) -> T:

        # this is for calculating the memory usage of stored activations
        # during training. Every non-module operation should store the
        # input and output
        if hasattr(self, "hook"):
            sample = torch.randn_like(self.mu) * F.softplus(self.sigma)
            self.hook(self, [self.sigma], sample)
            self.hook(self, [self.sigma], F.softplus(self.sigma))
            self.hook(self, [sample], sample + self.mu)

        if self.slot_type == "random":
            if self.training:
                return (  # type: ignore
                    torch.randn_like(self.mu) * \
                    F.softplus(self.sigma) + self.mu
                )
            return self.mu

        # returning the parameter S caused problems because it is an
        # nn.Parameter and the above random returns a tensor. Return
        # a tensor here to make it consistent
        return (  # type: ignore
            torch.ones_like(self.S) * self.S
        )


class Embedder(HashableModule):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        activation: str = "relu"
    ) -> None:
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        self.embedder = nn.Sequential(nn.Linear(in_dim, h_dim), act())

    def forward(self, x: T) -> T:
        return self.embedder(x)  # type: ignore


class SlotSetEncoder(MBCFunction):
    def __init__(
        self,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        slot_type: str = "random",
        ln_slots: bool = True,
        heads: int = 4,
        bias: bool = True,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        eps: float = EPS,
        ln_after: bool = True,
    ):
        super().__init__()
        self.name = "SlotSetEncoder"
        self.d = d                          # Input Dimension
        self.d_hat = d_hat                  # Linear Projection Dimension
        self.eps = eps                      # Additive epsilon for stability
        self.heads = heads                  # number of attention heads
        self.bias = bias
        self.ln_after = ln_after
        self.ln_slots = ln_slots            # never used
        self.slot_drop = slot_drop
        self.attn_act = attn_act
        self.slot_residual = slot_residual

        self.slots: Slots

        if d_hat % heads != 0:
            raise ValueError(
                f"for multihead attention, {d_hat} must be evenly divisible by {heads}")  # noqa

        # Normalization Term
        self.sqrt_d_hat = 1.0 / math.sqrt(d_hat // heads)
        self.slots = Slots(K=K, h=h, slot_type=slot_type)

        self.q = Linear(in_features=h, out_features=d_hat, bias=bias)
        self.v = Linear(in_features=d, out_features=d_hat, bias=bias)
        self.k = Linear(in_features=d, out_features=d_hat, bias=bias)

        self.norm_slots = nn.LayerNorm(
            normalized_shape=h) if ln_slots else nn.Identity()

        self.set_hashable_attrs([
            "heads", "d", "d_hat", "ln_slots",
            "slot_drop", "attn_act", "slot_residual",
            "ln_after"
        ])

        self.pre_sampled_slots = T()
        self.norm_after = nn.LayerNorm(d_hat) if ln_after else nn.Identity()
        self.pre_sampled = False
        self.hook: Any

    def sample_s(self) -> T:
        S = self.slots.sample_s()
        S = self.norm_slots(S)

        if self.slot_drop <= 0.0 or not self.training:
            return self.q(S)  # type: ignore

        idx = torch.rand(self.slots.K) > self.slot_drop
        # we need to ensure that at least one slot is not dropped
        if idx.sum() == 0:
            lucky_one = torch.randperm(self.slots.K)[0]
            idx[lucky_one] = True

        return self.q(S[:, idx])  # type: ignore

    def head_split(self, X: T) -> T:
        # this is for calculating the memory usage of stored activations
        # during training. Every non-module operation should store the
        # input and output
        if hasattr(self, "hook"):
            out = torch.cat(X.split(self.d_hat // self.heads, 2), 0)
            self.hook(self, [X], out)

        return torch.cat(X.split(self.d_hat // self.heads, 2), 0)

    def get_attn_v(self, X: T, S: OT = None) -> Tuple[T, T, T]:
        if S is None:   # S \in R^{B x K xh}
            S = self.sample_s().repeat(X.size(0), 1, 1)
        assert S is not None

        if S.size(0) == 1:  # in case S was passed in and not repeated
            S = S.repeat(X.size(0), 1, 1)

        # Linear Projections k \in R^{B x N x d_hat}, v \in R^{B x N x d_hat},
        # q \in R^{B x K x d_hat}
        Q, K, V = S, self.k(X), self.v(X)
        Q, K, V = self.head_split(Q), self.head_split(K), self.head_split(V)

        # this is for calculating the memory usage of stored activations
        # during training. Every non-module operation should store the
        # input and output
        if hasattr(self, "hook"):
            QK = Q.bmm(K.transpose(1, 2))
            self.hook(self, [Q, K], QK)
            self.hook(self, [QK], self.sqrt_d_hat * QK)

        # M \in R^{B x K x N}
        return Q, self.sqrt_d_hat * Q.bmm(K.transpose(1, 2)), V

    def get_attn_act(
        self,
        W: T,
        mask: OT = None,
        batch_process: bool = False
    ) -> Tuple[T, T]:

        # this is for calculating the memory usage of stored activations
        # during training. Every non-module operation should store the
        # input and output
        if hasattr(self, "hook"):
            Whook = attn_act_funcs[self.attn_act](
                W, eps=self.eps, mask=mask)
            Chook = update_set_normalization_constant(
                Whook, mask=mask, eps=self.eps)
            self.hook(self, [W], Whook)
            self.hook(self, [W], Chook)

        # W = set_preact_value(W, torch.rand_like(W) > 0.1)
        W = attn_act_funcs[self.attn_act](W, eps=self.eps, mask=mask)

        # if we are not batch processing then do the normalization here
        C = update_set_normalization_constant(W, mask=mask, eps=self.eps)

        if not batch_process:
            return W / (C + self.eps), C  # normalize over N

        # if we are batch processing, then do nothing and return the
        # pre-activation value and the normalization constant
        return W, C

    def forward(self, X: T, S: OT = None, mask: OT = None) -> T:
        if self.pre_sampled:
            S = self.pre_sampled_slots

        if mask is not None:
            mask = mask.repeat(self.heads, 1).unsqueeze(1)

        S, W, V = self.get_attn_v(X, S=S)
        A, _ = self.get_attn_act(W, mask=mask)

        S_hat = A.bmm(V)     # S_hat \in R^{B x K x D}
        if self.slot_residual:
            S_hat += S

            # this is for calculating the memory usage of stored activations
            # during training. Every non-module operation should store the
            # input and output
            if hasattr(self, "hook"):
                self.hook(self, [S_hat], S_hat + S)

        S_hat = torch.cat(S_hat.split(X.size(0), 0), 2)
        S_hat = self.norm_after(S_hat)
        return S_hat  # type: ignore

    def pre_forward_mbc(self) -> None:
        if self.pre_sampled:
            raise ValueError("called pre without calling post")

        self.pre_sampled_slots = self.sample_s()
        self.pre_sampled = True

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None,
        # detach_c: bool = False,
    ) -> Tuple[T, T]:
        if (X_prev is None) != (c_prev is None):
            raise ValueError("X_prev and c_prev must be provided together")

        if mask is not None:
            mask = mask.repeat(self.heads, 1).unsqueeze(1)

        s = self.pre_sampled_slots.repeat(X.size(0), 1, 1)
        with torch.set_grad_enabled(grad):
            _, x, c = self.process_batch(X, S=s, mask=mask)

        # if detach_c:
        #     c = c.detach()

        if X_prev is not None and c_prev is not None:
            # this is for calculating the memory usage of stored activations
            # during training. Every non-module operation should store the
            # input and output
            if hasattr(self, "hook"):
                self.hook(self, [c], c + c_prev)
                self.hook(self, [x], x + X_prev)

            c += c_prev
            x += X_prev

        return x, c

    def post_forward_mbc(self, X: T, c: OT = None, mask: OT = None) -> T:
        assert c is not None

        view_heads = (X.size(0), self.pre_sampled_slots.size(1),
                      self.heads, -1)
        view_std = (X.size(0), self.pre_sampled_slots.size(1), -1)

        x = X.view(*view_heads) / \
            (c.view(*view_heads) + self.eps)  # type: ignore
        x = x.view(*view_std)

        # this is for calculating the memory usage of stored activations
        # during training. Every non-module operation should store the
        # input and output
        if hasattr(self, "hook"):
            xhook = X.view(*view_heads) / (c.view(*view_heads) + self.eps)
            self.hook(self, [x], xhook)

        if self.slot_residual:
            if hasattr(self, "hook"):
                self.hook(self, [x], x + self.pre_sampled_slots)

            x = x + self.pre_sampled_slots

        x = self.norm_after(x)

        # reset the pre_sampled variable to False so we can check for safety
        # the next time pre is called
        # self.pre_sampled_slots = T()
        self.pre_sampled = False

        return x

    def grad_correct(self, c: float) -> None:
        for n, p in self.named_parameters():
            if "norm_after" not in n:
                p.grad.data.mul_(c)

    # ========================================================================
    # METHODS HERE MAY GET DELETED IN THE FUTURE
    # ========================================================================

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """
        for training the model where x_nograd has no gradient
        in the base layer.
        """
        s = self.sample_s().repeat(X.size(0), 1, 1)

        s_normed, x, c = self.process_batch(X, S=s)
        with torch.set_grad_enabled(False):
            _, x_nograd, c_nograd = self.process_batch(X_nograd, S=s)

        # store the normalization constant and the unnormalized
        # (QK)V for updating
        view_heads = (x.size(0), s.size(1), self.heads, -1)
        view_std = (x.size(0), s.size(1), -1)

        c += c_nograd
        x += x_nograd

        x = x.view(*view_heads) / \
            (c.view(*view_heads) + self.eps)  # type: ignore
        x = x.view(*view_std)  # type: ignore

        if self.slot_residual:
            return self.norm_after(x + s_normed)  # type: ignore
        return self.norm_after(x)  # type: ignore

    def process_batch(
        self,
        X: T,
        S: OT = None,
        mask: OT = None
    ) -> Tuple[T, T, T]:
        """
        this is a 'partial forward' which allows the user to aggreate the
        components manually returns:
        S: Slots
        S_hat: SSE output to be aggregated
        C: normalization constant
        """
        S, W, V = self.get_attn_v(X, S=S)

        W, C = self.get_attn_act(W, batch_process=True, mask=mask)
        S_hat = W.bmm(V)     # S_hat \in R^{B x K x D}

        S, S_hat, C = map(lambda x: torch.cat(
            x.split(X.size(0), 0), 2), (S, S_hat, C))
        return S, S_hat, C  # type: ignore


class ParallelSSE(ParallelMBCFunction):
    def __init__(
        self,
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
        n_parallel: int = 1,
        eps: float = EPS,
        ln_after: bool = True
    ):
        super().__init__()
        self.name = "parallel-sse"
        self.n_parallel = n_parallel
        self.eps = eps
        self.set_hashable_attrs(["n_parallel"])

        self.sse = nn.ModuleList([
            SlotSetEncoder(
                K=K, h=h, d=d, d_hat=d_hat, slot_type=slot_type,
                ln_slots=ln_slots, heads=heads, slot_drop=slot_drop,
                attn_act=attn_act, slot_residual=slot_residual,
                ln_after=ln_after
            )
            for i in range(n_parallel)
        ])

    def grad_correct(self, c: float) -> None:
        for sse in self.sse:
            sse.grad_correct(c)

    def forward(self, x: T, S: OT = None) -> T:
        if S is None:
            return torch.cat([lyr(x, S=S) for lyr in self.sse], dim=1)
        return torch.cat([lyr(x, S=s) for (lyr, s) in zip(self.sse, S)], dim=1)

    def pre_forward_mbc(self) -> None:
        for lyr in self.sse:
            lyr.pre_forward_mbc()

    def forward_mbc(
        self,
        X: T,
        X_prev: Optional[LT] = None,
        c_prev: Optional[LT] = None,
        grad: bool = True,
        mask: OT = None
    ) -> Tuple[LT, LT]:
        if (X_prev is None) != (c_prev is None):
            raise ValueError("X_prev and c_prev must be provided together")

        out_x, out_c = [], []
        with torch.set_grad_enabled(grad):
            for i, lyr in enumerate(self.sse):
                x_i = X_prev[i] if X_prev is not None else None
                c_i = c_prev[i] if c_prev is not None else None

                x, c = lyr.forward_mbc(X, x_i, c_i, grad=grad, mask=mask)

                out_x.append(x)
                out_c.append(c)

        return out_x, out_c

    def post_forward_mbc(self, X: LT, c: OLT = None, mask: OT = None) -> T:
        assert c is not None

        for i, (lyr_c, lyr_x) in enumerate(zip(c, X)):
            X[i] = self.sse[i].post_forward_mbc(lyr_x, lyr_c)

        x = torch.cat(X, dim=1)
        return x

    def set_drop_rate(self, p: float) -> None:
        for i, _ in enumerate(self.sse):
            self.sse[i].slot_drop = p

    # ========================================================================
    # METHODS HERE MAY GET DELETED IN THE FUTURE
    # ========================================================================
    def process_batch(
        self,
        x: T,
        S: OT = None,
        S_list: OLT = None,
        mask: OT = None
    ) -> Tuple[LT, LT, LT]:
        parallel_s, parallel_x, parallel_c = [], [], []
        for i, lyr in enumerate(self.sse):
            _s, _x, _c = lyr.process_batch(
                x, S=S_list[i] if S_list is not None else S, mask=mask)
            parallel_s.append(_s)
            parallel_x.append(_x)
            parallel_c.append(_c)
        return parallel_s, parallel_x, parallel_c

    def partitioned_forward(self, x: T, x_nograd: T) -> T:
        """
        for training the model where x2 (a large set) has no gradient in the
        base layer. Only tracking the gradient for a chunk of the set will
        allow for larger set sizes during training.
        """
        S_list = [lyr.sample_s() for lyr in self.sse]

        parallel_s, parallel_x, parallel_c = self.process_batch(
            x, S=None, S_list=S_list)
        with torch.no_grad():
            parallel_s_nograd, parallel_x_nograd, parallel_c_nograd = \
                self.process_batch(x_nograd, S=None, S_list=S_list)

        # store the normalization constant and the unnormalized
        # (QK)V for updating
        for i, (s, c, x) in enumerate(
            zip(parallel_s, parallel_c_nograd, parallel_x_nograd)
        ):
            view_heads = (x.size(0), s.size(1), self.sse[i].heads, -1)
            view_std = (x.size(0), s.size(1), -1)

            parallel_c[i] += c
            parallel_x[i] += x

            parallel_x[i] = parallel_x[i].view(
                *view_heads) / (parallel_c[i].view(*view_heads) + self.eps)  # type: ignore
            parallel_x[i] = parallel_x[i].view(*view_std)  # type: ignore
            if self.sse[i].slot_residual:
                parallel_x[i] = parallel_x[i] + s

            parallel_x[i] = self.sse[i].norm_after(parallel_x[i])

        x = torch.cat(parallel_x, dim=1)
        return x
