import argparse
import hashlib
import itertools
import logging
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple, Type)

import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
import torch
from matplotlib.colors import to_rgba  # type: ignore
from scipy import interpolate  # type: ignore
from torch import nn

__all__ = [
    "params_sans", "seed", "set_logger", "str2bool", "flatten",
    "unflatten_like", "get_mixture_mu_var", "gaussian_nll", "ensure_keys", "get_color",
    "to_cpu", "to_device", "get_test_name", "softmax_log_softmax_of_sample", "param_count",
    "get_linear_warmup_func", "get_module_root", "random_features", "mean_field_logits",
    "make_fgsm_adv", "list2string", "md5", "interpolate2d", "remove_dups", "set_sns"
]


T = torch.Tensor


def set_sns() -> None:
    sns.set_theme(style="white")
    sns.color_palette("tab10")
    sns.set_context(
        "notebook",
        font_scale=1.7,
        rc={
            "lines.linewidth": 3,
            "lines.markerscale": 4,
        }
    )


def md5(x: str) -> str:
    return hashlib.md5(x.encode()).hexdigest()


def list2string(lst: Any, delim: str = "_") -> str:
    return delim.join(map(str, lst))


def make_fgsm_adv(x: T, loss: T, eps: T) -> T:
    grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
    return x + eps * torch.sign(grad)


def mean_field_logits(logits: T, var: T, mean_field_factor: float = 1.0) -> T:
    """Adjust the predictive logits so its softmax approximates posterior mean."""

    logits_scale = torch.sqrt(1. + var * mean_field_factor)
    if mean_field_factor > 0:
        logits = logits / logits_scale

    return logits


# https://github.com/google/edward2/blob/720d7985a889362d421053965b6c528390557b05/edward2/tensorflow/initializers.py#L759
# based off of the implementation here. keras orthogonal first performs a QR decomposition to get an orthogonal matrix and samples
# rows until there are enough to form the right size. The SNGP implementation uses the random norms which rescales col norms.
# https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L161
def random_features(rows: int, cols: int, stddev: float = 0.05, orthogonal: bool = False, random_norm: bool = True) -> T:
    if orthogonal:
        cols_sampled, c = 0, []
        while cols_sampled < cols:
            # qr only returns orthogonal Q's as an (N, N) square matrix
            c.append(stddev * torch.linalg.qr(torch.randn(rows, rows, requires_grad=False), mode="complete")[0])
            cols_sampled += rows

        w = torch.cat(c, dim=-1)[:, :cols]

        # if not random norms for the columns, scale each norm column by the expected norm of each column.
        # https://github.com/google/edward2/blob/720d7985a889362d421053965b6c528390557b05/edward2/tensorflow/initializers.py#L814
        if not random_norm:
            return w * np.sqrt(rows)  # type: ignore

        col_norms = (torch.randn(rows, cols) ** 2).sum(dim=0).sqrt()
        return w * col_norms  # type: ignore
    return stddev * torch.randn(rows, cols, requires_grad=False)


def get_module_root() -> str:
    # return the module root which is the grandparent of this file
    return str(Path(os.path.abspath(__file__)).parent.parent)


def get_linear_warmup_func(total_epochs: int) -> Callable[[int], float]:
    return lambda epoch: epoch / float(total_epochs)


def param_count(model: nn.Module) -> int:
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def params_sans(model: nn.Module, without: Optional[Type[nn.Module]] = None) -> Iterator[Any]:
    if without is None:
        raise ValueError("need an argument for without")

    params = []
    for m in model.modules():
        if not isinstance(m, without):
            params.append(m.parameters(recurse=False))
    return itertools.chain(*params)


def seed(run: int) -> None:
    torch.cuda.manual_seed(run)
    torch.manual_seed(run)
    random.seed(run)
    np.random.seed(run)


def softmax_log_softmax_of_sample(x: T) -> Tuple[T, T]:
    """given that we have some logit samples in the shape of (samples, n, dim) stably return the softmax and the log softmax"""
    sample = x.size(0)

    out = torch.log_softmax(x, dim=-1)
    out = torch.logsumexp(out, dim=0) - np.log(sample)
    return out.exp(), out


def set_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger()


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def flatten(lst: Iterable[T]) -> T:
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector: T, like_tensor_list: Iterable[T]) -> Iterable[T]:
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors shaped like like_tensor_list
    if len(vector.size()) == 1:
        vector = vector.unsqueeze(0)

    outList = []
    i = 0
    for tensor in like_tensor_list:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i: i + n].view(tensor.shape))
        i += n
    return outList


def get_mixture_mu_var(mus: T, vars: T, dim: int = 0) -> Tuple[T, T]:
    mu = mus.mean(dim=dim)
    var = (vars + mus ** 2).mean(dim=dim) - (mu ** 2)
    return mu, var


def gaussian_nll(mu: T, var: T, y: T) -> T:
    s = len(mu.size())
    if s != len(var.size()) or s != len(y.size()):
        raise ValueError(f"mu: {mu.size()} var: {var.size()} and y: {y.size()} must all be the smae size")

    return ((1 / (2 * var)) * (y - mu) ** 2 + 0.5 * torch.log(var))  # type: ignore


def ensure_keys(o: Dict[str, Any], keys: List[Any], vals: List[Any]) -> None:
    for k, v in zip(keys, vals):
        if k not in o.keys():
            o[k] = v


colors = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    "mediumseagreen", "teal", "navy", "darkgoldenrod", "darkslateblue",
]


def get_color(i: int) -> Tuple[float, ...]:
    if i < len(colors):
        return to_rgba(colors[i])  # type: ignore
    return (np.random.rand(), np.random.rand(), np.random.rand(), 1.0)


def to_cpu(*args: Any) -> Any:
    return [v.cpu() for v in args]


def to_device(*args: Any, device: torch.device) -> Any:
    return [v.to(device) for v in args]


def get_test_name(args: Namespace) -> str:
    if not args.ood_test and not args.corrupt_test:
        return "standard"
    elif args.ood_test:
        return "ood"
    elif args.corrupt_test:
        return "corrupt"
    else:
        raise NotImplementedError(f"this combination of args has no known test name: {args=}")


def interpolate2d(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(data.shape) != 2:
        raise ValueError(f"data must be 2d. got: ({data.shape})")

    # create a grid of indices (x, y) and function evaluations (z) for the data we have
    xdim, ydim = data.shape
    x, y = np.arange(0, xdim, 1), np.arange(0, ydim, 1)
    print(x.shape, y.shape, data.shape)
    # xx, yy = np.meshgrid(x, y)
    f = interpolate.interp2d(x, y, data.T, kind='cubic')

    # create a finer grid which has some holes which need to be interpolated.
    # this should create a smoother look on the chart.
    xnew, ynew = np.arange(0, xdim, 1e-1), np.arange(0, ydim, 1e-1)
    return f(xnew, ynew), *np.meshgrid(xnew, ynew)  # type: ignore


def remove_dups(a: List[Any]) -> List[Any]:
    """remove duplicate entries from a list-like object"""
    for i in range(len(a)):
        j = i + 1
        while j < len(a):
            if a[i] == a[j]:
                a = a[:j] + a[j + 1:]
                continue
            j += 1
    return a
