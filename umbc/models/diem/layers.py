from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

from umbc.models.diem import networks

T = torch.Tensor


class DIEM(nn.Module):
    def __init__(
        self,
        d: int,
        H: int = 2,
        p: int = 5,
        L: int = 3,
        tau: float = 10.0,
        out: str = 'param_cat',
        distr_emb_args: Any = None
    ):
        super().__init__()
        self.priors = nn.ModuleList(
            [networks.DirNIWNet(p, d) for _ in range(H)])
        self.H = H
        self.L = L
        self.tau = tau
        self.out = out

        if out == 'param_cat':
            self.outdim = H * (1 + p + 2 * p * d)
        elif out == 'select_best':
            self.outdim = 1 * (p + 2 * p * d)
        elif out == 'select_best2':
            self.outdim = 1 * (H + p + 2 * p * d)
        else:
            raise NotImplementedError

    def forward(self, S: T, mask: Optional[T] = None) -> T:
        B, N_max, d = S.shape

        if mask is None:
            mask = torch.ones(B, N_max).to(S)

        pis, mus, Sigmas, alphas = [], [], [], []
        for h in range(self.H):
            pi, mu, Sigma = self.priors[h].map_em(
                S, mask=mask, num_iters=self.L,
                tau=self.tau, prior=self.priors[h]()
            )
            _, _, alpha = networks.mog_eval((pi, mu, Sigma), S)
            alpha = (alpha * mask).sum(-1)
            pis.append(pi)
            mus.append(mu)
            Sigmas.append(Sigma)
            alphas.append(alpha)

            if hasattr(self, "hook"):
                hook = partial(self.hook, self)
                _, _, alpha = networks.mog_eval((pi, mu, Sigma), S, hook=hook)

        pis = torch.stack(pis, dim=2)  # type: ignore
        mus = torch.stack(mus, dim=3)  # type: ignore
        Sigmas = torch.stack(Sigmas, dim=3)  # type: ignore
        alphas = torch.stack(alphas, dim=1)  # type: ignore
        alphas = (alphas - alphas.logsumexp(1, keepdim=True)).exp()  # type: ignore  # noqa

        if hasattr(self, "hook"):
            alphas_out = (alphas - alphas.logsumexp(1, keepdim=True)).exp()  # type: ignore  # noqa
            self.hook(self, [alphas], alphas_out)

        if self.out == 'param_cat':
            out = torch.cat([alphas, pis.reshape(B, -1),  # type: ignore
                             mus.reshape(B, -1), Sigmas.reshape(B, -1)], dim=1)  # type: ignore
        elif self.out == 'select_best':
            _, idx = alphas.max(1)
            pi = pis[range(B), :, idx]  # type: ignore
            mu = mus[range(B), :, :, idx]  # type: ignore
            Sigma = Sigmas[range(B), :, :, idx]  # type: ignore
            out = torch.cat(
                [pi.reshape(B, -1), mu.reshape(B, -1), Sigma.reshape(B, -1)], dim=1)
        elif self.out == 'select_best2':
            _, idx = alphas.max(1)
            pi = pis[range(B), :, idx]  # type: ignore
            mu = mus[range(B), :, :, idx]  # type: ignore
            Sigma = Sigmas[range(B), :, :, idx]  # type: ignore

            out = torch.cat([1.0 * F.one_hot(idx, self.H).to(alphas),
                            pi.reshape(B, -1), mu.reshape(B, -1), Sigma.reshape(B, -1)], dim=1)
        else:
            raise NotImplementedError

        return out
