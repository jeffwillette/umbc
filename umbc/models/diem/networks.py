from functools import partial
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = torch.Tensor


def mog_eval(
    mog: Tuple[T],
    data: T,
    hook: Optional[Any] = None
) -> Tuple[T, T, T]:

    B, N, d = data.shape
    pi, mu, Sigma = mog

    if hook is not None:
        jllhook = -0.5 * (d * np.log(2 * np.pi) +
                          Sigma.log().sum(-1).unsqueeze(1) +
                          torch.bmm(data ** 2, 1. / Sigma.permute(0, 2, 1)) +
                          ((mu ** 2) / Sigma).sum(-1).unsqueeze(1) +
                          -2. * torch.bmm(data, (mu / Sigma).permute(0, 2, 1))
                          ) + pi.log().unsqueeze(1)

        # they are already the correct shape
        # hook([pi], pi.unsqueeze(0).repeat(B, 1))
        # hook([mu], mu.unsqueeze(0).repeat(B, 1, 1))
        # hook([Sigma], Sigma.unsqueeze(0).repeat(B, 1, 1))

        # Sigma.log().sum(-1).unsqueeze(1) +
        sigma_log_final = Sigma.log().sum(-1).unsqueeze(1)
        hook([Sigma], Sigma.log())
        hook([Sigma], sigma_log_final)

        # torch.bmm(data ** 2, 1. / Sigma.permute(0, 2, 1)) +
        bmm_result1 = torch.bmm(data ** 2, 1. / Sigma.permute(0, 2, 1))
        hook([data], data ** 2)
        hook([Sigma], 1. / Sigma.permute(0, 2, 1))
        hook([data, Sigma], bmm_result1)

        # ((mu ** 2) / Sigma).sum(-1).unsqueeze(1) +
        musq_dsigma = mu ** 2 / Sigma
        musq_dsigma_sum = musq_dsigma.sum(-1).unsqueeze(1)
        hook([mu], mu ** 2)
        hook([mu, Sigma], mu ** 2)
        hook([mu, Sigma], musq_dsigma)
        hook([musq_dsigma], musq_dsigma_sum)

        # -2. * torch.bmm(data, (mu / Sigma).permute(0, 2, 1))
        bmm_result2 = -2 * torch.bmm(data, (mu / Sigma).permute(0, 2, 1))
        hook([mu, Sigma], mu / Sigma)
        hook([mu, Sigma], bmm_result2)

        # ) + pi.log().unsqueeze(1)
        hook([pi], pi.log().unsqueeze(1))

        one = sigma_log_final + bmm_result1
        two = one + musq_dsigma_sum
        three = two + bmm_result2
        four = three + pi.log().unsqueeze(1)
        hook([sigma_log_final, bmm_result1], one)
        hook([one, musq_dsigma_sum], two)
        hook([two, bmm_result2], three)
        hook([three, pi], four)

    if len(pi.shape) == 1:
        pi = pi.unsqueeze(0).repeat(B, 1)
        mu = mu.unsqueeze(0).repeat(B, 1, 1)
        Sigma = Sigma.unsqueeze(0).repeat(B, 1, 1)
    p = pi.shape[-1]

    jll = -0.5 * (d * np.log(2 * np.pi) +
                  Sigma.log().sum(-1).unsqueeze(1) +
                  torch.bmm(data ** 2, 1. / Sigma.permute(0, 2, 1)) +
                  ((mu ** 2) / Sigma).sum(-1).unsqueeze(1) +
                  -2. * torch.bmm(data, (mu / Sigma).permute(0, 2, 1))
                  ) + pi.log().unsqueeze(1)

    mll = jll.logsumexp(-1)
    cll = jll - mll.unsqueeze(-1)

    return jll, cll, mll


class DirNIWNet(nn.Module):
    def __init__(self, p, d):
        super(DirNIWNet, self).__init__()
        self.m = nn.Parameter(0.1 * torch.randn(p, d))
        self.V_ = nn.Parameter(np.log(np.exp(1) - 1) +
                               0.001 / np.sqrt(d) * torch.randn(p, d))
        self.p, self.d = p, d

    def forward(self):
        V = F.softplus(self.V_)
        if hasattr(self, "hook"):
            self.hook(self, [self.V_], F.softplus(self.V_))
        return self.m, V

    def mode(self, prior=None):
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior
        pi = torch.ones(self.p).to(m) / self.p
        mu = m
        Sigma = V
        return pi, mu, Sigma

    def loglik(self, theta):
        raise NotImplementedError

    def map_m_step(self, data, weight, tau=1.0, prior=None):
        B, N, d = data.shape

        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior

        if hasattr(self, "hook"):
            wsum = weight.sum(1)
            wsum_reg = wsum + tau
            self.hook(self, [weight], wsum)

            wxsumhook = torch.bmm(weight.permute(0, 2, 1), data)
            self.hook(self, [weight, data], wxsumhook)

            wxxsumhook = torch.bmm(weight.permute(0, 2, 1), data)
            self.hook(self, [weight, data], wxxsumhook)
            self.hook(self, [data], data ** 2)

            wsum_reg_sum = wsum_reg.sum(1, keepdim=True)
            self.hook(self, [wsum_reg], wsum_reg_sum)
            self.hook(self, [wsum_reg, wsum_reg_sum], wsum_reg / wsum_reg_sum)

            mudiv1 = wxsumhook + m.unsqueeze(0) * tau
            mudiv2 = mudiv1 / wsum_reg.unsqueeze(-1)
            self.hook(self, [wxsumhook, m], wxsumhook + m * tau)
            self.hook(self, [mudiv1, wsum_reg],
                      mudiv1 / wsum_reg.unsqueeze(-1))

            # Sigma = (wxxsum + (V + m ** 2).unsqueeze(0) * tau) / \
            #    wsum_reg.unsqueeze(-1) - mu ** 2
            vplusm = V + m
            vplusmsq = (vplusm ** 2).unsqueeze(0)
            wxxsumplus = wxxsumhook + vplusmsq
            wxxsumplus2 = wxxsumplus - wsum_reg.unsqueeze(-1)
            wxxsumplus3 = wxxsumplus2 - mudiv2 ** 2
            self.hook(self, [m], m ** 2)
            self.hook(self, [V, m], vplusmsq)
            self.hook(self, [wxxsumhook, vplusmsq], wxxsumplus)
            self.hook(self, [wxxsumplus, wsum_reg], wxxsumplus2)
            self.hook(self, [wxxsumplus2, mudiv2], wxxsumplus3)

        wsum = weight.sum(1)
        wsum_reg = wsum + tau
        wxsum = torch.bmm(weight.permute(0, 2, 1), data)
        wxxsum = torch.bmm(weight.permute(0, 2, 1), data ** 2)
        pi = wsum_reg / wsum_reg.sum(1, keepdim=True)
        mu = (wxsum + m.unsqueeze(0) * tau) / wsum_reg.unsqueeze(-1)
        Sigma = (wxxsum + (V + m ** 2).unsqueeze(0) * tau) / \
            wsum_reg.unsqueeze(-1) - mu ** 2

        return pi, mu, Sigma

    def map_em(self, data, mask=None, num_iters=3, tau=1.0, prior=None):
        B, N, d = data.shape

        if hasattr(self, "hook"):
            pi, mu, Sigma = self.mode(prior)
            self.hook(self, [pi], pi.unsqueeze(0).repeat(B, 1))
            self.hook(self, [mu], mu.unsqueeze(0).repeat(B, 1, 1))
            self.hook(self, [Sigma], Sigma.unsqueeze(0).repeat(B, 1, 1))

        if mask is None:
            mask = torch.ones(B, N).to(data)
        pi, mu, Sigma = self.mode(prior)
        pi = pi.unsqueeze(0).repeat(B, 1)
        mu = mu.unsqueeze(0).repeat(B, 1, 1)
        Sigma = Sigma.unsqueeze(0).repeat(B, 1, 1)

        for emiter in range(num_iters):
            _, qq, _ = mog_eval((pi, mu, Sigma), data)
            if hasattr(self, "hook"):
                hook = partial(self.hook, self)
                _, qq, _ = mog_eval((pi, mu, Sigma), data, hook=hook)

            qq = qq.exp() * mask.unsqueeze(-1)
            pi, mu, Sigma = self.map_m_step(
                data, weight=qq, tau=tau, prior=prior)

        return pi, mu, Sigma
