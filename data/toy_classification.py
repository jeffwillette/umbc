import math
from typing import Tuple

import torch
from torch.distributions import Categorical, Dirichlet
from torch.nn import functional as F
from torch.utils.data import Dataset

T = torch.Tensor


# MVN Diag and Mixture of Gaussians are adapted from
# https://github.com/juho-lee/set_transformer

class MVN:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def sample(self, B: int, K: int, labels: T) -> Tuple[T, T, T]:
        """Sample batch size B and classes K"""
        raise NotImplementedError

    def log_prob(self, X: T, mu: T, sigma: T, logdet: T) -> T:
        """calculate the log prob of X with the given mu and sigma"""
        X = X.unsqueeze(2)  # (B, N, 1, D)
        mu = mu.unsqueeze(1)  # (B, 1, K, D)
        sigma = sigma.unsqueeze(1)  # (B, 1, K, D, D)
        diff = (X - mu).unsqueeze(-1)  # (B, N, K, D, 1)

        energy = torch.einsum("bnkdx,bnkdz->bnkxz", diff, torch.inverse(sigma))
        energy = torch.einsum("bnkxd,bnkdz->bnkxz", energy, diff)
        nll = 0.5 * sigma.size(-1) * math.log(2 * math.pi) + 0.5 * \
            logdet.unsqueeze(1) + 0.5 * energy.view(*energy.size()[:-2])
        return -nll  # type: ignore

    def parse(self, raw: T) -> Tuple[T, T, T, T]:
        """parse the output of the network and return pi, mu, sigma"""
        pi = torch.softmax(raw[..., 0], -1)
        mu = raw[..., 1: 1 + self.dim]
        sigma = raw[..., 1 + self.dim:]
        if sigma.size(-1) == self.dim:
            # sigma = torch.clamp(sigma ** 2, 1e-2)
            sigma = torch.clamp(F.softplus(sigma), 1e-2)
            logdet = sigma.log().sum(dim=-1)
            sigma = torch.diag_embed(sigma)
            return pi, mu, sigma, logdet
        elif sigma.size(-1) == self.dim ** 2:
            b, k, _ = sigma.size()
            sigma = sigma.view(b, k, self.dim, self.dim)
            sigma = torch.einsum("bkij,bklj->bkil", sigma, sigma)
            logdet = torch.logdet(sigma)
            return pi, mu, sigma, logdet
        else:
            raise ValueError(
                f"sigma size must be [{self.dim}, {self.dim ** 2}]: got: {sigma.size(-1)}")


class MultivariateNormalFull(MVN):
    def __init__(self, dim: int):
        super().__init__(dim)

    def sample(self, B: int, K: int, labels: T) -> Tuple[T, T, T]:
        N = labels.size(-1)
        mu = -4 + (8 * torch.rand(B, K, self.dim))  # (B, K, d)
        D = torch.diag_embed(torch.rand(B, K, self.dim)
                             * 0.3 + 0.3)  # (B, K, d, d)

        # Q columns are orthogonal
        Q = torch.linalg.qr(torch.rand(
            B, K, self.dim, self.dim), mode="complete")[0]
        # Q = torch.eye(self.dim).view(1, 1, self.dim, self.dim).repeat(B, K, 1, 1)

        # O @ D (D is sqaure root of variance so S_sqrt @ S_sqrt^T = Cov)
        S_sqrt = torch.einsum("bkij,bkjl->bkil", Q, D)

        rlabels_mu = labels.view(B, N, 1).repeat(1, 1, self.dim)  # (B, N, d)
        rlabels_sigma = labels.view(B, N, 1, 1).repeat(
            1, 1, self.dim, self.dim)  # (B, N, d)

        # gather from dimension 1, so we are gathering from K dimensions N times to get --> (B, N, K)
        X = torch.gather(mu, 1, rlabels_mu) + torch.einsum("bnij,bnj->bni",
                                                           torch.gather(S_sqrt, 1, rlabels_sigma), torch.randn(B, N, self.dim))
        return X, mu, S_sqrt


class MultivariateNormalDiag(MVN):
    def __init__(self, dim: int):
        super().__init__(dim)

    def sample(self, B: int, K: int, labels: T) -> Tuple[T, T, T]:
        N = labels.size(-1)
        mu = -4 + 8 * torch.rand(B, K, self.dim)  # (B, K, d)
        sigma = torch.rand(B, K, self.dim) * 0.5 + 0.1  # (B, K, d)
        # sigma = torch.ones(B, K, self.dim) * 0.3  # (B, K, d)

        rlabels = labels.unsqueeze(-1).repeat(1, 1, self.dim)  # (B, N, d)

        # gather from dimension 1, so we are gathering from K dimensions N times to get --> (B, N, K)
        X = torch.gather(mu, 1, rlabels) + (torch.gather(sigma,
                                                         1, rlabels) * torch.randn(B, N, self.dim))
        return X, mu, torch.diag_embed(sigma ** 2)


class MixtureOfGaussians(Dataset):
    def __init__(self, dim: int, mvn_type: str = "full"):
        mvn = {"full": MultivariateNormalFull, "diag": MultivariateNormalDiag}
        self.mvn = mvn[mvn_type](dim)
        self.name = f"MixtureOfGaussians-{mvn_type}"

    def sample(self, B: int, N: int, K: int) -> Tuple[T, T, T, T, T]:
        """
        B: batch size
        K: classes
        labels: labels previously sampled
        """

        # get random class probabilities, sample labels
        pi = Dirichlet(torch.ones(K)).sample(torch.Size([B]))  # (B, K)
        labels = Categorical(probs=pi).sample(torch.Size([N]))  # (N, B)
        labels = labels.transpose(0, 1).contiguous()  # (B, N)

        X, mu, sigma = self.mvn.sample(B, K, labels)
        return X, labels, pi, mu, sigma

    def log_prob(self, X: T, pi: T, mu: T, sigma: T, logdet: T) -> Tuple[T, T]:
        ll = self.mvn.log_prob(X, mu, sigma, logdet)
        ll = ll + (pi + 1e-10).log().unsqueeze(1)
        labels = ll.argmax(-1)
        return ll.logsumexp(-1).mean(), labels

    def parse(self, raw: T) -> Tuple[T, T, T, T]:
        return self.mvn.parse(raw)

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, i: int) -> Tuple[T, T]:
        raise NotImplementedError
