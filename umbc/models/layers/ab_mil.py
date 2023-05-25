from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

T = torch.Tensor


class GatedAttention(nn.Module):
    def __init__(self, L: int, D: int, K: int) -> None:
        super(GatedAttention, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.inst_classifier = nn.Linear(L, K)

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
        )

    def forward(self, x: T) -> Tuple[T, T]:
        H = x
        inst_out = self.inst_classifier(H)

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        # element wise multiplication # NxK
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        out = self.classifier(M)
        return inst_out, out
