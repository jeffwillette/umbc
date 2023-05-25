import os

import torch
import torch.nn as nn
import torch.nn.functional as F

T = torch.Tensor

# this code is taken from:
# https://github.com/hrzhang1123/DTFD-MIL/blob/1a160d95e4ff4f084c5313afb2728e436f05de06/Main_DTFD_MIL.py


class Attention_Gated(nn.Module):
    def __init__(self, L: int = 512, D: int = 128, K: int = 1) -> None:
        super().__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x: T, isNorm: bool = True) -> T:
        # x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        # K x N
        return A  # type: ignore


class Attention_with_Classifier(nn.Module):
    def __init__(
        self,
        L: int = 512,
        D: int = 128,
        K: int = 1,
        num_cls: int = 2,
        droprate: float = 0.0
    ) -> None:
        super().__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x: T) -> T:  # x: N x L
        AA = self.attention(x)  # K x N
        afeat = torch.mm(AA, x)  # K x L
        pred = self.classifier(afeat)  # K x num_cls
        return pred  # type: ignore


class Classifier_1fc(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        droprate: float = 0.0
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return
