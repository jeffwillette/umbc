from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = torch.Tensor


# this code is taken from:
# https://github.com/binli123/dsmil-wsi/


class IClassifier(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_size: int,
        output_class: int
    ) -> None:
        super().__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

        # for lyr in [self.fc]:
        #     nn.init.xavier_uniform_(lyr.weight)

    def forward(self, x: T) -> Tuple[T, T]:
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_class: int,
        h_dim: int = 128,
        dropout_v: float = 0.0,
        nonlinear: bool = True,
        passing_v: bool = False
    ) -> None:  # K, L, N
        super().__init__()
        self.q: nn.Module
        self.v: nn.Module

        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, h_dim)

        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        # 1D convolutional layer that can handle multiple class
        # (including binary)
        self.fcc = nn.Conv1d(output_class, output_class,
                             kernel_size=input_size)

        # for lyr in [self.q[0], self.q[2]]:
        #     nn.init.xavier_uniform_(lyr.weight)

    def forward(self, feats: T, c: T) -> Tuple[T, T, T]:  # N x K, N x C
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        # sort class scores along the instance dimension, m_indices
        # in shape N x C
        _, m_indices = torch.sort(c, 0, descending=True)
        # select critical instances, m_feats in shape C x K
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        # compute queries of critical instances, q_max in shape C x Q
        q_max = self.q(m_feats)
        # compute inner product of Q to each entry of q_max, A in shape N x C,
        # each column contains unnormalized attention scores
        A = torch.mm(Q, q_max.transpose(0, 1))
        # normalize attention scores, A in shape N x C,
        A = F.softmax(A / np.sqrt(Q.shape[1]), 0)

        # compute bag representation, B in shape C x V
        B = torch.mm(A.transpose(0, 1), V)

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class MILNet(nn.Module):
    def __init__(self, i_classifier: nn.Module, b_classifier: nn.Module):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x: T) -> Tuple[T, T]:
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag
