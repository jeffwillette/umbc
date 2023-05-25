import os
from typing import Any, Dict, Tuple

import torch
import torchvision  # type: ignore
from torch import nn
from torch.nn import functional as F
from torchvision import models  # type: ignore

from umbc.models.layers.bigset import OT, GradCorrecter, MBCFunction, T

"""
https://github.com/Richarizardd/Self-Supervised-ViT-Path
https://github.com/mahmoodlab/HIPT
-------------------

work from the previous two links in the paper implies
that transferring from imagenet pretrained works well.


https://github.com/binli123/dsmil-wsi
------------------

work in the paper of the previous link implies
that tranferring from imagenet is very poor.

Many WSI works use transfer from imagenet, and there are multiple
confusing issues in the SimCLR pretrained features from the link above
so we use imagenet pretrained features from a ResNet50 and not the
SimCLR ResNet18 self supervised models.
"""


class ResNetMBC(MBCFunction):
    def layer(self, x: T) -> T:
        raise NotImplementedError()

    def forward(self, x: T) -> T:
        if len(x.size()) == 5:
            b, s, c, h, w = x.size()
            return (  # type: ignore
                self.layer(x.view(b * s, c, h, w)).view(b, s, -1)
            )

        return self.layer(x).unsqueeze(0)  # type: ignore

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """for training the model where x_nograd has no gradient."""
        x = self.layer(X)
        with torch.no_grad():
            x_nograd = self.layer(X_nograd)

        return torch.cat((x, x_nograd), dim=1)

    def grad_correct(self, c: float) -> None:
        for n, p in self.named_parameters():
            if p.requires_grad:
                p.grad.data.mul_(c)

    def pre_forward_mbc(self) -> None:
        pass

    def forward_mbc(
        self,
        X: T,
        X_prev: OT = None,
        c_prev: OT = None,
        grad: bool = True,
        mask: OT = None
    ) -> Tuple[T, OT]:
        with torch.set_grad_enabled(grad):
            return self.forward(X), c_prev

    def post_forward_mbc(self, X: T, c: OT = None, mask: OT = None) -> T:
        """
        for post processing the minibatched forward pooled
        vector and normalization constant
        """
        return X


class ResNet50Pretrained(ResNetMBC):
    def __init__(self) -> None:
        super().__init__()

        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.fc = nn.Identity()
        self.fn = resnet

    def layer(self, x: T) -> T:
        return self.fn(x)  # type: ignore


class ResNetSimCLR(ResNetMBC):
    def __init__(self, out_dim: int = 256, magnification: int = 20) -> None:
        super().__init__()
        self.name = "ResNetSimCLR"
        resnet = models.resnet18(
            pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        self.filename = {
            20: os.path.join(
                os.getenv("PYTHONPATH"),  # type: ignore
                ".pretrained", "mil-c16", "20x", "model-v2.pth"),
            5: os.path.join(
                os.getenv("PYTHONPATH"),  # type: ignore
                ".pretrained", "mil-c16", "5x", "model.pth"),
        }[magnification]

        sd = torch.load(self.filename)
        state_dict: Dict[str, Any] = {
            ".".join(k.split(".")[1:]): v for (k, v) in sd.items()}
        self.load_state_dict(state_dict)  # type: ignore

    def layer(self, x: T) -> T:
        h = self.features(x)
        h = h.view(h.size(0), h.size(1))

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return x
