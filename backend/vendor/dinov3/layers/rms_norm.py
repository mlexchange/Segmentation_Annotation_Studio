# Vendored from facebookresearch/dinov3 @ 6876159a11b4df116f30f667f8c9888617df0751
# Source: https://raw.githubusercontent.com/facebookresearch/dinov3/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/rms_norm.py
# Distributed under Meta's DINOv3 License (see backend/vendor/dinov3/LICENSE.md).
# Copied verbatim except where noted — see dino_runtime.py for why this is vendored
# instead of loaded via torch.hub (upstream's hubconf.py eagerly imports the full
# research repo — detectors/segmentors/eval utilities — pulling in torchvision,
# torchmetrics, termcolor, and more just to expose a ViT backbone constructor).
#

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
