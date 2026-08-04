# Vendored from facebookresearch/dinov3 @ 6876159a11b4df116f30f667f8c9888617df0751
# Source: https://raw.githubusercontent.com/facebookresearch/dinov3/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/layer_scale.py
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

from typing import Union

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
