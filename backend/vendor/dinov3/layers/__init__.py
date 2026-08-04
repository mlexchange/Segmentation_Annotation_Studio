# Vendored from facebookresearch/dinov3 @ 6876159a11b4df116f30f667f8c9888617df0751
# Source: https://raw.githubusercontent.com/facebookresearch/dinov3/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/__init__.py
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

from .attention import CausalSelfAttention, LinearKMaskedBias, SelfAttention
from .block import CausalSelfAttentionBlock, SelfAttentionBlock
from .ffn_layers import Mlp, SwiGLUFFN
from .fp8_linear import convert_linears_to_fp8
from .layer_scale import LayerScale
from .patch_embed import PatchEmbed
from .rms_norm import RMSNorm
from .rope_position_encoding import RopePositionEmbedding
