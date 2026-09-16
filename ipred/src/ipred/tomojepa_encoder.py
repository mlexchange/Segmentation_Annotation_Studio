"""Mark25 DINOv3ViTEncoder — ViT-S/16 + dense/global projectors.

Architecture reconstructed to match TomoJEPA checkpoints (``ckpt['net']``):
timm ``vit_small_patch16_dinov3`` backbone, BatchNorm dense MLP → ``proj_dim``
(Mark25: 64-D, Mark11: 256-D), LayerNorm global MLP → 16-D.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _dense_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(hidden, out_dim),
        nn.Identity(),
    )


def _global_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(hidden, out_dim),
        nn.Identity(),
    )


class DINOv3ViTEncoder(nn.Module):
    """Mark25 TomoJEPA encoder producing global and dense projections."""

    def __init__(
        self,
        proj_dim: int = 64,
        img_size: int = 512,
        in_chans: int = 1,
        pretrained: bool = False,
        *,
        embed_dim: int = 384,
        global_dim: int = 16,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        del pretrained  # checkpoint is loaded separately; never ImageNet init
        import timm

        self.backbone = timm.create_model(
            "vit_small_patch16_dinov3",
            pretrained=False,
            in_chans=in_chans,
            img_size=img_size,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.proj = _dense_mlp(embed_dim, hidden_dim, proj_dim)
        self.global_proj = _global_mlp(embed_dim, hidden_dim, global_dim)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 5))
        self.proj_dim = int(proj_dim)
        self.patch_size = int(getattr(self.backbone, "patch_size", 16) or 16)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder on multi-view batch.

        Args:
            x: Float tensor ``[B, C, V, H, W]`` (typically C=1).

        Returns:
            ``(global_proj, dense_proj, feats)`` where dense is
            ``[B, V, L, proj_dim]`` and global is ``[B, V, global_dim]``.
        """
        if x.ndim != 5:
            raise ValueError(f"expected [B,C,V,H,W], got shape {tuple(x.shape)}")
        b, c, v, h, w = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(b * v, c, h, w)
        feats = self.backbone.forward_features(flat)
        cls = feats[:, 0]
        patches = feats[:, self.num_prefix_tokens :]
        bv, length, dim = patches.shape
        dense = self.proj(patches.reshape(bv * length, dim)).reshape(
            b, v, length, -1
        )
        glob = self.global_proj(cls).reshape(b, v, -1)
        return glob, dense, feats


def build_encoder(**kwargs: Any) -> DINOv3ViTEncoder:
    """Construct an uninitialized Mark25 encoder."""
    return DINOv3ViTEncoder(**kwargs)
