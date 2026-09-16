"""Skimage multiscale feature module."""

from __future__ import annotations

from typing import Any

from ipred import features
from ipred.modules.base import ChannelBlock, ModuleContext, ModuleMeta


class SkimageMultiscaleModule:
    """Produce multiscale intensity/edges/texture channels."""

    meta = ModuleMeta(
        id="skimage_multiscale",
        name="Skimage multiscale",
        description="ilastik-style multiscale basic features",
        runtime="numpy",
        accepts_input_from=False,
        produces_channels=True,
        params_schema={
            "sigma_min": {"type": "number", "default": 1.0},
            "sigma_max": {"type": "number", "default": 8.0},
            "intensity": {"type": "boolean", "default": True},
            "edges": {"type": "boolean", "default": True},
            "texture": {"type": "boolean", "default": True},
            "clahe": {"type": "boolean", "default": True},
        },
    )

    def ready(self) -> bool:
        return True

    def preview_labels(self, params: dict[str, Any]) -> list[str]:
        return features.feature_channel_labels(
            sigma_min=float(params.get("sigma_min", 1.0)),
            sigma_max=float(params.get("sigma_max", 8.0)),
            intensity=bool(params.get("intensity", True)),
            edges=bool(params.get("edges", True)),
            texture=bool(params.get("texture", True)),
        )

    def run(self, ctx: ModuleContext) -> ChannelBlock:
        p = ctx.params
        _uint8, float_stack, labels = features.compute_feature_stacks(
            ctx.gray,
            sigma_min=float(p.get("sigma_min", 1.0)),
            sigma_max=float(p.get("sigma_max", 8.0)),
            intensity=bool(p.get("intensity", True)),
            edges=bool(p.get("edges", True)),
            texture=bool(p.get("texture", True)),
            clahe=bool(p.get("clahe", True)),
        )
        return ChannelBlock(float_stack=float_stack, labels=labels)
