"""CLAHE / intensity image module (feeds encoders or contributes a channel)."""

from __future__ import annotations

from typing import Any

from ipred import features
from ipred.modules.base import ChannelBlock, ModuleContext, ModuleMeta


class ClaheModule:
    """Single-channel CLAHE (or raw intensity) plus image_2d for encoders."""

    meta = ModuleMeta(
        id="clahe",
        name="CLAHE",
        description="Adaptive histogram equalization → one channel + encoder input",
        runtime="numpy",
        accepts_input_from=False,
        produces_channels=True,
        params_schema={
            "clahe": {"type": "boolean", "default": True},
            "clip_limit": {"type": "number", "default": 0.01},
            "kernel_size": {"type": ["integer", "null"], "default": None},
            "include_in_bank": {"type": "boolean", "default": True},
        },
    )

    def ready(self) -> bool:
        return True

    def preview_labels(self, params: dict[str, Any]) -> list[str]:
        if not bool(params.get("include_in_bank", True)):
            return []
        return ["clahe"] if bool(params.get("clahe", True)) else ["intensity"]

    def run(self, ctx: ModuleContext) -> ChannelBlock:
        p = ctx.params
        clahe_on = bool(p.get("clahe", True))
        clip_limit = float(p.get("clip_limit", 0.01))
        ks = p.get("kernel_size", None)
        kernel_size = int(ks) if ks not in (None, "", False) else None
        _uint8, float_stack, labels = features.compute_clahe_stack(
            ctx.gray,
            clip_limit=clip_limit,
            kernel_size=kernel_size,
            apply=clahe_on,
        )
        image_2d = float_stack[..., 0]
        include = bool(p.get("include_in_bank", True))
        return ChannelBlock(
            float_stack=float_stack if include else None,
            labels=labels if include else [],
            image_2d=image_2d,
        )
