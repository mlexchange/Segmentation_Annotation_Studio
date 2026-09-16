"""SlimSAM ONNX encoder module."""

from __future__ import annotations

from typing import Any

from ipred import feature_setups, sam_embed
from ipred.modules.base import ChannelBlock, ModuleContext, ModuleMeta


class SlimSamModule:
    """Dense SlimSAM vision-encoder embeddings (ONNX Runtime)."""

    meta = ModuleMeta(
        id="slimsam",
        name="SlimSAM",
        description="SlimSAM vision encoder (ONNX)",
        runtime="onnx",
        accepts_input_from=True,
        produces_channels=False,
        produces_embedding=True,
        params_schema={
            "weights_path": {"type": "string", "default": None},
        },
    )

    def ready(self) -> bool:
        path = feature_setups.resolve_slimsam_weights_path()
        return sam_embed.encoder_available(path)

    def preview_labels(self, params: dict[str, Any]) -> list[str]:
        del params
        return []  # PCA module consumes embedding

    def run(self, ctx: ModuleContext) -> ChannelBlock:
        src = ctx.input_image if ctx.input_image is not None else ctx.raw
        wpath = ctx.params.get("weights_path") or feature_setups.resolve_slimsam_weights_path()
        if not sam_embed.encoder_available(wpath):
            raise ValueError("SlimSAM ONNX weights not available")
        emb, orig_hw, reshaped_hw = sam_embed.encode_image_embeddings(
            src, weights_path=wpath
        )
        meta = {
            "orig_hw": list(orig_hw),
            "reshaped_hw": list(reshaped_hw),
            "weights_path": wpath,
            "encoder": "slimsam",
            "weights_format": feature_setups.WEIGHTS_ONNX_VISION,
        }
        return ChannelBlock(emb=emb, emb_meta=meta)
