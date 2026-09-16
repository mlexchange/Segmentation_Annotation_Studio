"""PCA reduce dense embeddings into bank channels."""

from __future__ import annotations

from typing import Any

import numpy as np

from ipred import features, sam_embed
from ipred.modules.base import ChannelBlock, ModuleContext, ModuleMeta


class PcaModule:
    """Bake encoder embedding grid → upsampled PCA float channels."""

    meta = ModuleMeta(
        id="pca",
        name="PCA reduce",
        description="PCA on dense embedding → heatmaps in feature bank",
        runtime="numpy",
        accepts_input_from=True,
        produces_channels=True,
        produces_embedding=False,
        params_schema={
            "dims": {"type": "integer", "default": 64},
        },
    )

    def ready(self) -> bool:
        return True

    def preview_labels(self, params: dict[str, Any]) -> list[str]:
        dims = max(1, int(params.get("dims", 64)))
        return [f"pca{i}" for i in range(dims)]

    def run(self, ctx: ModuleContext) -> ChannelBlock:
        src_id = None
        # Prefer explicitly linked node; else last embedding in context
        emb = None
        emb_meta: dict[str, Any] | None = None
        # input_from resolved upstream sets nothing on ctx for emb —
        # look up from node_outputs via params _input_from injected by runner
        input_from = ctx.params.get("_input_from")
        if input_from and input_from in ctx.node_outputs:
            block = ctx.node_outputs[input_from]
            emb, emb_meta = block.emb, block.emb_meta
            src_id = input_from
        if emb is None:
            for nid, block in reversed(list(ctx.node_outputs.items())):
                if block.emb is not None:
                    emb, emb_meta = block.emb, block.emb_meta
                    src_id = nid
                    break
        if emb is None or emb_meta is None:
            raise ValueError("pca module requires an upstream embedding node")
        dims = max(1, int(ctx.params.get("dims", 64)))
        h, w = int(ctx.gray.shape[0]), int(ctx.gray.shape[1])
        pca_float, pca_labels, info = sam_embed.emb_grid_to_pca_channels(
            emb,
            out_h=h,
            out_w=w,
            n_components=dims,
        )
        meta = dict(emb_meta)
        meta.update(info)
        meta["pca_from_node"] = src_id
        # Keep emb for bank persistence (sam_emb.npy)
        return ChannelBlock(
            float_stack=pca_float.astype(np.float32),
            labels=pca_labels,
            emb=emb,
            emb_meta=meta,
        )


def uint8_from_float(float_stack: np.ndarray) -> np.ndarray:
    """Display uint8 for a float HxWxC bank."""
    return features._float_feats_to_uint8(float_stack, clahe=False)
