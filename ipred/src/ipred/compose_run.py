"""Execute a composition document on a raw image array."""

from __future__ import annotations

from typing import Any

import numpy as np

from ipred import features
from ipred.compositions import validate_composition
from ipred.modules import get_module
from ipred.modules.base import ChannelBlock, ModuleContext
from ipred.modules.pca_mod import uint8_from_float


def run_composition(
    arr: np.ndarray,
    doc: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray | None,
    dict[str, Any] | None,
]:
    """Run composition nodes; concatenate ``outputs`` channel blocks.

    Returns:
        uint8_stack, float_stack, labels, sam_emb, sam_meta
        (emb/meta from the last PCA or encoder that produced them).
    """
    validate_composition(doc)
    gray = features.to_grayscale(arr)
    ctx = ModuleContext(raw=arr, gray=gray, params={})
    by_id = {n["id"]: n for n in doc["nodes"]}

    # Topological-ish: run in document order (authors must order deps first)
    for node in doc["nodes"]:
        nid = node["id"]
        mid = str(node["module"])
        mod = get_module(mid)
        params = dict(node.get("params") or {})
        input_from = node.get("input_from")
        input_image = None
        if input_from:
            upstream = ctx.node_outputs.get(input_from)
            if upstream is None:
                raise ValueError(
                    f"node {nid}: input_from {input_from} not run yet"
                )
            if upstream.image_2d is not None:
                input_image = upstream.image_2d
            elif upstream.float_stack is not None and upstream.float_stack.shape[-1] >= 1:
                input_image = upstream.float_stack[..., 0]
            params["_input_from"] = input_from
        ctx.params = params
        ctx.input_image = input_image
        block = mod.run(ctx)
        ctx.node_outputs[nid] = block

    float_parts: list[np.ndarray] = []
    labels: list[str] = []
    sam_emb: np.ndarray | None = None
    sam_meta: dict[str, Any] | None = None
    for oid in doc.get("outputs") or []:
        block = ctx.node_outputs[oid]
        if block.float_stack is not None and block.float_stack.size:
            float_parts.append(np.asarray(block.float_stack, dtype=np.float32))
            labels.extend(block.labels)
        if block.emb is not None:
            sam_emb = block.emb
            sam_meta = block.emb_meta
    # Also pick up emb from non-output encoder nodes if PCA was in outputs
    if sam_emb is None:
        for block in ctx.node_outputs.values():
            if block.emb is not None:
                sam_emb = block.emb
                sam_meta = block.emb_meta

    if not float_parts:
        raise ValueError("composition produced no channel outputs")
    float_stack = np.concatenate(float_parts, axis=-1)
    uint8_stack = uint8_from_float(float_stack)
    return uint8_stack, float_stack, labels, sam_emb, sam_meta
