"""Register a class-id mask/annotation volume as a real OME-NGFF multiscale
Tiled node, so the volume viewer's ``loadMask()``/``openOmeZarr()`` can
actually open it.

``tiled_mask_sync.write_masks_to_tiled`` used to write its ``semantic`` array
with a plain ``container.write_array()`` call, which sets no ``multiscales``
metadata — Tiled then serves it as an ordinary array node, and the viewer's
``openOmeZarr()`` rejects it with "missing multiscales" (see
``volume_nodes.has_multiscales``, which exists specifically to distinguish the
two). This module gives masks the same real multiscale registration the
primary volume already gets from ``volume_build.build_volume``.

Differs from ``volume_build.build_volume`` in two ways:
  - Downsamples by majority vote (mode), never by mean — an averaged class id
    is meaningless, the same reasoning the viewer's own ``mask-texture.ts``
    gives for using ``r8uint``/``textureLoad`` instead of ``r8unorm``/sampled.
  - Every level is computed directly from the native-resolution array rather
    than cascaded level-to-level: the whole mask is already fully in memory
    (unlike the primary volume, which is built by streaming a stack that may
    be tens of GB), so there is no incremental-IO reason to cascade.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

import tiff_stack_source as tss

logger = logging.getLogger("mask_pyramid")


def majority_downsample(volume: np.ndarray, factor: list[int]) -> np.ndarray:
    """Mode-downsample a uint8 class-id volume by the per-axis *factor*.

    Vectorized over the small set of class ids actually present in *volume*
    (real annotation taxonomies are a handful of classes even though uint8
    allows 256), rather than per-voxel. Ties favor the lower class id:
    ``np.unique`` is sorted ascending and a later id only overwrites the
    current winner on a strictly higher count.
    """
    fz, fy, fx = (int(f) for f in factor)
    if fz < 1 or fy < 1 or fx < 1:
        raise ValueError("downsample factors must be >= 1")
    z = (volume.shape[0] // fz) * fz
    y = (volume.shape[1] // fy) * fy
    x = (volume.shape[2] // fx) * fx
    trimmed = volume[:z, :y, :x]
    out_shape = (z // fz, y // fy, x // fx)

    class_ids = np.unique(trimmed)
    if class_ids.size <= 1:
        fill = int(class_ids[0]) if class_ids.size else 0
        return np.full(out_shape, fill, dtype=np.uint8)

    reshaped = trimmed.reshape(z // fz, fz, y // fy, fy, x // fx, fx)
    best_count = np.zeros(out_shape, dtype=np.int64)
    best_id = np.zeros(out_shape, dtype=np.uint8)
    for class_id in class_ids:
        count = (reshaped == class_id).sum(axis=(1, 3, 5))
        better = count > best_count
        best_count = np.where(better, count, best_count)
        best_id = np.where(better, class_id, best_id)
    return best_id


def build_mask_pyramid(semantic: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Compute ``{"scale0": semantic, "scale1": ..., ...}`` for a class-id volume.

    Returns the level dict plus the *generated* (non-scale0) plan entries from
    ``tiff_stack_source.pyramid_plan`` — the same shape ``volume_build.py``
    passes to ``multiscales_metadata``.
    """
    shape = tuple(int(s) for s in semantic.shape)
    generated = tss.pyramid_plan(shape)
    levels: dict[str, np.ndarray] = {"scale0": semantic.astype(np.uint8, copy=False)}
    for level in generated:
        levels[level["path"]] = majority_downsample(semantic, level["factor"])
    return levels, generated


def register_mask_pyramid(semantic: np.ndarray, key: str, container: Any) -> dict[str, Any]:
    """Build and register *semantic* (a ``(z, y, x)`` uint8 class-id volume) as
    a real OME-NGFF multiscale node named *key* inside *container* (the
    ``<source>__masks`` container ``tiled_mask_sync`` already manages).

    Reuses ``volume_build.build_volume``'s own write-pyramid-sidecar +
    ``register_single_item`` + ``multiscales_metadata`` machinery
    (``tiff_stack_source.write_pyramid_store``/``pyramid_plan``/
    ``multiscales_metadata``/``PYRAMID_KEY``) so Tiled serves this exactly the
    way it already serves the primary volume's pyramid — just downsampled by
    majority vote instead of mean.

    Replaces any previous registration for *key* — a re-sync must refresh the
    pyramid, not fail or accumulate duplicates, matching ``build_volume``'s
    same rule for the primary volume.
    """
    from tiled.client.register import Settings, register_single_item

    import ingest as ingest_mod

    levels, generated = build_mask_pyramid(semantic)
    sidecar = tss.write_pyramid_store(key, levels)

    if key in ingest_mod._child_keys(container):
        container.delete_contents(key, recursive=True, external_only=False)
    node = container.create_container(key=key, metadata={})

    try:
        asyncio.run(
            register_single_item(node, sidecar, is_directory=True, settings=Settings.init())
        )
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller, same as build_volume
        logger.warning("mask pyramid registration failed for %s: %s", sidecar, exc)
        raise

    pyramid_node = ingest_mod._walk(node, [tss.PYRAMID_KEY])
    if pyramid_node is None or not list(pyramid_node):
        raise RuntimeError(
            f"Tiled registered no levels for mask pyramid {key!r} from {sidecar}. "
            "The usual cause is that this path is not in the Tiled server's "
            "`readable_storage` (see tiled/config.yml)."
        )

    # scale0 is generated by THIS module rather than living in a pre-existing
    # per-slice sibling (the primary volume's case) — every level, scale0
    # included, is written into the one self-contained pyramid store above, so
    # it takes the SAME `f"{PYRAMID_KEY}/{level['path']}"` nesting the
    # generated levels get. Passing include_scale0=False and folding scale0
    # into `plan` (rather than True, which would emit it as a bare top-level
    # sibling path) is what keeps the metadata pointing at where the data
    # actually landed.
    plan = [{"path": "scale0", "factor": [1, 1, 1], "shape": list(semantic.shape)}, *generated]
    node.update_metadata(metadata=tss.multiscales_metadata(key, plan, include_scale0=False))
    return {"key": key, "full_shape": list(semantic.shape), "pyramid_plan": generated}


def read_mask_scale0(container: Any, key: str) -> np.ndarray:
    """Read back the native-resolution level of a mask pyramid registered by
    :func:`register_mask_pyramid` — the merge path's source of truth (the
    downsampled levels are viewer-only conveniences, never re-read)."""
    scale0 = container[key][tss.PYRAMID_KEY]["scale0"]
    return np.asarray(scale0[...])
