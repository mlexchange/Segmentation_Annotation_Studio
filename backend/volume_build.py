"""Build a renderable 3-D volume from a slice stack already in Tiled.

:mod:`tiff_stack_source` registers a *directory of TIFFs* and additionally
streams the full-resolution slices in place. That is the right path when the
source images are still on the server — but most datasets here arrived through
drag-and-drop ingest, which copies each slice into Tiled's own storage as a 2-D
array. For those there is no source directory to point at, and asking the user
to find one would be asking them to re-supply data the app already has.

So this module builds the pyramid from whatever :mod:`arrays` can already read.
It needs no arguments beyond the dataset that is open, which is what makes
"Build 3-D volume" a single button rather than a form.

What it does *not* do is copy the full-resolution data. Only the downsampled
levels are written, because those are the only ones the renderer ever uploads —
it picks a level that fits ``maxTextureDimension3D``, and full resolution never
does. Full resolution stays exactly where it is, read by the 2-D canvas as
always.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import numpy as np
from fastapi import HTTPException

import arrays as arrays_mod
import ingest as ingest_mod
import tiff_stack_source as tss
from tiled_clients import api_key_for_uri, get_tiled_client

logger = logging.getLogger("volume_build")


def _stack_shape(source: str, kind: str, server_uri: str | None) -> tuple[Any, dict, tuple[int, int, int]]:
    """Resolve *source* to a readable stack and report its ``(z, y, x)`` shape."""
    node = arrays_mod.resolve_array(source, kind, server_uri)
    meta = arrays_mod.array_shape_meta(node)
    n_slices = int(meta.get("n_slices") or 0)
    height = int(meta.get("height") or 0)
    width = int(meta.get("width") or 0)
    if n_slices < 2:
        raise HTTPException(
            422,
            "This dataset is a single image, not a stack — there is no volume to build.",
        )
    if meta.get("is_rgb"):
        raise HTTPException(
            422,
            "Colour images are not supported by the 3-D view, which renders a single "
            "scalar volume.",
        )
    return node, meta, (n_slices, height, width)


def inspect_volume_build(
    source: str, kind: str = "tiled", server_uri: str | None = None
) -> dict[str, Any]:
    """Describe the volume that would be built, without building it."""
    _node, meta, shape = _stack_shape(source, kind, server_uri)
    plan = tss.pyramid_plan(shape)
    return {
        "full_shape": list(shape),
        "dtype": str(meta.get("dtype") or "float32"),
        "pyramid_plan": plan,
        # Every source slice is read once; the coarser levels cascade in memory.
        "slices_to_read": shape[0] if plan else 0,
        "already_small": not plan,
    }


def build_volume(
    source: str,
    kind: str = "tiled",
    server_uri: str | None = None,
    container_path: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Build and register a 3-D volume sidecar for the stack open at *source*.

    Args:
        source: Tiled path of the per-slice dataset.
        kind: ``"tiled"`` (the only kind with a catalog to register into).
        server_uri: Connected Tiled server URI.
        container_path: Where to put the sidecar; defaults to *source*'s parent,
            so the volume lands next to the dataset it describes.
        progress: Optional ``(message, done, total)`` callback for the job UI.

    Returns:
        The registered ``key`` and ``tiled_path``, plus the build description.
    """
    if kind != "tiled":
        raise HTTPException(422, "Only datasets in the Tiled catalog can be built into volumes.")

    node, meta, shape = _stack_shape(source, kind, server_uri)
    plan = tss.pyramid_plan(shape)
    if not plan:
        raise HTTPException(
            422,
            f"This stack is already small enough ({shape[0]}x{shape[1]}x{shape[2]}) that "
            "no downsampling is needed — but it is stored as separate 2-D slices, which "
            "cannot be streamed as a volume. Re-register it from its source images.",
        )

    parts = [p for p in source.strip("/").split("/") if p]
    stem = parts[-1]
    key = f"{stem}{tss.VOLUME_SUFFIX}"
    target_parts = (
        [p for p in container_path.strip("/").split("/") if p]
        if container_path
        else parts[:-1]
    )
    if not target_parts:
        raise HTTPException(422, "Cannot place a volume at the catalog root.")

    dtype = np.dtype(meta.get("dtype") or "float32")
    total = shape[0]
    built: dict[str, np.ndarray] = {}
    previous: np.ndarray | None = None
    previous_factor: list[int] | None = None

    for level in plan:
        label = f"Building {level['path']}"
        if previous is None or previous_factor is None:
            # Finest generated level: read the stack once, a block of slices at a
            # time, so peak memory is the output level plus a few input slices.
            fz, fy, fx = (int(f) for f in level["factor"])
            out = np.empty(level["shape"], dtype=np.float32)
            for z in range(level["shape"][0]):
                block = []
                for k in range(fz):
                    index = z * fz + k
                    if index >= shape[0]:
                        break
                    block.append(arrays_mod.read_slice(node, meta, index))
                    if progress and (z * fz + k) % 25 == 0:
                        progress(label, z * fz + k, total)
                out[z] = tss.block_mean(np.stack(block), fy, fx)
            previous_working = out
        else:
            relative = [int(level["factor"][a] // previous_factor[a]) for a in range(3)]
            previous_working = tss.downsample_array(previous, relative)

        built[level["path"]] = tss._cast_like(previous_working, dtype)
        previous, previous_factor = previous_working, list(level["factor"])

    if progress:
        progress("Writing pyramid", total, total)
    sidecar = tss.write_pyramid_store(key, built)

    client = get_tiled_client(server_uri, api_key_for_uri(server_uri))
    target = ingest_mod._ensure_container(client, target_parts)

    # Replace any previous build for this dataset: re-running must refresh the
    # volume, not fail or accumulate duplicates.
    if key in ingest_mod._child_keys(target):
        target.delete_contents(key, recursive=True, external_only=False)
    volume = target.create_container(key=key, metadata={})

    from tiled.client.register import Settings, register_single_item

    try:
        asyncio.run(
            register_single_item(volume, sidecar, is_directory=True, settings=Settings.init())
        )
    except Exception as exc:  # noqa: BLE001 — classified for the UI
        logger.warning("volume registration failed for %s: %s", sidecar, exc)
        raise HTTPException(502, ingest_mod._classify_error(exc)["message"]) from exc

    pyramid_node = ingest_mod._walk(volume, [tss.PYRAMID_KEY])
    if pyramid_node is None or not list(pyramid_node):
        raise HTTPException(
            502,
            f"Tiled registered no levels for {key!r} from {sidecar}. The usual cause is "
            "that this path is not in the Tiled server's `readable_storage` (see "
            "tiled/config.yml).",
        )

    volume.update_metadata(
        metadata={
            "sample_name": key,
            "n_images": shape[0],
            "source_format": "tiled-stack-3d",
            "built_from": source,
            "full_shape": list(shape),
            "pyramid_plan": plan,
            # No scale0: full resolution stays in the per-slice nodes rather than
            # being duplicated. The renderer never uploads a level that large.
            **tss.multiscales_metadata(key, plan, include_scale0=False),
        }
    )

    if progress:
        progress("Done", total, total)
    return {
        "key": key,
        "tiled_path": "/".join([*target_parts, key]),
        "full_shape": list(shape),
        "pyramid_plan": plan,
    }
