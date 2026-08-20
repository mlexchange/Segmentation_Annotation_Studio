"""Downsampled 3-D volume assembly for the ``/api/image/volume`` endpoint.

The 3D tab renders the whole stack at once, so the browser never sees the
gigapixel-scale native resolution (e.g. 38 x 3232 x 3232) — this module builds
a small uint8 intensity volume server-side instead. The frontend rasterizes
its OWN class-label volume from live annotations onto the exact same grid
(see ``frontend/src/lib/volume/volumeDims.ts``), so the two must agree on
dimensions voxel-for-voxel — see :func:`volume_dims` for the shared contract.

Every slice is normalized with ``images.normalize_scalar_unit`` using the same
``RenderOpts``/``global_range`` the 2-D canvas uses for this source, so a
voxel's brightness matches what ``/api/image/slice`` would show at that
position. This is deliberate: a 3D view that looks different from the 2D
slice it was built from would be the exact bug this shared function prevents.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

import images as images_mod
from arrays import _stack_keys, read_slice

logger = logging.getLogger(__name__)

_VOLUME_WORKERS = 8


# Quality ladder offered by the 3-D tab, ascending. Mirrored EXACTLY by the
# frontend's `QUALITY_LADDER` — `effective_max_dim` walks it, so the two must
# agree or the client would rasterize its label volume onto a different grid
# than the raw volume it overlays.
QUALITY_LADDER = (128, 192, 256, 320, 384, 512, 640, 768, 1024)

# Voxel budget for ONE volume, as uint8 bytes. The 3-D tab holds two (raw plus
# the client-rasterized label volume), each also uploaded as a GL 3-D texture,
# so the real footprint is roughly double this.
#
# The per-axis cap alone is the wrong constraint: it makes an anisotropic stack
# needlessly coarse (63 x 3232^2 at max_dim=1024 is only 66 MB) while letting a
# cubic source reach 1024^3 = 1 GB. Budgeting total voxels lets a thin stack go
# to full quality and holds a cubic one to roughly 570^3.
MAX_VOLUME_VOXELS = int(os.getenv("MAX_VOLUME_VOXELS", str(192 * 1024 * 1024)))


def effective_max_dim(n_slices: int, height: int, width: int, requested: int) -> int:
    """Largest ladder quality <= *requested* whose grid fits the voxel budget.

    Mirrored byte-for-byte by the frontend's ``effectiveMaxDim``. Both sides
    clamp identically so the client can show (and rasterize onto) the grid it
    will actually receive, instead of discovering a silent server-side
    downgrade as a dimension mismatch.

    Only ever REDUCES: *requested* is returned untouched whenever it already
    fits, so arbitrary values (the route accepts any int in 32..1024, not just
    ladder entries) pass through unchanged and this can never hand back a
    coarser-than-asked *or* finer-than-asked grid.

    Falls back to the smallest of the ladder floor and *requested* when even
    that exceeds the budget — a coarse volume beats refusing to render one.
    """
    def _fits(quality: int) -> bool:
        d = volume_dims(n_slices, height, width, quality)
        return d["nz"] * d["ny"] * d["nx"] <= MAX_VOLUME_VOXELS

    if _fits(requested):
        return requested
    for quality in reversed([q for q in QUALITY_LADDER if q < requested]):
        if _fits(quality):
            return quality
    return min(requested, QUALITY_LADDER[0])


def volume_dims(n_slices: int, height: int, width: int, max_dim: int) -> dict[str, int]:
    """Compute the shared raw/label volume grid for an ``n_slices x height x
    width`` source downsampled so no axis exceeds ``max_dim``.

    This EXACT formula is mirrored by the frontend's ``volumeDimsFor``, which
    calls ``gridFor`` (``frontend/src/lib/rasterize.ts``) for ``nx``/``ny``/
    ``sxy`` so the two implementations can never independently drift — the
    backend raw volume and the client-rasterized label volume must line up
    voxel-for-voxel, or the segmentation overlay renders offset from the raw
    data it describes.

    ``gridFor`` FLOORS after striding (``floor(width / scale)``); a plain
    ``arr[::stride]`` CEILS. Callers of this function must therefore trim any
    strided read to ``(ny, nx)`` — never rely on the stride alone to produce
    the right shape.

    Returns:
        ``{"nz", "ny", "nx", "sxy", "sz"}`` — see the module for the
        z-index selection this implies (``range(0, n_slices, sz)``).
    """
    long_side = max(height, width)
    sxy = max(1, -(-long_side // max_dim))  # ceil division
    nx = max(1, width // sxy)
    ny = max(1, height // sxy)
    sz = 1 if n_slices <= max_dim else max(1, -(-n_slices // max_dim))
    nz = -(-n_slices // sz)  # ceil(n_slices / sz) == len(range(0, n_slices, sz))
    return {"nz": nz, "ny": ny, "nx": nx, "sxy": sxy, "sz": sz}


def _to_gray_u8(
    arr: np.ndarray,
    is_rgb: bool,
    opts: dict[str, Any],
    global_range: tuple[float, float] | None,
) -> np.ndarray:
    """Normalize one already-downsampled 2-D (or H x W x C) slice to uint8."""
    if is_rgb:
        rgb = np.asarray(arr)[:, :, :3].astype(np.float64)
        arr = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    unit = images_mod.normalize_scalar_unit(arr, opts, global_range)
    return (unit * 255.0).round().astype(np.uint8)


def build_volume(
    node: Any,
    meta: dict[str, Any],
    opts: dict[str, Any],
    max_dim: int,
    global_range: tuple[float, float] | None,
) -> tuple[bytes, dict[str, Any]]:
    """Assemble a downsampled uint8 raw-intensity volume for *node*.

    Args:
        node: Resolved array/container node (see ``arrays.resolve_array``).
        meta: Shape-dispatch dict from ``arrays.array_shape_meta``.
        opts: ``RenderOpts``-compatible dict (``norm``, ``scale``,
            ``vmin_pct``, ``vmax_pct``) — same shape ``images.render_slice``
            takes, minus ``cmap`` (a volume has no color map, only intensity).
        max_dim: Cap on any single axis of the output grid.
        global_range: ``(vmin, vmax)`` for ``norm == "global"``; ignored
            otherwise. Caller is responsible for computing it (typically via
            ``images._sample_global_stats``) so it matches the 2-D canvas.

    Returns:
        ``(payload_bytes, vol_meta)``. ``payload_bytes`` is C-order
        ``(nz, ny, nx)`` uint8. ``vol_meta`` is ``volume_dims(...)`` plus
        ``n_slices``, ``height``, ``width``, and ``skipped_z`` (z-indices
        into the OUTPUT volume — not source slice numbers — that could not
        be read and were left as zero).
    """
    n_slices, height, width = meta["n_slices"], meta["height"], meta["width"]
    is_rgb = meta["is_rgb"]
    dims = volume_dims(n_slices, height, width, max_dim)
    nz, ny, nx, sxy, sz = dims["nz"], dims["ny"], dims["nx"], dims["sxy"], dims["sz"]
    z_indices = list(range(0, n_slices, sz))

    volume = np.zeros((nz, ny, nx), dtype=np.uint8)
    skipped_z: list[int] = []
    shape_kind = meta["shape_kind"]

    if shape_kind in ("NHW", "NHWC"):
        try:
            raw = np.asarray(node[::sz, ::sxy, ::sxy])
            raw = raw[:nz, :ny, :nx, ...]
            for i in range(raw.shape[0]):
                volume[i] = _to_gray_u8(raw[i], is_rgb, opts, global_range)
        except Exception as exc:
            # A single bulk read failing (rather than a per-slice hiccup) means
            # the whole source is unreadable this way — fall back to reading
            # slice-by-slice via the same per-slice path STACK uses, so one
            # unusual array backend doesn't turn into a hard 500.
            logger.warning("Volume: bulk strided read failed, falling back per-slice: %s", exc)
            for i, z in enumerate(z_indices):
                try:
                    sl = np.asarray(read_slice(node, meta, z))[::sxy, ::sxy][:ny, :nx, ...]
                    volume[i] = _to_gray_u8(sl, is_rgb, opts, global_range)
                except Exception as inner_exc:
                    logger.warning("Volume: skipping unreadable slice %d: %s", z, inner_exc)
                    skipped_z.append(i)
    elif shape_kind == "STACK":
        keys = meta.get("keys") or _stack_keys(node)

        def _read_one(item: tuple[int, int]) -> tuple[int, np.ndarray | None]:
            i, z = item
            try:
                sl = np.asarray(node[keys[z]])[::sxy, ::sxy][:ny, :nx, ...]
                return i, _to_gray_u8(sl, is_rgb, opts, global_range)
            except Exception as exc:
                logger.warning("Volume: skipping unreadable slice %d (key %r): %s", z, keys[z] if z < len(keys) else "?", exc)
                return i, None

        with ThreadPoolExecutor(max_workers=min(_VOLUME_WORKERS, max(1, nz))) as ex:
            for i, gray in ex.map(_read_one, list(enumerate(z_indices))):
                if gray is None:
                    skipped_z.append(i)
                else:
                    volume[i] = gray
    else:  # HW / HWC — a single image, nz == 1
        sl = np.asarray(read_slice(node, meta, 0))[::sxy, ::sxy][:ny, :nx, ...]
        volume[0] = _to_gray_u8(sl, is_rgb, opts, global_range)

    vol_meta = {
        **dims,
        "n_slices": n_slices,
        "height": height,
        "width": width,
        "skipped_z": skipped_z,
    }
    return volume.tobytes(order="C"), vol_meta
