"""Slice-to-PNG rendering with explicit normalisation modes.

Two normalisation scopes are supported:

* ``"slice"`` — min/max percentiles computed from the current slice only.
* ``"global"`` — percentiles derived from up to 64 evenly-spaced slices
  sampled across the whole volume (result cached for 5 minutes).

Intensity transforms (``scale``) are applied *before* percentile clamping:

* ``"linear"`` — no transform.
* ``"log"``    — ``log1p`` of the min-shifted data.
* ``"symlog"`` — ``sign(x) * log1p(|x|)``.

Colour maps: ``"gray"`` (always available) and ``"viridis"`` (requires
``matplotlib``; gracefully degrades to grayscale if unavailable).
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import numpy as np
from fastapi import HTTPException
from PIL import Image as PILImage

from cache import TTLCache

logger = logging.getLogger(__name__)
_stats_cache: TTLCache = TTLCache(ttl_seconds=300.0, max_entries=64)


def _sample_global_stats(node: Any, meta: dict[str, Any]) -> tuple[float, float]:
    """Compute global vmin/vmax from up to 64 evenly-spaced slices.

    Results are cached by ``id(node)`` for 5 minutes.

    Args:
        node: Array node (Tiled or NumPy-compatible).
        meta: Shape-dispatch dict from :func:`arrays.array_shape_meta`.

    Returns:
        ``(vmin, vmax)`` floats across all sampled slices.
    """
    from concurrent.futures import ThreadPoolExecutor

    from arrays import read_slice

    cache_key = ("global_stats", id(node))
    cached = _stats_cache.get(cache_key)
    if cached is not None:
        return cached

    # Sample fewer slices, read them concurrently, and spatially subsample each
    # (every 4th pixel) — min/max are robust to this and the cost drops sharply.
    # Result still spans the whole volume, so it matches the viewer's contrast.
    n = meta["n_slices"]
    samples = 24
    indices = list(range(0, n, max(1, n // samples)))[:samples] or [0]

    def _minmax(i: int) -> tuple[float, float] | None:
        # A "stack" container can hold a sibling slice with a different shape
        # than the one that seeded `meta` (e.g. two unrelated single-image
        # samples living side by side) — read_slice rejects it. That sibling
        # is just one of many contrast samples, not the slice being viewed, so
        # skip it rather than failing the whole request over an index nobody
        # asked to see.
        try:
            sl = np.asarray(read_slice(node, meta, i))[::4, ::4].astype(np.float64)
        except Exception as exc:
            logger.warning("Global stats: skipping unreadable slice %d: %s", i, exc)
            return None
        return float(np.nanmin(sl)), float(np.nanmax(sl))

    with ThreadPoolExecutor(max_workers=min(8, len(indices))) as ex:
        pairs = [p for p in ex.map(_minmax, indices) if p is not None]

    if not pairs:
        raise HTTPException(422, "No readable slices found while sampling global contrast stats")

    result: tuple[float, float] = (min(p[0] for p in pairs), max(p[1] for p in pairs))
    _stats_cache.set(cache_key, result)
    return result


def normalize_scalar_unit(
    arr: np.ndarray,
    opts: dict[str, Any],
    global_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize a 2-D grayscale array to float64 in ``[0, 1]``.

    This is the exact intensity pipeline behind :func:`render_slice`'s
    grayscale branch (NaN handling, log/symlog transform, global-vs-slice
    percentile bounds), extracted so the 3-D volume endpoint (``volumes.py``)
    can normalize each slice identically to the 2-D canvas — same
    ``RenderOpts``, same ``global_range``, same output domain, just skipping
    the final uint8/colormap step. If you change intensity math here, do NOT
    duplicate it elsewhere; a 3D volume that looks different from the 2D
    slice it was built from is the exact bug this split is meant to prevent.

    Args:
        arr: 2-D grayscale array (native dtype).
        opts: :class:`~schemas.RenderOpts`-compatible dict with keys
            ``norm``, ``scale``, ``vmin_pct``, ``vmax_pct``.
        global_range: ``(vmin, vmax)`` used when ``opts["norm"] == "global"``.
            Ignored in slice-norm mode.

    Returns:
        float64 array of the same shape as *arr*, clamped to ``[0, 1]``.
    """
    data = arr.astype(np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    scale = opts.get("scale", "linear")
    use_global = opts.get("norm", "global") == "global" and global_range is not None

    # The intensity transform must be applied to the per-pixel data AND to the
    # display bounds through the exact same function/shift, or they end up in
    # different numeric domains — e.g. log-transformed pixels (small values)
    # clamped against an untransformed global min/max (large raw values)
    # collapses the whole image to near-black. `_sample_global_stats` samples
    # in the raw domain (cheap, dtype-agnostic), so that transform happens here.
    if scale == "log":
        # Global mode shifts by the raw global min (one baseline shared across
        # every slice, so log values stay comparable slice-to-slice). Slice mode
        # keeps the previous per-slice shift (this slice's own min).
        shift = float(global_range[0]) if use_global else float(np.nanmin(data))
        data = np.log1p(np.maximum(data - shift, 0.0))
    elif scale == "symlog":
        data = np.sign(data) * np.log1p(np.abs(data))

    if use_global:
        raw_vmin, raw_vmax = global_range
        if scale == "log":
            vmin_abs, vmax_abs = 0.0, float(np.log1p(max(raw_vmax - raw_vmin, 0.0)))
        elif scale == "symlog":
            vmin_abs = float(np.sign(raw_vmin) * np.log1p(abs(raw_vmin)))
            vmax_abs = float(np.sign(raw_vmax) * np.log1p(abs(raw_vmax)))
        else:
            vmin_abs, vmax_abs = raw_vmin, raw_vmax
    else:
        vmin_pct = opts.get("vmin_pct", 1.0)
        vmax_pct = opts.get("vmax_pct", 99.0)
        flat = data[np.isfinite(data)]
        if len(flat):
            vmin_abs = float(np.percentile(flat, vmin_pct))
            vmax_abs = float(np.percentile(flat, vmax_pct))
        else:
            vmin_abs, vmax_abs = 0.0, 1.0

    if vmax_abs > vmin_abs:
        data = (data - vmin_abs) / (vmax_abs - vmin_abs)
    else:
        data = np.zeros_like(data)
    return np.clip(data, 0.0, 1.0)


def render_slice(
    arr: np.ndarray,
    opts: dict[str, Any],
    global_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a 2-D or H×W×C array slice to uint8 RGB.

    Args:
        arr: 2-D grayscale or H×W×(3|4) colour array.
        opts: :class:`~schemas.RenderOpts`-compatible dict with keys
            ``norm``, ``scale``, ``vmin_pct``, ``vmax_pct``, ``cmap``.
        global_range: ``(vmin, vmax)`` used when ``opts["norm"] == "global"``.
            Ignored in slice-norm mode.

    Returns:
        uint8 RGB array of shape ``(H, W, 3)``.
    """
    is_rgb = arr.ndim == 3
    if is_rgb:
        rgb = arr[:, :, :3].astype(np.float64)
        mn, mx = float(rgb.min()), float(rgb.max())
        if mx > mn:
            rgb = (rgb - mn) / (mx - mn) * 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)

    data = normalize_scalar_unit(arr, opts, global_range)

    cmap = opts.get("cmap", "gray")
    if cmap == "viridis":
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt

            viridis = plt.get_cmap("viridis")
            # round(), not truncate: volumes.build_volume quantizes the same
            # [0,1] unit data with round() for its 3-D voxels (see
            # normalize_scalar_unit), so a bare cast here would make a 2-D
            # slice and the 3-D volume it should match disagree by up to 1
            # LSB at every pixel — exactly the drift normalize_scalar_unit
            # was split out to prevent.
            rgb = (viridis(data)[:, :, :3] * 255).round().astype(np.uint8)
        except Exception:
            gray = (data * 255).round().astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        gray = (data * 255).round().astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

    return rgb


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode a uint8 RGB array to PNG bytes.

    Args:
        rgb: uint8 array of shape ``(H, W, 3)``.

    Returns:
        PNG-encoded bytes suitable for an HTTP response body.
    """
    buf = BytesIO()
    PILImage.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()
