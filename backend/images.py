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

    def _minmax(i: int) -> tuple[float, float]:
        sl = np.asarray(read_slice(node, meta, i))[::4, ::4].astype(np.float64)
        return float(np.nanmin(sl)), float(np.nanmax(sl))

    with ThreadPoolExecutor(max_workers=min(8, len(indices))) as ex:
        pairs = list(ex.map(_minmax, indices))

    result: tuple[float, float] = (min(p[0] for p in pairs), max(p[1] for p in pairs))
    _stats_cache.set(cache_key, result)
    return result


def normalize_scalar_unit(
    arr: np.ndarray,
    opts: dict[str, Any],
    global_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Map a 2-D scalar array to ``[0, 1]`` via the scale transform + vmin/vmax
    normalisation ``render_slice``'s grayscale branch uses, stopping short of
    the final colormap/uint8 step.

    Extracted so :mod:`denoise_train` (``_slice_to_gray_uint8``) can train a
    denoiser on the exact same intensity pipeline the 2-D canvas renders,
    without duplicating this logic and risking the two drifting apart.

    Args:
        arr: 2-D grayscale array.
        opts: :class:`~schemas.RenderOpts`-compatible dict with keys
            ``norm``, ``scale``, ``vmin_pct``, ``vmax_pct``.
        global_range: ``(vmin, vmax)`` used when ``opts["norm"] == "global"``.
            Ignored in slice-norm mode.

    Returns:
        ``float64`` array the same shape as *arr*, values in ``[0, 1]``.
    """
    data = arr.astype(np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    scale = opts.get("scale", "linear")
    if scale == "log":
        shifted = data - float(np.nanmin(data))
        data = np.log1p(np.maximum(shifted, 0.0))
    elif scale == "symlog":
        data = np.sign(data) * np.log1p(np.abs(data))

    if opts.get("norm", "global") == "global" and global_range is not None:
        vmin_abs, vmax_abs = global_range
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
            rgb = (viridis(data)[:, :, :3] * 255).astype(np.uint8)
        except Exception:
            gray = (data * 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        gray = (data * 255).astype(np.uint8)
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
