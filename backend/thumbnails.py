"""PNG thumbnail rendering for Tiled array nodes.

Handles two common cases:

* 3D RGB arrays of shape ``(H, W, 3)`` or ``(H, W, 4)`` — rendered directly.
* 2D intensity arrays — log-scaled, min/max normalised, then colour-mapped
  with ``viridis`` (falls back to grayscale if matplotlib isn't available).

Matplotlib is imported at module load (not per-request) to avoid paying the
import cost on every thumbnail.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image as PILImage

try:  # pragma: no cover — optional dependency
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as _plt

    _VIRIDIS = _plt.get_cmap("viridis")
except Exception:  # pragma: no cover
    _VIRIDIS = None  # type: ignore[assignment]


def render_thumbnail(node: Any, size: int = 256) -> bytes | None:
    """Render a PNG thumbnail for *node* (array or container-of-arrays).

    Returns ``None`` when no suitable array can be found at *node*.
    """
    array_node = _resolve_array_node(node)
    if array_node is None:
        return None

    arr = np.squeeze(np.asarray(array_node.read()))

    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return _encode_png(_prepare_rgb(arr[:, :, :3]), size)
    if arr.ndim == 2:
        return _encode_png(_prepare_intensity(arr), size)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_array_node(node: Any) -> Any | None:
    """Return *node* if it's already an array, else its first array child."""
    if hasattr(node, "read"):
        return node
    try:
        children = list(node)
    except (TypeError, KeyError):
        return None
    for k in children:
        if k.endswith("_qmap"):
            continue
        try:
            child = node[k]
        except (KeyError, TypeError):
            continue
        if hasattr(child, "read"):
            return child
    return None


def _prepare_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    lo = float(rgb.min())
    hi = float(rgb.max())
    if hi <= lo:
        return np.zeros_like(rgb, dtype=np.uint8)
    scaled = (rgb - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def _prepare_intensity(arr: np.ndarray) -> np.ndarray:
    """Log-scale, normalise, and colour-map a 2D intensity array to uint8 RGB."""
    data = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    data = np.log1p(np.maximum(data, 0.0))

    lo = float(np.nanmin(data))
    hi = float(np.nanmax(data))
    if hi > lo:
        data = (data - lo) / (hi - lo)
    else:
        data = np.zeros_like(data)
    data = np.clip(data, 0.0, 1.0)

    if _VIRIDIS is not None:
        rgb = (_VIRIDIS(data)[:, :, :3] * 255).astype(np.uint8)
    else:
        gray = (data * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb


def _encode_png(rgb: np.ndarray, size: int) -> bytes:
    h, w = rgb.shape[:2]
    scale = min(size / h, size / w, 1.0)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    img = PILImage.fromarray(rgb).resize((nw, nh), PILImage.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
