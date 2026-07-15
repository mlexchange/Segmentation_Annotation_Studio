"""Backend SlimSAM vision-encoder embeddings for CatBoost pixel features.

Uses the same vendored ONNX as the frontend Magic/SAM tool when available
(``frontend/public/models/slimsam-77-uniform``), or ``backend/models/...``.
Embeddings stay at ``64×64×256`` — never upsampled to full HxW.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = Path(__file__).resolve().parent

# ImageNet mean/std matching SamImageProcessor / preprocessor_config.json
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_LONG_EDGE = 1024
_PAD = 1024
_EMB_SIDE = 64  # encoder spatial grid for 1024 input


def resolve_vision_encoder_path() -> Path | None:
    """Locate ``vision_encoder.onnx`` (backend models or frontend public)."""
    candidates = [
        _BACKEND_DIR / "models" / "slimsam-77-uniform" / "onnx" / "vision_encoder.onnx",
        _REPO_ROOT / "frontend" / "public" / "models" / "slimsam-77-uniform" / "onnx" / "vision_encoder.onnx",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def sam_available() -> bool:
    """True when the SlimSAM vision encoder ONNX file is present."""
    return resolve_vision_encoder_path() is not None


@lru_cache(maxsize=1)
def _session():
    """Lazy CPU InferenceSession (thread-safe via lock below)."""
    import onnxruntime as ort

    path = resolve_vision_encoder_path()
    if path is None:
        raise FileNotFoundError(
            "SlimSAM vision_encoder.onnx not found. Vendor via "
            "`node frontend/scripts/fetch-sam-model.mjs` or place under "
            "backend/models/slimsam-77-uniform/onnx/."
        )
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


_session_lock = threading.Lock()


def rgb_uint8_from_array(arr: np.ndarray) -> np.ndarray:
    """Convert a slice array to HxWx3 uint8 RGB (grayscale → triple)."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        rgb = a[..., :3].astype(np.float64)
    elif a.ndim == 2:
        g = a.astype(np.float64)
        rgb = np.stack([g, g, g], axis=-1)
    else:
        raise ValueError(f"unsupported array shape {a.shape}")
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        return np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
    scaled = (rgb - lo) / (hi - lo) * 255.0
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def preprocess_sam_rgb(rgb_u8: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Resize longest edge to 1024, pad to 1024², ImageNet-normalize NCHW float32.

    Returns:
        ``(pixel_values [1,3,1024,1024], (orig_h, orig_w), (reshaped_h, reshaped_w))``.
    """
    img = PILImage.fromarray(rgb_u8, mode="RGB")
    ow, oh = img.size  # PIL: width, height
    scale = _LONG_EDGE / float(max(oh, ow))
    nw = max(1, int(round(ow * scale)))
    nh = max(1, int(round(oh * scale)))
    # resample=2 → BILINEAR in HF preprocessor_config
    resized = img.resize((nw, nh), resample=PILImage.BILINEAR)
    canvas = PILImage.new("RGB", (_PAD, _PAD), (0, 0, 0))
    canvas.paste(resized, (0, 0))  # top-left pad (SAM processor default)
    arr = np.asarray(canvas, dtype=np.float32) * (1.0 / 255.0)
    nchw = np.transpose(arr, (2, 0, 1))  # 3,H,W
    nchw = (nchw - _MEAN) / _STD
    return nchw[np.newaxis, ...].astype(np.float32), (oh, ow), (nh, nw)


def encode_image_embeddings(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Run SlimSAM vision encoder; return ``(eh,ew,C) float32`` + size metadata.

    Output layout is ``(64, 64, 256)`` channels-last for bilinear sampling.
    """
    rgb = rgb_uint8_from_array(arr)
    pixel_values, orig_hw, reshaped_hw = preprocess_sam_rgb(rgb)
    with _session_lock:
        sess = _session()
        outs = sess.run(None, {sess.get_inputs()[0].name: pixel_values})
    # First output: image_embeddings [1, 256, 64, 64]
    emb = np.asarray(outs[0], dtype=np.float32)
    if emb.ndim != 4 or emb.shape[0] != 1:
        raise RuntimeError(f"unexpected embedding shape {emb.shape}")
    # → 64,64,256
    spatial = np.transpose(emb[0], (1, 2, 0))
    return np.ascontiguousarray(spatial), orig_hw, reshaped_hw


def bilinear_sample_emb(
    emb: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    orig_h: int,
    orig_w: int,
    reshaped_h: int,
    reshaped_w: int,
) -> np.ndarray:
    """Sample embedding channels at original-image pixel coords (bilinear).

    Maps ``(x,y)`` through the longest-edge resize into the 1024 pad, then into
    the ``64×64`` embedding grid.
    """
    eh, ew, c = emb.shape
    # resized coords in pad space (top-left paste)
    scale_y = reshaped_h / float(orig_h)
    scale_x = reshaped_w / float(orig_w)
    py = ys.astype(np.float64) * scale_y
    px = xs.astype(np.float64) * scale_x
    # pad → emb (1024 / 64 = 16)
    cell = _PAD / float(eh)
    fy = (py + 0.5) / cell - 0.5
    fx = (px + 0.5) / cell - 0.5
    fy = np.clip(fy, 0.0, eh - 1.001)
    fx = np.clip(fx, 0.0, ew - 1.001)
    y0 = np.floor(fy).astype(np.int32)
    x0 = np.floor(fx).astype(np.int32)
    y1 = np.minimum(y0 + 1, eh - 1)
    x1 = np.minimum(x0 + 1, ew - 1)
    wy = (fy - y0).astype(np.float32)[:, None]
    wx = (fx - x0).astype(np.float32)[:, None]
    v00 = emb[y0, x0]
    v01 = emb[y0, x1]
    v10 = emb[y1, x0]
    v11 = emb[y1, x1]
    top = v00 * (1.0 - wx) + v01 * wx
    bot = v10 * (1.0 - wx) + v11 * wx
    return top * (1.0 - wy) + bot * wy


def fit_pca(x: np.ndarray, n_components: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Fit a simple PCA; return ``(mean [C], components [k,C])``."""
    if x.ndim != 2:
        raise ValueError("x must be (n, c)")
    n, c = x.shape
    k = int(min(n_components, c, max(1, n - 1)))
    mean = x.mean(axis=0)
    xc = x - mean
    # Economy SVD on n×c (prefer when n >= c use XtX path via svd of xc)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:k].astype(np.float32)
    return mean.astype(np.float32), components


def transform_pca(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project rows of *x* with a fitted PCA."""
    return ((x - mean) @ components.T).astype(np.float32)
