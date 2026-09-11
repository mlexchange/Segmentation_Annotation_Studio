"""SlimSAM ONNX embeddings for feature banks (self-contained)."""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_LONG_EDGE = 1024
_PAD = 1024

_session_lock = threading.Lock()


def resolve_encoder_path(explicit: str | None = None) -> Path | None:
    """Resolve ONNX path from explicit path, env, or repo defaults."""
    if explicit and explicit != "(missing)":
        p = Path(explicit).expanduser().resolve()
        if p.is_file():
            return p
    env = os.getenv("FEATURE_ENCODER_ONNX")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    here = Path(__file__).resolve()
    repo = here.parents[3]
    for c in (
        repo / "backend" / "models" / "slimsam-77-uniform" / "onnx" / "vision_encoder.onnx",
        repo / "frontend" / "public" / "models" / "slimsam-77-uniform" / "onnx" / "vision_encoder.onnx",
    ):
        if c.is_file():
            return c.resolve()
    return None


def encoder_available(weights_path: str | None = None) -> bool:
    """True when an ONNX encoder file is present."""
    return resolve_encoder_path(weights_path) is not None


@lru_cache(maxsize=4)
def _session_for(path_str: str):
    import onnxruntime as ort

    return ort.InferenceSession(path_str, providers=["CPUExecutionProvider"])


def rgb_uint8_from_array(arr: np.ndarray) -> np.ndarray:
    """Convert slice to HxWx3 uint8 RGB."""
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
    return np.clip(scaled, 0, 255).astype(np.uint8)


def encode_image_embeddings(
    arr: np.ndarray,
    *,
    weights_path: str | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Return ``(emb eh×ew×C, orig_hw, reshaped_hw)``."""
    path = resolve_encoder_path(weights_path)
    if path is None:
        raise FileNotFoundError("SlimSAM vision_encoder.onnx not found")

    rgb = rgb_uint8_from_array(arr)
    oh, ow = int(rgb.shape[0]), int(rgb.shape[1])
    scale = _LONG_EDGE / max(oh, ow)
    rh = max(1, int(round(oh * scale)))
    rw = max(1, int(round(ow * scale)))
    resized = np.asarray(
        PILImage.fromarray(rgb).resize((rw, rh), PILImage.BILINEAR),
        dtype=np.uint8,
    )
    canvas = np.zeros((_PAD, _PAD, 3), dtype=np.uint8)
    canvas[:rh, :rw] = resized
    x = canvas.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = (x - _MEAN) / _STD
    x = np.expand_dims(x, 0)

    with _session_lock:
        sess = _session_for(str(path))
        inp = sess.get_inputs()[0].name
        out = sess.run(None, {inp: x})[0]

    # NCHW → HWC
    emb = np.transpose(np.asarray(out[0], dtype=np.float32), (1, 2, 0))
    return emb, (oh, ow), (rh, rw)


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
    """Sample embedding at pixel coords (orig image space)."""
    eh, ew, c = emb.shape
    # Map orig → reshaped → emb grid
    scale_y = reshaped_h / max(orig_h, 1)
    scale_x = reshaped_w / max(orig_w, 1)
    fy = (ys.astype(np.float64) * scale_y) * (eh / max(reshaped_h, 1))
    fx = (xs.astype(np.float64) * scale_x) * (ew / max(reshaped_w, 1))
    fy = np.clip(fy, 0, eh - 1.001)
    fx = np.clip(fx, 0, ew - 1.001)
    y0 = np.floor(fy).astype(np.int64)
    x0 = np.floor(fx).astype(np.int64)
    y1 = np.minimum(y0 + 1, eh - 1)
    x1 = np.minimum(x0 + 1, ew - 1)
    wy = fy - y0
    wx = fx - x0
    ia = emb[y0, x0]
    ib = emb[y0, x1]
    ic = emb[y1, x0]
    id_ = emb[y1, x1]
    wa = ((1 - wy) * (1 - wx))[:, None]
    wb = ((1 - wy) * wx)[:, None]
    wc = (wy * (1 - wx))[:, None]
    wd = (wy * wx)[:, None]
    return (wa * ia + wb * ib + wc * ic + wd * id_).astype(np.float32)


def fit_pca(x: np.ndarray, n_components: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA; return mean and components (n_comp × d)."""
    from sklearn.decomposition import PCA

    n = min(n_components, x.shape[0], x.shape[1])
    pca = PCA(n_components=n, svd_solver="randomized", random_state=0)
    pca.fit(x)
    return pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def transform_pca(
    x: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Project rows with fitted PCA."""
    return ((x - mean) @ components.T).astype(np.float32)


def emb_grid_to_pca_channels(
    emb: np.ndarray,
    *,
    out_h: int,
    out_w: int,
    n_components: int = 64,
    max_fit_samples: int = 50_000,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """PCA on dense emb tokens, upsample to ``out_h×out_w×K`` float channels.

    Args:
        emb: Encoder embedding grid ``eh×ew×C``.
        out_h / out_w: Full-resolution image size for heatmaps / float_stack.
        n_components: Target PCA dims (capped by tokens and channels).
        max_fit_samples: Subsample cap when fitting PCA on large grids.

    Returns:
        ``(float_hwk, labels, info)`` with labels ``pca0…pca{K-1}``.
    """
    from skimage.transform import resize

    eh, ew, c = emb.shape
    flat = np.asarray(emb, dtype=np.float32).reshape(-1, c)
    n_tok = flat.shape[0]
    k_target = max(1, int(n_components))
    if n_tok > max_fit_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_tok, size=max_fit_samples, replace=False)
        fit_x = flat[idx]
    else:
        fit_x = flat
    mean, comp = fit_pca(fit_x, n_components=k_target)
    projected = transform_pca(flat, mean, comp)
    k = projected.shape[1]
    grid = projected.reshape(eh, ew, k)
    up = resize(
        grid,
        (int(out_h), int(out_w), k),
        order=1,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float32)
    labels = [f"pca{i}" for i in range(k)]
    info = {
        "pca_dims": int(k),
        "emb_shape": [int(eh), int(ew), int(c)],
        "baked_into_float_stack": True,
    }
    return up, labels, info
