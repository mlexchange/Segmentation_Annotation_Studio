"""Multiscale feature computation (ported; no annotate-backend imports)."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image as PILImage
from skimage.exposure import equalize_adapthist
from skimage.feature import multiscale_basic_features


def list_sigmas(sigma_min: float, sigma_max: float, num_sigma: int | None = None) -> np.ndarray:
    """Return the σ grid used by multiscale_basic_features."""
    if sigma_min <= 0 or sigma_max < sigma_min:
        raise ValueError(f"invalid sigma range [{sigma_min}, {sigma_max}]")
    if num_sigma is None:
        num_sigma = int(np.log2(sigma_max) - np.log2(sigma_min) + 1)
    return np.logspace(
        np.log2(sigma_min),
        np.log2(sigma_max),
        num=int(num_sigma),
        base=2,
        endpoint=True,
    )


def _fmt_sigma(sigma: float) -> str:
    if abs(sigma - round(sigma)) < 1e-6:
        return str(int(round(sigma)))
    return f"{sigma:g}"


def feature_channel_labels(
    *,
    sigma_min: float = 1.0,
    sigma_max: float = 8.0,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    num_sigma: int | None = None,
) -> list[str]:
    """Human-readable labels matching skimage per-σ order."""
    if not any((intensity, edges, texture)):
        raise ValueError("at least one of intensity, edges, texture must be True")
    labels: list[str] = []
    for sigma in list_sigmas(sigma_min, sigma_max, num_sigma):
        s = _fmt_sigma(float(sigma))
        if intensity:
            labels.append(f"intensity σ={s}")
        if edges:
            labels.append(f"edges σ={s}")
        if texture:
            labels.append(f"texture λ− σ={s}")
            labels.append(f"texture λ+ σ={s}")
    return labels


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert a slice to float grayscale in [0, 1]."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        rgb = a[..., :3].astype(np.float64)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    elif a.ndim == 2:
        gray = a.astype(np.float64)
    else:
        raise ValueError(f"unsupported array shape {a.shape}")
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return np.zeros_like(gray, dtype=np.float64)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float64)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)


def _float_feats_to_uint8(feats: np.ndarray, *, clahe: bool) -> np.ndarray:
    """Convert float HxWxC to display uint8."""
    h, w, c = feats.shape
    out = np.empty((h, w, c), dtype=np.uint8)
    for i in range(c):
        ch = feats[..., i].astype(np.float64)
        lo, hi = float(np.nanmin(ch)), float(np.nanmax(ch))
        norm = (ch - lo) / (hi - lo) if hi > lo else np.zeros_like(ch)
        if clahe:
            norm = equalize_adapthist(np.clip(norm, 0.0, 1.0), clip_limit=0.01)
        out[..., i] = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    return out


def apply_clahe(
    gray: np.ndarray,
    *,
    clip_limit: float = 0.01,
    kernel_size: int | None = None,
) -> np.ndarray:
    """CLAHE on a [0, 1] grayscale image; returns float32 in [0, 1]."""
    g = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    kwargs: dict = {"clip_limit": float(clip_limit)}
    if kernel_size is not None and int(kernel_size) > 0:
        kwargs["kernel_size"] = int(kernel_size)
    out = equalize_adapthist(g, **kwargs)
    return np.asarray(out, dtype=np.float32)


def compute_clahe_stack(
    gray: np.ndarray,
    *,
    clip_limit: float = 0.01,
    kernel_size: int | None = None,
    apply: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return single-channel ``(uint8_stack, float_stack, labels)`` with optional CLAHE.

    Args:
        gray: Grayscale float image in ``[0, 1]``.
        clip_limit: skimage ``equalize_adapthist`` clip limit.
        kernel_size: Optional CLAHE tile size; ``None`` uses skimage default.
        apply: When False, store min-max gray only (no CLAHE).
    """
    g = np.asarray(gray, dtype=np.float32)
    if apply:
        ch = apply_clahe(g, clip_limit=clip_limit, kernel_size=kernel_size)
        label = "clahe"
    else:
        ch = np.clip(g, 0.0, 1.0).astype(np.float32)
        label = "intensity"
    float_stack = ch[..., None]
    uint8_stack = np.clip(np.round(float_stack * 255.0), 0, 255).astype(np.uint8)
    return uint8_stack, float_stack, [label]


def compute_feature_stacks(
    gray: np.ndarray,
    *,
    sigma_min: float = 1.0,
    sigma_max: float = 8.0,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    clahe: bool = True,
    num_sigma: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(uint8_stack, float_stack, labels)``."""
    labels = feature_channel_labels(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        intensity=intensity,
        edges=edges,
        texture=texture,
        num_sigma=num_sigma,
    )
    feats = multiscale_basic_features(
        np.asarray(gray, dtype=np.float32),
        intensity=intensity,
        edges=edges,
        texture=texture,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_sigma=num_sigma,
        workers=1,
    )
    if feats.shape[-1] != len(labels):
        raise RuntimeError(
            f"feature count mismatch: got {feats.shape[-1]}, expected {len(labels)}"
        )
    float_stack = np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    uint8_stack = _float_feats_to_uint8(float_stack, clahe=clahe)
    return uint8_stack, float_stack, labels


def encode_channel_png(stack: np.ndarray, index: int) -> bytes:
    """Encode one feature channel as grayscale PNG."""
    if index < 0 or index >= stack.shape[-1]:
        raise IndexError(f"channel index {index} out of range")
    img = PILImage.fromarray(stack[..., index], mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
