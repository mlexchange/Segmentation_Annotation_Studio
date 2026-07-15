"""Multiscale scale-space features via skimage (intensity / edges / texture).

Used by the Annotate UI to compute a feature bank on the current slice, CLAHE-
normalize each channel, and serve channels as gray PNGs for display / tools.
Also keeps a float stack for pixel classifiers (CatBoost), optionally plus a
SlimSAM embedding grid (never full-res).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image as PILImage
from skimage.exposure import equalize_adapthist
from skimage.feature import multiscale_basic_features

from cache import TTLCache

logger = logging.getLogger(__name__)

_job_cache: TTLCache = TTLCache(ttl_seconds=600.0, max_entries=32)


@dataclass(frozen=True)
class FeatureJob:
    """Cached multiscale feature computation result."""

    job_id: str
    labels: tuple[str, ...]
    stack: np.ndarray  # HxWxC uint8 (display)
    float_stack: np.ndarray  # HxWxC float32 (classifier)
    sam_emb: np.ndarray | None = None  # eh×ew×C float32 (optional)
    sam_orig_hw: tuple[int, int] | None = None  # (H, W) of source slice
    sam_reshaped_hw: tuple[int, int] | None = None  # resize before pad


def list_sigmas(sigma_min: float, sigma_max: float, num_sigma: int | None = None) -> np.ndarray:
    """Return the σ grid used by ``multiscale_basic_features`` (powers of 2 by default)."""
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


def feature_channel_labels(
    *,
    sigma_min: float = 1.0,
    sigma_max: float = 8.0,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    num_sigma: int | None = None,
) -> list[str]:
    """Human-readable labels matching skimage's per-σ feature order.

    Order per sigma: intensity (optional), edges (optional), texture λ− / λ+
    (optional, two eigenvalues).
    """
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


def _fmt_sigma(sigma: float) -> str:
    if abs(sigma - round(sigma)) < 1e-6:
        return str(int(round(sigma)))
    return f"{sigma:g}"


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert a slice to float grayscale in [0, 1] (approx)."""
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
    out = (gray - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _float_feats_to_uint8(feats: np.ndarray, *, clahe: bool) -> np.ndarray:
    """Convert float HxWxC features to display uint8 (optional per-channel CLAHE)."""
    h, w, c = feats.shape
    out = np.empty((h, w, c), dtype=np.uint8)
    for i in range(c):
        ch = feats[..., i].astype(np.float64)
        lo, hi = float(np.nanmin(ch)), float(np.nanmax(ch))
        if hi > lo:
            norm = (ch - lo) / (hi - lo)
        else:
            norm = np.zeros_like(ch)
        if clahe:
            norm = equalize_adapthist(np.clip(norm, 0.0, 1.0), clip_limit=0.01)
        out[..., i] = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    return out


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
    workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute display uint8 + float feature banks.

    Returns:
        ``(uint8_stack, float_stack, labels)`` both ``HxWxC``.
    """
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
        workers=workers,
    )
    if feats.shape[-1] != len(labels):
        raise RuntimeError(
            f"feature count mismatch: got {feats.shape[-1]} channels, "
            f"expected {len(labels)} labels"
        )
    float_stack = np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    uint8_stack = _float_feats_to_uint8(float_stack, clahe=clahe)
    return uint8_stack, float_stack, labels


def compute_feature_stack(
    gray: np.ndarray,
    *,
    sigma_min: float = 1.0,
    sigma_max: float = 8.0,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    clahe: bool = True,
    num_sigma: int | None = None,
    workers: int | None = 1,
) -> tuple[np.ndarray, list[str]]:
    """Compute the multiscale feature bank and optionally CLAHE each channel.

    Returns:
        ``(stack, labels)`` where ``stack`` is ``HxWxC`` uint8.
    """
    stack, _, labels = compute_feature_stacks(
        gray,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        intensity=intensity,
        edges=edges,
        texture=texture,
        clahe=clahe,
        num_sigma=num_sigma,
        workers=workers,
    )
    return stack, labels


def encode_channel_png(stack: np.ndarray, index: int) -> bytes:
    """Encode one feature channel as a grayscale PNG."""
    if index < 0 or index >= stack.shape[-1]:
        raise IndexError(f"channel index {index} out of range 0..{stack.shape[-1] - 1}")
    img = PILImage.fromarray(stack[..., index], mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def store_job(
    stack: np.ndarray,
    labels: list[str],
    *,
    float_stack: np.ndarray,
    sam_emb: np.ndarray | None = None,
    sam_orig_hw: tuple[int, int] | None = None,
    sam_reshaped_hw: tuple[int, int] | None = None,
) -> FeatureJob:
    """Cache a computed stack and return a job handle."""
    if stack.shape != float_stack.shape:
        raise ValueError("stack and float_stack shapes must match")
    job_id = uuid.uuid4().hex
    job = FeatureJob(
        job_id=job_id,
        labels=tuple(labels),
        stack=stack,
        float_stack=float_stack,
        sam_emb=sam_emb,
        sam_orig_hw=sam_orig_hw,
        sam_reshaped_hw=sam_reshaped_hw,
    )
    _job_cache.set(job_id, job)
    return job


def get_job(job_id: str) -> FeatureJob | None:
    """Look up a cached feature job (or ``None`` if missing/expired)."""
    job = _job_cache.get(job_id)
    return job if isinstance(job, FeatureJob) else None


def compute_and_store(
    arr: np.ndarray,
    *,
    sigma_min: float = 1.0,
    sigma_max: float = 8.0,
    intensity: bool = True,
    edges: bool = True,
    texture: bool = True,
    clahe: bool = True,
    include_sam: bool = False,
) -> FeatureJob:
    """Grayscale convert *arr*, compute features (+ optional SlimSAM), cache."""
    gray = to_grayscale(arr)
    stack, float_stack, labels = compute_feature_stacks(
        gray,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        intensity=intensity,
        edges=edges,
        texture=texture,
        clahe=clahe,
        workers=1,
    )
    sam_emb = None
    sam_orig_hw = None
    sam_reshaped_hw = None
    if include_sam:
        from sam_embed import encode_image_embeddings, sam_available

        if not sam_available():
            logger.warning("include_sam requested but vision_encoder.onnx missing")
        else:
            sam_emb, sam_orig_hw, sam_reshaped_hw = encode_image_embeddings(arr)
            logger.info(
                "SlimSAM embedding %s for slice %s (reshaped %s)",
                sam_emb.shape,
                sam_orig_hw,
                sam_reshaped_hw,
            )
    return store_job(
        stack,
        labels,
        float_stack=float_stack,
        sam_emb=sam_emb,
        sam_orig_hw=sam_orig_hw,
        sam_reshaped_hw=sam_reshaped_hw,
    )
