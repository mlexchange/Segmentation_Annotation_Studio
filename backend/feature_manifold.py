"""Greedy feature-variance box sampling for annotation guidance.

Sliding windows scored by whitened feature variance; picks suppress nearby
(spatial Chebyshev exclusion ≥ box side) and feature-similar boxes.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image as PILImage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cache import TTLCache
from multiscale_features import FeatureJob

logger = logging.getLogger(__name__)

_sample_cache: TTLCache = TTLCache(ttl_seconds=600.0, max_entries=32)
_COS_HARD = 0.95
_COS_ALPHA = 2.0


@dataclass(frozen=True)
class ManifoldSample:
    """Inducing boxes + residual interestingness heatmap."""

    job_id: str
    points: list[dict[str, Any]]
    heatmap: np.ndarray  # HxW float32 in [0, 1]
    meta: dict[str, Any]


@dataclass(frozen=True)
class CachedManifoldSample:
    """Server-side manifold sample for PNG GET."""

    sample_id: str
    heatmap_png: bytes
    points: list[dict[str, Any]]
    meta: dict[str, Any]


def derived_exclusion_radius(h: int, w: int, k: int) -> float:
    """Disk radius so roughly K disks pack the image plane."""
    k = max(1, int(k))
    area = float(h * w)
    r = math.sqrt(area / (math.pi * k))
    r_max = min(h, w) / 4.0
    return float(max(8.0, min(r, r_max)))


def sample_inducing_points(
    job: FeatureJob,
    *,
    k: int = 24,
    box_size: int | None = None,
    stride: int | None = None,
    pca_dims: int = 16,
    seed: int = 0,
    mask: np.ndarray | None = None,
) -> ManifoldSample:
    """Greedy variance boxes with spatial + feature-space exclusion.

    Args:
        job: Cached feature bank (``float_stack``).
        k: Number of inducing boxes (clamped to [2, 128]).
        box_size: Full square side length in image pixels. When omitted, derived
            from the K-based exclusion radius (``2 * round(r)``).
        stride: Optional hop override; default ``max(4, box_half)``.
        pca_dims: PCA dimensionality after standardization.
        seed: RNG for PCA solver stability.
        mask: Optional HxW placement mask. When set, only windows whose full
            box lies inside the mask are eligible.

    Returns:
        ManifoldSample with box centers, boxes, radius, and residual heatmap.
    """
    feats = np.asarray(job.float_stack, dtype=np.float32)
    if feats.ndim != 3:
        raise ValueError(f"float_stack must be HxWxC, got {feats.shape}")
    h, w, c = feats.shape
    if c < 1:
        raise ValueError("empty feature stack")

    place_mask: np.ndarray | None = None
    mask_pixels = 0
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != (h, w):
            raise ValueError(f"mask shape {m.shape} does not match image {(h, w)}")
        place_mask = m.astype(bool, copy=False)
        mask_pixels = int(place_mask.sum())
        if mask_pixels < 1:
            raise ValueError("placement mask is empty")

    k = int(max(2, min(128, k)))
    r_pack = derived_exclusion_radius(h, w, k)
    if box_size is None:
        side = int(max(8, 2 * round(r_pack)))
    else:
        side = int(box_size)
    # Clamp: at least 8px side, at most half the short image edge
    side = int(max(8, min(side, min(h, w) // 2 * 2)))  # even-ish via floor
    if side % 2 == 1:
        side += 1
    b = max(4, side // 2)  # box half-size
    # Spatial exclusion in Chebyshev (L∞) distance so axis-aligned boxes do
    # not overlap: centers must be ≥ side+1 apart (1px gap for stroke).
    # Also respect K-packing floor.
    r = float(max(r_pack, side + 1))
    hop = int(stride) if stride is not None else int(max(4, b))

    # Coarse feature grid for PCA fit + window stats
    grid_stride = max(2, hop // 2)
    gy = np.arange(0, h, grid_stride, dtype=np.int32)
    gx = np.arange(0, w, grid_stride, dtype=np.int32)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    flat_y = yy.ravel()
    flat_x = xx.ravel()
    X = feats[flat_y, flat_x, :].astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = X.shape[0]
    if n < 4:
        raise ValueError("too few pixels to score boxes")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    d = int(max(1, min(pca_dims, c, n - 1, 16)))
    pca = PCA(n_components=d, random_state=seed)
    Z = pca.fit_transform(Xs)
    explained = float(np.sum(pca.explained_variance_ratio_))

    # Map (y,x) -> Z row via nearest grid sample (for window aggregation)
    # Store Z as grid gh×gw×d
    gh, gw = len(gy), len(gx)
    Z_grid = Z.reshape(gh, gw, d)

    # Sliding window centers on hop grid
    cy = np.arange(b, h - b + 1, hop, dtype=np.int32)
    cx = np.arange(b, w - b + 1, hop, dtype=np.int32)
    if cy.size == 0:
        cy = np.array([h // 2], dtype=np.int32)
    if cx.size == 0:
        cx = np.array([w // 2], dtype=np.int32)
    win_yy, win_xx = np.meshgrid(cy, cx, indexing="ij")
    centers_y = win_yy.ravel()
    centers_x = win_xx.ravel()
    n_win = int(centers_y.size)

    scores = np.zeros(n_win, dtype=np.float64)
    means = np.zeros((n_win, d), dtype=np.float64)

    for i in range(n_win):
        y0 = int(centers_y[i] - b)
        y1 = int(centers_y[i] + b)
        x0 = int(centers_x[i] - b)
        x1 = int(centers_x[i] + b)
        # Grid indices covering the box
        gi0 = max(0, (y0 + grid_stride - 1) // grid_stride)
        gi1 = min(gh, (y1 // grid_stride) + 1)
        gj0 = max(0, (x0 + grid_stride - 1) // grid_stride)
        gj1 = min(gw, (x1 // grid_stride) + 1)
        patch = Z_grid[gi0:gi1, gj0:gj1, :].reshape(-1, d)
        if patch.shape[0] < 2:
            continue
        means[i] = patch.mean(axis=0)
        # tr(Cov) = sum of feature variances
        scores[i] = float(np.var(patch, axis=0).sum())

    residual = scores.copy()
    n_windows_in_mask = n_win
    if place_mask is not None:
        for i in range(n_win):
            cx_i = int(centers_x[i])
            cy_i = int(centers_y[i])
            x0 = max(0, cx_i - b)
            y0 = max(0, cy_i - b)
            x1 = min(w, cx_i + b)
            y1 = min(h, cy_i + b)
            if x1 <= x0 or y1 <= y0 or not bool(place_mask[y0:y1, x0:x1].all()):
                residual[i] = 0.0
                scores[i] = 0.0
        n_windows_in_mask = int(np.count_nonzero(residual > 0))
        if n_windows_in_mask < 1:
            raise ValueError(
                "no candidate boxes fit fully inside the placement mask; "
                "draw a larger mask or reduce box size"
            )

    points: list[dict[str, Any]] = []

    for pick_i in range(k):
        if not np.any(residual > 0):
            break
        i_star = int(np.argmax(residual))
        score = float(residual[i_star])
        if score <= 0:
            break
        cx_i = int(centers_x[i_star])
        cy_i = int(centers_y[i_star])
        mu = means[i_star]
        mu_norm = float(np.linalg.norm(mu))
        x0 = max(0, cx_i - b)
        y0 = max(0, cy_i - b)
        # Exclusive max so Konva/CSS width = x1 - x0 equals 2*b when unclipped
        x1 = min(w, cx_i + b)
        y1 = min(h, cy_i + b)
        points.append(
            {
                "x": cx_i,
                "y": cy_i,
                "cluster": pick_i,
                "score": score,
                "radius": float(r),
                "box_size": int(2 * b),
                "box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            }
        )
        residual[i_star] = 0.0

        # Spatial (Chebyshev) + feature suppress. Exclude with ≥ side so
        # axis-aligned boxes of width `side` do not share interior area.
        for j in range(n_win):
            if residual[j] <= 0:
                continue
            dx = abs(float(centers_x[j] - cx_i))
            dy = abs(float(centers_y[j] - cy_i))
            if max(dx, dy) < r:
                residual[j] = 0.0
                continue
            mj = means[j]
            mj_norm = float(np.linalg.norm(mj))
            if mu_norm < 1e-12 or mj_norm < 1e-12:
                continue
            cos = float(np.dot(mu, mj) / (mu_norm * mj_norm))
            cos = max(0.0, cos)
            if cos > _COS_HARD:
                residual[j] = 0.0
            else:
                residual[j] *= 1.0 - (cos ** _COS_ALPHA)

    # Hard guarantee: drop any pick whose AABB still overlaps an earlier one
    # (protects against hop / clip / float edge cases).
    points = _dedupe_overlapping_boxes(points)

    heatmap = _residual_heatmap(
        h, w, centers_y, centers_x, residual, hop=hop, half=b
    )

    meta = {
        "k": k,
        "n_picked": len(points),
        "n_windows": n_win,
        "n_windows_in_mask": n_windows_in_mask,
        "n_subsample": int(n),
        "pca_dims": d,
        "explained_variance": explained,
        "stride": hop,
        "radius": float(r),
        "box_half": b,
        "box_size": int(2 * b),
        "mask_pixels": mask_pixels,
        "has_mask": place_mask is not None,
    }
    return ManifoldSample(job_id=job.job_id, points=points, heatmap=heatmap, meta=meta)


def _boxes_overlap(a: dict[str, int], b: dict[str, int]) -> bool:
    """True if two exclusive-end AABBs share interior area."""
    return a["x0"] < b["x1"] and b["x0"] < a["x1"] and a["y0"] < b["y1"] and b["y0"] < a["y1"]


def _dedupe_overlapping_boxes(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep greedy order; drop later picks whose box overlaps an earlier one."""
    kept: list[dict[str, Any]] = []
    for p in points:
        box = p.get("box")
        if not isinstance(box, dict):
            kept.append(p)
            continue
        if any(_boxes_overlap(box, k["box"]) for k in kept if isinstance(k.get("box"), dict)):
            continue
        kept.append(p)
    for i, p in enumerate(kept):
        p["cluster"] = i
    return kept


def _residual_heatmap(
    h: int,
    w: int,
    centers_y: np.ndarray,
    centers_x: np.ndarray,
    residual: np.ndarray,
    *,
    hop: int,
    half: int,
) -> np.ndarray:
    """Paint residual scores onto a coarse grid and upsample to HxW."""
    out = np.zeros((h, w), dtype=np.float32)
    max_s = float(np.max(residual)) if residual.size else 0.0
    if max_s < 1e-12:
        return out
    for i in range(residual.size):
        if residual[i] <= 0:
            continue
        val = float(residual[i] / max_s)
        cy = int(centers_y[i])
        cx = int(centers_x[i])
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)
        # Take max so overlapping windows keep strongest residual
        patch = out[y0:y1, x0:x1]
        np.maximum(patch, val, out=patch)
    return out


def encode_heatmap_png(heatmap: np.ndarray) -> bytes:
    """Encode HxW float [0,1] residual map as grayscale PNG (0–255)."""
    u8 = np.clip(np.asarray(heatmap, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(u8, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def store_sample(result: ManifoldSample) -> CachedManifoldSample:
    """Cache heatmap PNG + points; return handle for GET."""
    sample_id = uuid.uuid4().hex
    cached = CachedManifoldSample(
        sample_id=sample_id,
        heatmap_png=encode_heatmap_png(result.heatmap),
        points=list(result.points),
        meta={
            "sample_id": sample_id,
            "job_id": result.job_id,
            **result.meta,
            "points": list(result.points),
        },
    )
    _sample_cache.set(sample_id, cached)
    return cached


def get_sample(sample_id: str) -> CachedManifoldSample | None:
    """Look up a cached manifold sample."""
    s = _sample_cache.get(sample_id)
    return s if isinstance(s, CachedManifoldSample) else None
