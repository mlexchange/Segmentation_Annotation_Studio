"""Orchestrate manifold suggest on persisted feature banks."""

from __future__ import annotations

from typing import Any

import numpy as np

from ipred import manifold
from ipred.catalog import Catalog
from ipred.labels import shape_to_mask
from ipred.preprocess import load_feature_bank_arrays


def run_manifold_sample(
    catalog: Catalog,
    *,
    feature_id: str,
    k: int = 24,
    box_size: int | None = None,
    stride: int | None = None,
    pca_dims: int = 16,
    shapes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sample inducing boxes on a feature bank; return API response dict."""
    row = catalog.get_feature_bank(feature_id)
    if row is None:
        raise KeyError(f"unknown feature bank {feature_id}")
    bank = load_feature_bank_arrays(row["blob_dir"])
    float_stack = bank["float_stack"]
    h, w = float_stack.shape[:2]

    place_mask = None
    if shapes:
        place_mask = np.zeros((h, w), dtype=bool)
        for shape in shapes:
            place_mask |= shape_to_mask(shape, h, w)
        if not bool(place_mask.any()):
            raise ValueError("placement mask is empty")

    result = manifold.sample_inducing_points(
        float_stack,
        feature_id=feature_id,
        k=k,
        box_size=box_size,
        stride=stride,
        pca_dims=pca_dims,
        mask=place_mask,
    )
    cached = manifold.store_sample(result)
    return {
        "sample_id": cached.sample_id,
        "feature_id": feature_id,
        "points": cached.points,
        "k": result.meta["k"],
        "n_picked": result.meta["n_picked"],
        "n_subsample": result.meta["n_subsample"],
        "pca_dims": result.meta["pca_dims"],
        "explained_variance": result.meta["explained_variance"],
        "stride": result.meta["stride"],
        "radius": result.meta["radius"],
        "box_size": result.meta["box_size"],
        "mask_pixels": result.meta.get("mask_pixels", 0),
        "n_windows_in_mask": result.meta.get("n_windows_in_mask"),
        "has_mask": result.meta.get("has_mask", False),
    }


def heatmap_png(sample_id: str) -> bytes:
    """Return cached heatmap PNG bytes."""
    cached = manifold.get_sample(sample_id)
    if cached is None:
        raise KeyError(f"unknown manifold sample {sample_id}")
    return cached.heatmap_png
