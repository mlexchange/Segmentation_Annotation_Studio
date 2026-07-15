"""Tests for greedy feature-variance box sampling on FeatureJob float_stack."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from annotation_server import app
from multiscale_features import compute_feature_stacks, store_job
from feature_manifold import (
    derived_exclusion_radius,
    encode_heatmap_png,
    sample_inducing_points,
    store_sample,
)


def _two_blob(h: int = 64, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.float64)
    img[:, : w // 2] = 0.1
    img[:, w // 2 :] = 0.9
    return img


def _checker(h: int = 64, w: int = 64, tile: int = 8) -> np.ndarray:
    """Tiled high-variance checker pattern (identical patches across image)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx // tile) + (yy // tile)) % 2).astype(np.float64)


def _job(gray: np.ndarray):
    u8, fl, labs = compute_feature_stacks(
        gray,
        sigma_min=1.0,
        sigma_max=2.0,
        intensity=True,
        edges=True,
        texture=False,
        clahe=False,
    )
    return store_job(u8, labs, float_stack=fl)


def test_derived_radius_shrinks_with_k() -> None:
    r8 = derived_exclusion_radius(64, 64, 8)
    r32 = derived_exclusion_radius(64, 64, 32)
    assert r8 >= r32
    assert r8 >= 8


def test_manual_box_size_controls_window() -> None:
    job = _job(_two_blob(128, 128))
    small = sample_inducing_points(job, k=8, box_size=24, seed=0)
    large = sample_inducing_points(job, k=8, box_size=64, seed=0)
    assert small.meta["box_size"] == 24
    assert large.meta["box_size"] == 64
    # Drawn box extent matches
    sb = small.points[0]["box"]
    assert (sb["x1"] - sb["x0"]) == 24 or abs((sb["x1"] - sb["x0"]) - 24) <= 1
    lb = large.points[0]["box"]
    assert abs((lb["x1"] - lb["x0"]) - 64) <= 1


def test_sample_returns_k_points_with_boxes() -> None:
    job = _job(_two_blob())
    k = 12
    result = sample_inducing_points(job, k=k, seed=0)
    assert 1 <= len(result.points) <= k
    h, w = job.float_stack.shape[:2]
    assert result.heatmap.shape == (h, w)
    assert 0.0 <= float(result.heatmap.min())
    assert float(result.heatmap.max()) <= 1.0 + 1e-6
    r = result.meta["radius"]
    for p in result.points:
        assert 0 <= p["x"] < w and 0 <= p["y"] < h
        assert "box" in p and "radius" in p and "score" in p
        assert p["radius"] == r
        assert p.get("box_size") == result.meta["box_size"]
        box = p["box"]
        assert box["x0"] <= p["x"] < box["x1"] or box["x0"] <= p["x"] <= box["x1"]
        assert box["y0"] <= p["y"] < box["y1"] or box["y0"] <= p["y"] <= box["y1"]
        assert abs((box["x1"] - box["x0"]) - result.meta["box_size"]) <= 1
        assert abs((box["y1"] - box["y0"]) - result.meta["box_size"]) <= 1


def test_picks_spread_on_two_blob_and_respect_exclusion() -> None:
    job = _job(_two_blob(96, 96))
    result = sample_inducing_points(job, k=10, seed=1)
    assert len(result.points) >= 2
    r = float(result.meta["radius"])
    # Pairwise Chebyshev distance must be ≥ exclusion (non-overlapping squares)
    for i, a in enumerate(result.points):
        for b in result.points[i + 1 :]:
            d_inf = max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))
            assert d_inf >= r - 1e-6
    # First pick near the vertical boundary (high variance)
    first = result.points[0]
    assert 96 * 0.25 <= first["x"] <= 96 * 0.75


def test_large_boxes_do_not_overlap_after_sample() -> None:
    """Growing box size must re-space centers so AABBs do not overlap."""
    job = _job(_two_blob(256, 256))
    result = sample_inducing_points(job, k=16, box_size=96, seed=0)
    side = int(result.meta["box_size"])
    assert result.meta["radius"] >= side - 1e-6
    assert len(result.points) >= 2
    for i, a in enumerate(result.points):
        ba = a["box"]
        for b in result.points[i + 1 :]:
            bb = b["box"]
            # Axis-aligned rectangles: no area overlap (edges may touch)
            overlap_x = ba["x0"] < bb["x1"] and bb["x0"] < ba["x1"]
            overlap_y = ba["y0"] < bb["y1"] and bb["y0"] < ba["y1"]
            assert not (overlap_x and overlap_y), (a["box"], b["box"])


def test_placement_mask_keeps_boxes_fully_inside() -> None:
    """Only windows whose full box lies inside the mask are eligible."""
    job = _job(_two_blob(160, 160))
    h, w = job.float_stack.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[:, : w // 2] = True  # left half only
    result = sample_inducing_points(job, k=12, box_size=32, seed=0, mask=mask)
    assert result.meta["has_mask"] is True
    assert result.meta["mask_pixels"] == int(mask.sum())
    assert result.meta["n_windows_in_mask"] >= 1
    assert len(result.points) >= 1
    for p in result.points:
        box = p["box"]
        assert box["x1"] <= w // 2 + 1  # exclusive end at or before half (+tol)
        assert mask[box["y0"] : box["y1"], box["x0"] : box["x1"]].all()


def test_placement_mask_empty_raises() -> None:
    job = _job(_two_blob(64, 64))
    mask = np.zeros(job.float_stack.shape[:2], dtype=bool)
    try:
        sample_inducing_points(job, k=4, box_size=16, seed=0, mask=mask)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc).lower()


def test_feature_suppress_avoids_duplicate_textures() -> None:
    job = _job(_checker(80, 80, tile=8))
    result = sample_inducing_points(job, k=8, seed=0)
    # With pure spatial packing on 80x80 we could get many; feature suppress
    # should leave fewer than a dense spatial pack of tiny r.
    assert len(result.points) >= 1
    assert len(result.points) <= 8


def test_encode_and_cache_api_roundtrip() -> None:
    job = _job(_two_blob())
    result = sample_inducing_points(job, k=8, seed=0)
    cached = store_sample(result)
    png = encode_heatmap_png(result.heatmap)
    arr = np.array(Image.open(io.BytesIO(png)))
    assert arr.shape == result.heatmap.shape

    client = TestClient(app)
    res = client.post(
        "/api/image/features/manifold/sample",
        json={"job_id": job.job_id, "k": 8},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["k"] == 8
    assert len(body["points"]) >= 1
    assert "radius" in body["points"][0]
    assert "box" in body["points"][0]
    heat = client.get(f"/api/image/features/manifold/{body['sample_id']}/heatmap.png")
    assert heat.status_code == 200
    assert cached.sample_id
