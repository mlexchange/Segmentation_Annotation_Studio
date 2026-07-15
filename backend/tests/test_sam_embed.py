"""Tests for SlimSAM embedding helpers (no full ONNX required for core math)."""

from __future__ import annotations

import numpy as np

from sam_embed import (
    bilinear_sample_emb,
    fit_pca,
    preprocess_sam_rgb,
    rgb_uint8_from_array,
    transform_pca,
)


def test_rgb_uint8_from_gray() -> None:
    g = np.linspace(0, 1, 16 * 16, dtype=np.float64).reshape(16, 16)
    rgb = rgb_uint8_from_array(g)
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.uint8


def test_preprocess_shape() -> None:
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    rgb[:, 40:] = 255
    pv, orig, reshaped = preprocess_sam_rgb(rgb)
    assert pv.shape == (1, 3, 1024, 1024)
    assert orig == (40, 80)
    assert reshaped[1] == 1024  # width is long edge
    assert reshaped[0] < 1024


def test_bilinear_sample_and_pca() -> None:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((64, 64, 8)).astype(np.float32)
    ys = np.linspace(0, 39, 20)
    xs = np.linspace(0, 79, 20)
    samples = bilinear_sample_emb(
        emb, ys, xs, orig_h=40, orig_w=80, reshaped_h=512, reshaped_w=1024
    )
    assert samples.shape == (20, 8)
    mean, comps = fit_pca(samples, n_components=4)
    proj = transform_pca(samples, mean, comps)
    assert proj.shape == (20, 4)
