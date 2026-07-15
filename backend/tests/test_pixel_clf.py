"""Tests for CatBoost pixel classifier (sparse labels → predict)."""

from __future__ import annotations

import numpy as np
import pytest

from multiscale_features import compute_feature_stacks, store_job
from pixel_clf import (
    build_label_map,
    encode_label_png,
    predict_labels,
    train_classifier,
    train_result_dict,
    tree_view,
)


def _two_blob_image(h: int = 48, w: int = 48) -> np.ndarray:
    """Synthetic gray with left dark / right bright regions."""
    img = np.zeros((h, w), dtype=np.float64)
    img[:, : w // 2] = 0.2
    img[:, w // 2 :] = 0.8
    return img


def test_build_label_map_sparse_and_last_wins() -> None:
    shapes = [
        {
            "kind": "rectangle",
            "classId": 1,
            "x": 0,
            "y": 0,
            "w": 10,
            "h": 10,
        },
        {
            "kind": "rectangle",
            "classId": 2,
            "x": 5,
            "y": 5,
            "w": 10,
            "h": 10,
        },
    ]
    labels = build_label_map(shapes, 20, 20)
    assert labels.dtype == np.uint8
    assert labels[0, 0] == 1
    assert labels[7, 7] == 2  # overlap: second wins
    assert labels[19, 19] == 0  # unlabeled


def test_train_requires_labeled_pixels() -> None:
    gray = _two_blob_image()
    u8, fl, labs = compute_feature_stacks(
        gray,
        sigma_min=1.0,
        sigma_max=2.0,
        intensity=True,
        edges=False,
        texture=False,
        clahe=False,
    )
    job = store_job(u8, labs, float_stack=fl)
    with pytest.raises(ValueError, match="labeled"):
        train_classifier(job, shapes=[])


def test_train_predict_two_class_blobs() -> None:
    gray = _two_blob_image(64, 64)
    u8, fl, labs = compute_feature_stacks(
        gray,
        sigma_min=1.0,
        sigma_max=2.0,
        intensity=True,
        edges=True,
        texture=False,
        clahe=False,
    )
    job = store_job(u8, labs, float_stack=fl)

    # Sparse scribbles only (ilastik): left → class 1, right → class 2
    shapes = [
        {"kind": "rectangle", "classId": 1, "x": 4, "y": 28, "w": 8, "h": 8},
        {"kind": "rectangle", "classId": 2, "x": 52, "y": 28, "w": 8, "h": 8},
    ]
    model = train_classifier(
        job,
        shapes,
        iterations=50,
        depth=4,
        learning_rate=0.2,
        max_samples=50_000,
    )
    assert set(model.class_ids) == {1, 2}
    assert model.n_samples >= 50

    pred = predict_labels(model, job)
    assert pred.shape == (64, 64)
    # Exactly one class id per pixel (argmax)
    assert set(np.unique(pred).tolist()).issubset({1, 2})
    left = pred[:, :32]
    right = pred[:, 32:]
    assert np.mean(left == 1) > 0.7
    assert np.mean(right == 2) > 0.7

    preserved = predict_labels(model, job, preserve_shapes=shapes)
    # Scribbles zeroed out
    assert preserved[28:36, 4:12].sum() == 0
    assert preserved[28:36, 52:60].sum() == 0

    png = encode_label_png(pred)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    result = train_result_dict(model)
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert result["n_trees"] >= 1
    assert result["feature_importances"]
    assert result["tree_preview"]["splits"] is not None
    tv = tree_view(model, 0)
    assert tv["tree_index"] == 0
    assert "splits" in tv


def test_train_with_fake_sam_emb() -> None:
    """CatBoost concat path with a synthetic SlimSAM grid (no ONNX)."""
    gray = _two_blob_image(48, 48)
    u8, fl, labs = compute_feature_stacks(
        gray,
        sigma_min=1.0,
        sigma_max=2.0,
        intensity=True,
        edges=False,
        texture=False,
        clahe=False,
    )
    rng = np.random.default_rng(1)
    # Fake emb: left vs right distinguishable
    emb = np.zeros((64, 64, 16), dtype=np.float32)
    emb[:, :32] = 1.0
    emb[:, 32:] = -1.0
    emb += 0.01 * rng.standard_normal(emb.shape).astype(np.float32)
    job = store_job(
        u8,
        labs,
        float_stack=fl,
        sam_emb=emb,
        sam_orig_hw=(48, 48),
        sam_reshaped_hw=(1024, 1024),
    )
    shapes = [
        {"kind": "rectangle", "classId": 1, "x": 2, "y": 20, "w": 8, "h": 8},
        {"kind": "rectangle", "classId": 2, "x": 38, "y": 20, "w": 8, "h": 8},
    ]
    model = train_classifier(job, shapes, iterations=30, depth=3, learning_rate=0.3, sam_pca_dims=8)
    assert model.uses_sam
    assert any(lab.startswith("sam") for lab, _ in model.feature_importances)
    pred = predict_labels(model, job, row_chunk=16)
    assert np.mean(pred[:, :24] == 1) > 0.6
    assert np.mean(pred[:, 24:] == 2) > 0.6


def test_compute_stack_returns_float() -> None:
    gray = np.random.default_rng(0).random((32, 32))
    u8, fl, labs = compute_feature_stacks(
        gray,
        sigma_min=1.0,
        sigma_max=2.0,
        intensity=True,
        edges=False,
        texture=False,
        clahe=True,
    )
    assert u8.dtype == np.uint8
    assert fl.dtype == np.float32
    assert u8.shape == fl.shape
    assert fl.shape[-1] == len(labs)
