"""Tests for Mondrian split-conformal CatBoost pixel prediction."""

from __future__ import annotations

import numpy as np
import pytest

from multiscale_features import compute_feature_stacks, store_job
from pixel_clf import (
    conformal_quantile,
    mondrian_thresholds,
    predict_conformal,
    stratified_train_cal_split,
    train_classifier,
)


def _two_blob(h: int = 64, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.float64)
    img[:, : w // 2] = 0.15
    img[:, w // 2 :] = 0.85
    return img


def test_stratified_train_cal_split_sizes() -> None:
    y = np.array([1] * 50 + [2] * 50)
    train_idx, cal_idx = stratified_train_cal_split(y, train_frac=0.8, rng=np.random.default_rng(0))
    assert len(train_idx) + len(cal_idx) == 100
    assert abs(len(train_idx) / 100 - 0.8) < 0.05
    # Both classes in both splits
    assert set(y[train_idx].tolist()) == {1, 2}
    assert set(y[cal_idx].tolist()) == {1, 2}


def test_conformal_quantile_finite_sample() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    # (1-alpha)(n+1)/n level — for alpha=0.2, n=5 → ceiling index
    q = conformal_quantile(scores, alpha=0.2)
    assert 0.1 <= q <= 0.5


def test_train_stores_cal_scores_and_split_sizes() -> None:
    gray = _two_blob()
    u8, fl, labs = compute_feature_stacks(
        gray, sigma_min=1.0, sigma_max=2.0, intensity=True, edges=True, texture=False, clahe=False
    )
    job = store_job(u8, labs, float_stack=fl)
    # Dense enough scribbles for 80/20
    shapes = [
        {"kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 20, "h": 50},
        {"kind": "rectangle", "classId": 2, "x": 42, "y": 2, "w": 20, "h": 50},
    ]
    model = train_classifier(job, shapes, iterations=40, depth=4, learning_rate=0.2, train_frac=0.8)
    assert model.n_train > 0
    assert model.n_cal > 0
    assert model.n_train + model.n_cal == model.n_samples
    assert set(model.cal_scores_by_class.keys()) == {1, 2}
    for cid, scores in model.cal_scores_by_class.items():
        assert scores.ndim == 1 and scores.size >= 1


def test_predict_conformal_commit_only_singletons() -> None:
    gray = _two_blob()
    u8, fl, labs = compute_feature_stacks(
        gray, sigma_min=1.0, sigma_max=2.0, intensity=True, edges=True, texture=False, clahe=False
    )
    job = store_job(u8, labs, float_stack=fl)
    shapes = [
        {"kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 24, "h": 56},
        {"kind": "rectangle", "classId": 2, "x": 38, "y": 2, "w": 24, "h": 56},
    ]
    model = train_classifier(job, shapes, iterations=50, depth=4, learning_rate=0.2)
    result = predict_conformal(model, job, alpha=0.05, row_chunk=32)
    commit = result.commit_map
    status = result.status_map
    assert commit.shape == status.shape == (64, 64)
    # Where status != singleton, commit must be 0
    assert np.all(commit[status != 1] == 0)
    # Singleton pixels have a class id
    assert np.all(commit[status == 1] > 0)
    assert result.counts["singleton"] + result.counts["multi"] + result.counts["abstain"] == 64 * 64

    loose = predict_conformal(model, job, alpha=0.25, row_chunk=32)
    # Larger alpha → lower score quantile → more abstain (emptier sets)
    assert result.counts["abstain"] <= loose.counts["abstain"] + 500


def test_mondrian_thresholds_change_with_alpha() -> None:
    scores = {1: np.linspace(0.0, 1.0, 40), 2: np.linspace(0.0, 0.5, 40)}
    q_tight = mondrian_thresholds(scores, alpha=0.05)  # high coverage → high q
    q_loose = mondrian_thresholds(scores, alpha=0.30)
    assert q_tight[1] >= q_loose[1] - 1e-9
