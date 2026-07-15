"""API smoke tests for CatBoost train/predict endpoints."""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import multiscale_features as features_mod
from annotation_server import app


def test_clf_train_predict_roundtrip() -> None:
    gray = np.zeros((48, 48), dtype=np.float64)
    gray[:, 24:] = 1.0
    stack, float_stack, labels = features_mod.compute_feature_stacks(
        gray,
        sigma_min=1.0,
        sigma_max=2.0,
        intensity=True,
        edges=True,
        texture=False,
        clahe=False,
    )
    job = features_mod.store_job(stack, labels, float_stack=float_stack)
    client = TestClient(app)

    shapes = [
        {"kind": "rectangle", "classId": 1, "x": 2, "y": 4, "w": 16, "h": 40},
        {"kind": "rectangle", "classId": 2, "x": 30, "y": 4, "w": 16, "h": 40},
    ]
    train = client.post(
        "/api/image/features/clf/train",
        json={
            "job_id": job.job_id,
            "shapes": shapes,
            "iterations": 40,
            "depth": 4,
            "learning_rate": 0.2,
        },
    )
    assert train.status_code == 200, train.text
    body = train.json()
    assert body["n_samples"] > 0
    assert body["n_train"] > 0
    assert body["n_cal"] > 0
    assert set(body["class_ids"]) == {1, 2}

    pred = client.post(
        "/api/image/features/clf/predict",
        json={"model_id": body["model_id"], "job_id": job.job_id, "alpha": 0.1},
    )
    assert pred.status_code == 200, pred.text
    meta = pred.json()
    assert "pred_id" in meta
    assert "counts" in meta
    assert set(meta["counts"]) >= {"singleton", "multi", "abstain"}

    commit = client.get(f"/api/image/features/clf/predict/{meta['pred_id']}/commit.png")
    status = client.get(f"/api/image/features/clf/predict/{meta['pred_id']}/status.png")
    assert commit.status_code == 200
    assert status.status_code == 200
    carr = np.array(Image.open(io.BytesIO(commit.content)))
    sarr = np.array(Image.open(io.BytesIO(status.content)))
    assert carr.shape == (48, 48)
    assert sarr.shape == (48, 48)
    assert np.all(carr[sarr != 1] == 0)


def test_clf_train_empty_shapes_400() -> None:
    gray = np.linspace(0, 1, 16 * 16).reshape(16, 16)
    stack, float_stack, labels = features_mod.compute_feature_stacks(
        gray, sigma_min=1.0, sigma_max=1.0, intensity=True, edges=False, texture=False
    )
    job = features_mod.store_job(stack, labels, float_stack=float_stack)
    client = TestClient(app)
    res = client.post(
        "/api/image/features/clf/train",
        json={"job_id": job.job_id, "shapes": []},
    )
    assert res.status_code == 400
