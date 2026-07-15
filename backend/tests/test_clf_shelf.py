"""Tests for persistent CatBoost model shelf."""

from __future__ import annotations

import os
from unittest import mock

import numpy as np

from multiscale_features import compute_feature_stacks, store_job
from pixel_clf import predict_conformal, train_classifier
import clf_shelf


def _two_blob(h: int = 48, w: int = 48) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.float64)
    img[:, : w // 2] = 0.15
    img[:, w // 2 :] = 0.85
    return img


def test_shelf_save_load_predict_roundtrip(tmp_path) -> None:
    with mock.patch.dict(os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}):
        gray = _two_blob()
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
        shapes = [
            {"kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 16, "h": 40},
            {"kind": "rectangle", "classId": 2, "x": 30, "y": 2, "w": 16, "h": 40},
        ]
        model = train_classifier(job, shapes, iterations=30, depth=4, learning_rate=0.2)
        recipe = {
            "sigma_min": 1.0,
            "sigma_max": 2.0,
            "intensity": True,
            "edges": True,
            "texture": False,
            "clahe": False,
            "include_sam": False,
        }
        meta = clf_shelf.save_model(model, name="blob-clf", feature_recipe=recipe)
        listed = clf_shelf.list_models()
        assert any(m["id"] == meta.id for m in listed)

        loaded = clf_shelf.load_into_cache(meta.id)
        assert loaded.feature_job_id.startswith("shelf:")
        # Apply on a fresh feature job (same recipe dims)
        job2 = store_job(u8, labs, float_stack=fl)
        result = predict_conformal(loaded, job2, alpha=0.1, row_chunk=24)
        assert result.commit_map.shape == (48, 48)
        assert result.counts["singleton"] + result.counts["multi"] + result.counts["abstain"] == 48 * 48

        assert clf_shelf.delete_model(meta.id) is True
        assert clf_shelf.list_models() == []
