"""Softmax proba heatmaps + per-class threshold maps."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from ipred.catalog import Catalog
from ipred import feature_setups, preprocess, train_infer
from ipred.paths import project_blob_dir


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def _fake_run(catalog: Catalog, tmp_path) -> str:
    """Insert a minimal run with synthetic HxWxK softmax proba."""
    feature_setups.ensure_default_setups(catalog)
    session = catalog.open_session(kind="local", source="x.png", root=str(tmp_path))
    project_id = session.project_id
    run_id = "runproba01"
    blob = project_blob_dir(project_id) / "runs" / run_id
    blob.mkdir(parents=True, exist_ok=True)
    # Two classes; left half prefers class 1, right prefers class 2
    h, w, k = 8, 10, 2
    proba = np.zeros((h, w, k), dtype=np.float16)
    proba[:, :5, 0] = 0.8
    proba[:, :5, 1] = 0.2
    proba[:, 5:, 0] = 0.3
    proba[:, 5:, 1] = 0.7
    np.save(blob / "proba.npy", proba)
    meta = {
        "run_id": run_id,
        "model_id": "m",
        "feature_id": "f",
        "alpha": 0.05,
        "class_ids": [1, 2],
        "counts": {"singleton": 0, "multi": 0, "abstain": 0},
    }
    import json
    from datetime import datetime, timezone

    (blob / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    catalog.insert_run(
        {
            "run_id": run_id,
            "project_id": project_id,
            "model_id": "m",
            "feature_id": "f",
            "alpha": 0.05,
            "blob_dir": str(blob),
            "meta_json": json.dumps(meta),
            "created_at": now,
            "updated_at": now,
        }
    )
    return run_id


def test_proba_heatmap_png(catalog: Catalog, tmp_path) -> None:
    run_id = _fake_run(catalog, tmp_path)
    png = train_infer.proba_heatmap_png(catalog, run_id, 0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    png1 = train_infer.proba_heatmap_png(catalog, run_id, 1)
    assert png1[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        train_infer.proba_heatmap_png(catalog, run_id, 9)


def test_threshold_class_label_map(catalog: Catalog, tmp_path) -> None:
    run_id = _fake_run(catalog, tmp_path)
    out = train_infer.threshold_class_label_map(
        catalog, run_id, class_id=1, threshold=0.5
    )
    assert out["width"] == 10
    assert out["height"] == 8
    assert out["class_id"] == 1
    raw = base64.b64decode(out["label_map_b64"])
    labels = np.frombuffer(raw, dtype=np.uint8).reshape(8, 10)
    assert np.all(labels[:, :5] == 1)
    assert np.all(labels[:, 5:] == 0)
    assert out["n_positive"] == 8 * 5

    out2 = train_infer.threshold_class_label_map(
        catalog, run_id, class_id=2, threshold=0.6
    )
    labels2 = np.frombuffer(
        base64.b64decode(out2["label_map_b64"]), dtype=np.uint8
    ).reshape(8, 10)
    assert np.all(labels2[:, 5:] == 2)
    assert np.all(labels2[:, :5] == 0)
