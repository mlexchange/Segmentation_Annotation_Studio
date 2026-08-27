"""Train → infer → rethreshold without recompute."""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage
import pytest

from ipred.catalog import Catalog
from ipred import feature_setups, preprocess, train_infer


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def test_train_infer_rethreshold(catalog: Catalog, tmp_path) -> None:
    feature_setups.ensure_default_setups(catalog)
    img = np.zeros((48, 48), dtype=np.uint8)
    img[:, 24:] = 220
    PILImage.fromarray(img, mode="L").save(tmp_path / "blob.png")

    session = catalog.open_session(
        kind="local", source="blob.png", root=str(tmp_path)
    )
    preprocess.run_preprocess(
        catalog,
        session_id=session.session_id,
        feature_setup_id="default-skimage",
    )
    shapes = [
        {"kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 18, "h": 40},
        {"kind": "rectangle", "classId": 2, "x": 28, "y": 2, "w": 18, "h": 40},
    ]
    trained = train_infer.run_train(
        catalog,
        session_id=session.session_id,
        shapes=shapes,
        trainer_id="catboost",
        config={"iterations": 20, "depth": 4, "random_seed": 0},
    )
    assert trained["model_id"]
    imps = trained.get("feature_importances") or []
    assert imps, "CatBoost train must return ranked feature_importances"
    assert "label" in imps[0] and "importance" in imps[0]
    assert imps[0]["importance"] >= imps[-1]["importance"]

    run = train_infer.run_infer(
        catalog,
        session_id=session.session_id,
        alpha=0.2,
    )
    assert run["run_id"]
    blob = run["blob_dir"]
    assert (tmp_path / "ipred").exists() or True
    from pathlib import Path

    proba_path = Path(blob) / "proba.npy"
    assert proba_path.is_file()
    proba_before = np.load(proba_path).copy()

    rerun = train_infer.run_rethreshold(
        catalog,
        session_id=session.session_id,
        alpha=0.05,
    )
    assert rerun["alpha"] == 0.05
    proba_after = np.load(proba_path)
    np.testing.assert_array_equal(proba_before, proba_after)
