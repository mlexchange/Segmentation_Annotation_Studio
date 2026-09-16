"""run_infer(store_probabilities=False) — the option volume-apply batch jobs
use (backend/ipred_batch_jobs.py) to skip writing the largest thing a run
saves, since that flow only ever reads commit.png back."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from ipred import feature_setups, preprocess, train_infer
from ipred.catalog import Catalog


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def _trained_session(catalog: Catalog, tmp_path) -> str:
    feature_setups.ensure_default_setups(catalog)
    img = np.zeros((48, 48), dtype=np.uint8)
    img[:, 24:] = 220
    PILImage.fromarray(img, mode="L").save(tmp_path / "blob.png")

    session = catalog.open_session(kind="local", source="blob.png", root=str(tmp_path))
    preprocess.run_preprocess(catalog, session_id=session.session_id, feature_setup_id="default-skimage")
    shapes = [
        {"kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 18, "h": 40},
        {"kind": "rectangle", "classId": 2, "x": 28, "y": 2, "w": 18, "h": 40},
    ]
    train_infer.run_train(
        catalog, session_id=session.session_id, shapes=shapes,
        trainer_id="catboost", config={"iterations": 20, "depth": 4, "random_seed": 0},
    )
    return session.session_id


def test_store_probabilities_false_skips_proba_npy(catalog: Catalog, tmp_path) -> None:
    session_id = _trained_session(catalog, tmp_path)

    run = train_infer.run_infer(catalog, session_id=session_id, alpha=0.2, store_probabilities=False)

    blob = Path(run["blob_dir"])
    assert not (blob / "proba.npy").exists()
    # Everything volume-apply's commit step actually needs must still exist.
    assert (blob / "commit.png").is_file()
    assert (blob / "commit.npy").exists()
    assert (blob / "status.npy").exists()
    assert (blob / "membership.npy").exists()


def test_store_probabilities_true_is_still_the_default(catalog: Catalog, tmp_path) -> None:
    session_id = _trained_session(catalog, tmp_path)

    run = train_infer.run_infer(catalog, session_id=session_id, alpha=0.2)

    assert (Path(run["blob_dir"]) / "proba.npy").is_file()


def test_store_probabilities_false_produces_the_same_commit_map(catalog: Catalog, tmp_path) -> None:
    """The math must be identical either way — only persistence changes."""
    session_id = _trained_session(catalog, tmp_path)

    with_proba = train_infer.run_infer(catalog, session_id=session_id, alpha=0.2, store_probabilities=True)
    without_proba = train_infer.run_infer(catalog, session_id=session_id, alpha=0.2, store_probabilities=False)

    a = np.load(Path(with_proba["blob_dir"]) / "commit.npy")
    b = np.load(Path(without_proba["blob_dir"]) / "commit.npy")
    assert np.array_equal(a, b)
