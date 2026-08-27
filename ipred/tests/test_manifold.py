"""Manifold Suggest Labels on persisted ipred feature banks."""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage
import pytest

from ipred.catalog import Catalog
from ipred import feature_setups, manifold_jobs, preprocess


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def test_manifold_sample_on_feature_bank(catalog: Catalog, tmp_path) -> None:
    feature_setups.ensure_default_setups(catalog)
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 200
    img[16:48, 16:48] = 120
    PILImage.fromarray(img, mode="L").save(tmp_path / "m.png")

    session = catalog.open_session(kind="local", source="m.png", root=str(tmp_path))
    bank = preprocess.run_preprocess(
        catalog,
        session_id=session.session_id,
        feature_setup_id="default-skimage",
    )
    out = manifold_jobs.run_manifold_sample(
        catalog,
        feature_id=bank["feature_id"],
        k=8,
        box_size=16,
    )
    assert out["sample_id"]
    assert out["n_picked"] >= 1
    assert len(out["points"]) >= 1
    png = manifold_jobs.heatmap_png(out["sample_id"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
