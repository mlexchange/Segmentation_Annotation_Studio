"""Cache-aware preprocess."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import pytest

from ipred.catalog import Catalog
from ipred import feature_setups, preprocess, tomojepa_embed

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "tomojepa25.pth"


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def test_preprocess_cache_hit(catalog: Catalog, tmp_path) -> None:
    feature_setups.ensure_default_setups(catalog)
    # Write a local PNG under LOCAL_DATA_ROOT
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:, 16:] = 200
    rel = "sample.png"
    PILImage.fromarray(img, mode="L").save(tmp_path / rel)

    session = catalog.open_session(kind="local", source=rel, root=str(tmp_path))
    first = preprocess.run_preprocess(
        catalog,
        session_id=session.session_id,
        feature_setup_id="default-skimage",
        slice_index=0,
    )
    assert first["cache_hit"] is False
    assert first["n_channels"] > 0

    second = preprocess.run_preprocess(
        catalog,
        session_id=session.session_id,
        feature_setup_id="default-skimage",
        slice_index=0,
    )
    assert second["cache_hit"] is True
    assert second["feature_id"] == first["feature_id"]


@pytest.mark.skipif(
    not WEIGHTS.is_file() or not tomojepa_embed.encoder_available(str(WEIGHTS)),
    reason="tomojepa25.pth or torch/timm not available",
)
def test_preprocess_mark25_writes_dense_emb(catalog: Catalog, tmp_path) -> None:
    feature_setups.ensure_default_setups(catalog)
    img = np.linspace(0, 255, 64 * 48, dtype=np.float32).reshape(64, 48).astype(
        np.uint8
    )
    rel = "tomo.png"
    PILImage.fromarray(img, mode="L").save(tmp_path / rel)

    session = catalog.open_session(kind="local", source=rel, root=str(tmp_path))
    out = preprocess.run_preprocess(
        catalog,
        session_id=session.session_id,
        feature_setup_id="default-skimage-mark25",
        slice_index=0,
    )
    assert out["cache_hit"] is False
    bank = preprocess.load_feature_bank_arrays(out["blob_dir"])
    assert bank["sam_emb"] is not None
    assert bank["sam_emb"].shape == (32, 32, 64)
    assert bank["sam_meta"]["encoder"] == "mark25"
    assert bank["sam_meta"]["input_size"] == 512


@pytest.mark.skipif(
    not WEIGHTS.is_file() or not tomojepa_embed.encoder_available(str(WEIGHTS)),
    reason="tomojepa25.pth or torch/timm not available",
)
def test_preprocess_mark25_clahe_no_skimage(catalog: Catalog, tmp_path) -> None:
    feature_setups.ensure_default_setups(catalog)
    img = np.linspace(0, 255, 80 * 64, dtype=np.float32).reshape(80, 64).astype(
        np.uint8
    )
    rel = "tomo_clahe.png"
    PILImage.fromarray(img, mode="L").save(tmp_path / rel)

    session = catalog.open_session(kind="local", source=rel, root=str(tmp_path))
    out = preprocess.run_preprocess(
        catalog,
        session_id=session.session_id,
        feature_setup_id="default-mark25-clahe",
        slice_index=0,
    )
    assert out["cache_hit"] is False
    assert out["n_channels"] == 1 + 64
    bank = preprocess.load_feature_bank_arrays(out["blob_dir"])
    assert bank["labels"][0] == "clahe"
    assert bank["labels"][1:5] == ["pca0", "pca1", "pca2", "pca3"]
    assert bank["float_stack"].shape[-1] == 65
    assert bank["float_stack"].shape[:2] == (80, 64)
    assert bank["sam_emb"] is not None
    assert bank["sam_emb"].shape == (32, 32, 64)
    assert bank["sam_meta"]["encoder"] == "mark25"
    assert bank["sam_meta"]["resize"] is True
    assert bank["sam_meta"]["input_size"] == 512
    assert bank["sam_meta"]["baked_into_float_stack"] is True
    assert bank["sam_meta"]["pca_dims"] == 64
    # Channel PNGs written for heatmap browsing
    from pathlib import Path

    ch_dir = Path(out["blob_dir"]) / "channels"
    assert (ch_dir / "0000.png").is_file()
    assert (ch_dir / "0064.png").is_file()
