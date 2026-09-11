"""Feature Setup shelf defaults."""

from __future__ import annotations

import pytest

from ipred.catalog import Catalog
from ipred import feature_setups


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def test_ensure_defaults(catalog: Catalog) -> None:
    defaults = feature_setups.ensure_default_setups(catalog)
    assert len(defaults) == 10
    ids = {d["id"] for d in defaults}
    assert "default-skimage" in ids
    assert "default-slimsam" in ids
    assert "default-skimage-slimsam" in ids
    assert "default-slimsam-clahe" in ids
    assert "default-mark25" in ids
    assert "default-skimage-mark25" in ids
    assert "default-mark25-clahe" in ids
    assert "default-mark11" in ids
    assert "default-skimage-mark11" in ids
    assert "default-mark11-clahe" in ids
    sk = feature_setups.resolve_setup("default-skimage")
    assert sk["kind"] == "procedure"
    assert sk["procedure_id"] == feature_setups.PROCEDURE_SKIMAGE
    mark = feature_setups.resolve_setup("default-mark25")
    assert mark["kind"] == "weights"
    assert mark["weights_format"] == feature_setups.WEIGHTS_TOMOJEPA_MARK25
    assert mark["inference"]["input_size"] == 512
    combo = feature_setups.resolve_setup("default-skimage-mark25")
    assert combo["encoder_setup_id"] == "default-mark25"
    assert combo["params"]["input_size"] == 512
    assert combo["params"]["resize"] is True
    assert combo["params"]["pca_dims"] == 64
    clahe = feature_setups.resolve_setup("default-mark25-clahe")
    assert clahe["procedure_id"] == feature_setups.PROCEDURE_CLAHE_ENCODER
    assert clahe["encoder_setup_id"] == "default-mark25"
    assert clahe["params"]["resize"] is True
    assert clahe["params"]["input_size"] == 512
    assert clahe["params"]["clahe"] is True
    assert clahe["params"]["pca_dims"] == 64
    mark11 = feature_setups.resolve_setup("default-mark11")
    assert mark11["weights_format"] == feature_setups.WEIGHTS_TOMOJEPA_MARK11
    combo11 = feature_setups.resolve_setup("default-skimage-mark11")
    assert combo11["encoder_setup_id"] == "default-mark11"
    clahe11 = feature_setups.resolve_setup("default-mark11-clahe")
    assert clahe11["encoder_setup_id"] == "default-mark11"
    slim_clahe = feature_setups.resolve_setup("default-slimsam-clahe")
    assert slim_clahe["procedure_id"] == feature_setups.PROCEDURE_CLAHE_ENCODER
    assert slim_clahe["encoder_setup_id"] == "default-slimsam"
