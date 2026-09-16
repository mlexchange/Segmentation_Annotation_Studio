"""Composition validation, builtins, and legacy setup mapping."""

from __future__ import annotations

import numpy as np
import pytest

from ipred import compositions, compose_run, feature_setups
from ipred.catalog import Catalog
from ipred.modules import list_module_catalog


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def test_module_catalog_lists_core_modules() -> None:
    ids = {m["id"] for m in list_module_catalog()}
    assert ids >= {
        "skimage_multiscale",
        "clahe",
        "slimsam",
        "tomojepa",
        "pca",
    }


def test_ensure_default_compositions(catalog: Catalog) -> None:
    comps = compositions.ensure_default_compositions(catalog)
    ids = {c["id"] for c in comps}
    assert "comp-skimage" in ids
    assert "comp-skimage-mark11" in ids
    assert "comp-mark25-clahe" in ids
    assert "comp-slimsam-clahe" in ids


def test_legacy_setup_maps_to_composition(catalog: Catalog) -> None:
    compositions.ensure_default_compositions(catalog)
    assert (
        compositions.resolve_preprocess_id("default-skimage-mark25")
        == "comp-skimage-mark25"
    )
    assert compositions.resolve_preprocess_id("default-mark11") == "comp-mark11-clahe"


def test_preview_concat_skimage(catalog: Catalog) -> None:
    compositions.ensure_default_compositions(catalog)
    doc = compositions.resolve_composition("comp-skimage")
    labels = compositions.preview_concat_labels(doc)
    assert any("intensity" in lab for lab in labels)


def test_run_composition_skimage_only(catalog: Catalog) -> None:
    compositions.ensure_default_compositions(catalog)
    doc = compositions.resolve_composition("comp-skimage")
    arr = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    uint8, floats, labels, emb, meta = compose_run.run_composition(arr, doc)
    assert emb is None
    assert floats.ndim == 3
    assert uint8.shape == floats.shape
    assert len(labels) == floats.shape[-1]


def test_content_hash_stable() -> None:
    doc = {
        "nodes": [{"id": "n1", "module": "skimage_multiscale", "params": {}}],
        "outputs": ["n1"],
    }
    compositions.validate_composition(doc)
    h1 = compositions.content_hash(doc)
    h2 = compositions.content_hash({**doc, "name": "x"})
    assert h1 == h2


def test_feature_setups_still_seed(catalog: Catalog) -> None:
    feature_setups.ensure_default_setups(catalog)
    compositions.ensure_default_compositions(catalog)
    assert feature_setups.load_setup("default-skimage") is not None
