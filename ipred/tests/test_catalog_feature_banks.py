"""Catalog.delete_feature_bank — cleanup for batch-apply jobs (see
backend/ipred_batch_jobs.py's volume-apply job, which relies on this to avoid
accumulating a ~1 GB feature bank per slice with no eviction)."""
from __future__ import annotations

import pytest

from ipred.catalog import Catalog


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def _insert_bank(catalog: Catalog, feature_id: str, blob_dir: str, slice_index: int = 0) -> None:
    catalog.insert_feature_bank(
        {
            "feature_id": feature_id,
            "project_id": "proj",
            "setup_id": "setup",
            "content_hash": "hash",
            "slice_index": slice_index,
            "n_channels": 3,
            "height": 8,
            "width": 8,
            "blob_dir": blob_dir,
            "setup_snapshot": "{}",
            "status": "ready",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )


def test_delete_feature_bank_removes_the_row_and_returns_its_blob_dir(catalog: Catalog) -> None:
    _insert_bank(catalog, "fid-1", "/blobs/fid-1")

    blob_dir = catalog.delete_feature_bank("fid-1")

    assert blob_dir == "/blobs/fid-1"
    assert catalog.get_feature_bank("fid-1") is None


def test_delete_feature_bank_is_a_noop_for_an_unknown_id(catalog: Catalog) -> None:
    assert catalog.delete_feature_bank("does-not-exist") is None


def test_delete_feature_bank_does_not_touch_other_rows(catalog: Catalog) -> None:
    _insert_bank(catalog, "fid-1", "/blobs/fid-1", slice_index=0)
    _insert_bank(catalog, "fid-2", "/blobs/fid-2", slice_index=1)

    catalog.delete_feature_bank("fid-1")

    assert catalog.get_feature_bank("fid-1") is None
    assert catalog.get_feature_bank("fid-2") is not None
