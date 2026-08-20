"""``n_images`` when an upload lands in a container that already holds slices.

Uploading a second batch into an existing sample used to relabel it with only
the newest batch's file count — a 63-slice dataset reporting ``n_images: 25``
after a 25-file top-up. Browse derives ``n_slices`` from the real children, so
the two disagreed visibly in the UI.

Reuses ``test_ingest_grouped_write.FakeNode`` rather than defining a fourth
fake-Tiled harness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ingest
from tests.test_ingest_grouped_write import FakeNode


def _npy(tmp_path: Path, name: str) -> tuple[str, Path]:
    path = tmp_path / f"{name}.npy"
    np.save(path, np.zeros((4, 4), dtype=np.uint8))
    return (f"{name}.npy", path)


def _target_meta(node: FakeNode) -> dict:
    """The last metadata written to a container (that's the sample description)."""
    assert node.metadata_updates, "expected the container to be described"
    return node.metadata_updates[-1]


def _run(target: str, temp_files, grouping: str) -> dict:
    jid = ingest.new_job(len(temp_files), None, target)
    ingest.run_ingest_job(jid, None, target, temp_files, "", "fail", grouping)
    return ingest.get_job(jid)


@pytest.fixture
def client_with_existing(monkeypatch):
    """``browse/ds`` already holding three slices from an earlier upload."""
    existing = FakeNode({"old_0001": "array", "old_0002": "array", "old_0003": "array"})
    root = FakeNode({"browse": FakeNode({"ds": existing})})
    monkeypatch.setattr(ingest, "get_tiled_client", lambda uri: root)
    return root


def test_n_images_counts_existing_slices_plus_the_new_batch(client_with_existing, tmp_path) -> None:
    """The reported bug: a top-up upload must not relabel the sample with just
    its own file count."""
    ds = client_with_existing["browse"]["ds"]
    files = [_npy(tmp_path, f"new_{i:04d}") for i in range(1, 3)]

    job = _run("browse/ds", files, "single")

    assert job["done"] == 2
    assert _target_meta(ds)["n_images"] == 5  # 3 already there + 2 new


def test_n_images_equals_the_batch_size_for_a_fresh_container(monkeypatch, tmp_path) -> None:
    """Regression guard on the fix itself — an empty target must be unaffected."""
    ds = FakeNode()
    root = FakeNode({"browse": FakeNode({"ds": ds})})
    monkeypatch.setattr(ingest, "get_tiled_client", lambda uri: root)
    files = [_npy(tmp_path, f"img_{i:04d}") for i in range(1, 4)]

    _run("browse/ds", files, "single")

    assert _target_meta(ds)["n_images"] == 3


def test_a_reuploaded_sample_subcontainer_also_accumulates(monkeypatch, tmp_path) -> None:
    """Same fix on the grouped path: a sample container that already has slices
    must report the total, not just the newest batch."""
    sample = FakeNode({"almond__old_01": "array", "almond__old_02": "array"})
    ds = FakeNode({"almond": sample})
    root = FakeNode({"browse": FakeNode({"ds": ds})})
    monkeypatch.setattr(ingest, "get_tiled_client", lambda uri: root)
    files = [_npy(tmp_path, "almond__new_01")]

    _run("browse/ds", files, "prefix")

    assert _target_meta(sample)["n_images"] == 3  # 2 existing + 1 new


def test_n_images_matches_the_child_count_after_a_merge(client_with_existing, tmp_path) -> None:
    """The property that actually matters: whatever n_images says, it agrees with
    the container's real children, since Browse reads n_slices from those."""
    ds = client_with_existing["browse"]["ds"]
    files = [_npy(tmp_path, f"new_{i:04d}") for i in range(1, 5)]

    _run("browse/ds", files, "single")

    assert _target_meta(ds)["n_images"] == len(ds.keys())
