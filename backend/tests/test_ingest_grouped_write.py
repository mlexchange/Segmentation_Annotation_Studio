"""run_ingest_job's grouped write path — writing into per-sample sub-containers.

test_ingest_conflict.py's fixtures only ever use unprefixed filenames, so
`sample_name_for` returns None for all of them and every assertion there
exercises the ungrouped ("everything in the target") code path. These tests
exercise `sample_nodes`/`dest`/per-sample `existing` — the parts of
`run_ingest_job` that only run when a file actually declares a sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ingest


class FakeNode:
    """Minimal stand-in for a Tiled container client (see test_ingest_conflict)."""

    def __init__(self, children: dict | None = None) -> None:
        self._children: dict = dict(children or {})
        self.written: dict = {}
        self.deleted: list[str] = []
        self.metadata_updates: list[dict] = []

    def __getitem__(self, key: str):
        if key not in self._children:
            raise KeyError(key)
        return self._children[key]

    def keys(self) -> list[str]:
        return list(self._children)

    def create_container(self, key: str, metadata: dict | None = None) -> "FakeNode":
        node = FakeNode()
        self._children[key] = node
        return node

    def update_metadata(self, metadata: dict | None = None) -> None:
        self.metadata_updates.append(metadata or {})

    def write_array(self, arr, key=None, metadata=None, dims=None) -> None:
        if key in self._children:
            raise RuntimeError(f"409: /browse/ds/{key} http://127.0.0.1:8010/api/v1/metadata/browse/ds")
        self._children[key] = "array"
        self.written[key] = (arr, metadata)

    def delete_contents(self, keys=None, recursive: bool = False, external_only: bool = True) -> None:
        assert keys, "delete_contents(None) would delete every sample"
        assert external_only is False, "internally-managed arrays need external_only=False"
        for key in [keys] if isinstance(keys, str) else keys:
            self.deleted.append(key)
            self._children.pop(key, None)


@pytest.fixture
def fake_client(monkeypatch):
    """Root client with an empty ``browse/ds`` — the grouped-upload target."""
    root = FakeNode({"browse": FakeNode({"ds": FakeNode()})})
    monkeypatch.setattr(ingest, "get_tiled_client", lambda uri: root)
    return root


def _npy(tmp_path: Path, name: str) -> tuple[str, Path]:
    path = tmp_path / f"{name}.npy"
    np.save(path, np.zeros((4, 4), dtype=np.uint8))
    return (f"{name}.npy", path)


def _bad(tmp_path: Path, name: str) -> tuple[str, Path]:
    """A file that will fail to decode — .npy magic bytes required, not present."""
    path = tmp_path / f"{name}.npy"
    path.write_bytes(b"not a numpy file")
    return (f"{name}.npy", path)


def _run(jid: str, temp_files, on_conflict: str = "fail", grouping: str = "prefix") -> dict:
    ingest.run_ingest_job(jid, None, "browse/ds", temp_files, "", on_conflict, grouping)
    return ingest.get_job(jid)


def test_grouped_files_land_in_their_own_sample_subcontainer(fake_client, tmp_path) -> None:
    ds = fake_client["browse"]["ds"]
    files = [_npy(tmp_path, "almond_control__slice_01"), _npy(tmp_path, "almond_control__slice_02")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files)

    assert (job["done"], job["failed"]) == (2, 0)
    assert "almond_control" in ds.keys()
    sample_node = ds["almond_control"]
    assert sorted(sample_node.written) == ["almond_control__slice_01", "almond_control__slice_02"]
    # Not written directly into the target.
    assert "almond_control__slice_01" not in ds.keys()


def test_two_samples_get_independent_subcontainers(fake_client, tmp_path) -> None:
    ds = fake_client["browse"]["ds"]
    files = [
        _npy(tmp_path, "almond_control__slice_01"),
        _npy(tmp_path, "almond_drought__slice_01"),
    ]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files)

    assert job["done"] == 2
    assert sorted(k for k in ds.keys()) == ["almond_control", "almond_drought"]
    assert list(ds["almond_control"].written) == ["almond_control__slice_01"]
    assert list(ds["almond_drought"].written) == ["almond_drought__slice_01"]


def test_skip_resolves_against_the_samples_own_children_not_the_target(fake_client, tmp_path) -> None:
    """A stem can legitimately repeat across different samples — this is the bug
    that made preflight and run_ingest_job disagree about where conflicts live:
    the target's children are SAMPLE names, never file stems."""
    ds = fake_client["browse"]["ds"]
    existing_sample = ds.create_container("almond_control")
    # The node key inside a sample keeps the FULL original stem (see run_ingest_job's
    # write_array(key=stem)) — the sample name is a grouping label, not a rename.
    existing_sample._children["almond_control__slice_01"] = "array"

    files = [_npy(tmp_path, "almond_control__slice_01"), _npy(tmp_path, "almond_drought__slice_01")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files, on_conflict="skip")

    assert (job["done"], job["skipped"]) == (1, 1)
    # The almond_drought file, whose stem never collided anywhere, still lands.
    assert list(ds["almond_drought"].written) == ["almond_drought__slice_01"]


def test_replace_deletes_from_the_sample_not_the_target(fake_client, tmp_path) -> None:
    ds = fake_client["browse"]["ds"]
    existing_sample = ds.create_container("almond_control")
    existing_sample._children["almond_control__slice_01"] = "array"

    files = [_npy(tmp_path, "almond_control__slice_01")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files, on_conflict="replace")

    assert job["done"] == 1
    assert existing_sample.deleted == ["almond_control__slice_01"]
    assert ds.deleted == []  # the target itself was never touched


def test_duplicate_stem_within_one_batch_is_caught_by_skip(fake_client, tmp_path) -> None:
    """Regression: `existing` used to be a pre-loop snapshot, so a second file in
    the SAME batch with the same stem (e.g. two folders each containing an
    'img_001') was not recognised as a duplicate and clobbered the first."""
    ds = fake_client["browse"]["ds"]
    files = [
        ("almond_control__slice_01.npy", (tmp_path / "a.npy")),
        ("almond_control__slice_01.npy", (tmp_path / "b.npy")),
    ]
    np.save(files[0][1], np.zeros((4, 4), dtype=np.uint8))
    np.save(files[1][1], np.ones((4, 4), dtype=np.uint8))
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files, on_conflict="skip")

    assert (job["done"], job["skipped"]) == (1, 1)
    assert list(ds["almond_control"].written) == ["almond_control__slice_01"]


def test_parent_container_describes_total_images_and_sample_count(fake_client, tmp_path) -> None:
    """n_images on the parent is the total FILE count, not the sample count —
    the two were confused for each other before this fix."""
    ds = fake_client["browse"]["ds"]
    files = [
        _npy(tmp_path, "almond_control__slice_01"),
        _npy(tmp_path, "almond_control__slice_02"),
        _npy(tmp_path, "almond_drought__slice_01"),
    ]
    jid = ingest.new_job(len(files), None, "browse/ds")

    _run(jid, files)

    parent_meta = ds.metadata_updates[0]
    assert parent_meta["n_images"] == 3
    assert parent_meta["n_samples"] == 2

    sample_meta = ds["almond_control"].metadata_updates[0]
    assert sample_meta["n_images"] == 2
    assert "n_samples" not in sample_meta


def test_all_unprefixed_batch_gets_no_n_samples_field(fake_client, tmp_path) -> None:
    """A z-stack (single sample) must not claim n_samples — it isn't grouped."""
    ds = fake_client["browse"]["ds"]
    files = [_npy(tmp_path, "img_00001"), _npy(tmp_path, "img_00002")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    _run(jid, files)

    assert "n_samples" not in ds.metadata_updates[0]
    assert ds.metadata_updates[0]["n_images"] == 2


def test_sample_with_zero_successful_writes_is_removed(fake_client, tmp_path) -> None:
    """A sample container that never got a real array is worse than useless: it
    claims images it doesn't have, and opening it 422s. It must be cleaned up,
    and the parent's counts corrected."""
    ds = fake_client["browse"]["ds"]
    files = [
        _npy(tmp_path, "almond_control__slice_01"),
        _bad(tmp_path, "almond_drought__slice_01"),
    ]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files)

    assert (job["done"], job["failed"]) == (1, 1)
    assert "almond_drought" not in ds.keys()  # cleaned up
    assert "almond_control" in ds.keys()

    final_parent_meta = ds.metadata_updates[-1]
    assert final_parent_meta["n_images"] == 1
    assert final_parent_meta["n_samples"] == 1


def test_first_path_points_at_the_first_written_array(fake_client, tmp_path) -> None:
    files = [_npy(tmp_path, "almond_control__slice_01"), _npy(tmp_path, "almond_drought__slice_01")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files)

    assert job["first_path"] == "browse/ds/almond_control/almond_control__slice_01"


def test_first_path_skips_a_failed_first_file(fake_client, tmp_path) -> None:
    files = [_bad(tmp_path, "almond_control__slice_01"), _npy(tmp_path, "almond_drought__slice_01")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files)

    assert job["first_path"] == "browse/ds/almond_drought/almond_drought__slice_01"


def test_first_path_has_no_sample_segment_for_an_ungrouped_upload(fake_client, tmp_path) -> None:
    files = [_npy(tmp_path, "img_00001")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files)

    assert job["first_path"] == "browse/ds/img_00001"
