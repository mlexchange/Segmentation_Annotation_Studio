"""Tests for duplicate-upload handling in the ingest flow (issue #8).

Covers ``preflight`` (what already exists at the destination), the three
``on_conflict`` modes of ``run_ingest_job``, and the error classifier that keeps
raw Tiled URLs out of user-facing messages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ingest


class FakeNode:
    """Minimal stand-in for a Tiled container client."""

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
            # Mirrors tiled's ClientError text: "<status>: <path> <internal url>".
            raise RuntimeError(
                f"409: /browse/ds/{key} http://127.0.0.1:8010/api/v1/metadata/browse/ds"
            )
        self._children[key] = "array"
        self.written[key] = (arr, metadata)

    def delete_contents(
        self, keys=None, recursive: bool = False, external_only: bool = True
    ) -> None:
        # Guard against the real API's footgun: keys=None wipes the container.
        assert keys, "delete_contents(None) would delete every sample"
        assert external_only is False, "internally-managed arrays need external_only=False"
        for key in [keys] if isinstance(keys, str) else keys:
            self.deleted.append(key)
            self._children.pop(key, None)

    def delete(self, recursive: bool = False, external_only: bool = True) -> None:
        raise AssertionError("Container.delete() deletes the container itself — never call it")


class FakeArray:
    """Leaf node: arrays have no children, so keys() raises."""

    def keys(self):
        raise AttributeError("arrays have no children")


@pytest.fixture
def fake_client(monkeypatch):
    """Root client with ``browse/ds`` holding two samples."""
    ds = FakeNode({"img_00001": "array", "img_00002": "array"})
    root = FakeNode({"browse": FakeNode({"ds": ds})})
    monkeypatch.setattr(ingest, "api_key_for_uri", lambda uri: None)
    monkeypatch.setattr(ingest, "get_tiled_client", lambda uri, key=None: root)
    return root, ds


def _npy(tmp_path: Path, name: str) -> tuple[str, Path]:
    """Write a tiny .npy temp file and return the (original_name, path) pair."""
    path = tmp_path / f"{name}.npy"
    np.save(path, np.zeros((4, 4), dtype=np.uint8))
    return (f"{name}.npy", path)


def _run(jid: str, temp_files, on_conflict: str) -> dict:
    ingest.run_ingest_job(jid, None, "browse/ds", temp_files, "", on_conflict)
    return ingest.get_job(jid)


# --- preflight ---------------------------------------------------------------


def test_preflight_reports_overlapping_keys(fake_client) -> None:
    result = ingest.preflight(None, "browse/ds", ["img_00002.tif", "img_00009.tif"])

    assert result["container_exists"] is True
    assert result["existing_count"] == 2
    assert result["conflicts"] == [{"filename": "img_00002.tif", "key": "img_00002"}]
    assert result["suggested_container_path"] == "browse/ds_2"


def test_preflight_missing_container_has_no_conflicts(fake_client) -> None:
    result = ingest.preflight(None, "browse/brand_new", ["img_00001.tif"])

    assert result["container_exists"] is False
    assert result["existing_count"] == 0
    assert result["conflicts"] == []


def test_preflight_tolerates_leaf_destination(fake_client, monkeypatch) -> None:
    """Pointing at an array (not a container) must not blow up."""
    root, _ = fake_client
    root["browse"]._children["leaf"] = FakeArray()

    result = ingest.preflight(None, "browse/leaf", ["img_00001.tif"])

    assert result["existing_count"] == 0
    assert result["conflicts"] == []


def test_suggested_path_skips_taken_names_and_bumps_suffix(fake_client) -> None:
    root, _ = fake_client
    browse = root["browse"]
    browse._children["ds_2"] = FakeNode()
    browse._children["ds_3"] = FakeNode()

    assert ingest.preflight(None, "browse/ds", [])["suggested_container_path"] == "browse/ds_4"
    # An existing numeric suffix is bumped, not stacked (never "ds_2_2").
    assert ingest.preflight(None, "browse/ds_2", [])["suggested_container_path"] == "browse/ds_4"


# --- on_conflict modes -------------------------------------------------------


def test_skip_leaves_existing_nodes_untouched(fake_client, tmp_path) -> None:
    _, ds = fake_client
    files = [_npy(tmp_path, "img_00002"), _npy(tmp_path, "img_00009")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files, "skip")

    assert (job["done"], job["skipped"], job["failed"]) == (1, 1, 0)
    assert job["state"] == "done"
    assert job["errors"] == []
    assert ds.deleted == []
    assert list(ds.written) == ["img_00009"]


def test_replace_deletes_then_rewrites(fake_client, tmp_path) -> None:
    _, ds = fake_client
    files = [_npy(tmp_path, "img_00002"), _npy(tmp_path, "img_00009")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files, "replace")

    assert (job["done"], job["skipped"], job["failed"]) == (2, 0, 0)
    assert ds.deleted == ["img_00002"]
    assert sorted(ds.written) == ["img_00002", "img_00009"]


def test_fail_mode_classifies_the_conflict_and_hides_the_url(fake_client, tmp_path) -> None:
    files = [_npy(tmp_path, "img_00001"), _npy(tmp_path, "img_00002")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files, "fail")

    assert (job["done"], job["failed"]) == (0, 2)
    # Nothing landed → 'error', not a green 'done' (issue #8's misleading status).
    assert job["state"] == "error"
    assert [e["kind"] for e in job["errors"]] == ["conflict", "conflict"]
    assert job["errors"][0]["filename"] == "img_00001.npy"
    assert all("http" not in e["message"] for e in job["errors"])


def test_partial_success_still_reports_done(fake_client, tmp_path) -> None:
    files = [_npy(tmp_path, "img_00001"), _npy(tmp_path, "img_00009")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files, "fail")

    assert (job["done"], job["failed"]) == (1, 1)
    assert job["state"] == "done"


def test_unknown_mode_falls_back_to_fail(fake_client, tmp_path) -> None:
    _, ds = fake_client
    files = [_npy(tmp_path, "img_00002")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    job = _run(jid, files, "obliterate")

    assert job["failed"] == 1
    assert ds.deleted == []


def test_temp_files_are_cleaned_up_even_when_skipped(fake_client, tmp_path) -> None:
    files = [_npy(tmp_path, "img_00002"), _npy(tmp_path, "img_00009")]
    jid = ingest.new_job(len(files), None, "browse/ds")

    _run(jid, files, "skip")

    assert not any(path.exists() for _, path in files)


# --- error classification ----------------------------------------------------


@pytest.mark.parametrize(
    "exc, kind",
    [
        (RuntimeError("409: /browse/ds/img http://127.0.0.1:8010/api/v1/metadata/browse/ds"), "conflict"),
        (RuntimeError("401: unauthorized http://127.0.0.1:8010/api/v1/metadata"), "auth"),
        (RuntimeError("403: forbidden http://127.0.0.1:8010/api/v1/metadata"), "auth"),
        (ValueError("unsupported extension '.bin'"), "unreadable"),
        (OSError("truncated file"), "unreadable"),
        (RuntimeError("something odd happened"), "unknown"),
    ],
)
def test_classify_error_kinds(exc: Exception, kind: str) -> None:
    assert ingest._classify_error(exc)["kind"] == kind


def test_classify_error_recognizes_connection_failures() -> None:
    class ConnectError(Exception):
        pass

    assert ingest._classify_error(ConnectError("nope"))["kind"] == "unreachable"


def test_classify_error_strips_urls_from_unknown_messages() -> None:
    exc = RuntimeError("weird failure at http://127.0.0.1:8010/api/v1/metadata/browse/ds")
    message = ingest._classify_error(exc)["message"]

    assert "http" not in message
    assert message.startswith("weird failure at")


def test_filename_digits_are_not_mistaken_for_a_status_code() -> None:
    """'409' inside a message body must not be read as a 409 status."""
    result = ingest._classify_error(RuntimeError("could not read IMG_409.tif"))

    assert result["kind"] == "unknown"


# --- container navigation ----------------------------------------------------


def test_ensure_container_reuses_existing_nodes(fake_client) -> None:
    root, ds = fake_client

    assert ingest._ensure_container(root, ["browse", "ds"]) is ds


def test_ensure_container_creates_missing_nodes(fake_client) -> None:
    root, _ = fake_client
    node = ingest._ensure_container(root, ["browse", "fresh"])

    assert node is root["browse"]["fresh"]


def test_ensure_container_propagates_transport_errors(fake_client) -> None:
    """A transport blip must NOT be misread as 'missing' → create → 409."""

    class Flaky(FakeNode):
        def __getitem__(self, key):
            raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError):
        ingest._ensure_container(Flaky(), ["browse"])
