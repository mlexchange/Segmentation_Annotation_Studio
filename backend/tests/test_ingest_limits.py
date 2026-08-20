"""Resource and target-boundary tests for image ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import annotation_server
import ingest


class _NoopThread:
    """Prevent the pre-fix route from starting a real Tiled worker in limit tests."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass


class _SyncThread:
    """Runs `target` immediately instead of on a real thread.

    Unlike `_NoopThread`, this actually exercises the route's background
    closure — needed to test the cache-invalidation wrapper around
    ingest_mod.run_ingest_job, which only runs inside that closure.
    """

    def __init__(self, target=None, args=(), kwargs=None, daemon=None) -> None:
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self) -> None:
        self._target(*self._args, **self._kwargs)


def test_upload_rejects_too_many_files_before_starting_job(monkeypatch) -> None:
    """File-count limits are enforced before allocating temporary upload storage."""
    monkeypatch.setattr(annotation_server, "MAX_INGEST_FILES", 2, raising=False)
    monkeypatch.setattr(annotation_server.threading, "Thread", _NoopThread)
    response = TestClient(annotation_server.app).post(
        "/api/ingest/upload",
        data={"container_path": "browse/test"},
        files=[
            ("files", (f"file-{index}.txt", b"x", "text/plain"))
            for index in range(3)
        ],
    )

    assert response.status_code == 413


def test_upload_rejects_per_file_byte_limit(monkeypatch) -> None:
    """Streaming stops and rejects a file as soon as its byte quota is exceeded."""
    monkeypatch.setattr(annotation_server, "MAX_INGEST_FILES", 10, raising=False)
    monkeypatch.setattr(annotation_server, "MAX_INGEST_FILE_BYTES", 4, raising=False)
    monkeypatch.setattr(annotation_server, "MAX_INGEST_TOTAL_BYTES", 100, raising=False)
    monkeypatch.setattr(annotation_server.threading, "Thread", _NoopThread)
    monkeypatch.setattr(annotation_server.ingest_mod, "new_job", lambda *args: "no-job")

    response = TestClient(annotation_server.app).post(
        "/api/ingest/upload",
        data={"container_path": "browse/test"},
        files=[("files", ("image.npy", b"12345", "application/octet-stream"))],
    )

    assert response.status_code == 413


def test_upload_invalidates_browse_caches_once_the_job_finishes(monkeypatch) -> None:
    """Regression: nothing cleared Browse's caches after an ingest, so a listing
    fetched just before an upload could keep being served for the rest of
    BROWSE_CACHE_TTL_SECONDS — a freshly-uploaded sample invisible in Browse."""
    monkeypatch.setattr(annotation_server.threading, "Thread", _SyncThread)
    monkeypatch.setattr(annotation_server.ingest_mod, "new_job", lambda *args: "job-1")
    monkeypatch.setattr(annotation_server.ingest_mod, "run_ingest_job", lambda *args, **kwargs: None)

    annotation_server._items_cache.set(("items", "probe"), {"stale": True})
    annotation_server._column_cache.set(("column", "probe"), {"stale": True})
    annotation_server._field_mapping_cache.set(("mapping", "probe"), {"stale": True})

    response = TestClient(annotation_server.app).post(
        "/api/ingest/upload",
        data={"container_path": "browse/test"},
        files=[("files", ("image.npy", b"12345", "application/octet-stream"))],
    )

    assert response.status_code == 200
    assert annotation_server._items_cache.get(("items", "probe")) is None
    assert annotation_server._column_cache.get(("column", "probe")) is None
    assert annotation_server._field_mapping_cache.get(("mapping", "probe")) is None


def test_upload_invalidates_browse_caches_even_when_the_job_errors(monkeypatch) -> None:
    """The invalidation must run in a `finally`, since a failed ingest can still
    have partially written samples worth refreshing Browse for."""
    monkeypatch.setattr(annotation_server.threading, "Thread", _SyncThread)
    monkeypatch.setattr(annotation_server.ingest_mod, "new_job", lambda *args: "job-1")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated ingest failure")

    monkeypatch.setattr(annotation_server.ingest_mod, "run_ingest_job", _boom)
    annotation_server._items_cache.set(("items", "probe2"), {"stale": True})

    with pytest.raises(RuntimeError):
        TestClient(annotation_server.app).post(
            "/api/ingest/upload",
            data={"container_path": "browse/test"},
            files=[("files", ("image.npy", b"12345", "application/octet-stream"))],
        )

    assert annotation_server._items_cache.get(("items", "probe2")) is None


def test_upload_rejects_an_unknown_grouping_mode(monkeypatch) -> None:
    monkeypatch.setattr(annotation_server.threading, "Thread", _NoopThread)
    monkeypatch.setattr(annotation_server.ingest_mod, "new_job", lambda *args: "no-job")

    response = TestClient(annotation_server.app).post(
        "/api/ingest/upload",
        data={"container_path": "browse/test", "grouping": "by_vibes"},
        files=[("files", ("image.npy", b"12345", "application/octet-stream"))],
    )

    assert response.status_code == 400


def test_preflight_rejects_an_unknown_grouping_mode() -> None:
    """The preflight body field is a Literal, so this 422s at the schema
    boundary rather than surfacing as preflight()'s broad-except 502."""
    response = TestClient(annotation_server.app).post(
        "/api/ingest/preflight",
        json={"container_path": "browse/test", "names": [], "grouping": "by_vibes"},
    )

    assert response.status_code == 422


def test_upload_rejects_aggregate_byte_limit(monkeypatch) -> None:
    """The aggregate upload quota applies even when each file is individually small."""
    monkeypatch.setattr(annotation_server, "MAX_INGEST_FILES", 10, raising=False)
    monkeypatch.setattr(annotation_server, "MAX_INGEST_FILE_BYTES", 10, raising=False)
    monkeypatch.setattr(annotation_server, "MAX_INGEST_TOTAL_BYTES", 6, raising=False)
    monkeypatch.setattr(annotation_server.threading, "Thread", _NoopThread)
    monkeypatch.setattr(annotation_server.ingest_mod, "new_job", lambda *args: "no-job")

    response = TestClient(annotation_server.app).post(
        "/api/ingest/upload",
        data={"container_path": "browse/test"},
        files=[
            ("files", ("one.npy", b"1234", "application/octet-stream")),
            ("files", ("two.npy", b"5678", "application/octet-stream")),
        ],
    )

    assert response.status_code == 413


def test_npy_decoded_size_is_checked_before_use(tmp_path: Path, monkeypatch) -> None:
    """Small encoded files cannot request arrays larger than the decoded quota."""
    path = tmp_path / "array.npy"
    np.save(path, np.zeros((4, 4), dtype=np.uint8))
    monkeypatch.setattr(ingest, "MAX_DECODED_BYTES", 8, raising=False)

    with pytest.raises(ValueError, match="decoded byte limit"):
        ingest._read_array(path)


@pytest.mark.parametrize("path", ["", "/", "admin", "browse/../admin", "browse//dataset"])
def test_ingest_target_is_restricted_to_safe_child_of_configured_root(path: str) -> None:
    """Ingest cannot write or replace arbitrary nodes elsewhere in Tiled."""
    with pytest.raises(ValueError):
        ingest.validate_container_path(path)


def test_ingest_target_accepts_safe_browse_child(monkeypatch) -> None:
    """The documented browse upload workflow remains available."""
    monkeypatch.setenv("TILED_INGEST_ROOT", "browse")
    assert ingest.validate_container_path("browse/project/images") == [
        "browse",
        "project",
        "images",
    ]


# ---------------------------------------------------------------------------
# Starlette's multipart file-count default
# ---------------------------------------------------------------------------

def _tiny_tiff() -> bytes:
    import io

    import numpy as np
    import tifffile

    buf = io.BytesIO()
    tifffile.imwrite(buf, np.zeros((2, 2), dtype=np.uint16))
    return buf.getvalue()


def _upload(client, n_files: int, raw: bytes):
    files = [("files", (f"img_{i:05d}.tif", raw, "image/tiff")) for i in range(n_files)]
    return client.post(
        "/api/ingest/upload",
        data={"container_path": "browse/big", "grouping": "single"},
        files=files,
    )


def test_more_than_1000_files_is_accepted(monkeypatch) -> None:
    """Regression guard for a limit nobody chose. Starlette's Request.form()
    defaults to max_files=1000 and rejects past it with "Too many files.
    Maximum number of files is 1000." — which fires during FastAPI's own body
    parsing, BEFORE the route's MAX_INGEST_FILES check, making our 5000 limit
    unreachable. _LargeUploadRoute pre-parses with our quota instead.
    """
    from fastapi.testclient import TestClient

    import annotation_server

    monkeypatch.setattr(annotation_server.ingest_mod, "run_ingest_job", lambda *a, **k: None)
    raw = _tiny_tiff()
    client = TestClient(annotation_server.app)

    response = _upload(client, 1200, raw)

    assert response.status_code == 200, response.json()
    assert response.json()["total"] == 1200


def test_our_own_file_limit_is_reachable_and_reports_our_wording(monkeypatch) -> None:
    """The other half: raising the ceiling must not remove it. Our limit has to
    be the one that actually fires, with our message and a 413 (Starlette's was
    a 400 phrased differently)."""
    from fastapi.testclient import TestClient

    import annotation_server

    monkeypatch.setattr(annotation_server.ingest_mod, "run_ingest_job", lambda *a, **k: None)
    monkeypatch.setattr(annotation_server, "MAX_INGEST_FILES", 50)
    raw = _tiny_tiff()
    client = TestClient(annotation_server.app)

    assert _upload(client, 50, raw).status_code == 200

    over = _upload(client, 51, raw)
    assert over.status_code == 413
    assert "max 50" in over.json()["detail"]
