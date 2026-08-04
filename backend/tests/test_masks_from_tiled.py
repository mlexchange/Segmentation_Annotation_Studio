"""Tests for the masks-READ-BACK path: turning a previously-written Tiled
``<stem>__masks`` sidecar (semantic class-index volume + legend metadata) back
into vectorized annotation shapes, so the Annotate tab can "Load saved masks"
instead of only being able to write them (``tiled_mask_sync.build_mask_volumes``
/ ``write_masks_to_tiled``, covered separately in ``test_tiled_mask_sync.py``).

Covers, at the pure-function/worker level (``tiled_mask_sync.*`` called
directly, no HTTP):

  * ``_legend_lut`` remaps a STORED semantic pixel value (the legend's own id,
    which may be non-contiguous or not start at 1) to the 1-based positional
    convention ``infer_jobs._vectorize_label_map`` requires.
  * ``read_masks_summary`` is a metadata-only probe that never raises.
  * ``run_masks_readback_job`` reads the semantic volume PER FRAME (never a
    whole-volume ``[...]`` read — the single most important property here,
    since a whole-volume read is the ~397MB-vs-~10MB memory regression this
    feature is specifically designed to avoid), maps axis-0 position to the
    real stored slice number, tolerates a bad frame without aborting the
    others, supports cooperative cancellation, and fails the job cleanly with
    a specific message for each of the ways saved masks can be missing or
    inconsistent.

And, at the HTTP boundary (ASGITransport, mirroring ``test_reset_tiled.py``):
the two new routes' request/response shapes, the 404 when nothing is saved,
job registration on success, and OpenAPI registration.
"""
from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server
import export_jobs
import tiled_mask_sync
from annotation_server import app
from schemas import MasksFromTiledRequest

# ---------------------------------------------------------------------------
# Test doubles
#
# Deliberately NOT a reuse of test_tiled_mask_sync.py's `_FakeArray` — that one
# asserts `key is Ellipsis` (the OLD whole-volume-read pattern the WRITE path
# still uses for a one-shot merge). The read-back path added here reads one
# frame at a time, so `_FakeFrameArray` asserts the opposite: it is a hard
# test failure if it is EVER indexed with Ellipsis.
# ---------------------------------------------------------------------------


class _FakeFrameArray:
    """Stand-in for ``container["semantic"]``: integer positional indexing
    only, returning one (H, W) frame per position — proves the code under
    test never materializes the whole (n, H, W) volume at once."""

    def __init__(self, frames: list[np.ndarray], fail_at: dict[int, Exception] | None = None) -> None:
        self.frames = frames
        self.fail_at = fail_at or {}
        self.shape = (len(frames),) + (frames[0].shape if frames else ())
        self.accessed: list[int] = []

    def __getitem__(self, key):
        assert key is not Ellipsis, "expected a per-frame read, got a whole-volume read"
        self.accessed.append(key)
        if key in self.fail_at:
            raise self.fail_at[key]
        return self.frames[key]


class _FakeMasksContainer:
    """Stand-in for the ``<stem>__masks`` container: ``.metadata`` plus an
    optional ``"semantic"`` child (``None`` simulates it being absent)."""

    def __init__(self, metadata: dict, semantic: _FakeFrameArray | None = None) -> None:
        self._metadata = dict(metadata)
        self._semantic = semantic

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def __getitem__(self, key: str):
        if key == "semantic" and self._semantic is not None:
            return self._semantic
        raise KeyError(key)


class _FakeRoot:
    """Minimal stand-in for a Tiled client / container node: dict-style
    ``__getitem__`` only, so it can represent either the client root or one
    level of the parent chain ``_resolve_masks_parent`` walks."""

    def __init__(self, children: dict) -> None:
        self._children = dict(children)

    def __getitem__(self, key: str):
        return self._children[key]  # raises KeyError naturally when absent


class _BrokenMetaContainer:
    """A container whose metadata is unreadable — simulates a malformed /
    transiently-unreachable sidecar for ``read_masks_summary``'s must-never-
    raise contract."""

    @property
    def metadata(self):
        raise RuntimeError("bad metadata")


def _legend(*pairs: tuple[int, str]) -> list[dict]:
    return [{"id": i, "name": n, "color": "#ffffff"} for i, n in pairs]


def _request(source: str = "sample", server_uri: str | None = None, min_area: int = 10, simplify_tol: float = 0.0) -> MasksFromTiledRequest:
    return MasksFromTiledRequest(kind="tiled", source=source, server_uri=server_uri, min_area=min_area, simplify_tol=simplify_tol)


def _make_root(
    source: str,
    *,
    legend: list[dict] | None,
    slice_indices: list[int] | None,
    frames: list[np.ndarray] | None = None,
    fail_at: dict[int, Exception] | None = None,
) -> tuple[_FakeRoot, _FakeMasksContainer, _FakeFrameArray | None]:
    """Build a fake client rooted so that ``_resolve_masks_parent(root, source)``
    finds the ``<stem>__masks`` sibling, wrapping any needed intermediate path
    segments in nested ``_FakeRoot``s (so sources containing "/" work too)."""
    parts = [p for p in source.strip("/").split("/") if p]
    stem = parts[-1]
    masks_key = f"{stem}__masks"

    metadata: dict = {"studio_type": "segmentation_masks", "updated_at": "2024-01-01T00:00:00+00:00"}
    if legend is not None:
        metadata["legend"] = legend
    if slice_indices is not None:
        metadata["slice_indices"] = slice_indices

    fake_array = _FakeFrameArray(frames, fail_at=fail_at) if frames is not None else None
    container = _FakeMasksContainer(metadata, semantic=fake_array)

    # Build bottom-up: innermost node is {masks_key: container} (what
    # _resolve_masks_parent's final `parent` looks up); each remaining path
    # segment (walked outer-to-inner by _resolve_masks_parent) wraps that in
    # one more level, so e.g. "browse/ds/img" yields
    # root["browse"]["ds"]["img__masks"] is container.
    node: object = _FakeRoot({masks_key: container})
    for part in reversed(parts[:-1]):
        node = _FakeRoot({part: node})
    root = node
    return root, container, fake_array  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# _legend_lut
# ---------------------------------------------------------------------------


def test_legend_lut_maps_non_contiguous_ids_by_sorted_position() -> None:
    """Legend ids 5 and 9 (not 1/2) must round-trip to classId 5/9 in
    run_classes, while the LUT maps the STORED value to the 1-based
    positional index _vectorize_label_map's convention needs."""
    legend = [{"id": 9, "name": "Air", "color": "#222222"}, {"id": 5, "name": "Sand", "color": "#111111"}]

    run_classes, lut = tiled_mask_sync._legend_lut(legend)

    # Sorted ascending by id, regardless of input order.
    assert [c["classId"] for c in run_classes] == [5, 9]
    assert run_classes[0] == {"classId": 5, "label": "Sand", "color": "#111111"}
    assert run_classes[1] == {"classId": 9, "label": "Air", "color": "#222222"}

    assert lut[5] == 1
    assert lut[9] == 2
    other_values = [int(v) for i, v in enumerate(lut) if i not in (5, 9)]
    assert all(v == 0 for v in other_values)


# ---------------------------------------------------------------------------
# read_masks_summary
# ---------------------------------------------------------------------------


def test_read_masks_summary_reports_unavailable_when_no_masks_container(monkeypatch) -> None:
    root = _FakeRoot({})
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    summary = tiled_mask_sync.read_masks_summary("sample", None)

    assert summary == {"available": False}


def test_read_masks_summary_reports_full_shape_when_masks_exist(monkeypatch) -> None:
    legend = _legend((1, "Cell"), (2, "Wall"))
    root, _container, _arr = _make_root("sample", legend=legend, slice_indices=[3, 7])
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    summary = tiled_mask_sync.read_masks_summary("sample", None)

    assert summary["available"] is True
    assert summary["path"] == "sample__masks"
    assert summary["slice_indices"] == [3, 7]
    assert summary["n_slices"] == 2  # falls back to len(slice_indices) when metadata omits it
    assert summary["updated_at"] == "2024-01-01T00:00:00+00:00"
    assert summary["classes"] == legend


def test_read_masks_summary_never_raises_on_unreadable_metadata(monkeypatch) -> None:
    root = _FakeRoot({"sample__masks": _BrokenMetaContainer()})
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    summary = tiled_mask_sync.read_masks_summary("sample", None)

    assert summary == {"available": False, "error": "bad metadata"}


def test_read_masks_summary_never_raises_when_client_lookup_fails(monkeypatch) -> None:
    def _boom(uri=None):
        raise RuntimeError("no such tiled server")

    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", _boom)

    summary = tiled_mask_sync.read_masks_summary("sample", None)

    assert summary == {"available": False, "error": "no such tiled server"}


# ---------------------------------------------------------------------------
# run_masks_readback_job
# ---------------------------------------------------------------------------


def test_readback_job_maps_axis0_position_to_real_slice_and_never_reads_whole_volume(monkeypatch) -> None:
    """2 frames at axis-0 positions 0,1 but slice_indices [3, 7] (deliberately
    NOT [0, 1]) with non-contiguous legend ids — the definitive round-trip
    proving position != real slice number, and that only per-frame reads
    ever happen."""
    legend = _legend((9, "Air"), (5, "Sand"))
    frame0 = np.zeros((12, 12), dtype=np.uint8)
    frame0[2:10, 2:10] = 5  # Sand: legend id 5 -> run_classes position 1
    frame1 = np.zeros((12, 12), dtype=np.uint8)
    frame1[2:10, 2:10] = 9  # Air: legend id 9 -> run_classes position 2

    root, _container, fake_array = _make_root(
        "browse/ds/img", legend=legend, slice_indices=[3, 7], frames=[frame0, frame1],
    )
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request(source="browse/ds/img"))

    job = export_jobs.get_job(jid)
    assert job["state"] == "done"
    result = job["result"]

    assert set(result["slices"]) == {"3", "7"}
    assert all(shape["classId"] == 5 for shape in result["slices"]["3"])
    assert all(shape["classId"] == 9 for shape in result["slices"]["7"])
    assert result["errors"] == []
    assert result["cancelled"] is False
    assert result["path"] == "browse/ds/img__masks"

    # The whole point: only integer per-frame reads happened, never Ellipsis
    # (also hard-enforced inside _FakeFrameArray.__getitem__ itself).
    assert fake_array.accessed == [0, 1]


def test_readback_job_cancellation_after_first_frame_keeps_partial_result_and_stays_done(monkeypatch) -> None:
    legend = _legend((1, "Grain"))
    frame = np.zeros((10, 10), dtype=np.uint8)
    frame[2:8, 2:8] = 1
    frames = [frame.copy(), frame.copy(), frame.copy()]

    root, _container, fake_array = _make_root("sample", legend=legend, slice_indices=[10, 20, 30], frames=frames)
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    calls = {"n": 0}

    def _fake_cancel_requested(jid: str) -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # not cancelled on the first (pre-frame-0) check, cancelled after

    monkeypatch.setattr(export_jobs, "cancel_requested", _fake_cancel_requested)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request())

    job = export_jobs.get_job(jid)
    assert job["state"] == "done"  # a clean stop, not a job failure
    result = job["result"]
    assert result["cancelled"] is True
    assert set(result["slices"]) == {"10"}
    # Frame 0 was processed before the cancel check tripped; frames 1/2 never read.
    assert fake_array.accessed == [0]


def test_readback_job_one_bad_frame_does_not_abort_the_others(monkeypatch) -> None:
    legend = _legend((7, "Grain"))
    good0 = np.zeros((10, 10), dtype=np.uint8)
    good0[2:8, 2:8] = 7
    bad = np.zeros((10, 10), dtype=np.uint8)
    good2 = np.zeros((10, 10), dtype=np.uint8)
    good2[2:8, 2:8] = 7

    root, _container, _arr = _make_root(
        "sample",
        legend=legend,
        slice_indices=[10, 20, 30],
        frames=[good0, bad, good2],
        fail_at={1: RuntimeError("corrupt frame")},
    )
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request())

    job = export_jobs.get_job(jid)
    assert job["state"] == "done"  # partial success, not a job failure
    result = job["result"]
    assert set(result["slices"]) == {"10", "30"}
    assert all(shape["classId"] == 7 for shape in result["slices"]["10"])
    assert all(shape["classId"] == 7 for shape in result["slices"]["30"])
    assert result["errors"] == [{"slice": 20, "error": "corrupt frame"}]


def test_readback_job_all_frames_failing_ends_the_job_in_error_state(monkeypatch) -> None:
    legend = _legend((1, "Grain"))
    frames = [np.zeros((5, 5), dtype=np.uint8), np.zeros((5, 5), dtype=np.uint8)]
    fail_at = {0: RuntimeError("bad frame 0"), 1: RuntimeError("bad frame 1")}

    root, _container, _arr = _make_root(
        "sample", legend=legend, slice_indices=[1, 2], frames=frames, fail_at=fail_at,
    )
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request())

    job = export_jobs.get_job(jid)
    assert job["state"] == "error"
    assert isinstance(job["error"], str) and job["error"]


def test_readback_job_missing_container_errors_with_no_saved_masks_message(monkeypatch) -> None:
    root = _FakeRoot({})
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request())

    job = export_jobs.get_job(jid)
    assert job["state"] == "error"
    assert "no saved masks" in job["error"].lower()


def test_readback_job_empty_legend_errors_mentioning_legend(monkeypatch) -> None:
    root, _container, _arr = _make_root("sample", legend=[], slice_indices=[1, 2])
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request())

    job = export_jobs.get_job(jid)
    assert job["state"] == "error"
    assert "legend" in job["error"].lower()


def test_readback_job_missing_slice_indices_errors_mentioning_slices(monkeypatch) -> None:
    legend = _legend((1, "Grain"))
    root, _container, _arr = _make_root("sample", legend=legend, slice_indices=[])
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request())

    job = export_jobs.get_job(jid)
    assert job["state"] == "error"
    assert "slice" in job["error"].lower()


def test_readback_job_missing_semantic_key_errors_mentioning_semantic(monkeypatch) -> None:
    legend = _legend((1, "Grain"))
    # frames=None -> the fake container has no "semantic" child at all.
    root, _container, _arr = _make_root("sample", legend=legend, slice_indices=[1, 2], frames=None)
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request())

    job = export_jobs.get_job(jid)
    assert job["state"] == "error"
    assert "semantic" in job["error"].lower()


def test_readback_job_shape_mismatch_errors_suggesting_rewrite(monkeypatch) -> None:
    legend = _legend((1, "Grain"))
    frames = [np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)]  # only 2 frames
    root, _container, _arr = _make_root(
        "sample", legend=legend, slice_indices=[1, 2, 3], frames=frames,  # but 3 recorded slices
    )
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request())

    job = export_jobs.get_job(jid)
    assert job["state"] == "error"
    assert "inconsistent" in job["error"].lower()


def test_readback_job_single_class_filling_whole_frame_still_yields_a_shape(monkeypatch) -> None:
    """Cheap sanity check that a dominant/border-touching class (the whole
    frame is one class, no background) doesn't break the read-back path —
    _vectorize_label_map's own border-padding trick is exercised indirectly."""
    legend = _legend((3, "OnlyClass"))
    frame = np.full((6, 6), 3, dtype=np.uint8)

    root, _container, _arr = _make_root("sample", legend=legend, slice_indices=[0], frames=[frame])
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri=None: root)

    jid = export_jobs.new_job("")
    tiled_mask_sync.run_masks_readback_job(jid, _request(min_area=1))

    job = export_jobs.get_job(jid)
    assert job["state"] == "done"
    shapes = job["result"]["slices"]["0"]
    assert len(shapes) >= 1
    assert shapes[0]["classId"] == 3
    assert shapes[0]["kind"] == "polygon"


# ---------------------------------------------------------------------------
# HTTP routes: GET /api/masks/from-tiled/preview, POST /api/masks/from-tiled
#
# annotation_server.py imports tiled_mask_sync LAZILY inside each route
# function body (not at module level), so `annotation_server.tiled_mask_sync`
# is not an attribute to monkeypatch. Python's `import tiled_mask_sync` there
# just looks up the same module object already in sys.modules, so patching
# attributes directly on the `tiled_mask_sync` module object (imported at the
# top of this file) affects the route regardless of where/when it imports it
# — verified by the passing tests below.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_route_returns_summary_dict_verbatim_when_available(monkeypatch) -> None:
    fake_summary = {
        "available": True,
        "path": "sample__masks",
        "n_slices": 1,
        "slice_indices": [0],
        "updated_at": "2024-01-01T00:00:00+00:00",
        "classes": [{"id": 1, "name": "Cell", "color": "#ff0000"}],
    }
    monkeypatch.setattr(tiled_mask_sync, "read_masks_summary", lambda source, server_uri: dict(fake_summary))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/masks/from-tiled/preview", params={"source": "sample"})

    assert response.status_code == 200
    assert response.json() == fake_summary


@pytest.mark.asyncio
async def test_preview_route_returns_unavailable_dict_verbatim(monkeypatch) -> None:
    monkeypatch.setattr(tiled_mask_sync, "read_masks_summary", lambda source, server_uri: {"available": False})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/masks/from-tiled/preview", params={"source": "sample"})

    assert response.status_code == 200
    assert response.json() == {"available": False}


@pytest.mark.asyncio
async def test_post_returns_404_when_no_saved_masks(monkeypatch) -> None:
    monkeypatch.setattr(tiled_mask_sync, "read_masks_summary", lambda source, server_uri: {"available": False})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/masks/from-tiled", json={"kind": "tiled", "source": "sample"})

    assert response.status_code == 404
    detail = str(response.json().get("detail", "")).lower()
    assert "saved masks" in detail or "write masks to tiled" in detail


@pytest.mark.asyncio
async def test_post_returns_job_id_and_registers_a_real_job_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        tiled_mask_sync, "read_masks_summary",
        lambda source, server_uri: {"available": True, "path": "sample__masks", "n_slices": 1,
                                     "slice_indices": [0], "updated_at": None, "classes": []},
    )

    captured: dict = {}

    def _fake_worker(jid: str, request) -> None:
        captured["jid"] = jid
        captured["source"] = request.source
        export_jobs.update(jid, state="done", result={"classes": [], "slices": {}, "n_shapes": 0})

    monkeypatch.setattr(tiled_mask_sync, "run_masks_readback_job", _fake_worker)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/masks/from-tiled", json={"kind": "tiled", "source": "sample"})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("job_id"), str) and body["job_id"]
    assert export_jobs.get_job(body["job_id"]) is not None


@pytest.mark.asyncio
async def test_post_rejects_non_tiled_kind_via_real_pydantic_validation() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/masks/from-tiled", json={"kind": "local", "source": "sample"})

    assert response.status_code == 422


def test_openapi_registers_both_masks_from_tiled_routes() -> None:
    schema = annotation_server.app.openapi()
    assert "post" in schema["paths"]["/api/masks/from-tiled"]
    assert "get" in schema["paths"]["/api/masks/from-tiled/preview"]
