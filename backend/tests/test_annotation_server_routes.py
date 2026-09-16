"""HTTP-level tests for annotation_server.py's session-persistence and
export-job-status routes — mirrors the httpx.AsyncClient + ASGITransport
pattern already established in test_ipred_routes.py/test_train_routes.py.

Scoped to the routes testable with local filesystem + in-memory job registry
state (drafts, guide, measure, export status/cancel/download) — the
biggest, most tractable slice of this file's route coverage. Browse/ingest/
zarr/volume/tiff-stack/denoise routes need a real or heavily-mocked Tiled
client and are left for a further pass.
"""
from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import arrays as arrays_mod
import drafts
import export_jobs
import guides
from annotation_server import app


@pytest.fixture()
def local_data_root(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """drafts.py/guides.py compute their storage dir at IMPORT time
    (module-level `_DRAFT_DIR`), so setting LOCAL_DATA_ROOT alone has no
    effect on an already-imported module — patch the attribute directly."""
    draft_dir = tmp_path / ".drafts"
    monkeypatch.setattr(drafts, "_DRAFT_DIR", draft_dir)
    monkeypatch.setattr(guides, "_DRAFT_DIR", draft_dir)
    return draft_dir


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_draft_404_when_none_saved(client, local_data_root):
    response = await client.get("/api/annotations/draft", params={"source_key": "local:x.tif"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_put_then_get_draft_round_trips(client, local_data_root):
    payload = {"classes": [{"classId": 1, "label": "A", "color": "#f00"}], "slices": {"0": []}}
    put_res = await client.put(
        "/api/annotations/draft", params={"source_key": "local:x.tif"}, json=payload,
    )
    assert put_res.status_code == 200

    get_res = await client.get("/api/annotations/draft", params={"source_key": "local:x.tif"})
    assert get_res.status_code == 200
    body = get_res.json()
    assert body["source_key"] == "local:x.tif"
    assert body["payload"]["classes"][0]["label"] == "A"


@pytest.mark.asyncio
async def test_list_drafts_includes_saved_ones(client, local_data_root):
    await client.put(
        "/api/annotations/draft", params={"source_key": "local:x.tif"},
        json={"classes": [], "slices": {}},
    )
    response = await client.get("/api/annotations/drafts")
    assert response.status_code == 200
    keys = [d["source_key"] for d in response.json()]
    assert "local:x.tif" in keys


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_guide_404_when_none_saved(client, local_data_root):
    response = await client.get("/api/guide", params={"source_key": "local:x.tif"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_put_then_get_guide_round_trips(client, local_data_root):
    payload = {"classes": [{"label": "A", "color": "#f00", "description": "an example"}], "notes": "n"}
    put_res = await client.put("/api/guide", params={"source_key": "local:x.tif"}, json=payload)
    assert put_res.status_code == 200

    get_res = await client.get("/api/guide", params={"source_key": "local:x.tif"})
    assert get_res.status_code == 200
    assert get_res.json()["guide"]["notes"] == "n"


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_measure_source(monkeypatch: pytest.MonkeyPatch):
    arr = np.zeros((10, 10), dtype=np.float32)
    # shape_to_mask's rectangle rasterization is inclusive of both endpoints
    # (skimage.draw.rectangle "like mlex") — x=2,y=2,w=3,h=3 covers columns/
    # rows 2..5 inclusive, a 4x4=16 region, not the 3x3=9 a half-open range
    # would give. Filled to match exactly so every measured pixel is 10.0.
    arr[2:6, 2:6] = 10.0

    monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri: arr)
    monkeypatch.setattr(
        arrays_mod, "array_shape_meta",
        lambda node, pyramid=None: {"height": 10, "width": 10, "n_slices": 1},
    )
    monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: node)
    return arr


@pytest.mark.asyncio
async def test_measure_returns_stats_for_shape_region(client, fake_measure_source):
    shape = {"id": "a", "kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 3, "h": 3}
    response = await client.post(
        "/api/measure", params={"source_key": "local:x.tif"},
        json={"slice_index": 0, "shapes": [shape]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pixel_count"] == 16
    assert body["min"] == 10.0
    assert body["max"] == 10.0
    assert body["mean"] == 10.0


@pytest.mark.asyncio
async def test_measure_with_no_shapes_returns_empty_stats(client, fake_measure_source):
    response = await client.post(
        "/api/measure", params={"source_key": "local:x.tif"},
        json={"slice_index": 0, "shapes": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pixel_count"] == 0
    assert body["min"] is None


@pytest.mark.asyncio
async def test_measure_ignores_a_shape_that_fails_to_rasterize(client, fake_measure_source):
    bad_shape = {"id": "bad", "kind": "polygon", "classId": 1, "points": []}
    good_shape = {"id": "good", "kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 3, "h": 3}
    response = await client.post(
        "/api/measure", params={"source_key": "local:x.tif"},
        json={"slice_index": 0, "shapes": [bad_shape, good_shape]},
    )
    assert response.status_code == 200
    assert response.json()["pixel_count"] == 16


# ---------------------------------------------------------------------------
# Export job status/cancel/download
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_export_status_unknown_job_404s(client):
    response = await client.get("/api/export/status/does-not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_export_status_returns_real_job_state(client):
    jid = export_jobs.new_job("some/path")
    export_jobs.update(jid, state="running", phase="working")
    export_jobs.set_total(jid, 10)
    export_jobs.bump(jid, 3)

    response = await client.get(f"/api/export/status/{jid}")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "running"
    assert body["done"] == 3
    assert body["total"] == 10


@pytest.mark.asyncio
async def test_export_cancel_unknown_job_404s(client):
    response = await client.post("/api/export/cancel/does-not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_export_cancel_sets_flag_on_real_job(client):
    jid = export_jobs.new_job("x")
    response = await client.post(f"/api/export/cancel/{jid}")
    assert response.status_code == 200
    assert response.json() == {"cancelled": True}
    assert export_jobs.cancel_requested(jid) is True


@pytest.mark.asyncio
async def test_export_download_404s_when_zip_not_ready(client):
    jid = export_jobs.new_job("x")
    response = await client.get(f"/api/export/download/{jid}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_export_download_streams_the_real_zip_file(client, tmp_path):
    jid = export_jobs.new_job("x")
    zip_path = tmp_path / "out.zip"
    zip_path.write_bytes(b"PK\x03\x04fakezip")
    export_jobs.update(jid, zip_path=str(zip_path))

    response = await client.get(f"/api/export/download/{jid}")
    assert response.status_code == 200
    assert response.content == b"PK\x03\x04fakezip"
    assert response.headers["content-type"] == "application/zip"
