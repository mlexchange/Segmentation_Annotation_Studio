"""HTTP-level tests for annotation_server.py's data-source registration
(zarr/tiff-stack), denoise, volume, and remaining train-route wiring.

The underlying modules (zarr_source, tiff_stack_source, volume_build,
volume_nodes, denoise_bake, infer_jobs, train_jobs) already have their own
direct unit tests — these tests are about routing, request validation, and
background-job wiring, so the delegate functions are monkeypatched rather
than re-proven here. Background threads are made synchronous via a fake
`threading.Thread` so job results are observable without polling/sleeping.
"""
from __future__ import annotations

import types

import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server
import denoise as denoise_mod
import export_jobs
import infer_jobs
import tiff_stack_source
import train_common
import volume_build
import volume_nodes
import zarr_source

app = annotation_server.app


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class ImmediateThread:
    """Runs its target synchronously instead of spawning a real thread, so a
    background-job route's result is observable right after the request
    returns, with no polling or sleeping needed."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


@pytest.fixture()
def sync_threads(monkeypatch: pytest.MonkeyPatch):
    # Replace the NAME `threading` inside annotation_server's own module
    # namespace (not the shared stdlib module object) — otherwise this would
    # also intercept unrelated internal thread creation, e.g. inside
    # asyncio.to_thread's own ThreadPoolExecutor, which every route here uses.
    monkeypatch.setattr(annotation_server, "threading", types.SimpleNamespace(Thread=ImmediateThread))


# ---------------------------------------------------------------------------
# /api/zarr/*
# ---------------------------------------------------------------------------

class TestZarrRoutes:
    @pytest.mark.asyncio
    async def test_inspect_delegates_to_zarr_source(self, client, monkeypatch):
        monkeypatch.setattr(zarr_source, "inspect_zarr", lambda path: {"path": path, "levels": 3})
        response = await client.get("/api/zarr/inspect", params={"path": "/data/x.zarr"})
        assert response.status_code == 200
        assert response.json() == {"path": "/data/x.zarr", "levels": 3}

    @pytest.mark.asyncio
    async def test_preflight_delegates_with_request_fields(self, client, monkeypatch):
        calls = []
        monkeypatch.setattr(
            zarr_source, "preflight_zarr",
            lambda server_uri, path, container_path: calls.append((server_uri, path, container_path)) or {"ok": True},
        )
        response = await client.post(
            "/api/zarr/preflight",
            json={"path": "/data/x.zarr", "container_path": "browse/x"},
        )
        assert response.status_code == 200
        assert calls == [(None, "/data/x.zarr", "browse/x")]

    @pytest.mark.asyncio
    async def test_register_rejects_bad_on_conflict(self, client):
        response = await client.post(
            "/api/zarr/register",
            json={"path": "/data/x.zarr", "on_conflict": "explode"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_register_delegates_on_valid_request(self, client, monkeypatch):
        monkeypatch.setattr(zarr_source, "register_zarr", lambda *a: {"key": "x"})
        response = await client.post(
            "/api/zarr/register",
            json={"path": "/data/x.zarr", "on_conflict": "replace"},
        )
        assert response.status_code == 200
        assert response.json() == {"key": "x"}


# ---------------------------------------------------------------------------
# /api/tiff-stack/*
# ---------------------------------------------------------------------------

class TestTiffStackRoutes:
    @pytest.mark.asyncio
    async def test_inspect_delegates(self, client, monkeypatch):
        monkeypatch.setattr(tiff_stack_source, "inspect_tiff_stack", lambda path: {"slices_to_read": 5})
        response = await client.get("/api/tiff-stack/inspect", params={"path": "/data/stack"})
        assert response.status_code == 200
        assert response.json() == {"slices_to_read": 5}

    @pytest.mark.asyncio
    async def test_preflight_delegates(self, client, monkeypatch):
        monkeypatch.setattr(
            tiff_stack_source, "preflight_tiff_stack",
            lambda server_uri, path, container_path: {"collision": False},
        )
        response = await client.post(
            "/api/tiff-stack/preflight",
            json={"path": "/data/stack", "container_path": "browse/s"},
        )
        assert response.status_code == 200
        assert response.json() == {"collision": False}

    @pytest.mark.asyncio
    async def test_register_rejects_bad_on_conflict(self, client):
        response = await client.post(
            "/api/tiff-stack/register",
            json={"path": "/data/stack", "on_conflict": "nope"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_register_runs_job_and_reports_done(self, client, monkeypatch, sync_threads):
        monkeypatch.setattr(tiff_stack_source, "inspect_tiff_stack", lambda path: {"slices_to_read": 3})
        monkeypatch.setattr(
            tiff_stack_source, "register_tiff_stack",
            lambda server_uri, path, container_path, description, on_conflict, progress: {"key": "stack1"},
        )
        response = await client.post(
            "/api/tiff-stack/register",
            json={"path": "/data/stack", "on_conflict": "fail"},
        )
        assert response.status_code == 200
        jid = response.json()["job_id"]
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"] == {"key": "stack1"}

    @pytest.mark.asyncio
    async def test_register_job_reports_error_on_exception(self, client, monkeypatch, sync_threads):
        monkeypatch.setattr(tiff_stack_source, "inspect_tiff_stack", lambda path: {"slices_to_read": 3})

        def boom(*a, **k):
            raise RuntimeError("registration blew up")

        monkeypatch.setattr(tiff_stack_source, "register_tiff_stack", boom)
        response = await client.post(
            "/api/tiff-stack/register",
            json={"path": "/data/stack", "on_conflict": "fail"},
        )
        jid = response.json()["job_id"]
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "registration blew up" in job["error"]


# ---------------------------------------------------------------------------
# /api/denoise/*
# ---------------------------------------------------------------------------

class TestDenoiseMethods:
    @pytest.mark.asyncio
    async def test_lists_described_methods(self, client, monkeypatch):
        monkeypatch.setattr(denoise_mod, "describe_methods", lambda: [{"id": "median"}])
        response = await client.get("/api/denoise/methods")
        assert response.json() == {"methods": [{"id": "median"}]}


class TestDenoiseAuto:
    @pytest.mark.asyncio
    async def test_unknown_method_is_400(self, client):
        response = await client.get(
            "/api/denoise/auto", params={"source": "x", "kind": "local", "method": "not_real"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_suggests_strength_for_a_real_slice(self, client, monkeypatch):
        import numpy as np

        import arrays as arrays_mod

        arr = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri, root: "node")
        monkeypatch.setattr(arrays_mod, "pyramid_info", lambda source, kind, server_uri, root: None)
        monkeypatch.setattr(arrays_mod, "array_shape_meta", lambda node, pyramid: {"shape_kind": "HW"})
        monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: arr)
        response = await client.get(
            "/api/denoise/auto", params={"source": "x", "kind": "local", "method": "median"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "strength" in body
        assert "noise_sigma" in body


class TestDenoiseBakeRoute:
    @pytest.mark.asyncio
    async def test_method_none_is_422(self, client):
        response = await client.post(
            "/api/denoise/bake", json={"source": "browse/s", "method": "none"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_unknown_method_is_422(self, client):
        response = await client.post(
            "/api/denoise/bake", json={"source": "browse/s", "method": "not_real"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_unavailable_method_is_422(self, client, monkeypatch):
        monkeypatch.setattr(denoise_mod, "available_methods", lambda: ("median",))
        response = await client.post(
            "/api/denoise/bake", json={"source": "browse/s", "method": "wavelet"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_target_path_is_422(self, client):
        response = await client.post(
            "/api/denoise/bake",
            json={"source": "browse/s", "method": "median", "target_path": "../escape"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_valid_request_starts_a_job(self, client, monkeypatch, sync_threads):
        import denoise_bake as denoise_bake_mod

        monkeypatch.setattr(denoise_bake_mod, "run_denoise_bake_job", lambda jid, payload: export_jobs.update(jid, state="done"))
        response = await client.post(
            "/api/denoise/bake", json={"source": "browse/s", "method": "median"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "job_id" in body
        assert body["target_path"] == "browse/s_denoised"
        assert export_jobs.get_job(body["job_id"])["state"] == "done"


# ---------------------------------------------------------------------------
# /api/train/start — remaining guard paths not covered by test_train_routes.py
# ---------------------------------------------------------------------------

class TestTrainStartGuards:
    def _payload(self, **overrides):
        base = {
            "task": "segmentation",
            "sources": [{"kind": "local", "source": "fake.tif", "slices": {"0": []}}],
            "classes": [{"classId": 1, "label": "a", "color": "#ff0000"}],
            "model": {"model_family": "dlsia_tunet"},
        }
        base.update(overrides)
        return base

    @pytest.mark.asyncio
    async def test_torch_unavailable_is_503(self, client, monkeypatch):
        monkeypatch.setattr(train_common, "torch_available", lambda: False)
        response = await client.post("/api/train/start", json=self._payload())
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_dlsia_unavailable_for_tunet_is_503(self, client, monkeypatch):
        monkeypatch.setattr(train_common, "torch_available", lambda: True)
        monkeypatch.setattr(train_common, "dlsia_available", lambda: False)
        response = await client.post("/api/train/start", json=self._payload())
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_busy_ml_lock_is_409(self, client, monkeypatch):
        monkeypatch.setattr(train_common, "torch_available", lambda: True)
        monkeypatch.setattr(train_common, "dlsia_available", lambda: True)
        train_common.ML_LOCK.acquire()
        try:
            response = await client.post("/api/train/start", json=self._payload())
            assert response.status_code == 409
        finally:
            train_common.ML_LOCK.release()

    @pytest.mark.asyncio
    async def test_resume_incompatible_is_400(self, client, monkeypatch):
        import train_jobs

        monkeypatch.setattr(train_common, "torch_available", lambda: True)
        monkeypatch.setattr(train_common, "dlsia_available", lambda: True)
        monkeypatch.setattr(train_common, "load_run_config", lambda run_id: {"some": "config"})

        def raise_incompatible(parent_config, payload):
            raise ValueError("architectures differ")

        monkeypatch.setattr(train_jobs, "check_resume_compatible", raise_incompatible)
        response = await client.post(
            "/api/train/start", json=self._payload(resume_from_run_id="parent1"),
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_success_creates_a_job(self, client, monkeypatch, sync_threads):
        import train_jobs

        monkeypatch.setattr(train_common, "torch_available", lambda: True)
        monkeypatch.setattr(train_common, "dlsia_available", lambda: True)
        monkeypatch.setattr(train_jobs, "run_train_job", lambda jid, payload, run_id: export_jobs.update(jid, state="done"))
        response = await client.post("/api/train/start", json=self._payload())
        assert response.status_code == 200
        body = response.json()
        assert "job_id" in body and "run_id" in body
        assert export_jobs.get_job(body["job_id"])["state"] == "done"


class TestTrainInferPreviewAndWriteTiled:
    @pytest.mark.asyncio
    async def test_preview_delegates_to_infer_jobs(self, client, monkeypatch):
        monkeypatch.setattr(infer_jobs, "preview_png", lambda job_id, slice_index: b"pngbytes")
        response = await client.get("/api/train/infer/preview/job1/3")
        assert response.status_code == 200
        assert response.content == b"pngbytes"
        assert response.headers["content-type"] == "image/png"

    @pytest.mark.asyncio
    async def test_write_tiled_starts_a_new_job(self, client, monkeypatch, sync_threads):
        monkeypatch.setattr(
            infer_jobs, "run_write_tiled_job",
            lambda write_jid, infer_job_id: export_jobs.update(write_jid, state="done", result={"source": infer_job_id}),
        )
        response = await client.post("/api/train/infer/write-tiled/infer-job-1")
        assert response.status_code == 200
        jid = response.json()["job_id"]
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"] == {"source": "infer-job-1"}


# ---------------------------------------------------------------------------
# /api/volume/*
# ---------------------------------------------------------------------------

class TestVolumeRoutes:
    @pytest.mark.asyncio
    async def test_resolve_delegates(self, client, monkeypatch):
        monkeypatch.setattr(volume_nodes, "resolve_volume", lambda server_uri, source: {"key": "vol1"})
        response = await client.get("/api/volume/resolve", params={"source": "browse/s"})
        assert response.json() == {"key": "vol1"}

    @pytest.mark.asyncio
    async def test_build_inspect_delegates(self, client, monkeypatch):
        monkeypatch.setattr(
            volume_build, "inspect_volume_build",
            lambda source, kind, server_uri: {"slices_to_read": 10},
        )
        response = await client.get("/api/volume/build/inspect", params={"source": "browse/s"})
        assert response.json() == {"slices_to_read": 10}

    @pytest.mark.asyncio
    async def test_build_start_runs_job_and_reports_done(self, client, monkeypatch, sync_threads):
        monkeypatch.setattr(
            volume_build, "inspect_volume_build",
            lambda source, kind, server_uri: {"slices_to_read": 4},
        )
        monkeypatch.setattr(
            volume_build, "build_volume",
            lambda source, kind, server_uri, container_path, progress: {"key": "vol1"},
        )
        response = await client.post("/api/volume/build", json={"source": "browse/s"})
        assert response.status_code == 200
        body = response.json()
        assert body["slices_to_read"] == 4
        job = export_jobs.get_job(body["job_id"])
        assert job["state"] == "done"
        assert job["result"] == {"key": "vol1"}

    @pytest.mark.asyncio
    async def test_build_start_job_reports_error_on_exception(self, client, monkeypatch, sync_threads):
        monkeypatch.setattr(
            volume_build, "inspect_volume_build",
            lambda source, kind, server_uri: {"slices_to_read": 4},
        )

        def boom(*a, **k):
            raise RuntimeError("build blew up")

        monkeypatch.setattr(volume_build, "build_volume", boom)
        response = await client.post("/api/volume/build", json={"source": "browse/s"})
        job = export_jobs.get_job(response.json()["job_id"])
        assert job["state"] == "error"
        assert "build blew up" in job["error"]
