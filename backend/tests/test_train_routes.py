"""HTTP-level tests for the /api/train/* routes.

Route wiring, validation, and job-status polling for the parts that don't
need a real registered dataset (capability, runs list/delete, batch-size
probe). The core train/save/load contract itself is proven for real (no
mocks) in test_train_e2e_real_ml.py; a full train-via-HTTP round trip would
also need real Tiled/local array registration, which is exercised elsewhere
for export and is out of scope for this file.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

import export_jobs
import train_common
from annotation_server import app


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


async def _await_job(jid: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = export_jobs.get_job(jid)
        assert job is not None
        if job["state"] in ("done", "error"):
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {jid} did not finish within {timeout}s")


@pytest.mark.asyncio
async def test_capability_route(runs_dir) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/train/capability")
    assert response.status_code == 200
    body = response.json()
    assert "dinov3" not in body
    assert "torch_available" in body


@pytest.mark.asyncio
async def test_runs_list_and_delete_unknown(runs_dir) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/train/runs")
        assert response.status_code == 200
        assert response.json() == {"runs": []}

        response = await client.delete("/api/train/runs/nonexistent")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_train_start_rejects_segmentation_with_no_classes(runs_dir) -> None:
    """Pydantic validation (TrainRequest.validate_train_taxonomy) surfaces as 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/train/start",
            json={
                "task": "segmentation",
                "sources": [{"kind": "local", "source": "fake.tif", "slices": {"0": []}}],
                "classes": [],
                "model": {"model_family": "dlsia_tunet"},
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_train_infer_unknown_run_is_404(runs_dir) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/train/infer",
            json={
                "run_id": "nonexistent",
                "kind": "local",
                "source": "fake.tif",
                "slice_indices": [0],
            },
        )
    if not train_common.torch_available():
        assert response.status_code == 503
    else:
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_estimate_batch_probe_runs_for_real_when_torch_available(runs_dir) -> None:
    """A real (not mocked) tiny batch-size probe, when torch is installed —
    otherwise just confirms the clean 503 degradation."""
    payload = {
        "model": {
            "model_family": "dlsia_tunet",
            "hyperparams": {"depth": 2, "base_channels": 4, "image_size": 64, "batch_size": 2},
        },
        "n_classes": 2,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/train/estimate-batch", json=payload)
        if not train_common.torch_available():
            assert response.status_code == 503
            return
        assert response.status_code == 200
        jid = response.json()["job_id"]
        job = await _await_job(jid)
        assert job["state"] == "done"
        assert job["result"]["suggested_batch_size"] is not None
        assert job["result"]["largest_ok"] >= 1
