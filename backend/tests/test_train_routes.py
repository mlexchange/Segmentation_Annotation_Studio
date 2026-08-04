"""HTTP-boundary tests for the /api/train/* routes (Train tab).

Runs without torch installed (as CI does) — every route must degrade to a
clear error rather than a 500 when torch/dlsia/a checkpoint/a run is missing.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import export_jobs
import train_common
from annotation_server import app

# Minimal valid /api/train/start body, reused by the resume tests below.
_TRAIN_PAYLOAD = {
    "sources": [
        {
            "kind": "local",
            "source": "s.tif",
            "slices": {"0": [{"id": "s1", "classId": 1, "kind": "rectangle", "x": 0, "y": 0, "w": 5, "h": 5}]},
        }
    ],
    "classes": [{"classId": 1, "label": "pore", "color": "#ff0000"}],
    "model": {"model_family": "dinov3_lora", "arch": "vitb16", "checkpoint": "ckpt.pth"},
}


@pytest.mark.asyncio
async def test_capability_endpoint_never_errors_without_torch(monkeypatch) -> None:
    monkeypatch.setattr(train_common, "torch_available", lambda: False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/train/capability")
    assert response.status_code == 200
    body = response.json()
    assert body["torch_available"] is False
    assert body["dinov3"]["available"] is False


@pytest.mark.asyncio
async def test_runs_endpoint_returns_empty_list_when_none_saved(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/train/runs")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_delete_run_endpoint_removes_run(monkeypatch, tmp_path) -> None:
    import json

    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    d = tmp_path / "run_a"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"run_id": "run_a", "created_at": "2026-01-01T00:00:00+00:00"}))
    (d / "metrics.json").write_text("{}")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        list_response = await client.get("/api/train/runs")
        assert [r["run_id"] for r in list_response.json()] == ["run_a"]

        delete_response = await client.delete("/api/train/runs/run_a")
        assert delete_response.status_code == 200

        list_after = await client.get("/api/train/runs")
    assert list_after.json() == []


@pytest.mark.asyncio
async def test_delete_run_endpoint_404s_for_unknown_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/api/train/runs/does_not_exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_train_start_returns_503_when_torch_unavailable(monkeypatch) -> None:
    # The route does `import train_common` locally, but that's the same cached
    # module object as this top-level import (sys.modules is a singleton), so
    # patching it here affects the route's lookup too.
    monkeypatch.setattr(train_common, "torch_available", lambda: False)
    payload = {
        "sources": [
            {
                "kind": "local",
                "source": "s.tif",
                "slices": {
                    "0": [
                        {"id": "s1", "classId": 1, "kind": "rectangle", "x": 0, "y": 0, "w": 5, "h": 5},
                    ]
                },
            }
        ],
        "classes": [{"classId": 1, "label": "pore", "color": "#ff0000"}],
        "model": {"model_family": "dinov3_lora", "arch": "vitb16", "checkpoint": "ckpt.pth"},
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/train/start", json=payload)
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_train_cancel_returns_404_for_unknown_job() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/train/cancel/does-not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_train_cancel_sets_flag_on_known_job() -> None:
    jid = export_jobs.new_job("")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/api/train/cancel/{jid}")
    assert response.status_code == 200
    assert export_jobs.get_job(jid)["cancel_requested"] is True


@pytest.mark.asyncio
async def test_train_infer_returns_404_for_unknown_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(train_common, "torch_available", lambda: True)
    payload = {"run_id": "no_such_run", "kind": "local", "source": "s.tif", "slice_indices": [0]}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/train/infer", json=payload)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_infer_preview_returns_404_for_uncached_job() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/train/infer/preview/no-such-job/0")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_write_tiled_returns_404_for_unknown_job() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/train/infer/write-tiled/no-such-job")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_train_start_404s_for_an_unknown_resume_run(monkeypatch, tmp_path) -> None:
    """Resuming from a run that isn't there must be an immediate 404, not a job
    that starts and then fails in its progress log."""
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(train_common, "torch_available", lambda: True)
    payload = {**_TRAIN_PAYLOAD, "resume_from_run_id": "no_such_run"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/train/start", json=payload)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_train_start_400s_when_the_resume_classes_changed(monkeypatch, tmp_path) -> None:
    """The compatibility guard has to reject synchronously — the saved head's
    channels are positional per class, so a changed list makes the weights mean
    something else. The message must name what the run expects."""
    import json

    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(train_common, "torch_available", lambda: True)
    d = tmp_path / "parent"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "run_id": "parent",
        "model_family": "dinov3_lora",
        "classes": [{"classId": 1, "label": "grain", "color": "#00ff00"}],
        "model_config": {"arch": "vitb16", "checkpoint": "ckpt.pth"},
        "hyperparams": {},
        "created_at": "2026-01-01T00:00:00+00:00",
    }))
    (d / "metrics.json").write_text("{}")

    # _TRAIN_PAYLOAD's only class is "pore"; the parent was trained on "grain".
    payload = {**_TRAIN_PAYLOAD, "resume_from_run_id": "parent"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/train/start", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "grain" in detail and "pore" in detail


@pytest.mark.asyncio
async def test_train_start_rejects_an_unsafe_resume_run_id(monkeypatch) -> None:
    """resume_from_run_id becomes a path component — traversal must 422 at the
    schema, never reach the filesystem."""
    monkeypatch.setattr(train_common, "torch_available", lambda: True)
    payload = {**_TRAIN_PAYLOAD, "resume_from_run_id": "../../etc/passwd"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/train/start", json=payload)
    assert response.status_code == 422


def test_openapi_train_routes_are_registered() -> None:
    schema = app.openapi()
    for path in [
        "/api/train/capability",
        "/api/train/runs",
        "/api/train/start",
        "/api/train/cancel/{job_id}",
        "/api/train/infer",
        "/api/train/infer/preview/{job_id}/{slice_index}",
        "/api/train/infer/write-tiled/{job_id}",
    ]:
        assert path in schema["paths"], path
