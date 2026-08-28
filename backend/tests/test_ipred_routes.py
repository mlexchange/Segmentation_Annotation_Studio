"""Tests for the /api/ipred/* proxy — mocks ipred_client, never hits :8003."""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import export_jobs
import ipred_client as ipred_client_mod
from annotation_server import app


@pytest.mark.asyncio
async def test_health_proxies_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """A healthy ipred returns its body verbatim."""
    monkeypatch.setattr(ipred_client_mod, "health", lambda: {"status": "ok"})
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/ipred/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_reports_503_when_ipred_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A down/missing ipred surfaces as 503, not a 500 or crash."""

    def _raise() -> dict:
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(ipred_client_mod, "health", _raise)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/ipred/health")
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upstream_http_error_status_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4xx from ipred itself is forwarded with the same status + detail."""

    def _raise() -> dict:
        req = httpx.Request("GET", "http://127.0.0.1:8003/modules")
        resp = httpx.Response(422, json={"detail": "bad params"}, request=req)
        raise httpx.HTTPStatusError("bad", request=req, response=resp)

    monkeypatch.setattr(ipred_client_mod, "list_modules", _raise)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/ipred/modules")
    assert response.status_code == 422
    assert response.json()["detail"] == {"detail": "bad params"}


@pytest.mark.asyncio
async def test_open_session_forwards_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /sessions passes kind/source/server_uri/root through to the client."""
    captured: dict = {}

    def _open_session(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {"session_id": "s1", "project_id": "p1"}

    monkeypatch.setattr(ipred_client_mod, "open_session", _open_session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ipred/sessions",
            json={"kind": "local", "source": "foo.tiff"},
        )
    assert response.status_code == 200
    assert response.json() == {"session_id": "s1", "project_id": "p1"}
    assert captured == {
        "kind": "local",
        "source": "foo.tiff",
        "server_uri": None,
        "root": None,
    }


@pytest.mark.asyncio
async def test_list_modules_reports_ready_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Module readiness (e.g. tomojepa without weights) passes through unchanged."""
    modules = [
        {"id": "slimsam", "ready": True, "runtime": "onnx"},
        {"id": "tomojepa", "ready": False, "runtime": "torch"},
    ]
    monkeypatch.setattr(ipred_client_mod, "list_modules", lambda: modules)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/ipred/modules")
    assert response.status_code == 200
    assert response.json() == {"modules": modules}


@pytest.mark.asyncio
async def test_run_proba_png_proxies_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binary PNG proxies return the raw bytes with an image content-type."""
    monkeypatch.setattr(ipred_client_mod, "run_proba_png", lambda run_id, idx: b"\x89PNG\r\n")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/ipred/runs/run1/proba/0.png")
    assert response.status_code == 200
    assert response.content == b"\x89PNG\r\n"
    assert response.headers["content-type"] == "image/png"


async def _await_job(jid: str, timeout: float = 5.0) -> dict:
    """Poll export_jobs until the (thread-backed) job finishes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = export_jobs.get_job(jid)
        assert job is not None
        if job["state"] in ("done", "error"):
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {jid} did not finish within {timeout}s")


@pytest.mark.asyncio
async def test_batch_train_pools_slices_and_reports_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /batch/train preprocesses every slice, then trains once, pooled."""
    preprocessed: list[int] = []

    def _preprocess(*, session_id, feature_setup_id, composition_id, slice_index, array_ref=None):
        preprocessed.append(slice_index)
        return {"feature_id": f"feat-{slice_index}", "cache_hit": False}

    captured_train: dict = {}

    def _train_multi(*, session_id, slices, feature_ids, trainer_id, config):
        captured_train.update(
            session_id=session_id, slices=slices, feature_ids=feature_ids, trainer_id=trainer_id
        )
        return {"model_id": "m1", "class_ids": [1, 2], "n_samples": 4000}

    monkeypatch.setattr(ipred_client_mod, "preprocess", _preprocess)
    monkeypatch.setattr(ipred_client_mod, "train_multi", _train_multi)

    shapes = [{"kind": "rectangle", "classId": 1, "x": 0, "y": 0, "w": 5, "h": 5}]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ipred/batch/train",
            json={
                "session_id": "s1",
                "slices": {"0": shapes, "2": shapes},
                "composition_id": "comp-skimage",
                "trainer_id": "catboost",
            },
        )
    assert response.status_code == 200
    jid = response.json()["job_id"]
    job = await _await_job(jid)
    assert job["state"] == "done"
    assert job["result"] == {"model_id": "m1", "class_ids": [1, 2], "n_samples": 4000}
    assert sorted(preprocessed) == [0, 2]
    assert set(captured_train["feature_ids"]) == {"0", "2"}


@pytest.mark.asyncio
async def test_batch_apply_tolerates_one_bad_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /batch/apply keeps going after one slice fails and reports it in result.errors."""

    def _preprocess(*, session_id, feature_setup_id, composition_id, slice_index, array_ref=None):
        if slice_index == 1:
            raise RuntimeError("boom")
        return {"feature_id": f"feat-{slice_index}"}

    def _infer(*, session_id, model_id, feature_id, alpha):
        return {"run_id": f"run-{feature_id}"}

    monkeypatch.setattr(ipred_client_mod, "preprocess", _preprocess)
    monkeypatch.setattr(ipred_client_mod, "infer", _infer)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ipred/batch/apply",
            json={
                "session_id": "s1",
                "model_id": "m1",
                "slice_indices": [0, 1, 2],
            },
        )
    assert response.status_code == 200
    jid = response.json()["job_id"]
    job = await _await_job(jid)
    assert job["state"] == "done"
    assert job["result"]["runs"] == {"0": "run-feat-0", "2": "run-feat-2"}
    assert job["result"]["errors"] == [{"slice": 1, "error": "boom"}]
