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

    def _preprocess(*, session_id, feature_setup_id, composition_id, slice_index, array_ref=None, client=None):
        if slice_index == 1:
            raise RuntimeError("boom")
        return {"feature_id": f"feat-{slice_index}"}

    def _infer(*, session_id, model_id, feature_id, alpha, store_probabilities=True, client=None):
        return {"run_id": f"run-{feature_id}"}

    monkeypatch.setattr(ipred_client_mod, "preprocess", _preprocess)
    monkeypatch.setattr(ipred_client_mod, "infer", _infer)
    monkeypatch.setattr(ipred_client_mod, "delete_feature_bank", lambda feature_id, *, client=None: None)

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


@pytest.mark.asyncio
async def test_batch_train_rejects_empty_slices() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ipred/batch/train", json={"session_id": "s1", "slices": {}},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_apply_rejects_empty_slice_indices() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ipred/batch/apply",
            json={"session_id": "s1", "model_id": "m1", "slice_indices": []},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_generic_exception_reports_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception that is neither an HTTPStatusError nor a ConnectError
    (e.g. ipred is unreachable via a different transport failure) reports 500
    with the exception's own message rather than crashing the request."""
    def _raise() -> dict:
        raise ValueError("something unexpected")

    monkeypatch.setattr(ipred_client_mod, "list_trainers", _raise)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/ipred/trainers")
    assert response.status_code == 500
    assert "something unexpected" in response.json()["detail"]


@pytest.mark.asyncio
async def test_http_status_error_falls_back_to_text_when_not_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> dict:
        req = httpx.Request("GET", "http://127.0.0.1:8003/setups")
        resp = httpx.Response(500, content=b"plain text error", request=req)
        raise httpx.HTTPStatusError("bad", request=req, response=resp)

    monkeypatch.setattr(ipred_client_mod, "list_setups", _raise)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/ipred/setups")
    assert response.status_code == 500
    assert response.json()["detail"] == "plain text error"


class TestSimpleProxyRoutes:
    """Each of these is a thin call-through + exception translation; one happy
    path per route is enough since _ipred_http_error's branches are already
    covered above and shared by all of them."""

    @pytest.mark.asyncio
    async def test_get_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "get_setup", lambda setup_id: {"id": setup_id})
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/ipred/setups/s1")
        assert response.json() == {"id": "s1"}

    @pytest.mark.asyncio
    async def test_upsert_setup_excludes_none_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}
        monkeypatch.setattr(ipred_client_mod, "upsert_setup", lambda payload: captured.update(payload) or {"id": "s1"})
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/setups", json={"name": "s", "kind": "onnx"},
            )
        assert response.status_code == 200
        assert "procedure_id" not in captured
        assert captured["name"] == "s"

    @pytest.mark.asyncio
    async def test_list_trainers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "list_trainers", lambda: ["catboost"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/ipred/trainers")
        assert response.json() == {"trainers": ["catboost"]}

    @pytest.mark.asyncio
    async def test_list_compositions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "list_compositions", lambda: [{"id": "c1"}])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/ipred/compositions")
        assert response.json() == {"compositions": [{"id": "c1"}]}

    @pytest.mark.asyncio
    async def test_get_composition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "get_composition", lambda cid: {"id": cid})
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/ipred/compositions/c1")
        assert response.json() == {"id": "c1"}

    @pytest.mark.asyncio
    async def test_upsert_composition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "upsert_composition", lambda payload: {"id": "c2"})
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/compositions",
                json={"name": "comp", "nodes": [], "outputs": []},
            )
        assert response.json() == {"id": "c2"}

    @pytest.mark.asyncio
    async def test_preview_composition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "preview_composition", lambda payload: {"preview": True})
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/compositions/preview",
                json={"name": "comp", "nodes": [], "outputs": []},
            )
        assert response.json() == {"preview": True}

    @pytest.mark.asyncio
    async def test_upload_array_injects_session_id_into_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        def _upload(session_id, payload):
            captured.update(payload)
            return {"ok": True}

        monkeypatch.setattr(ipred_client_mod, "upload_session_array", _upload)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/sessions/s1/arrays",
                json={"session_id": "ignored", "shape": [2, 2], "data_b64": "AAA="},
            )
        assert response.status_code == 200
        assert captured["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_manifold_sample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "manifold_sample", lambda payload: {"sample_id": "sm1"})
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/manifold/sample", json={"feature_id": "f1"},
            )
        assert response.json() == {"sample_id": "sm1"}

    @pytest.mark.asyncio
    async def test_manifold_heatmap_png(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "manifold_heatmap_png", lambda sample_id: b"heatbytes")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/ipred/manifold/sm1/heatmap.png")
        assert response.content == b"heatbytes"
        assert response.headers["content-type"] == "image/png"

    @pytest.mark.asyncio
    async def test_preprocess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        def _preprocess(**kwargs):
            captured.update(kwargs)
            return {"feature_id": "f1"}

        monkeypatch.setattr(ipred_client_mod, "preprocess", _preprocess)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/preprocess", json={"session_id": "s1", "slice_index": 3},
            )
        assert response.json() == {"feature_id": "f1"}
        assert captured["slice_index"] == 3

    @pytest.mark.asyncio
    async def test_feature_channel_png(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "feature_channel_bytes", lambda fid, idx: b"chanbytes")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/ipred/features/f1/channels/2")
        assert response.content == b"chanbytes"

    @pytest.mark.asyncio
    async def test_train(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        def _train(**kwargs):
            captured.update(kwargs)
            return {"model_id": "m1"}

        monkeypatch.setattr(ipred_client_mod, "train", _train)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/train", json={"session_id": "s1", "shapes": []},
            )
        assert response.json() == {"model_id": "m1"}
        assert captured["trainer_id"] == "catboost"

    @pytest.mark.asyncio
    async def test_infer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "infer", lambda **kwargs: {"run_id": "r1"})
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/infer", json={"session_id": "s1"},
            )
        assert response.json() == {"run_id": "r1"}

    @pytest.mark.asyncio
    async def test_rethreshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "rethreshold", lambda **kwargs: {"ok": True})
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/rethreshold", json={"session_id": "s1", "alpha": 0.1},
            )
        assert response.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_run_commit_png(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "run_commit_png", lambda run_id: b"commitbytes")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/ipred/runs/r1/commit.png")
        assert response.content == b"commitbytes"

    @pytest.mark.asyncio
    async def test_run_status_png(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ipred_client_mod, "run_status_png", lambda run_id: b"statusbytes")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/ipred/runs/r1/status.png")
        assert response.content == b"statusbytes"

    @pytest.mark.asyncio
    async def test_threshold_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        def _threshold(run_id, *, class_id, threshold):
            captured.update(run_id=run_id, class_id=class_id, threshold=threshold)
            return {"ok": True}

        monkeypatch.setattr(ipred_client_mod, "threshold_class_map", _threshold)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/ipred/runs/r1/threshold-class", json={"class_id": 2, "threshold": 0.7},
            )
        assert response.json() == {"ok": True}
        assert captured == {"run_id": "r1", "class_id": 2, "threshold": 0.7}
