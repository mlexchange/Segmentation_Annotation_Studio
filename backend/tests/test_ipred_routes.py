"""Tests for the /api/ipred/* proxy — mocks ipred_client, never hits :8003."""
from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

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
