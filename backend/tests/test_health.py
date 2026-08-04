"""Tests for the health endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server
from annotation_server import app


@pytest.mark.asyncio
async def test_health_returns_ok() -> None:
    """Health endpoint must return status ok."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_servers_endpoint_no_api_key() -> None:
    """Servers endpoint must not expose api_key in response."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/config/servers")
    assert response.status_code == 200
    for server in response.json():
        assert "api_key" not in server, "api_key must not be exposed to frontend"


def test_openapi_does_not_accept_client_supplied_tiled_keys() -> None:
    """Public API operations must never define a server_api_key parameter."""
    schema = app.openapi()
    for path, path_item in schema["paths"].items():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            names = {parameter["name"] for parameter in operation.get("parameters", [])}
            assert "server_api_key" not in names, path


@pytest.mark.asyncio
async def test_unknown_tiled_server_is_rejected_without_networking(monkeypatch) -> None:
    """Public URI parameters cannot turn the API into an SSRF proxy."""
    monkeypatch.setattr(
        annotation_server,
        "get_tiled_client",
        lambda uri: pytest.fail("unknown server reached the network client"),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/tiled/list",
            params={"server_uri": "https://attacker.example"},
        )

    assert response.status_code == 403
    assert response.json() == {"detail": "Tiled server is not configured"}


@pytest.mark.asyncio
async def test_image_errors_do_not_disclose_internal_paths(monkeypatch) -> None:
    """Unexpected backend failures return a stable, redacted error."""

    def _fail(*args, **kwargs):
        raise RuntimeError("sensitive path: /Users/example/private/data.npy")

    monkeypatch.setattr(annotation_server.arrays_mod, "resolve_array", _fail)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/image/meta",
            params={"kind": "local", "source": "sample.npy"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to read image metadata"}
    assert "/Users/example" not in response.text


@pytest.mark.asyncio
async def test_annotation_thumbnail_uses_private_cache(monkeypatch) -> None:
    """Annotation images must not be stored in shared HTTP caches."""
    monkeypatch.setattr(
        annotation_server.drafts_mod,
        "get_version_thumbnail",
        lambda source_key, version: b"png",
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/annotations/versions/1/thumbnail",
            params={"source_key": "local:sample.npy"},
        )

    assert response.status_code == 200
    assert response.headers["cache-control"].startswith("private")
