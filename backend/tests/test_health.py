"""Tests for the health endpoint."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

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
