"""API health + session open."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ipred import api


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    api._catalog = None
    with TestClient(api.app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_open_session_and_list_setups(client: TestClient) -> None:
    r = client.post(
        "/sessions",
        json={"kind": "local", "source": "x.png", "root": "/tmp"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert "project_id" in body

    r2 = client.get("/setups")
    assert r2.status_code == 200
    ids = {s["id"] for s in r2.json()["setups"]}
    assert "default-skimage" in ids
    assert "default-mark25" in ids
    assert "default-skimage-mark25" in ids
    assert "default-mark25-clahe" in ids
    assert "default-mark11" in ids
    assert "default-skimage-mark11" in ids
    assert "default-mark11-clahe" in ids
    assert "default-slimsam-clahe" in ids

    r3 = client.get("/compositions")
    assert r3.status_code == 200
    cids = {c["id"] for c in r3.json()["compositions"]}
    assert "comp-skimage" in cids
    assert "comp-skimage-mark11" in cids
