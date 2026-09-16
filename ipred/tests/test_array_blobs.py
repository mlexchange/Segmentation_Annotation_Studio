"""Array blob data plane for remote preprocess."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ipred import api, array_blobs, compositions


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    api._catalog = None
    with TestClient(api.app) as c:
        yield c


def test_save_load_array_blob(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    rec = array_blobs.save_array_blob("proj1", arr)
    out = array_blobs.load_array_blob("proj1", rec["array_ref"])
    np.testing.assert_array_equal(out, arr)


def test_modules_and_compositions_api(client: TestClient) -> None:
    r = client.get("/modules")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["modules"]}
    assert "tomojepa" in ids

    r2 = client.get("/compositions")
    assert r2.status_code == 200
    cids = {c["id"] for c in r2.json()["compositions"]}
    assert "comp-skimage" in cids

    r3 = client.get("/compositions/comp-skimage")
    assert r3.status_code == 200
    assert "preview_labels" in r3.json()


def test_upload_array_and_preprocess(client: TestClient) -> None:
    sess = client.post(
        "/sessions",
        json={"kind": "local", "source": "x.png", "root": "/tmp"},
    ).json()
    arr = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    raw = arr.astype(np.float32).tobytes()
    up = client.post(
        f"/sessions/{sess['session_id']}/arrays",
        json={
            "session_id": sess["session_id"],
            "shape": [16, 16],
            "dtype": "float32",
            "data_b64": base64.b64encode(raw).decode("ascii"),
        },
    )
    assert up.status_code == 200
    ref = up.json()["array_ref"]
    compositions.ensure_default_compositions(api.get_catalog())
    prep = client.post(
        "/preprocess",
        json={
            "session_id": sess["session_id"],
            "composition_id": "comp-skimage",
            "array_ref": ref,
            "slice_index": 0,
        },
    )
    assert prep.status_code == 200, prep.text
    body = prep.json()
    assert body["n_channels"] > 0
    assert body["composition_id"] == "comp-skimage"
