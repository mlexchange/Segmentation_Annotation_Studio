"""POST /train/multi over HTTP."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from ipred import api


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    api._catalog = None
    with TestClient(api.app) as c:
        yield c


def test_train_multi_over_http(client: TestClient, tmp_path) -> None:
    stack = np.zeros((2, 48, 48), dtype=np.uint8)
    stack[:, :, 24:] = 200
    tifffile.imwrite(tmp_path / "stack.tif", stack, photometric="minisblack")

    r = client.post(
        "/sessions", json={"kind": "local", "source": "stack.tif", "root": str(tmp_path)}
    )
    assert r.status_code == 200
    session_id = r.json()["session_id"]

    feature_ids: dict[str, str] = {}
    for slice_index in (0, 1):
        r = client.post(
            "/preprocess",
            json={
                "session_id": session_id,
                "feature_setup_id": "default-skimage",
                "slice_index": slice_index,
            },
        )
        assert r.status_code == 200, r.text
        feature_ids[str(slice_index)] = r.json()["feature_id"]

    shapes = [
        {"kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 18, "h": 40},
        {"kind": "rectangle", "classId": 2, "x": 28, "y": 2, "w": 18, "h": 40},
    ]
    r = client.post(
        "/train/multi",
        json={
            "session_id": session_id,
            "slices": {"0": shapes, "1": shapes},
            "feature_ids": feature_ids,
            "trainer_id": "catboost",
            "config": {"iterations": 20, "depth": 4, "random_seed": 0},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["class_ids"] == [1, 2]
    assert body["trained_slice_indices"] == [0, 1]
