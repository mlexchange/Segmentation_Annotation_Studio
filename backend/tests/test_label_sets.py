"""Tests for reusable annotation label sets."""

from __future__ import annotations

import os
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from annotation_server import app
import label_sets as label_sets_mod


@pytest.fixture()
def label_root(tmp_path):
    with mock.patch.dict(os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}):
        yield tmp_path


def test_save_list_get_delete_label_set(label_root) -> None:
    classes = [
        {"classId": 1, "label": "sample", "color": "#1f77b4", "isVisible": True},
        {"classId": 2, "label": "void", "color": "#ff7f0e", "isVisible": True},
    ]
    saved = label_sets_mod.save_label_set(name="Petiole", classes=classes)
    assert saved["id"]
    assert saved["name"] == "Petiole"
    assert len(saved["classes"]) == 2

    listed = label_sets_mod.list_label_sets()
    assert len(listed) == 1
    assert listed[0]["n_classes"] == 2

    got = label_sets_mod.get_label_set(saved["id"])
    assert got is not None
    assert got["classes"][0]["label"] == "sample"

    assert label_sets_mod.delete_label_set(saved["id"]) is True
    assert label_sets_mod.get_label_set(saved["id"]) is None


def test_empty_classes_rejected(label_root) -> None:
    with pytest.raises(ValueError):
        label_sets_mod.save_label_set(name="x", classes=[])


def test_label_sets_api_roundtrip(label_root) -> None:
    client = TestClient(app)
    res = client.post(
        "/api/label-sets",
        json={
            "name": "Default",
            "classes": [
                {"classId": 1, "label": "air", "color": "#aaaaaa", "isVisible": True},
            ],
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    set_id = body["id"]

    listed = client.get("/api/label-sets")
    assert listed.status_code == 200
    assert any(x["id"] == set_id for x in listed.json()["sets"])

    got = client.get(f"/api/label-sets/{set_id}")
    assert got.status_code == 200
    assert got.json()["classes"][0]["label"] == "air"

    deleted = client.delete(f"/api/label-sets/{set_id}")
    assert deleted.status_code == 200
