"""Tests for source key parsing and annotation metadata helpers."""

from __future__ import annotations

from source_keys import parse_source_key
from tiled_annotation_sync import annotation_metadata, STUDIO_ANNOTATED


def test_parse_local_source_key() -> None:
    parsed = parse_source_key("local:imgs_00/foo.png")
    assert parsed["kind"] == "local"
    assert parsed["path"] == "imgs_00/foo.png"


def test_annotation_metadata_yes_when_shapes() -> None:
    meta = annotation_metadata({
        "classes": [{"classId": 1}],
        "slices": {"0": [{"id": "s1", "kind": "rectangle"}]},
    })
    assert meta[STUDIO_ANNOTATED] == "yes"
    assert meta["studio_shape_count"] == 1


def test_annotation_metadata_no_when_empty() -> None:
    meta = annotation_metadata({"classes": [], "slices": {}})
    assert meta[STUDIO_ANNOTATED] == "no"
    assert meta["studio_shape_count"] == 0
