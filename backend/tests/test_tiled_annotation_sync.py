"""Tests for source key parsing and annotation metadata helpers."""

from __future__ import annotations

import pytest

import arrays as arrays_mod
from source_keys import parse_source_key
from tiled_annotation_sync import STUDIO_ANNOTATED, annotation_metadata, sync_annotation_metadata


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


def test_annotation_metadata_counts_classes_and_multiple_slices() -> None:
    meta = annotation_metadata({
        "classes": [{"classId": 1}, {"classId": 2}],
        "slices": {"0": [{"id": "s1"}, {"id": "s2"}], "1": [{"id": "s3"}]},
    })
    assert meta["studio_shape_count"] == 3
    assert meta["studio_class_count"] == 2


def test_annotation_metadata_non_list_classes_counts_as_zero() -> None:
    meta = annotation_metadata({"classes": "not-a-list", "slices": {}})
    assert meta["studio_class_count"] == 0


def test_annotation_metadata_includes_iso_timestamp() -> None:
    meta = annotation_metadata({"classes": [], "slices": {}})
    assert "T" in meta["studio_updated_at"]


class FakeTiledNode:
    def __init__(self):
        self.updates: list[dict] = []

    def update_metadata(self, metadata):
        self.updates.append(metadata)


class TestSyncAnnotationMetadata:
    def test_local_source_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch):
        called = []
        monkeypatch.setattr(arrays_mod, "resolve_array", lambda *a, **k: called.append(1))
        sync_annotation_metadata("local:foo.tif", {"classes": [], "slices": {}})
        assert called == []

    def test_tiled_source_updates_node_metadata(self, monkeypatch: pytest.MonkeyPatch):
        node = FakeTiledNode()
        monkeypatch.setattr(arrays_mod, "resolve_array", lambda path, kind, server_uri: node)
        sync_annotation_metadata(
            "tiled::browse/sample", {"classes": [{"classId": 1}], "slices": {"0": [{"id": "s1"}]}},
        )
        assert len(node.updates) == 1
        assert node.updates[0][STUDIO_ANNOTATED] == "yes"
        assert node.updates[0]["studio_shape_count"] == 1

    def test_node_without_update_metadata_is_skipped_not_raised(self, monkeypatch: pytest.MonkeyPatch):
        class NoUpdateNode:
            pass

        monkeypatch.setattr(arrays_mod, "resolve_array", lambda path, kind, server_uri: NoUpdateNode())
        # Should not raise even though the resolved node can't be updated.
        sync_annotation_metadata("tiled::browse/sample", {"classes": [], "slices": {}})

    def test_empty_path_after_tiled_prefix_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch):
        called = []
        monkeypatch.setattr(arrays_mod, "resolve_array", lambda *a, **k: called.append(1))
        sync_annotation_metadata("tiled:", {"classes": [], "slices": {}})
        assert called == []

    def test_passes_parsed_server_uri_through(self, monkeypatch: pytest.MonkeyPatch):
        import source_keys
        import tiled_config

        monkeypatch.setattr(
            tiled_config, "get_tiled_servers", lambda: {"local": {"uri": "http://x:1"}},
        )
        monkeypatch.setattr(source_keys, "get_tiled_servers", tiled_config.get_tiled_servers)

        seen = []
        node = FakeTiledNode()

        def fake_resolve(path, kind, server_uri):
            seen.append((path, kind, server_uri))
            return node

        monkeypatch.setattr(arrays_mod, "resolve_array", fake_resolve)
        sync_annotation_metadata(
            "tiled:http://x:1:browse/sample", {"classes": [], "slices": {}},
        )
        assert seen[0] == ("browse/sample", "tiled", "http://x:1")
