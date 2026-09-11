"""Tests for drafts.py — real filesystem (tmp_path), no mocks needed. Sets
`drafts._DRAFT_DIR` directly rather than the LOCAL_DATA_ROOT env var, since
it's computed once at import time (see MEMORY.md's import-time-constant
gotcha) — the pattern already used in test_annotation_server_routes.py."""
from __future__ import annotations

import json

import pytest

import drafts


@pytest.fixture(autouse=True)
def draft_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    d = tmp_path / ".drafts"
    monkeypatch.setattr(drafts, "_DRAFT_DIR", d)
    return d


# ---------------------------------------------------------------------------
# save_draft / load_draft
# ---------------------------------------------------------------------------

class TestSaveAndLoadDraft:
    def test_round_trips_payload(self):
        result = drafts.save_draft("local:foo.tif", {"classes": [], "slices": {}})
        assert "saved_at" in result
        assert "path" in result

        loaded = drafts.load_draft("local:foo.tif")
        assert loaded["source_key"] == "local:foo.tif"
        assert loaded["payload"] == {"classes": [], "slices": {}}

    def test_missing_draft_returns_none(self):
        assert drafts.load_draft("nope") is None

    def test_different_source_keys_do_not_collide(self):
        drafts.save_draft("a", {"v": 1})
        drafts.save_draft("b", {"v": 2})
        assert drafts.load_draft("a")["payload"] == {"v": 1}
        assert drafts.load_draft("b")["payload"] == {"v": 2}

    def test_overwrites_previous_draft_for_same_key(self):
        drafts.save_draft("a", {"v": 1})
        drafts.save_draft("a", {"v": 2})
        assert drafts.load_draft("a")["payload"] == {"v": 2}

    def test_corrupt_draft_file_returns_none(self, draft_dir):
        draft_dir.mkdir(parents=True, exist_ok=True)
        path = drafts._draft_path("bad")
        path.write_text("{not valid json")
        assert drafts.load_draft("bad") is None

    def test_write_is_atomic_no_leftover_tmp_file(self, draft_dir):
        drafts.save_draft("a", {"v": 1})
        tmp_files = list(draft_dir.glob("*.tmp"))
        assert tmp_files == []


class TestDraftPath:
    def test_same_key_produces_same_path(self):
        assert drafts._draft_path("x") == drafts._draft_path("x")

    def test_different_keys_produce_different_paths(self):
        assert drafts._draft_path("x") != drafts._draft_path("y")


# ---------------------------------------------------------------------------
# list_drafts
# ---------------------------------------------------------------------------

class TestListDrafts:
    def test_empty_when_dir_does_not_exist(self):
        assert drafts.list_drafts() == []

    def test_lists_saved_drafts(self):
        drafts.save_draft("a", {"slices": {}})
        drafts.save_draft("b", {"slices": {}})
        result = drafts.list_drafts()
        assert len(result) == 2
        keys = {d["source_key"] for d in result}
        assert keys == {"a", "b"}

    def test_has_annotations_true_when_a_slice_has_shapes(self):
        drafts.save_draft("a", {"slices": {"0": [{"id": "s1"}]}})
        result = drafts.list_drafts()
        assert result[0]["has_annotations"] is True

    def test_has_annotations_false_when_all_slices_empty(self):
        drafts.save_draft("a", {"slices": {"0": []}})
        result = drafts.list_drafts()
        assert result[0]["has_annotations"] is False

    def test_corrupt_draft_file_is_skipped_not_raised(self, draft_dir):
        drafts.save_draft("a", {"slices": {}})
        draft_dir.mkdir(parents=True, exist_ok=True)
        (draft_dir / "corrupt.json").write_text("{not json")
        result = drafts.list_drafts()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# save_version / list_versions / get_version
# ---------------------------------------------------------------------------

class TestSaveVersion:
    def test_first_version_is_1(self):
        result = drafts.save_version("a", {"classes": [], "slices": {}})
        assert result["version"] == 1

    def test_versions_increment(self):
        drafts.save_version("a", {"classes": [], "slices": {}})
        result2 = drafts.save_version("a", {"classes": [], "slices": {}})
        assert result2["version"] == 2

    def test_shape_count_and_class_count_computed(self):
        payload = {
            "classes": [{"classId": 1}, {"classId": 2}],
            "slices": {"0": [{"id": "s1"}, {"id": "s2"}], "1": [{"id": "s3"}]},
        }
        result = drafts.save_version("a", payload)
        assert result["shape_count"] == 3

    def test_also_updates_the_crash_recovery_draft(self):
        drafts.save_version("a", {"classes": [], "slices": {"0": [{"id": "s1"}]}})
        loaded = drafts.load_draft("a")
        assert loaded["payload"]["slices"] == {"0": [{"id": "s1"}]}

    def test_annotated_by_and_notes_are_trimmed_and_none_when_blank(self):
        result_blank = drafts.save_version("a", {"classes": [], "slices": {}}, annotated_by="  ", notes="")
        vdoc = json.loads((drafts._versions_dir("a") / "v0001.json").read_text())
        assert vdoc["annotated_by"] is None
        assert vdoc["notes"] is None

        drafts.save_version("a", {"classes": [], "slices": {}}, annotated_by="  Alice  ", notes=" hi ")
        vdoc2 = json.loads((drafts._versions_dir("a") / "v0002.json").read_text())
        assert vdoc2["annotated_by"] == "Alice"
        assert vdoc2["notes"] == "hi"
        assert result_blank["version"] == 1


class TestListVersions:
    def test_empty_when_no_versions_dir(self):
        assert drafts.list_versions("nope") == []

    def test_sorted_oldest_first(self):
        drafts.save_version("a", {"classes": [], "slices": {}})
        drafts.save_version("a", {"classes": [], "slices": {}})
        result = drafts.list_versions("a")
        assert [v["version"] for v in result] == [1, 2]

    def test_has_thumbnail_reflects_saved_thumbnail(self):
        drafts.save_version("a", {"classes": [], "slices": {}})
        result = drafts.list_versions("a")
        assert result[0]["has_thumbnail"] is False
        drafts.save_version_thumbnail("a", 1, b"pngbytes")
        result2 = drafts.list_versions("a")
        assert result2[0]["has_thumbnail"] is True

    def test_corrupt_version_file_is_skipped(self):
        drafts.save_version("a", {"classes": [], "slices": {}})
        (drafts._versions_dir("a") / "vbad.json").write_text("not json at all {")
        result = drafts.list_versions("a")
        assert len(result) == 1


class TestGetVersion:
    def test_returns_full_document(self):
        drafts.save_version("a", {"classes": [{"classId": 1}], "slices": {}}, notes="test")
        doc = drafts.get_version("a", 1)
        assert doc["payload"]["classes"] == [{"classId": 1}]
        assert doc["notes"] == "test"

    def test_missing_version_returns_none(self):
        drafts.save_version("a", {"classes": [], "slices": {}})
        assert drafts.get_version("a", 99) is None

    def test_corrupt_version_returns_none(self):
        vdir = drafts._versions_dir("a")
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "v0001.json").write_text("{ broken")
        assert drafts.get_version("a", 1) is None


class TestVersionThumbnails:
    def test_missing_thumbnail_returns_none(self):
        assert drafts.get_version_thumbnail("a", 1) is None

    def test_round_trips_thumbnail_bytes(self):
        drafts.save_version_thumbnail("a", 3, b"\x89PNGdata")
        assert drafts.get_version_thumbnail("a", 3) == b"\x89PNGdata"

    def test_thumbnail_write_is_atomic_no_leftover_tmp(self):
        drafts.save_version_thumbnail("a", 1, b"data")
        vdir = drafts._versions_dir("a")
        assert list(vdir.glob("*.tmp")) == []
