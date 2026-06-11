"""Tests for session-draft persistence."""

from __future__ import annotations

import importlib
import os


def test_save_and_load_draft(tmp_path) -> None:
    """A saved draft should be loadable with the correct source_key."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import drafts

        importlib.reload(drafts)
        meta = drafts.save_draft("test-source", {"classes": [], "slices": {}})
        assert "saved_at" in meta
        loaded = drafts.load_draft("test-source")
        assert loaded is not None
        assert loaded["source_key"] == "test-source"


def test_load_missing_draft(tmp_path) -> None:
    """Loading a draft that was never saved should return None."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import drafts

        importlib.reload(drafts)
        result = drafts.load_draft("nonexistent-key")
        assert result is None


def test_list_drafts_returns_saved(tmp_path) -> None:
    """list_drafts should include a previously saved draft."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import drafts

        importlib.reload(drafts)
        drafts.save_draft("my-source", {"classes": [{"classId": 1, "label": "cell", "color": "#f00", "isVisible": True}]})
        all_drafts = drafts.list_drafts()
    assert any(d["source_key"] == "my-source" for d in all_drafts)


def test_list_drafts_empty_when_none(tmp_path) -> None:
    """list_drafts should return an empty list when no drafts exist."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import drafts

        importlib.reload(drafts)
        result = drafts.list_drafts()
    assert result == []
