"""Tests for session-draft persistence."""

from __future__ import annotations

import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor


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


def test_list_drafts_excludes_guide_documents(tmp_path) -> None:
    """Guide JSON stored beside drafts must never appear in the draft listing."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import drafts
        import guides

        importlib.reload(drafts)
        importlib.reload(guides)
        drafts.save_draft("draft-source", {"classes": [], "slices": {}})
        guides.save_guide("guide-source", {"classes": [], "notes": "guide"})
        listed = drafts.list_drafts()

    assert [item["source_key"] for item in listed] == ["draft-source"]


def test_persisted_documents_include_schema_version_and_type(tmp_path) -> None:
    """Persisted formats need an explicit migration discriminator."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import drafts

        importlib.reload(drafts)
        meta = drafts.save_draft("typed-source", {"classes": [], "slices": {}})
        document = json.loads(__import__("pathlib").Path(meta["path"]).read_text())

    assert document["schema_version"] == 1
    assert document["document_type"] == "draft"


def test_version_number_uses_max_and_never_overwrites_after_gap(tmp_path) -> None:
    """Deleting an old version must not make the next save reuse an existing number."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import drafts

        importlib.reload(drafts)
        first = drafts.save_version("versioned", {"classes": [], "slices": {}})
        second = drafts.save_version("versioned", {"classes": [], "slices": {}})
        __import__("pathlib").Path(first["path"]).unlink()
        third = drafts.save_version("versioned", {"classes": [], "slices": {}})

        assert second["version"] == 2
        assert third["version"] == 3
        assert json.loads(__import__("pathlib").Path(second["path"]).read_text())["version"] == 2


def test_concurrent_version_saves_get_unique_immutable_numbers(tmp_path) -> None:
    """Parallel save requests must create distinct version files without overwrites."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import drafts

        importlib.reload(drafts)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda i: drafts.save_version(
                        "shared-source",
                        {"classes": [], "slices": {}, "marker": i},
                    ),
                    range(20),
                )
            )

        versions = sorted(result["version"] for result in results)
        stored = drafts.list_versions("shared-source")

    assert versions == list(range(1, 21))
    assert [item["version"] for item in stored] == list(range(1, 21))


def test_concurrent_draft_writes_remain_valid_and_leave_no_shared_tmp(tmp_path) -> None:
    """Overlapping autosaves must not race on one fixed temporary filename."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import drafts

        importlib.reload(drafts)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda i: drafts.save_draft(
                        "shared-draft",
                        {"classes": [], "slices": {}, "marker": i},
                    ),
                    range(20),
                )
            )
        loaded = drafts.load_draft("shared-draft")
        draft_dir = __import__("pathlib").Path(results[0]["path"]).parent

    assert loaded is not None
    assert loaded["payload"]["marker"] in range(20)
    assert list(draft_dir.glob("*.tmp")) == []
