"""Tests for local filesystem sandboxing."""

from __future__ import annotations

import importlib
import os

import pytest
from fastapi import HTTPException


def test_path_traversal_rejected(tmp_path) -> None:
    """Paths escaping LOCAL_DATA_ROOT must raise HTTP 403."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import local_fs

        importlib.reload(local_fs)
        with pytest.raises(HTTPException) as exc_info:
            local_fs._safe("../../etc/passwd")
        assert exc_info.value.status_code == 403


def test_list_dir_returns_entries(tmp_path) -> None:
    """list_dir should return file entries for an existing directory."""
    (tmp_path / "sample.npy").write_bytes(b"\x93NUMPY")
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import local_fs

        importlib.reload(local_fs)
        entries = local_fs.list_dir("")
    names = [e["name"] for e in entries]
    assert "sample.npy" in names


def test_list_dir_missing_raises_404(tmp_path) -> None:
    """list_dir on a non-existent path should raise HTTP 404."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import local_fs

        importlib.reload(local_fs)
        with pytest.raises(HTTPException) as exc_info:
            local_fs.list_dir("does_not_exist")
        assert exc_info.value.status_code == 404


def test_open_array_unsupported_type_raises_422(tmp_path) -> None:
    """open_array on an unsupported extension should raise HTTP 422."""
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b\n1,2\n")
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}
    ):
        import local_fs

        importlib.reload(local_fs)
        with pytest.raises(HTTPException) as exc_info:
            local_fs.open_array("data.csv")
        assert exc_info.value.status_code == 422


def test_caller_cannot_select_an_unconfigured_root(tmp_path) -> None:
    """A request root outside LOCAL_DATA_ROOT must be rejected."""
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(allowed)}, clear=True
    ):
        import local_fs

        importlib.reload(local_fs)
        with pytest.raises(HTTPException) as exc_info:
            local_fs.list_dir("", str(outside))

    assert exc_info.value.status_code == 403


def test_absolute_source_must_stay_under_a_configured_root(tmp_path) -> None:
    """Re-opening an absolute source must not bypass the configured sandbox."""
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    inside_file = allowed / "inside.npy"
    outside_file = outside / "outside.npy"
    inside_file.write_bytes(b"placeholder")
    outside_file.write_bytes(b"placeholder")
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(allowed)}, clear=True
    ):
        import local_fs

        importlib.reload(local_fs)
        assert local_fs._safe(str(inside_file)) == inside_file.resolve()
        with pytest.raises(HTTPException) as exc_info:
            local_fs._safe(str(outside_file))

    assert exc_info.value.status_code == 403


def test_symlink_cannot_escape_a_configured_root(tmp_path) -> None:
    """Resolved symlink targets outside the allowed root must be rejected."""
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "escape").symlink_to(outside, target_is_directory=True)
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, {"LOCAL_DATA_ROOT": str(allowed)}, clear=True
    ):
        import local_fs

        importlib.reload(local_fs)
        with pytest.raises(HTTPException) as exc_info:
            local_fs.list_dir("escape")

    assert exc_info.value.status_code == 403


def test_additional_configured_root_is_allowed(tmp_path) -> None:
    """Operators can opt in to multiple roots without granting arbitrary paths."""
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir()
    secondary.mkdir()
    (secondary / "sample.npy").write_bytes(b"placeholder")
    env = {
        "LOCAL_DATA_ROOT": str(primary),
        "LOCAL_DATA_ROOTS": str(secondary),
    }
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ, env, clear=True
    ):
        import local_fs

        importlib.reload(local_fs)
        entries = local_fs.list_dir("", str(secondary))

    assert [entry["name"] for entry in entries] == ["sample.npy"]
