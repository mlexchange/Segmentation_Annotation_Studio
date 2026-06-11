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
