"""Tests for local_fs.py — real filesystem (tmp_path), no mocks. Sets
`local_fs._DEFAULT_ROOT` directly rather than the LOCAL_DATA_ROOT env var,
since it's computed once at import time (see MEMORY.md's import-time-constant
gotcha)."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image as PILImage

import local_fs


@pytest.fixture(autouse=True)
def default_root(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(local_fs, "_DEFAULT_ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# _resolve_root / _within / _safe
# ---------------------------------------------------------------------------

class TestResolveRoot:
    def test_none_falls_back_to_default_root(self, default_root):
        assert local_fs._resolve_root(None) == default_root

    def test_explicit_root_is_expanded_and_resolved(self, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        assert local_fs._resolve_root(str(granted)) == granted.resolve()


class TestWithin:
    def test_base_itself_is_within(self, tmp_path):
        assert local_fs._within(tmp_path, tmp_path) is True

    def test_descendant_is_within(self, tmp_path):
        assert local_fs._within(tmp_path, tmp_path / "a" / "b") is True

    def test_sibling_with_shared_prefix_is_not_within(self, tmp_path):
        sibling = tmp_path.parent / (tmp_path.name + "2")
        assert local_fs._within(tmp_path, sibling) is False


class TestSafe:
    def test_traversal_with_explicit_root_is_403(self, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        with pytest.raises(HTTPException) as exc:
            local_fs._safe("../../etc/passwd", root=str(granted))
        assert exc.value.status_code == 403

    def test_traversal_without_root_under_default_is_403(self):
        with pytest.raises(HTTPException) as exc:
            local_fs._safe("../../etc/passwd")
        assert exc.value.status_code == 403

    def test_relative_path_resolved_under_granted_root(self, tmp_path):
        granted = tmp_path / "granted"
        granted.mkdir()
        result = local_fs._safe("sub/file.tif", root=str(granted))
        assert result == (granted / "sub" / "file.tif").resolve()

    def test_absolute_path_without_root_returned_as_is_no_sandbox(self, tmp_path):
        absolute = tmp_path / "elsewhere" / "file.tif"
        result = local_fs._safe(str(absolute))
        assert result == absolute.resolve()

    def test_relative_path_without_root_resolved_under_default(self, default_root):
        result = local_fs._safe("sub/file.tif")
        assert result == (default_root / "sub" / "file.tif").resolve()


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

class TestListDir:
    def test_lists_files_and_dirs_with_sizes(self, default_root):
        (default_root / "a.npy").write_bytes(b"12345")
        (default_root / "sub").mkdir()
        entries = local_fs.list_dir("")
        by_name = {e["name"]: e for e in entries}
        assert by_name["a.npy"]["is_dir"] is False
        assert by_name["a.npy"]["size"] == 5
        assert by_name["sub"]["is_dir"] is True
        assert by_name["sub"]["size"] is None

    def test_missing_root_itself_returns_empty_not_404(self, tmp_path, monkeypatch):
        monkeypatch.setattr(local_fs, "_DEFAULT_ROOT", tmp_path / "does_not_exist")
        assert local_fs.list_dir("") == []
        assert local_fs.list_dir(".") == []

    def test_missing_subpath_raises_404(self):
        with pytest.raises(HTTPException) as exc:
            local_fs.list_dir("no/such/dir")
        assert exc.value.status_code == 404

    def test_file_instead_of_dir_raises_400(self, default_root):
        (default_root / "afile.npy").write_bytes(b"x")
        with pytest.raises(HTTPException) as exc:
            local_fs.list_dir("afile.npy")
        assert exc.value.status_code == 400

    def test_paths_are_relative_to_root(self, default_root):
        (default_root / "sub").mkdir()
        (default_root / "sub" / "f.npy").write_bytes(b"x")
        entries = local_fs.list_dir("sub")
        assert entries[0]["path"] == "sub/f.npy"


# ---------------------------------------------------------------------------
# count_image_files / list_image_files
# ---------------------------------------------------------------------------

class TestCountImageFiles:
    def test_counts_only_supported_extensions_recursively(self, default_root):
        (default_root / "a.tif").write_bytes(b"x")
        (default_root / "b.npy").write_bytes(b"x")
        (default_root / "c.txt").write_bytes(b"x")
        (default_root / "sub").mkdir()
        (default_root / "sub" / "d.png").write_bytes(b"x")
        assert local_fs.count_image_files("") == 3

    def test_nonexistent_path_returns_zero(self):
        assert local_fs.count_image_files("nope") == 0

    def test_file_instead_of_dir_returns_zero(self, default_root):
        (default_root / "a.tif").write_bytes(b"x")
        assert local_fs.count_image_files("a.tif") == 0

    def test_extension_matching_is_case_insensitive(self, default_root):
        (default_root / "A.TIF").write_bytes(b"x")
        assert local_fs.count_image_files("") == 1


class TestListImageFiles:
    def test_returns_sorted_flat_list(self, default_root):
        (default_root / "sub").mkdir()
        (default_root / "sub" / "b.tif").write_bytes(b"x")
        (default_root / "a.tif").write_bytes(b"x")
        (default_root / "readme.md").write_bytes(b"x")
        entries = local_fs.list_image_files("")
        assert [e["path"] for e in entries] == ["a.tif", "sub/b.tif"]

    def test_missing_path_raises_404(self):
        with pytest.raises(HTTPException) as exc:
            local_fs.list_image_files("nope")
        assert exc.value.status_code == 404

    def test_file_instead_of_dir_raises_400(self, default_root):
        (default_root / "a.tif").write_bytes(b"x")
        with pytest.raises(HTTPException) as exc:
            local_fs.list_image_files("a.tif")
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# open_array
# ---------------------------------------------------------------------------

class TestOpenArray:
    def test_missing_file_raises_404(self):
        with pytest.raises(HTTPException) as exc:
            local_fs.open_array("nope.npy")
        assert exc.value.status_code == 404

    def test_unsupported_extension_raises_422(self, default_root):
        (default_root / "data.csv").write_text("a,b\n1,2\n")
        with pytest.raises(HTTPException) as exc:
            local_fs.open_array("data.csv")
        assert exc.value.status_code == 422

    def test_opens_npy_file(self, default_root):
        arr = np.arange(12).reshape(3, 4).astype(np.float32)
        np.save(default_root / "a.npy", arr)
        result = local_fs.open_array("a.npy")
        assert np.array_equal(np.asarray(result), arr)

    def test_opens_png_file(self, default_root):
        img = PILImage.new("L", (5, 5), color=128)
        img.save(default_root / "a.png")
        result = local_fs.open_array("a.png")
        assert result.shape == (5, 5)

    def test_opens_tif_file(self, default_root):
        tifffile = pytest.importorskip("tifffile")
        arr = np.arange(16).reshape(4, 4).astype(np.uint16)
        tifffile.imwrite(default_root / "a.tif", arr)
        result = local_fs.open_array("a.tif")
        assert np.array_equal(np.asarray(result), arr)

    def test_corrupt_npy_raises_500(self, default_root):
        (default_root / "bad.npy").write_bytes(b"not a real npy file")
        with pytest.raises(HTTPException) as exc:
            local_fs.open_array("bad.npy")
        assert exc.value.status_code == 500
