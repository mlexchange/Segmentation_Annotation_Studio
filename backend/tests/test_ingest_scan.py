"""Tests for scan_and_register_image_stacks — bulk auto-discovery of image
folders under a mounted directory, mirroring zarr_source's
scan_and_register_zarrs for the non-Zarr case. register_zarr/run_ingest_job
themselves are already covered elsewhere (test_zarr_source.py,
test_ingest_conflict.py); these tests are about candidate discovery,
temp-file safety, and result aggregation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

import ingest


class FakeNode:
    """Minimal stand-in for a Tiled container client (see test_ingest_conflict.py)."""

    def __init__(self, children: dict | None = None, metadata: dict | None = None) -> None:
        self._children: dict = dict(children or {})
        self.metadata: dict = dict(metadata or {})
        self.written: dict = {}

    def __getitem__(self, key: str):
        if key not in self._children:
            raise KeyError(key)
        return self._children[key]

    def keys(self) -> list[str]:
        return list(self._children)

    def create_container(self, key: str, metadata: dict | None = None) -> "FakeNode":
        node = FakeNode(metadata=metadata)
        self._children[key] = node
        return node

    def update_metadata(self, metadata: dict | None = None) -> None:
        self.metadata = dict(metadata or {})

    def write_array(self, arr, key=None, metadata=None, dims=None) -> None:
        self._children[key] = "array"
        self.written[key] = (arr, metadata)

    def delete_contents(self, keys=None, recursive: bool = False, external_only: bool = True) -> None:
        for k in [keys] if isinstance(keys, str) else keys:
            self._children.pop(k, None)


@pytest.fixture
def fake_client(monkeypatch):
    root = FakeNode({"browse": FakeNode()})
    monkeypatch.setattr(ingest, "api_key_for_uri", lambda uri: None)
    monkeypatch.setattr(ingest, "get_tiled_client", lambda uri, key=None: root)
    return root


def _make_stack(root: Path, name: str, n: int = 3, ext: str = ".tif") -> Path:
    """A folder of *n* tiny real image files (real bytes, no mocks)."""
    stack = root / name
    stack.mkdir()
    for i in range(n):
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(stack / f"{i:03d}{ext}")
    return stack


class TestIsImageStackDir:
    def test_two_or_more_images_qualifies(self, tmp_path: Path) -> None:
        stack = _make_stack(tmp_path, "s1", n=2)
        assert ingest._is_image_stack_dir(stack) is True

    def test_a_single_image_does_not_qualify(self, tmp_path: Path) -> None:
        stack = _make_stack(tmp_path, "s1", n=1)
        assert ingest._is_image_stack_dir(stack) is False

    def test_a_plain_file_does_not_qualify(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.txt"
        f.write_text("hi")
        assert ingest._is_image_stack_dir(f) is False

    def test_unsupported_extensions_do_not_count(self, tmp_path: Path) -> None:
        d = tmp_path / "junk"
        d.mkdir()
        (d / "a.txt").write_text("x")
        (d / "b.txt").write_text("y")
        assert ingest._is_image_stack_dir(d) is False


class TestScanAndRegisterImageStacks:
    def test_finds_and_ingests_top_level_stacks_only(self, tmp_path: Path, fake_client) -> None:
        _make_stack(tmp_path, "stack_a", n=3)
        _make_stack(tmp_path, "stack_b", n=2)
        (tmp_path / "not_a_stack").mkdir()
        # A stack's OWN internals must never be treated as a second candidate.
        (tmp_path / "stack_a" / "nested_stack").mkdir()

        result = ingest.scan_and_register_image_stacks(None, str(tmp_path), "browse")
        assert result["scanned"] == 2
        assert sorted(r["name"] for r in result["registered"]) == ["stack_a", "stack_b"]
        assert result["skipped"] == []
        assert result["errors"] == []

        # Real per-slice ingest actually happened (not a no-op / dry run).
        browse = fake_client["browse"]
        assert "stack_a" in browse.keys()
        assert len(browse["stack_a"].written) == 3

    def test_a_zarr_directory_is_never_treated_as_an_image_stack(self, tmp_path: Path, fake_client) -> None:
        zarr_dir = tmp_path / "vol.zarr"
        zarr_dir.mkdir()
        (zarr_dir / ".zarray").write_text("{}")
        # Even if it happens to also hold loose image files at its root.
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(zarr_dir / "a.tif")
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(zarr_dir / "b.tif")

        result = ingest.scan_and_register_image_stacks(None, str(tmp_path), "browse")
        assert result["scanned"] == 0
        assert result["registered"] == []

    def test_does_not_delete_the_original_source_files(self, tmp_path: Path, fake_client) -> None:
        stack = _make_stack(tmp_path, "stack_a", n=3)
        originals = sorted(stack.iterdir())
        assert len(originals) == 3

        ingest.scan_and_register_image_stacks(None, str(tmp_path), "browse")

        # The whole point of copying to a real temp dir first: run_ingest_job
        # unlinks its inputs when done, and must never touch the originals.
        assert sorted(stack.iterdir()) == originals
        for f in originals:
            assert f.exists()

    def test_skips_an_already_registered_stack_by_default(self, tmp_path: Path, fake_client) -> None:
        _make_stack(tmp_path, "stack_a", n=2)
        fake_client["browse"].create_container("stack_a", metadata={"source_format": "image-stack"})

        result = ingest.scan_and_register_image_stacks(None, str(tmp_path), "browse")
        assert result["registered"] == []
        assert result["skipped"] == ["stack_a"]

    def test_replace_on_conflict_re_ingests(self, tmp_path: Path, fake_client) -> None:
        _make_stack(tmp_path, "stack_a", n=2)
        fake_client["browse"].create_container("stack_a", metadata={"source_format": "image-stack"})

        result = ingest.scan_and_register_image_stacks(None, str(tmp_path), "browse", on_conflict="replace")
        assert [r["name"] for r in result["registered"]] == ["stack_a"]
        assert result["skipped"] == []

    def test_a_different_kind_collision_is_shadowed_not_skipped(self, tmp_path: Path, fake_client) -> None:
        _make_stack(tmp_path, "stack_a", n=2)
        # Same stem, but an unrelated Zarr registration got there first.
        fake_client["browse"].create_container("stack_a", metadata={"source_format": "zarr"})

        result = ingest.scan_and_register_image_stacks(None, str(tmp_path), "browse")
        assert result["registered"] == []
        assert result["skipped"] == []
        assert result["shadowed"] == [
            {"name": "stack_a", "key": "stack_a", "existing_kind": "zarr", "suggested_key": "stack_a_images"}
        ]

    def test_renames_lets_a_shadowed_candidate_ingest_under_an_alternate_key(
        self, tmp_path: Path, fake_client
    ) -> None:
        _make_stack(tmp_path, "stack_a", n=2)
        fake_client["browse"].create_container("stack_a", metadata={"source_format": "zarr"})

        result = ingest.scan_and_register_image_stacks(
            None, str(tmp_path), "browse", renames={"stack_a": "stack_a_images"}
        )
        assert result["shadowed"] == []
        assert [r["key"] for r in result["registered"]] == ["stack_a_images"]
        assert fake_client["browse"]["stack_a_images"].written

    def test_rejects_relative_scan_root(self) -> None:
        with pytest.raises(HTTPException) as exc:
            ingest.scan_and_register_image_stacks(None, "relative/dir", "browse")
        assert exc.value.status_code == 400

    def test_rejects_missing_scan_root(self, tmp_path: Path) -> None:
        with pytest.raises(HTTPException) as exc:
            ingest.scan_and_register_image_stacks(None, str(tmp_path / "nope"), "browse")
        assert exc.value.status_code == 404

    def test_empty_directory_scans_cleanly_with_nothing_found(self, tmp_path: Path, fake_client) -> None:
        result = ingest.scan_and_register_image_stacks(None, str(tmp_path), "browse")
        assert result == {"scanned": 0, "registered": [], "skipped": [], "shadowed": [], "errors": []}

    def test_invalid_on_conflict_falls_back_to_skip(self, tmp_path: Path, fake_client) -> None:
        _make_stack(tmp_path, "stack_a", n=2)
        fake_client["browse"].create_container("stack_a", metadata={"source_format": "image-stack"})

        result = ingest.scan_and_register_image_stacks(None, str(tmp_path), "browse", on_conflict="bogus")
        assert result["skipped"] == ["stack_a"]
