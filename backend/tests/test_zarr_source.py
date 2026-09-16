"""Tests for on-disk Zarr registration and multiscale slice resolution.

Fixtures build tiny multiscale stores on the fly, so these run anywhere without
the multi-GB reference data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import arrays as arrays_mod  # noqa: E402
import zarr_source  # noqa: E402


def _make_multiscale(root: Path, shapes: list[tuple[int, int, int]]) -> Path:
    """Write a v2 multiscale store shaped like the reference data.

    Layout mirrors the real volumes: ``scaleN/image`` arrays plus a
    ``multiscales`` attribute naming them, so path discovery is exercised rather
    than shape guessing.
    """
    import zarr

    store = root / "vol.zarr"
    group = zarr.open_group(str(store), mode="w")
    for i, shape in enumerate(shapes):
        sub = group.create_group(f"scale{i}")
        arr = sub.create_array("image", shape=shape, dtype="float32", chunks=(2, 8, 8))
        arr[:] = np.linspace(0, 1, int(np.prod(shape)), dtype="float32").reshape(shape)

    base = shapes[0]
    datasets = []
    for i, shape in enumerate(shapes):
        factor = base[1] / shape[1]
        datasets.append(
            {
                "path": f"scale{i}/image",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [0.5 * factor, 0.5 * factor, 0.5 * factor]}
                ],
            }
        )
    (store / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": datasets,
                    }
                ]
            }
        )
    )
    return store


@pytest.fixture
def pyramid(tmp_path: Path) -> Path:
    # Deliberately NOT a clean power of two in z (9 -> 5 -> 3), matching the real
    # data where 690 -> 172 gives a factor of 4.0116.
    return _make_multiscale(tmp_path, [(9, 32, 32), (5, 16, 16), (3, 8, 8)])


class TestInspect:
    def test_lists_levels_finest_first(self, pyramid: Path) -> None:
        info = zarr_source.inspect_zarr(str(pyramid))
        assert [lv["path"] for lv in info["levels"]] == [
            "scale0/image",
            "scale1/image",
            "scale2/image",
        ]
        assert info["full_shape"] == [9, 32, 32]
        assert info["dtype"] == "float32"

    def test_reports_downsample_and_voxel_size(self, pyramid: Path) -> None:
        info = zarr_source.inspect_zarr(str(pyramid))
        assert info["levels"][0]["downsample"] == [1.0, 1.0, 1.0]
        assert info["levels"][1]["downsample"][1] == 2.0
        assert info["voxel_size"] == [0.5, 0.5, 0.5]
        assert info["voxel_unit"] == "micrometer"

    def test_rejects_missing_path(self) -> None:
        with pytest.raises(HTTPException) as exc:
            zarr_source.inspect_zarr("/nope/missing.zarr")
        assert exc.value.status_code == 404

    def test_rejects_relative_path(self) -> None:
        with pytest.raises(HTTPException) as exc:
            zarr_source.inspect_zarr("relative/vol.zarr")
        assert exc.value.status_code == 400

    def test_rejects_non_zarr_directory(self, tmp_path: Path) -> None:
        plain = tmp_path / "images"
        plain.mkdir()
        with pytest.raises(HTTPException) as exc:
            zarr_source.inspect_zarr(str(plain))
        assert exc.value.status_code == 422
        assert "not a Zarr store" in exc.value.detail

    def test_rejects_zip_with_actionable_message(self, tmp_path: Path) -> None:
        archive = tmp_path / "vol.zarr.zip"
        archive.write_bytes(b"PK\x03\x04")
        with pytest.raises(HTTPException) as exc:
            zarr_source.inspect_zarr(str(archive))
        assert exc.value.status_code == 422
        assert "unzip" in exc.value.detail.lower()

    def test_rejects_group_with_no_arrays(self, tmp_path: Path) -> None:
        import zarr

        empty = tmp_path / "empty.zarr"
        zarr.open_group(str(empty), mode="w")
        with pytest.raises(HTTPException) as exc:
            zarr_source.inspect_zarr(str(empty))
        assert exc.value.status_code == 422
        assert "no 3-D arrays" in exc.value.detail

    def test_discovers_arrays_without_multiscales_metadata(self, tmp_path: Path) -> None:
        import zarr

        store = tmp_path / "bare.zarr"
        group = zarr.open_group(str(store), mode="w")
        group.create_array("volume", shape=(4, 8, 8), dtype="float32", chunks=(2, 4, 4))
        info = zarr_source.inspect_zarr(str(store))
        assert [lv["path"] for lv in info["levels"]] == ["volume"]
        assert info["full_shape"] == [4, 8, 8]

    def test_handles_a_bare_array_store_with_no_group_wrapper(self, tmp_path: Path) -> None:
        """A store that's just a 3-D array at its own root (.zarray, no
        .zgroup) — e.g. written via `zarr.open(mode='w', ...)` / `dask.array
        .to_zarr()` with no OME-NGFF multiscale wrapper. `_is_zarr_dir`
        already recognizes this shape; `inspect_zarr` must not reject it."""
        import zarr

        store = tmp_path / "plain.zarr"
        arr = zarr.open(str(store), mode="w", shape=(6, 10, 12), dtype="uint16", zarr_format=2)
        arr[:] = np.arange(6 * 10 * 12, dtype="uint16").reshape(6, 10, 12)
        assert not (store / ".zgroup").exists()  # confirm this really is group-less

        info = zarr_source.inspect_zarr(str(store))
        assert len(info["levels"]) == 1
        level = info["levels"][0]
        assert level["path"] == ""
        assert level["shape"] == [6, 10, 12]
        assert level["downsample"] == [1.0, 1.0, 1.0]
        assert info["full_shape"] == [6, 10, 12]
        assert info["dtype"] == "uint16"
        assert info["voxel_size"] is None

    def test_rejects_a_bare_array_that_is_not_3d(self, tmp_path: Path) -> None:
        import zarr

        store = tmp_path / "plain2d.zarr"
        zarr.open(str(store), mode="w", shape=(10, 12), dtype="uint16", zarr_format=2)
        with pytest.raises(HTTPException) as exc:
            zarr_source.inspect_zarr(str(store))
        assert exc.value.status_code == 422
        assert "not 3-D" in exc.value.detail


class TestScanAndRegisterZarrs:
    """scan_and_register_zarrs's own logic (candidate discovery, shadow
    detection, aggregation) — register_zarr itself needs a live Tiled server
    (see TestPreflightZarr's own docstring), so it's mocked here rather than
    re-proven. The shadow pre-check navigates a fake Tiled client (see
    FakeContainer, defined below in this file) the same way preflight_zarr's
    own tests do."""

    def _fake_client(self, monkeypatch, browse_children=None):
        client = FakeContainer({"browse": FakeContainer(browse_children or {})})
        monkeypatch.setattr(zarr_source, "get_tiled_client", lambda uri, key: client)
        monkeypatch.setattr(zarr_source, "api_key_for_uri", lambda uri: None)
        return client

    def test_finds_only_top_level_zarr_dirs_and_registers_each(self, tmp_path: Path, monkeypatch) -> None:
        import zarr

        for name in ("a.zarr", "b.zarr"):
            store = tmp_path / name
            zarr.open(str(store), mode="w", shape=(2, 4, 4), dtype="uint8", zarr_format=2)
        (tmp_path / "not_a_store").mkdir()
        (tmp_path / "readme.txt").write_text("hi")
        # A Zarr store's OWN internals must never be treated as a second
        # candidate — only immediate children of scan_root are considered.
        (tmp_path / "a.zarr" / "nested.zarr").mkdir()

        self._fake_client(monkeypatch)
        calls: list[str] = []

        def fake_register_zarr(server_uri, path, container_path, description="", on_conflict="fail"):
            calls.append(Path(path).name)
            return {"key": Path(path).stem, "tiled_path": f"browse/{Path(path).stem}", "skipped": False}

        monkeypatch.setattr(zarr_source, "register_zarr", fake_register_zarr)

        result = zarr_source.scan_and_register_zarrs(None, str(tmp_path), "browse")
        assert result["scanned"] == 2
        assert sorted(calls) == ["a.zarr", "b.zarr"]
        assert sorted(r["name"] for r in result["registered"]) == ["a.zarr", "b.zarr"]
        assert result["skipped"] == []
        assert result["shadowed"] == []
        assert result["errors"] == []

    def test_reports_skipped_entries_separately_from_newly_registered(self, tmp_path: Path, monkeypatch) -> None:
        import zarr

        zarr.open(str(tmp_path / "existing.zarr"), mode="w", shape=(2, 4, 4), dtype="uint8", zarr_format=2)
        # A same-kind ("zarr") existing node — a legitimate re-scan match, not
        # a shadow — so register_zarr's own on_conflict=skip path is what
        # reports it, exactly as before.
        self._fake_client(monkeypatch, {"existing": FakeContainer({}, metadata={"source_format": "zarr"})})

        def fake_register_zarr(server_uri, path, container_path, description="", on_conflict="fail"):
            return {"key": "existing", "tiled_path": "browse/existing", "skipped": True}

        monkeypatch.setattr(zarr_source, "register_zarr", fake_register_zarr)

        result = zarr_source.scan_and_register_zarrs(None, str(tmp_path), "browse")
        assert result["registered"] == []
        assert result["skipped"] == ["existing"]
        assert result["shadowed"] == []

    def test_a_different_kind_collision_is_reported_as_shadowed_not_skipped(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import zarr

        zarr.open(str(tmp_path / "existing.zarr"), mode="w", shape=(2, 4, 4), dtype="uint8", zarr_format=2)
        self._fake_client(
            monkeypatch, {"existing": FakeContainer({}, metadata={"source_format": "image-stack"})}
        )
        register_calls: list[str] = []
        monkeypatch.setattr(
            zarr_source, "register_zarr",
            lambda *a, **kw: register_calls.append(a) or {"key": "existing", "tiled_path": "x", "skipped": False},
        )

        result = zarr_source.scan_and_register_zarrs(None, str(tmp_path), "browse")
        assert result["registered"] == []
        assert result["skipped"] == []
        assert result["shadowed"] == [
            {"name": "existing.zarr", "key": "existing", "existing_kind": "image-stack", "suggested_key": "existing_zarr"}
        ]
        # register_zarr must never be called for a shadowed candidate — no
        # blind replace, no misleading "skip" of someone else's data.
        assert register_calls == []

    def test_renames_lets_a_shadowed_candidate_register_under_an_alternate_key(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import zarr

        zarr.open(str(tmp_path / "existing.zarr"), mode="w", shape=(2, 4, 4), dtype="uint8", zarr_format=2)
        self._fake_client(
            monkeypatch, {"existing": FakeContainer({}, metadata={"source_format": "image-stack"})}
        )
        calls = []

        def fake_register_zarr(server_uri, path, container_path, on_conflict="fail"):
            calls.append(container_path)
            return {"key": "existing", "tiled_path": f"{container_path}/existing", "skipped": False}

        monkeypatch.setattr(zarr_source, "register_zarr", fake_register_zarr)

        result = zarr_source.scan_and_register_zarrs(
            None, str(tmp_path), "browse", renames={"existing.zarr": "existing_zarr"}
        )
        assert result["shadowed"] == []
        assert [r["key"] for r in result["registered"]] == ["existing_zarr"]
        # Registered one level deeper, under a container named for the
        # chosen alternate key — the only way to make Tiled's own
        # filename-derived key land under a different name.
        assert calls == ["browse/existing_zarr"]
        assert result["registered"][0]["tiled_path"] == "browse/existing_zarr/existing"

    def test_one_bad_store_does_not_abort_the_rest(self, tmp_path: Path, monkeypatch) -> None:
        import zarr

        for name in ("good.zarr", "bad.zarr"):
            zarr.open(str(tmp_path / name), mode="w", shape=(2, 4, 4), dtype="uint8", zarr_format=2)

        self._fake_client(monkeypatch)

        def fake_register_zarr(server_uri, path, container_path, description="", on_conflict="fail"):
            if Path(path).name == "bad.zarr":
                raise HTTPException(502, "boom")
            return {"key": "good", "tiled_path": "browse/good", "skipped": False}

        monkeypatch.setattr(zarr_source, "register_zarr", fake_register_zarr)

        result = zarr_source.scan_and_register_zarrs(None, str(tmp_path), "browse")
        assert [r["name"] for r in result["registered"]] == ["good.zarr"]
        assert result["errors"] == [{"name": "bad.zarr", "error": "boom"}]

    def test_rejects_relative_scan_root(self) -> None:
        with pytest.raises(HTTPException) as exc:
            zarr_source.scan_and_register_zarrs(None, "relative/dir", "browse")
        assert exc.value.status_code == 400

    def test_rejects_missing_scan_root(self, tmp_path: Path) -> None:
        with pytest.raises(HTTPException) as exc:
            zarr_source.scan_and_register_zarrs(None, str(tmp_path / "nope"), "browse")
        assert exc.value.status_code == 404

    def test_empty_directory_scans_cleanly_with_nothing_found(self, tmp_path: Path, monkeypatch) -> None:
        self._fake_client(monkeypatch)
        result = zarr_source.scan_and_register_zarrs(None, str(tmp_path), "browse")
        assert result == {"scanned": 0, "registered": [], "skipped": [], "shadowed": [], "errors": []}

    def test_invalid_on_conflict_falls_back_to_skip(self, tmp_path: Path, monkeypatch) -> None:
        import zarr

        zarr.open(str(tmp_path / "a.zarr"), mode="w", shape=(2, 4, 4), dtype="uint8", zarr_format=2)
        self._fake_client(monkeypatch)
        seen_on_conflict = []

        def fake_register_zarr(server_uri, path, container_path, description="", on_conflict="fail"):
            seen_on_conflict.append(on_conflict)
            return {"key": "a", "tiled_path": "browse/a", "skipped": False}

        monkeypatch.setattr(zarr_source, "register_zarr", fake_register_zarr)
        zarr_source.scan_and_register_zarrs(None, str(tmp_path), "browse", on_conflict="fail")
        assert seen_on_conflict == ["skip"]


class TestRegisteredKey:
    def test_strips_the_zarr_extension(self, pyramid: Path) -> None:
        # Tiled keys on the stem; the collision check must use the same key or it
        # silently misses an existing node and fails deep inside registration.
        assert zarr_source.registered_key(pyramid) == "vol"


class _FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.dtype = np.dtype("float32")
        self.structure_family = "array"

    def __getitem__(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.shape[0], f"slice {idx} out of range for {self.shape}"
        return np.zeros(self.shape[1:], dtype="float32")


class _FakeContainer(dict):
    structure_family = "container"

    def __init__(self, children: dict) -> None:
        super().__init__(children)


def _fake_pyramid() -> _FakeContainer:
    """Container mimicking a registered volume: scaleN -> {image: array}."""
    return _FakeContainer(
        {
            "scale0": _FakeContainer({"image": _FakeArray((9, 32, 32))}),
            "scale1": _FakeContainer({"image": _FakeArray((5, 16, 16))}),
            "scale2": _FakeContainer({"image": _FakeArray((3, 8, 8))}),
        }
    )


class TestMultiscaleResolution:
    def test_detects_pyramid_levels_in_order(self) -> None:
        assert arrays_mod.multiscale_levels(_fake_pyramid()) == ["scale0", "scale1", "scale2"]

    def test_ignores_non_pyramid_containers(self) -> None:
        stack = _FakeContainer({"a": _FakeArray((4, 4)), "b": _FakeArray((4, 4))})
        assert arrays_mod.multiscale_levels(stack) is None

    def test_selecting_the_volume_opens_the_finest_array(self) -> None:
        # The generic walk would return `scale0` as a one-element stack, showing a
        # whole 3-D volume as a single slice.
        node = arrays_mod._descend_to_stack(_fake_pyramid())
        assert isinstance(node, _FakeArray)
        assert node.shape == (9, 32, 32)


class TestFullResolutionCoordinates:
    def _meta(self, level_shape: tuple[int, int, int], full: list[int]) -> dict:
        pyramid = {
            "level_key": "scale2",
            "level_index": 2,
            "level_count": 3,
            "full_shape": full,
            "z_downsample": full[0] / level_shape[0],
        }
        return arrays_mod.array_shape_meta(_FakeArray(level_shape), pyramid)

    def test_reports_finest_geometry_for_a_coarse_level(self) -> None:
        meta = self._meta((3, 8, 8), [9, 32, 32])
        # Annotation coordinates must be full-resolution whichever level is shown.
        assert (meta["n_slices"], meta["height"], meta["width"]) == (9, 32, 32)
        assert (meta["level_n_slices"], meta["level_height"]) == (3, 8)

    def test_maps_full_resolution_index_onto_the_level(self) -> None:
        level = _FakeArray((3, 8, 8))
        meta = self._meta((3, 8, 8), [9, 32, 32])
        # Every full-res index must land in range — _FakeArray asserts otherwise.
        for idx in range(9):
            assert arrays_mod.read_slice(level, meta, idx).shape == (8, 8)

    def test_handles_non_power_of_two_z_ratios(self) -> None:
        # 690 -> 172 is 4.0116; a fixed integer factor would run off the end.
        level = _FakeArray((172, 160, 160))
        meta = self._meta((172, 160, 160), [690, 2560, 2560])
        assert arrays_mod.read_slice(level, meta, 689).shape == (160, 160)
        assert meta["width"] == 2560

    def test_leaves_plain_volumes_untouched(self) -> None:
        meta = arrays_mod.array_shape_meta(_FakeArray((7, 16, 16)))
        assert (meta["n_slices"], meta["height"], meta["width"]) == (7, 16, 16)
        assert "z_downsample" not in meta


class FakeContainer:
    """Duck-typed fake Tiled container — enough surface for preflight_zarr's
    navigation (`_walk`/`_child_keys`) without touching a real Tiled server."""

    def __init__(self, children=None, metadata=None):
        self._children = dict(children or {})
        self.metadata = metadata or {}

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, key):
        return self._children[key]

    def __len__(self):
        return len(self._children)

    def keys(self):
        return list(self._children.keys())


class TestPreflightZarr:
    """register_zarr itself needs a live Tiled server; preflight_zarr never
    calls it — it only inspects the local store and navigates a fake Tiled
    client, so it is fully testable without one."""

    def test_no_collision_when_key_absent(self, pyramid: Path, monkeypatch) -> None:
        client = FakeContainer({"browse": FakeContainer({})})
        monkeypatch.setattr(zarr_source, "get_tiled_client", lambda uri, key: client)
        monkeypatch.setattr(zarr_source, "api_key_for_uri", lambda uri: None)
        result = zarr_source.preflight_zarr(None, str(pyramid), "browse")
        assert result["exists"] is False
        assert result["existing"] is None
        assert result["key"] == zarr_source.registered_key(pyramid)

    def test_collision_reports_external_registration(self, pyramid: Path, monkeypatch) -> None:
        key = zarr_source.registered_key(pyramid)
        existing_node = FakeContainer(
            {"scale0": object()},
            metadata={"source_format": "zarr", "sample_name": "vol", "n_images": 9},
        )
        client = FakeContainer({"browse": FakeContainer({key: existing_node})})
        monkeypatch.setattr(zarr_source, "get_tiled_client", lambda uri, key: client)
        monkeypatch.setattr(zarr_source, "api_key_for_uri", lambda uri: None)
        result = zarr_source.preflight_zarr(None, str(pyramid), "browse")
        assert result["exists"] is True
        assert result["existing"]["external"] is True
        assert result["existing"]["sample_name"] == "vol"
        assert result["existing"]["n_images"] == 9

    def test_collision_with_internally_managed_data_is_not_external(self, pyramid: Path, monkeypatch) -> None:
        key = zarr_source.registered_key(pyramid)
        existing_node = FakeContainer({"img_0000.tif": object()}, metadata={})
        client = FakeContainer({"browse": FakeContainer({key: existing_node})})
        monkeypatch.setattr(zarr_source, "get_tiled_client", lambda uri, key: client)
        monkeypatch.setattr(zarr_source, "api_key_for_uri", lambda uri: None)
        result = zarr_source.preflight_zarr(None, str(pyramid), "browse")
        assert result["existing"]["external"] is False

    def test_missing_target_container_reports_no_collision(self, pyramid: Path, monkeypatch) -> None:
        client = FakeContainer({})
        monkeypatch.setattr(zarr_source, "get_tiled_client", lambda uri, key: client)
        monkeypatch.setattr(zarr_source, "api_key_for_uri", lambda uri: None)
        result = zarr_source.preflight_zarr(None, str(pyramid), "browse/missing")
        assert result["exists"] is False

    def test_invalid_path_propagates_the_http_exception(self) -> None:
        with pytest.raises(HTTPException) as exc:
            zarr_source.preflight_zarr(None, "relative/path")
        assert exc.value.status_code == 400
