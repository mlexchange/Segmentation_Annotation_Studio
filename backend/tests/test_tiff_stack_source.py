"""Tests for exposing a TIFF directory as a streamable 3-D Zarr volume.

The registration path needs a live Tiled server, so what is pinned here is
everything that decides *correctness* and can be checked without one: the
pyramid plan, the downsampling arithmetic, the OME-NGFF metadata shape, and the
inspection guards that stop a shuffled or unusable stack from being registered.
Fixtures write tiny TIFFs on the fly, so these run anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiff_stack_source as tss  # noqa: E402

tifffile = pytest.importorskip("tifffile")


def write_stack(
    root: Path, n: int = 8, h: int = 16, w: int = 16, pad: int = 4, dtype=np.uint16
) -> Path:
    """A directory of zero-padded 2-D TIFFs whose value encodes the slice index."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        frame = np.full((h, w), i, dtype=dtype)
        tifffile.imwrite(str(root / f"img_{i:0{pad}d}.tif"), frame)
    return root


class TestPyramidPlan:
    def test_no_levels_when_already_small(self):
        # Nothing to generate: scale0 alone already fits a GPU texture.
        assert tss.pyramid_plan((10, 64, 64), target_dim=384) == []

    def test_first_level_brings_every_axis_under_target(self):
        plan = tss.pyramid_plan((2000, 3232, 3232), target_dim=384)
        assert plan, "a 3232-wide stack must generate levels"
        first = plan[0]
        assert max(first["shape"]) <= 384
        # Power-of-two factors keep the block mean exact.
        for f in first["factor"]:
            assert f & (f - 1) == 0

    def test_factor_is_the_smallest_that_fits(self):
        # 3232/8 = 404 > 384, so 8 is not enough and 16 is the answer.
        assert tss.pyramid_plan((2000, 3232, 3232), target_dim=384)[0]["factor"] == [8, 16, 16]

    def test_levels_double_and_are_named_in_order(self):
        plan = tss.pyramid_plan((2000, 3232, 3232), target_dim=384, levels=3)
        assert [p["path"] for p in plan] == ["scale1", "scale2", "scale3"]
        assert [p["factor"] for p in plan] == [[8, 16, 16], [16, 32, 32], [32, 64, 64]]

    def test_anisotropic_stack_still_gets_levels(self):
        # 4 slices of 4096x4096 is an ordinary tomography shape. A single shared
        # factor picked for the wide axes would ask for 4//16 = 0 slices and drop
        # every level, leaving nothing renderable — which looks exactly like a
        # broken viewer. Per-axis factors keep z alive.
        plan = tss.pyramid_plan((4, 4096, 4096), target_dim=384, levels=3)
        assert plan, "the xy axes still need downsampling"
        for level in plan:
            assert min(level["shape"]) >= 1
        # z has nowhere to go, so it is left alone rather than collapsed.
        assert all(level["shape"][0] >= 1 for level in plan)
        assert plan[0]["factor"][0] == 1

    def test_does_not_emit_a_level_identical_to_the_one_above(self):
        # Once every axis has bottomed out, another level would cost bytes and
        # add no detail.
        plan = tss.pyramid_plan((2, 1024, 1024), target_dim=384, levels=6)
        shapes = [tuple(level["shape"]) for level in plan]
        assert len(shapes) == len(set(shapes))

    def test_degenerate_shape_yields_nothing(self):
        assert tss.pyramid_plan((0, 512, 512)) == []


class TestBlockMean:
    def test_averages_over_the_whole_block(self):
        block = np.stack([np.full((4, 4), 10.0), np.full((4, 4), 20.0)])
        out = tss.block_mean(block, 2, 2)
        assert out.shape == (2, 2)
        # Mean over z as well as over each 2x2 tile.
        assert np.allclose(out, 15.0)

    def test_drops_partial_tiles_rather_than_averaging_them(self):
        # 5 columns with fx=2: the last column cannot fill a tile. Including it
        # would make that output column brighter purely because of where the
        # edge fell.
        block = np.ones((1, 4, 5), dtype=np.float32)
        assert tss.block_mean(block, 2, 2).shape == (2, 2)

    def test_does_not_overflow_integer_inputs(self):
        # uint16 sums overflow almost immediately; the mean must be exact.
        block = np.full((4, 4, 4), 60000, dtype=np.uint16)
        assert np.allclose(tss.block_mean(block, 2, 2), 60000.0)

    def test_rejects_zero_factor(self):
        with pytest.raises(ValueError):
            tss.block_mean(np.ones((1, 2, 2), dtype=np.float32), 0, 1)


class TestMultiscalesMetadata:
    def test_nests_under_attributes(self):
        # Tiled's .zattrs route returns metadata["attributes"] verbatim; anywhere
        # else and the viewer reports "missing multiscales".
        meta = tss.multiscales_metadata("sample", [])
        assert "multiscales" in meta["attributes"]

    def test_always_includes_scale0_first(self):
        datasets = tss.multiscales_metadata("s", [])["attributes"]["multiscales"][0]["datasets"]
        assert [d["path"] for d in datasets] == ["scale0"]

    def test_lists_every_generated_level_with_a_scaled_transform(self):
        plan = tss.pyramid_plan((2000, 3232, 3232), target_dim=384)
        ms = tss.multiscales_metadata("s", plan)["attributes"]["multiscales"][0]
        # scale0 is the in-place TIFF sequence; the generated levels live in the
        # registered sidecar sub-group, so their paths are nested.
        assert [d["path"] for d in ms["datasets"]] == ["scale0"] + [
            f"{tss.PYRAMID_KEY}/{p['path']}" for p in plan
        ]
        # A level downsampled by f on an axis covers f times as much space per
        # voxel on that axis — anisotropic factors must survive into the transform.
        for dataset, level in zip(ms["datasets"][1:], plan):
            scale = dataset["coordinateTransformations"][0]["scale"]
            assert scale == [float(f) for f in level["factor"]]

    def test_declares_zyx_axes(self):
        ms = tss.multiscales_metadata("s", [])["attributes"]["multiscales"][0]
        assert [a["name"] for a in ms["axes"]] == ["z", "y", "x"]


class TestInspect:
    def test_describes_a_well_formed_stack(self, tmp_path):
        info = tss.inspect_tiff_stack(str(write_stack(tmp_path / "scan", n=8, h=16, w=16)))
        assert info["full_shape"] == [8, 16, 16]
        assert info["dtype"] == "uint16"
        assert info["name"] == "scan"

    def test_files_are_in_slice_order(self, tmp_path):
        root = write_stack(tmp_path / "scan", n=12)
        names = [p.name for p in tss.tiff_files(root)]
        assert names == sorted(names)
        assert names[0].endswith("0000.tif") and names[-1].endswith("0011.tif")

    def test_rejects_inconsistent_numbering(self, tmp_path):
        # Unpadded names sort lexically as img_1, img_10, img_2 — registering
        # that as-is would build a shuffled volume, silently.
        root = tmp_path / "scan"
        root.mkdir()
        for i in (1, 2, 10):
            tifffile.imwrite(str(root / f"img_{i}.tif"), np.zeros((4, 4), np.uint16))
        with pytest.raises(HTTPException) as excinfo:
            tss.inspect_tiff_stack(str(root))
        assert excinfo.value.status_code == 422
        assert "order" in excinfo.value.detail

    def test_rejects_a_single_image(self, tmp_path):
        root = write_stack(tmp_path / "scan", n=1)
        with pytest.raises(HTTPException) as excinfo:
            tss.inspect_tiff_stack(str(root))
        assert excinfo.value.status_code == 422
        assert "volume" in excinfo.value.detail

    def test_rejects_a_directory_with_no_tiffs(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(HTTPException) as excinfo:
            tss.inspect_tiff_stack(str(tmp_path / "empty"))
        assert excinfo.value.status_code == 422

    def test_rejects_a_relative_path(self, tmp_path):
        with pytest.raises(HTTPException) as excinfo:
            tss.inspect_tiff_stack("relative/scan")
        assert excinfo.value.status_code == 400

    def test_reports_no_reads_when_no_pyramid_is_needed(self, tmp_path):
        # A small stack needs no generated levels, so registration is instant —
        # the UI should not warn about a long job.
        info = tss.inspect_tiff_stack(str(write_stack(tmp_path / "scan", n=4, h=8, w=8)))
        assert info["pyramid_plan"] == []
        assert info["slices_to_read"] == 0


class TestPyramidStore:
    """The generated levels are written as a real on-disk Zarr store.

    Not with Tiled's ``write_array``: its ``/zarr/v2`` chunk route only serves
    externally-managed arrays, and the Zarr façade is exactly what the 3-D viewer
    reads. See this module's docstring.
    """

    def test_writes_a_readable_zarr_v2_group(self, tmp_path, monkeypatch):
        zarr = pytest.importorskip("zarr")
        monkeypatch.setenv("VOLUME_CACHE_DIR", str(tmp_path / "cache"))
        levels = {
            "scale1": np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4),
            "scale2": np.ones((1, 2, 2), dtype=np.float32),
        }
        store = tss.write_pyramid_store("sample__volume", levels)

        assert store.is_dir()
        group = zarr.open_group(str(store), mode="r")
        for name, expected in levels.items():
            assert np.array_equal(group[name][:], expected)

    def test_is_zarr_v2_not_v3(self, tmp_path, monkeypatch):
        # Tiled's /zarr/v2 facade and the viewer's reader both speak v2; a v3
        # store would write `zarr.json` and the viewer would find no `.zarray`.
        monkeypatch.setenv("VOLUME_CACHE_DIR", str(tmp_path / "cache"))
        store = tss.write_pyramid_store(
            "sample__volume", {"scale1": np.zeros((1, 2, 2), np.float32)}
        )
        assert (store / ".zgroup").exists()
        assert (store / "scale1" / ".zarray").exists()

    def test_chunks_one_slice_at_a_time(self, tmp_path, monkeypatch):
        zarr = pytest.importorskip("zarr")
        monkeypatch.setenv("VOLUME_CACHE_DIR", str(tmp_path / "cache"))
        store = tss.write_pyramid_store(
            "sample__volume", {"scale1": np.zeros((6, 8, 8), np.float32)}
        )
        assert zarr.open_group(str(store), mode="r")["scale1"].chunks == (1, 8, 8)

    def test_replaces_a_previous_build(self, tmp_path, monkeypatch):
        # Re-registering must not leave a stale volume behind for the same key.
        zarr = pytest.importorskip("zarr")
        monkeypatch.setenv("VOLUME_CACHE_DIR", str(tmp_path / "cache"))
        tss.write_pyramid_store("sample__volume", {"scale1": np.zeros((4, 4, 4), np.float32)})
        store = tss.write_pyramid_store(
            "sample__volume", {"scale1": np.ones((2, 2, 2), np.float32)}
        )
        group = zarr.open_group(str(store), mode="r")
        assert group["scale1"].shape == (2, 2, 2)
        assert list(group.array_keys()) == ["scale1"]

    def test_honours_the_cache_dir_override(self, tmp_path, monkeypatch):
        # Beamline reconstruction directories are routinely read-only, so the
        # pyramid must never be written next to the source.
        monkeypatch.setenv("VOLUME_CACHE_DIR", str(tmp_path / "elsewhere"))
        store = tss.write_pyramid_store(
            "sample__volume", {"scale1": np.zeros((1, 2, 2), np.float32)}
        )
        assert str(store).startswith(str(tmp_path / "elsewhere"))


class TestRegisteredKey:
    def test_is_a_sidecar_of_the_dataset_key(self, tmp_path):
        # The per-slice container Annotate reads already owns the unsuffixed key.
        # Colliding with it would make the 3-D view impossible for precisely the
        # datasets it exists to serve.
        root = write_stack(tmp_path / "rec20230224_sea_shell")
        key = tss.registered_key(root)
        assert key == f"rec20230224_sea_shell{tss.VOLUME_SUFFIX}"

    def test_matches_the_existing_sidecar_convention(self, tmp_path):
        # Same shape as __v_thumbs / __masks elsewhere in the catalog.
        assert tss.VOLUME_SUFFIX.startswith("__")


class TestBuildLevel:
    def test_downsamples_a_real_stack_correctly(self, tmp_path):
        # Slice i has value 2i, so a factor-2 level's slice j is the mean of
        # source slices 2j and 2j+1 — i.e. 4j+1. Even values keep the expected
        # result an exact integer, so this checks the averaging rather than the
        # rounding (which has its own test below).
        root = tmp_path / "scan"
        root.mkdir()
        for i in range(8):
            tifffile.imwrite(str(root / f"img_{i:04d}.tif"), np.full((8, 8), 2 * i, np.uint16))
        out = tss._build_level(tss.tiff_files(root), [2, 2, 2], [4, 4, 4], np.dtype(np.uint16))
        assert out.shape == (4, 4, 4)
        assert out.dtype == np.uint16
        for j in range(4):
            assert np.all(out[j] == 4 * j + 1)

    def test_rounds_rather_than_truncating_on_integer_output(self, tmp_path):
        # Mean of 0 and 3 is 1.5. Truncation would give 1; the cast must round.
        root = tmp_path / "scan"
        root.mkdir()
        for i, value in enumerate((0, 3)):
            tifffile.imwrite(str(root / f"img_{i:04d}.tif"), np.full((4, 4), value, np.uint16))
        out = tss._build_level(tss.tiff_files(root), [2, 1, 1], [1, 4, 4], np.dtype(np.uint16))
        assert np.all(out == 2)

    def test_counts_every_source_slice_once(self, tmp_path):
        root = write_stack(tmp_path / "scan", n=8, h=8, w=8)
        reads = []
        tss._build_level(
            tss.tiff_files(root), [2, 2, 2], [4, 4, 4], np.dtype(np.uint16), lambda: reads.append(1)
        )
        assert len(reads) == 8

    def test_supports_anisotropic_factors(self, tmp_path):
        # z left alone, xy halved — the shape a short, wide stack produces.
        root = write_stack(tmp_path / "scan", n=4, h=8, w=8, dtype=np.float32)
        out = tss._build_level(tss.tiff_files(root), [1, 2, 2], [4, 4, 4], np.dtype(np.float32))
        assert out.shape == (4, 4, 4)
        for z in range(4):
            assert np.allclose(out[z], z)  # no z averaging, so values survive

    def test_preserves_float_input_without_rounding(self, tmp_path):
        root = write_stack(tmp_path / "scan", n=4, h=8, w=8, dtype=np.float32)
        out = tss._build_level(tss.tiff_files(root), [2, 2, 2], [2, 4, 4], np.dtype(np.float32))
        assert out.dtype == np.float32
        assert np.allclose(out[0], 0.5)  # mean of slices valued 0 and 1


class TestCascade:
    """Coarse levels are built from the level above, not from the source.

    That is what makes registration read every source slice once instead of once
    per level. It is only a valid shortcut because the factors are powers of two,
    so averaging an average equals averaging the source — these tests pin that.
    """

    def test_matches_a_direct_downsample(self):
        rng = np.random.default_rng(7)
        source = rng.random((16, 32, 32), dtype=np.float32)

        direct = tss.downsample_array(source, [4, 4, 4])
        cascaded = tss.downsample_array(tss.downsample_array(source, [2, 2, 2]), [2, 2, 2])

        assert direct.shape == cascaded.shape
        assert np.allclose(direct, cascaded, atol=1e-5)

    def test_matches_a_direct_downsample_with_anisotropic_factors(self):
        rng = np.random.default_rng(11)
        source = rng.random((4, 32, 32), dtype=np.float32)

        direct = tss.downsample_array(source, [1, 4, 4])
        cascaded = tss.downsample_array(tss.downsample_array(source, [1, 2, 2]), [1, 2, 2])

        assert np.allclose(direct, cascaded, atol=1e-5)

    def test_relative_factors_between_plan_levels_are_whole_numbers(self):
        # The cascade divides each level's factor by the previous level's; a
        # non-integer ratio would silently truncate and misalign the level.
        plan = tss.pyramid_plan((604, 2560, 2560), target_dim=384, levels=3)
        for previous, level in zip(plan, plan[1:]):
            for axis in range(3):
                assert level["factor"][axis] % previous["factor"][axis] == 0

    def test_shrinks_on_every_axis_that_still_can(self):
        out = tss.downsample_array(np.ones((8, 8, 8), dtype=np.float32), [2, 2, 2])
        assert out.shape == (4, 4, 4)
        assert np.allclose(out, 1.0)  # a constant volume stays constant
