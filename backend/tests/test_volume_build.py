"""Tests for building a 3-D volume from a slice stack already in Tiled.

The registration half needs a live catalog, so what is pinned here is the part
that decides whether the feature is usable at all: what gets inspected, which
inputs are refused (and with a message that says what to do instead), and that
full resolution is deliberately not duplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiff_stack_source as tss  # noqa: E402
import volume_build  # noqa: E402


@pytest.fixture
def fake_stack(monkeypatch):
    """Stand in for a Tiled stack: `resolve_array` + `array_shape_meta` + `read_slice`."""

    # 4096 > TARGET_DIM (2048) by default, so an un-overridden fake_stack still
    # represents "a large stack that needs a pyramid" — this used to be true at
    # 1024 back when TARGET_DIM was 384.
    def install(n_slices=64, height=4096, width=4096, dtype="uint16", is_rgb=False):
        meta = {
            "n_slices": n_slices,
            "height": height,
            "width": width,
            "dtype": dtype,
            "is_rgb": is_rgb,
        }
        monkeypatch.setattr(volume_build.arrays_mod, "resolve_array", lambda *a, **k: object())
        monkeypatch.setattr(volume_build.arrays_mod, "array_shape_meta", lambda *a, **k: meta)
        # Slice i is filled with i, so a downsampled level's values are checkable.
        monkeypatch.setattr(
            volume_build.arrays_mod,
            "read_slice",
            lambda node, m, idx: np.full((height, width), idx, dtype=np.dtype(dtype)),
        )
        return meta

    return install


class TestInspect:
    def test_describes_the_pyramid_that_would_be_built(self, fake_stack):
        fake_stack(n_slices=64, height=4096, width=4096)
        info = volume_build.inspect_volume_build("browse/stack")
        assert info["full_shape"] == [64, 4096, 4096]
        assert info["pyramid_plan"]
        assert info["already_small"] is False

    def test_reads_every_source_slice_once(self, fake_stack):
        # The coarse levels cascade in memory, so the cost is one pass.
        fake_stack(n_slices=690)
        assert volume_build.inspect_volume_build("browse/stack")["slices_to_read"] == 690

    def test_flags_a_stack_that_needs_no_downsampling(self, fake_stack):
        fake_stack(n_slices=8, height=64, width=64)
        info = volume_build.inspect_volume_build("browse/stack")
        assert info["already_small"] is True
        assert info["slices_to_read"] == 0

    def test_refuses_a_single_image(self, fake_stack):
        fake_stack(n_slices=1)
        with pytest.raises(HTTPException) as excinfo:
            volume_build.inspect_volume_build("browse/one")
        assert excinfo.value.status_code == 422
        assert "single image" in excinfo.value.detail

    def test_refuses_colour_images(self, fake_stack):
        # The renderer draws one scalar volume; RGB has no meaning there.
        fake_stack(is_rgb=True)
        with pytest.raises(HTTPException) as excinfo:
            volume_build.inspect_volume_build("browse/rgb")
        assert excinfo.value.status_code == 422
        assert "Colour" in excinfo.value.detail


class TestBuildGuards:
    def test_refuses_non_tiled_sources(self):
        with pytest.raises(HTTPException) as excinfo:
            volume_build.build_volume("foo.tif", kind="local")
        assert excinfo.value.status_code == 422

    def test_refuses_an_already_small_stack_with_advice(self, fake_stack):
        # No pyramid is needed, but the data is still per-slice and unstreamable.
        # Saying only "nothing to do" would leave the user stuck.
        fake_stack(n_slices=8, height=64, width=64)
        with pytest.raises(HTTPException) as excinfo:
            volume_build.build_volume("browse/small")
        assert excinfo.value.status_code == 422
        assert "source images" in excinfo.value.detail

    def test_refuses_to_place_a_volume_at_the_catalog_root(self, fake_stack):
        fake_stack()
        with pytest.raises(HTTPException) as excinfo:
            volume_build.build_volume("stack")
        assert excinfo.value.status_code == 422
        assert "root" in excinfo.value.detail


class TestMetadataShape:
    def test_omits_scale0_when_full_resolution_is_not_copied(self):
        # Full resolution stays in the per-slice nodes. Declaring a scale0 that
        # does not exist would make the viewer request a missing level.
        plan = tss.pyramid_plan((690, 2560, 2560))
        ms = tss.multiscales_metadata("v", plan, include_scale0=False)["attributes"][
            "multiscales"
        ][0]
        paths = [d["path"] for d in ms["datasets"]]
        assert all(p.startswith(f"{tss.PYRAMID_KEY}/") for p in paths)
        assert "scale0" not in paths

    def test_scale_transforms_stay_relative_to_full_resolution(self):
        # Dropping scale0 must not renumber the others: a level downsampled 16x
        # still describes 16 full-res voxels per voxel, or the volume renders at
        # the wrong physical size.
        plan = tss.pyramid_plan((690, 2560, 2560))
        ms = tss.multiscales_metadata("v", plan, include_scale0=False)["attributes"][
            "multiscales"
        ][0]
        for dataset, level in zip(ms["datasets"], plan):
            scale = dataset["coordinateTransformations"][0]["scale"]
            assert scale == [float(f) for f in level["factor"]]
