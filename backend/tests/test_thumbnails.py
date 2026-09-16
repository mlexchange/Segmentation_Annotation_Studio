"""Tests for thumbnails.py — real numpy/PIL, no mocks needed since the whole
module is pure array->PNG logic."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PILImage

import thumbnails


class FakeArrayNode:
    def __init__(self, arr):
        self._arr = arr

    def read(self):
        return self._arr


class FakeContainer:
    def __init__(self, children):
        self._children = children

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, key):
        return self._children[key]


def _decode(png_bytes):
    return np.array(PILImage.open(BytesIO(png_bytes)))


class TestRenderThumbnailRgb:
    def test_uint8_rgb_array_round_trips(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :, 0] = 200
        png = thumbnails.render_thumbnail(FakeArrayNode(arr), size=32)
        assert png is not None
        decoded = _decode(png)
        assert decoded.shape[:2] == (10, 10)

    def test_rgba_array_drops_alpha(self):
        arr = np.zeros((8, 8, 4), dtype=np.uint8)
        png = thumbnails.render_thumbnail(FakeArrayNode(arr), size=32)
        assert png is not None
        decoded = _decode(png)
        assert decoded.shape[:2] == (8, 8)

    def test_non_uint8_rgb_is_normalized(self):
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        arr[0, 0, :] = 100.0
        arr[1, 1, :] = 50.0
        png = thumbnails.render_thumbnail(FakeArrayNode(arr), size=32)
        assert png is not None

    def test_constant_rgb_array_yields_all_zero(self):
        rgb = thumbnails._prepare_rgb(np.full((4, 4, 3), 7.0, dtype=np.float32))
        assert np.all(rgb == 0)


class TestRenderThumbnailIntensity:
    def test_2d_array_gets_colormapped(self):
        arr = np.random.default_rng(0).random((16, 16)) * 1000
        png = thumbnails.render_thumbnail(FakeArrayNode(arr), size=32)
        assert png is not None
        decoded = _decode(png)
        assert decoded.shape[:2] == (16, 16)

    def test_constant_intensity_array_does_not_crash(self):
        arr = np.full((4, 4), 5.0)
        png = thumbnails.render_thumbnail(FakeArrayNode(arr), size=16)
        assert png is not None

    def test_nan_and_inf_are_sanitized(self):
        arr = np.array([[np.nan, np.inf], [-np.inf, 1.0]])
        png = thumbnails.render_thumbnail(FakeArrayNode(arr), size=16)
        assert png is not None

    def test_negative_values_are_clamped_before_log(self):
        arr = np.array([[-5.0, -1.0], [0.0, 10.0]])
        png = thumbnails.render_thumbnail(FakeArrayNode(arr), size=16)
        assert png is not None

    def test_grayscale_fallback_when_no_viridis(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(thumbnails, "_VIRIDIS", None)
        rgb = thumbnails._prepare_intensity(np.array([[0.0, 1.0], [2.0, 3.0]]))
        assert rgb.shape == (2, 2, 3)
        assert np.all(rgb[:, :, 0] == rgb[:, :, 1])
        assert np.all(rgb[:, :, 1] == rgb[:, :, 2])


class TestRenderThumbnailUnsupportedShapes:
    def test_1d_array_returns_none(self):
        assert thumbnails.render_thumbnail(FakeArrayNode(np.zeros(10)), size=32) is None

    def test_4d_array_returns_none_after_squeeze(self):
        arr = np.zeros((1, 5, 5, 2))
        assert thumbnails.render_thumbnail(FakeArrayNode(arr), size=32) is None

    def test_no_array_child_returns_none(self):
        assert thumbnails.render_thumbnail(FakeContainer({}), size=32) is None


class TestResolveArrayNode:
    def test_direct_array_node_returned_as_is(self):
        node = FakeArrayNode(np.zeros((2, 2)))
        assert thumbnails._resolve_array_node(node) is node

    def test_finds_first_array_child(self):
        child = FakeArrayNode(np.zeros((2, 2)))
        container = FakeContainer({"data": child})
        assert thumbnails._resolve_array_node(container) is child

    def test_skips_qmap_suffixed_children(self):
        qmap_child = FakeArrayNode(np.zeros((2, 2)))
        real_child = FakeArrayNode(np.ones((3, 3)))
        container = FakeContainer({"foo_qmap": qmap_child, "bar": real_child})
        assert thumbnails._resolve_array_node(container) is real_child

    def test_non_iterable_node_returns_none(self):
        assert thumbnails._resolve_array_node(object()) is None

    def test_container_with_no_array_children_returns_none(self):
        container = FakeContainer({"nested": FakeContainer({})})
        assert thumbnails._resolve_array_node(container) is None


class TestEncodePng:
    def test_downscales_to_fit_size(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        png = thumbnails._encode_png(rgb, size=50)
        decoded = _decode(png)
        assert decoded.shape[0] <= 50
        assert decoded.shape[1] <= 50

    def test_never_upscales(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        png = thumbnails._encode_png(rgb, size=256)
        decoded = _decode(png)
        assert decoded.shape[:2] == (10, 10)
