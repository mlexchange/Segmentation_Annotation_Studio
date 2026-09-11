"""Tests for annotation_thumbnails.py — real PIL/numpy rendering, arrays.py
resolve/meta/read functions monkeypatched (same fake-array-source pattern as
test_annotation_server_routes.py's measure fixture); Tiled upload path uses a
lightweight duck-typed fake container."""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PILImage

import annotation_thumbnails
import arrays as arrays_mod

# ---------------------------------------------------------------------------
# decode_thumbnail_base64
# ---------------------------------------------------------------------------


class TestDecodeThumbnailBase64:
    def test_empty_string_returns_none(self):
        assert annotation_thumbnails.decode_thumbnail_base64("") is None

    def test_plain_base64_decodes(self):
        raw = base64.b64encode(b"hello").decode()
        assert annotation_thumbnails.decode_thumbnail_base64(raw) == b"hello"

    def test_data_url_prefix_is_stripped(self):
        raw = base64.b64encode(b"pngdata").decode()
        data_url = f"data:image/png;base64,{raw}"
        assert annotation_thumbnails.decode_thumbnail_base64(data_url) == b"pngdata"

    def test_invalid_base64_returns_none(self):
        assert annotation_thumbnails.decode_thumbnail_base64("not-@@base64!!") is None


# ---------------------------------------------------------------------------
# _stride_downsample
# ---------------------------------------------------------------------------

class TestStrideDownsample:
    def test_small_array_returned_unchanged(self):
        arr = np.zeros((10, 10))
        out = annotation_thumbnails._stride_downsample(arr, 512)
        assert out is arr

    def test_large_2d_array_is_strided(self):
        arr = np.zeros((1000, 1000))
        out = annotation_thumbnails._stride_downsample(arr, 500)
        assert max(out.shape) <= 500

    def test_large_3d_array_preserves_channel_dim(self):
        arr = np.zeros((1000, 1000, 3))
        out = annotation_thumbnails._stride_downsample(arr, 500)
        assert out.shape[2] == 3
        assert max(out.shape[:2]) <= 500


# ---------------------------------------------------------------------------
# _hex_to_rgba
# ---------------------------------------------------------------------------

class TestHexToRgba:
    def test_full_hex_with_hash(self):
        assert annotation_thumbnails._hex_to_rgba("#ff0080", 100) == (255, 0, 128, 100)

    def test_hex_without_hash(self):
        assert annotation_thumbnails._hex_to_rgba("00ff00", 200) == (0, 255, 0, 200)

    def test_short_hex_is_padded(self):
        # "abc" -> ljust(6, "0") -> "abc000"
        assert annotation_thumbnails._hex_to_rgba("#abc", 50) == (0xAB, 0xC0, 0x00, 50)


# ---------------------------------------------------------------------------
# _draw_shapes (via a real PIL ImageDraw)
# ---------------------------------------------------------------------------

class TestDrawShapes:
    def _draw(self, shapes, class_colors=None):
        img = PILImage.new("RGBA", (50, 50), (0, 0, 0, 0))
        draw = annotation_thumbnails.ImageDraw.Draw(img)
        annotation_thumbnails._draw_shapes(draw, shapes, class_colors or {}, scale=1.0)
        return np.array(img)

    def test_polygon_is_drawn(self):
        shape = {"kind": "polygon", "classId": 1, "points": [5, 5, 20, 5, 20, 20, 5, 20]}
        pixels = self._draw([shape], {1: "#ff0000"})
        assert pixels[10, 10, 3] > 0  # alpha channel non-zero inside the fill

    def test_polygon_with_too_few_points_is_skipped(self):
        shape = {"kind": "polygon", "classId": 1, "points": [5, 5]}
        pixels = self._draw([shape])
        assert np.all(pixels[:, :, 3] == 0)

    def test_rectangle_is_drawn(self):
        shape = {"kind": "rectangle", "classId": 1, "x": 5, "y": 5, "w": 10, "h": 10}
        pixels = self._draw([shape], {1: "#00ff00"})
        assert pixels[10, 10, 3] > 0

    def test_ellipse_is_drawn(self):
        shape = {"kind": "ellipse", "classId": 1, "cx": 25, "cy": 25, "rx": 10, "ry": 10}
        pixels = self._draw([shape], {1: "#0000ff"})
        assert pixels[25, 25, 3] > 0

    def test_brush_stroke_is_drawn(self):
        shape = {
            "kind": "brush",
            "classId": 1,
            "strokes": [{"mode": "paint", "points": [5, 5, 30, 30], "radius": 3}],
        }
        pixels = self._draw([shape], {1: "#ffffff"})
        assert pixels.sum() > 0

    def test_brush_erase_stroke_is_skipped(self):
        shape = {
            "kind": "brush",
            "classId": 1,
            "strokes": [{"mode": "erase", "points": [5, 5, 30, 30], "radius": 3}],
        }
        pixels = self._draw([shape])
        assert np.all(pixels[:, :, 3] == 0)

    def test_brush_stroke_with_one_point_is_skipped(self):
        shape = {
            "kind": "brush",
            "classId": 1,
            "strokes": [{"mode": "paint", "points": [5, 5], "radius": 3}],
        }
        pixels = self._draw([shape])
        assert np.all(pixels[:, :, 3] == 0)

    def test_unknown_class_id_falls_back_to_default_color(self):
        shape = {"kind": "rectangle", "classId": 999, "x": 0, "y": 0, "w": 5, "h": 5}
        pixels = self._draw([shape], {1: "#00ff00"})
        assert pixels[2, 2, 3] > 0


# ---------------------------------------------------------------------------
# render_annotated_thumbnail
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_array_source(monkeypatch: pytest.MonkeyPatch):
    arr = np.random.default_rng(0).integers(0, 255, size=(20, 20), dtype=np.uint8).astype(np.float32)

    def fake_read_slice(node, meta, idx):
        return arr + idx  # differ per slice so "best slice" pick is checkable

    monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri: "node")
    monkeypatch.setattr(arrays_mod, "array_shape_meta", lambda node: {"n_slices": 5})
    monkeypatch.setattr(arrays_mod, "read_slice", fake_read_slice)
    return arr


class TestRenderAnnotatedThumbnail:
    def test_returns_png_bytes_for_grayscale_source(self, fake_array_source):
        payload = {"classes": [], "slices": {}}
        png = annotation_thumbnails.render_annotated_thumbnail("local:foo.tif", payload)
        assert png is not None
        decoded = np.array(PILImage.open(BytesIO(png)))
        assert decoded.shape[:2] == (20, 20)

    def test_picks_slice_with_most_shapes(self, fake_array_source, monkeypatch):
        calls = []
        real_read_slice = arrays_mod.read_slice

        def spy_read_slice(node, meta, idx):
            calls.append(idx)
            return real_read_slice(node, meta, idx)

        monkeypatch.setattr(arrays_mod, "read_slice", spy_read_slice)
        payload = {
            "classes": [{"classId": 1, "color": "#ff0000"}],
            "slices": {
                "0": [{"kind": "rectangle", "classId": 1, "x": 0, "y": 0, "w": 2, "h": 2}],
                "3": [
                    {"kind": "rectangle", "classId": 1, "x": 0, "y": 0, "w": 2, "h": 2},
                    {"kind": "rectangle", "classId": 1, "x": 5, "y": 5, "w": 2, "h": 2},
                ],
            },
        }
        png = annotation_thumbnails.render_annotated_thumbnail("local:foo.tif", payload)
        assert png is not None
        assert calls == [3]

    def test_best_slice_clamped_to_valid_range(self, fake_array_source):
        payload = {"classes": [], "slices": {"999": [{"kind": "rectangle", "x": 0, "y": 0, "w": 1, "h": 1}]}}
        png = annotation_thumbnails.render_annotated_thumbnail("local:foo.tif", payload)
        assert png is not None

    def test_non_integer_slice_key_is_ignored(self, fake_array_source):
        payload = {"classes": [], "slices": {"not-a-number": [{"kind": "rectangle"}]}}
        png = annotation_thumbnails.render_annotated_thumbnail("local:foo.tif", payload)
        assert png is not None

    def test_missing_arrays_module_returns_none(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "arrays":
                raise ImportError("no arrays")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert annotation_thumbnails.render_annotated_thumbnail("local:x", {}) is None

    def test_resolve_array_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            arrays_mod, "resolve_array",
            lambda source, kind, server_uri: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert annotation_thumbnails.render_annotated_thumbnail("local:x", {}) is None

    def test_unsupported_array_shape_returns_none(self, monkeypatch):
        monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri: "node")
        monkeypatch.setattr(arrays_mod, "array_shape_meta", lambda node: {"n_slices": 1})
        monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: np.zeros(10))
        assert annotation_thumbnails.render_annotated_thumbnail("local:x", {}) is None

    def test_read_slice_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri: "node")
        monkeypatch.setattr(arrays_mod, "array_shape_meta", lambda node: {"n_slices": 1})

        def boom(node, meta, idx):
            raise RuntimeError("read failed")

        monkeypatch.setattr(arrays_mod, "read_slice", boom)
        assert annotation_thumbnails.render_annotated_thumbnail("local:x", {}) is None

    def test_rgb_source_uses_prepare_rgb_path(self, monkeypatch):
        rgb_arr = np.zeros((15, 15, 3), dtype=np.uint8)
        monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri: "node")
        monkeypatch.setattr(arrays_mod, "array_shape_meta", lambda node: {"n_slices": 1})
        monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: rgb_arr)
        png = annotation_thumbnails.render_annotated_thumbnail("local:x", {"classes": [], "slices": {}})
        assert png is not None
        decoded = np.array(PILImage.open(BytesIO(png)))
        assert decoded.shape[:2] == (15, 15)


# ---------------------------------------------------------------------------
# upload_thumbnail_to_tiled
# ---------------------------------------------------------------------------

class FakeArrayContainer:
    def __init__(self):
        self.written = []

    def write_array(self, arr, key, metadata):
        self.written.append((key, arr, metadata))


class FakeContainerNode(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.created = []

    def create_container(self, key, metadata):
        c = FakeArrayContainer()
        self[key] = c
        self.created.append((key, metadata))
        return c


@pytest.fixture()
def fake_tiled(monkeypatch: pytest.MonkeyPatch):
    import tiled_clients

    browse = FakeContainerNode()
    root = FakeContainerNode({"browse": browse})
    monkeypatch.setattr(tiled_clients, "api_key_for_uri", lambda uri: None)
    monkeypatch.setattr(tiled_clients, "get_tiled_client", lambda uri, key: root)
    return browse


class TestUploadThumbnailToTiled:
    def _png_bytes(self):
        img = PILImage.new("RGB", (4, 4), (10, 20, 30))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_local_source_is_a_no_op(self, fake_tiled):
        annotation_thumbnails.upload_thumbnail_to_tiled("local:foo.tif", 1, "2024-01-01", self._png_bytes())
        assert fake_tiled.created == []

    def test_creates_container_and_writes_array_on_first_version(self, fake_tiled):
        annotation_thumbnails.upload_thumbnail_to_tiled(
            "tiled::browse/sample1", 3, "2024-01-01T00:00:00", self._png_bytes(),
        )
        assert len(fake_tiled.created) == 1
        key, metadata = fake_tiled.created[0]
        assert key == "sample1__v_thumbs"
        container = fake_tiled[key]
        assert container.written[0][0] == "v0003"
        assert container.written[0][2]["version"] == 3

    def test_reuses_existing_container_on_subsequent_versions(self, fake_tiled):
        annotation_thumbnails.upload_thumbnail_to_tiled(
            "tiled::browse/sample1", 1, "t1", self._png_bytes(),
        )
        annotation_thumbnails.upload_thumbnail_to_tiled(
            "tiled::browse/sample1", 2, "t2", self._png_bytes(),
        )
        assert len(fake_tiled.created) == 1
        container = fake_tiled["sample1__v_thumbs"]
        assert [w[0] for w in container.written] == ["v0001", "v0002"]

    def test_missing_tiled_clients_module_is_a_no_op(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tiled_clients":
                raise ImportError("no tiled_clients")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Should not raise.
        annotation_thumbnails.upload_thumbnail_to_tiled("tiled::browse/s", 1, "t", self._png_bytes())

    def test_tiled_error_is_swallowed(self, monkeypatch):
        import tiled_clients
        monkeypatch.setattr(tiled_clients, "api_key_for_uri", lambda uri: None)
        monkeypatch.setattr(
            tiled_clients, "get_tiled_client",
            lambda uri, key: (_ for _ in ()).throw(RuntimeError("down")),
        )
        # Should not raise even though the Tiled call blows up.
        annotation_thumbnails.upload_thumbnail_to_tiled("tiled::browse/s", 1, "t", self._png_bytes())
