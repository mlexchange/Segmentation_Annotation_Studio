"""Tests for arrays.py — the shape-dispatch/slice-reading core shared by
almost every route. Fake Tiled containers are plain duck-typed classes (same
minimal-surface pattern as test_browse_helpers.py's FakeNode); leaf arrays
are either real numpy arrays or a thin FakeArrayNode wrapper exposing
.shape/.dtype/.metadata/__array__/__getitem__, matching what a real Tiled
array client node looks like."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

import arrays
import local_fs


class FakeArrayNode:
    def __init__(self, arr, metadata=None):
        self._arr = np.asarray(arr)
        self.shape = self._arr.shape
        self.dtype = self._arr.dtype
        self.metadata = metadata or {}

    def __array__(self, dtype=None):
        return self._arr if dtype is None else self._arr.astype(dtype)

    def __getitem__(self, idx):
        return self._arr[idx]


class FakeContainer:
    def __init__(self, children=None, metadata=None):
        self.structure_family = "container"
        self._children = dict(children or {})
        self.metadata = metadata or {}

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, key):
        return self._children[key]

    def __len__(self):
        return len(self._children)


@pytest.fixture(autouse=True)
def clear_node_cache():
    arrays._node_cache.clear()
    yield
    arrays._node_cache.clear()


# ---------------------------------------------------------------------------
# array_shape_meta — direct array shape dispatch
# ---------------------------------------------------------------------------

class TestArrayShapeMetaDirect:
    def test_2d_array_is_hw(self):
        meta = arrays.array_shape_meta(np.zeros((10, 20), dtype=np.uint8))
        assert meta == {
            "n_slices": 1, "height": 10, "width": 20,
            "dtype": "uint8", "is_rgb": False, "shape_kind": "HW",
        }

    def test_3d_rgb_array_is_hwc(self):
        meta = arrays.array_shape_meta(np.zeros((10, 20, 3), dtype=np.uint8))
        assert meta["shape_kind"] == "HWC"
        assert meta["is_rgb"] is True
        assert meta["n_slices"] == 1

    def test_3d_rgba_array_is_hwc(self):
        meta = arrays.array_shape_meta(np.zeros((10, 20, 4), dtype=np.uint8))
        assert meta["shape_kind"] == "HWC"

    def test_3d_stack_is_nhw(self):
        meta = arrays.array_shape_meta(np.zeros((5, 10, 20), dtype=np.float32))
        assert meta["shape_kind"] == "NHW"
        assert meta["n_slices"] == 5
        assert meta["height"] == 10
        assert meta["width"] == 20

    def test_4d_rgb_stack_is_nhwc(self):
        meta = arrays.array_shape_meta(np.zeros((5, 10, 20, 3), dtype=np.uint8))
        assert meta["shape_kind"] == "NHWC"
        assert meta["n_slices"] == 5
        assert meta["is_rgb"] is True

    def test_1d_array_raises(self):
        with pytest.raises(HTTPException) as exc:
            arrays.array_shape_meta(np.zeros(10))
        assert exc.value.status_code == 422

    def test_5d_array_raises(self):
        with pytest.raises(HTTPException) as exc:
            arrays.array_shape_meta(np.zeros((1, 2, 3, 4, 5)))
        assert exc.value.status_code == 422

    def test_array_like_without_shape_attr_goes_through_asarray(self):
        class ArrayLikeNoShape:
            def __array__(self, dtype=None):
                return np.zeros((4, 4), dtype=np.uint8)

        meta = arrays.array_shape_meta(ArrayLikeNoShape())
        assert meta["shape_kind"] == "HW"

    def test_pyramid_kwarg_reports_finest_geometry(self):
        pyramid = {
            "full_shape": [100, 512, 512], "z_downsample": 4.0,
            "level_key": "scale2", "level_index": 2, "level_count": 3,
        }
        meta = arrays.array_shape_meta(np.zeros((25, 128, 128)), pyramid=pyramid)
        assert meta["n_slices"] == 100
        assert meta["height"] == 512
        assert meta["width"] == 512
        assert meta["z_downsample"] == 4.0
        assert meta["level_height"] == 128
        assert meta["level_n_slices"] == 25


# ---------------------------------------------------------------------------
# array_shape_meta — container/stack dispatch
# ---------------------------------------------------------------------------

class TestArrayShapeMetaStack:
    def test_container_of_2d_arrays_is_stack(self):
        node = FakeContainer({"b": np.zeros((4, 4)), "a": np.ones((4, 4))})
        meta = arrays.array_shape_meta(node)
        assert meta["shape_kind"] == "STACK"
        assert meta["n_slices"] == 2
        assert meta["keys"] == ["a", "b"]  # sorted lexically
        assert meta["is_rgb"] is False

    def test_container_of_rgb_arrays_is_rgb_stack(self):
        node = FakeContainer({"a": np.zeros((4, 4, 3))})
        meta = arrays.array_shape_meta(node)
        assert meta["is_rgb"] is True

    def test_empty_container_raises(self):
        with pytest.raises(HTTPException) as exc:
            arrays.array_shape_meta(FakeContainer({}))
        assert exc.value.status_code == 422

    def test_container_with_unsupported_child_shape_raises(self):
        node = FakeContainer({"a": np.zeros((4, 4, 5))})
        with pytest.raises(HTTPException) as exc:
            arrays.array_shape_meta(node)
        assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# read_slice
# ---------------------------------------------------------------------------

class TestReadSliceDirect:
    def test_hw_ignores_index(self):
        arr = np.arange(9).reshape(3, 3)
        meta = {"shape_kind": "HW"}
        out = arrays.read_slice(arr, meta, 5)
        assert np.array_equal(out, arr)

    def test_hwc_ignores_index(self):
        arr = np.zeros((3, 3, 3))
        out = arrays.read_slice(arr, {"shape_kind": "HWC"}, 2)
        assert out.shape == (3, 3, 3)

    def test_nhw_indexes_by_slice(self):
        node = FakeArrayNode(np.arange(2 * 4 * 4).reshape(2, 4, 4))
        out = arrays.read_slice(node, {"shape_kind": "NHW"}, 1)
        assert np.array_equal(out, node._arr[1])

    def test_nhw_maps_full_res_index_through_z_downsample(self):
        node = FakeArrayNode(np.arange(3 * 2 * 2).reshape(3, 2, 2))
        meta = {"shape_kind": "NHW", "z_downsample": 3.0, "level_n_slices": 3}
        # full-res idx 6 -> round(6/3)=2, clamped to level_n_slices-1=2
        out = arrays.read_slice(node, meta, 6)
        assert np.array_equal(out, node._arr[2])

    def test_nhw_z_downsample_of_one_is_a_no_op(self):
        node = FakeArrayNode(np.arange(3 * 2 * 2).reshape(3, 2, 2))
        meta = {"shape_kind": "NHW", "z_downsample": 1.0}
        out = arrays.read_slice(node, meta, 1)
        assert np.array_equal(out, node._arr[1])

    def test_unrecognized_shape_kind_raises(self):
        with pytest.raises(HTTPException) as exc:
            arrays.read_slice(np.zeros((2, 2)), {"shape_kind": "WEIRD"}, 0)
        assert exc.value.status_code == 422


class TestReadSliceStack:
    def test_reads_by_sorted_key_order(self):
        node = FakeContainer({"b": np.full((2, 2), 2.0), "a": np.full((2, 2), 1.0)})
        meta = {"shape_kind": "STACK", "keys": ["a", "b"]}
        out = arrays.read_slice(node, meta, 1)
        assert np.all(out == 2.0)

    def test_out_of_range_index_falls_back_to_first_key(self):
        node = FakeContainer({"a": np.full((2, 2), 1.0), "b": np.full((2, 2), 2.0)})
        meta = {"shape_kind": "STACK", "keys": ["a", "b"]}
        out = arrays.read_slice(node, meta, 99)
        assert np.all(out == 1.0)

    def test_missing_keys_in_meta_recomputes_from_node(self):
        node = FakeContainer({"a": np.full((2, 2), 1.0)})
        meta = {"shape_kind": "STACK"}
        out = arrays.read_slice(node, meta, 0)
        assert np.all(out == 1.0)

    def test_empty_stack_raises(self):
        node = FakeContainer({})
        with pytest.raises(HTTPException) as exc:
            arrays.read_slice(node, {"shape_kind": "STACK", "keys": []}, 0)
        assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# _is_container_node / multiscale_levels / _level_array / _descend_to_stack
# ---------------------------------------------------------------------------

class TestIsContainerNode:
    def test_plain_array_is_not_a_container(self):
        assert arrays._is_container_node(np.zeros((2, 2))) is False

    def test_fake_container_is_a_container(self):
        assert arrays._is_container_node(FakeContainer({})) is True

    def test_enum_like_structure_family_value_is_read(self):
        class FakeEnum:
            value = "container"

        class Node:
            structure_family = FakeEnum()

        assert arrays._is_container_node(Node()) is True


class TestMultiscaleLevels:
    def test_non_container_returns_none(self):
        assert arrays.multiscale_levels(np.zeros((2, 2))) is None

    def test_container_without_scale_children_returns_none(self):
        assert arrays.multiscale_levels(FakeContainer({"foo": 1, "bar": 2})) is None

    def test_single_scale_child_returns_none(self):
        assert arrays.multiscale_levels(FakeContainer({"scale0": 1})) is None

    def test_multiple_scale_children_sorted_numerically(self):
        node = FakeContainer({"scale10": 1, "scale2": 2, "scale0": 3})
        assert arrays.multiscale_levels(node) == ["scale0", "scale2", "scale10"]

    def test_non_enumerable_container_returns_none(self):
        class BrokenContainer:
            structure_family = "container"

            def __iter__(self):
                raise RuntimeError("boom")

        assert arrays.multiscale_levels(BrokenContainer()) is None


class TestLevelArray:
    def test_level_wrapping_a_single_array_child(self):
        arr = np.zeros((2, 2))
        level = FakeContainer({"image": arr})
        node = FakeContainer({"scale0": level})
        assert arrays._level_array(node, "scale0") is arr

    def test_level_that_is_already_an_array(self):
        arr = np.zeros((2, 2))
        node = FakeContainer({"scale0": arr})
        assert arrays._level_array(node, "scale0") is arr

    def test_missing_level_key_returns_none(self):
        node = FakeContainer({})
        assert arrays._level_array(node, "scale0") is None

    def test_empty_level_container_returns_none(self):
        node = FakeContainer({"scale0": FakeContainer({})})
        assert arrays._level_array(node, "scale0") is None


class TestDescendToStack:
    def test_non_container_returned_unchanged(self):
        arr = np.zeros((2, 2))
        assert arrays._descend_to_stack(arr) is arr

    def test_descends_through_wrapper_container_to_array_stack(self):
        stack = FakeContainer({"slice_0": np.zeros((2, 2))})
        wrapper = FakeContainer({"dataset": stack})
        assert arrays._descend_to_stack(wrapper) is stack

    def test_stops_at_container_of_arrays(self):
        stack = FakeContainer({"slice_0": np.zeros((2, 2)), "slice_1": np.zeros((2, 2))})
        assert arrays._descend_to_stack(stack) is stack

    def test_empty_container_returned_as_is(self):
        empty = FakeContainer({})
        assert arrays._descend_to_stack(empty) is empty

    def test_max_depth_stops_infinite_wrapper_chain(self):
        node = FakeContainer({})
        for _ in range(20):
            node = FakeContainer({"only": node})
        result = arrays._descend_to_stack(node, max_depth=3)
        # Should not infinite-loop or raise; just stop after max_depth hops.
        assert arrays._is_container_node(result)

    def test_multiscale_volume_resolves_to_finest_level_array(self):
        finest = np.zeros((3, 3))
        level0 = FakeContainer({"image": finest})
        level1 = FakeContainer({"image": np.zeros((2, 2))})
        node = FakeContainer({"scale0": level0, "scale1": level1})
        assert arrays._descend_to_stack(node) is finest


# ---------------------------------------------------------------------------
# node_keywords
# ---------------------------------------------------------------------------

class TestNodeKeywords:
    def test_no_metadata_returns_empty(self):
        assert arrays.node_keywords(np.zeros((2, 2))) == []

    def test_list_keywords_on_node_itself(self):
        node = FakeArrayNode(np.zeros((2, 2)), metadata={"keywords": ["a", "b"]})
        assert arrays.node_keywords(node) == ["a", "b"]

    def test_string_keyword_is_wrapped_in_a_list(self):
        node = FakeArrayNode(np.zeros((2, 2)), metadata={"keywords": "solo"})
        assert arrays.node_keywords(node) == ["solo"]

    def test_blank_string_values_filtered_out(self):
        node = FakeArrayNode(np.zeros((2, 2)), metadata={"keywords": ["a", "  ", ""]})
        assert arrays.node_keywords(node) == ["a"]

    def test_falls_back_to_first_child_of_container(self):
        child = FakeArrayNode(np.zeros((2, 2)), metadata={"keywords": ["child-tag"]})
        container = FakeContainer({"a": child})
        assert arrays.node_keywords(container) == ["child-tag"]

    def test_container_metadata_wins_over_child(self):
        child = FakeArrayNode(np.zeros((2, 2)), metadata={"keywords": ["child-tag"]})
        container = FakeContainer({"a": child}, metadata={"keywords": ["container-tag"]})
        assert arrays.node_keywords(container) == ["container-tag"]

    def test_container_with_no_tags_anywhere_returns_empty(self):
        container = FakeContainer({"a": FakeArrayNode(np.zeros((2, 2)))})
        assert arrays.node_keywords(container) == []

    def test_error_reading_child_metadata_is_swallowed(self):
        class BrokenContainer(FakeContainer):
            def __getitem__(self, key):
                raise RuntimeError("boom")

        container = BrokenContainer({"a": 1})
        assert arrays.node_keywords(container) == []


# ---------------------------------------------------------------------------
# resolve_array / resolve_container — kind dispatch + caching
# ---------------------------------------------------------------------------

class TestResolveArrayLocal:
    def test_local_kind_delegates_to_local_fs(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        arr = np.zeros((3, 3))
        monkeypatch.setattr(local_fs, "open_array", lambda source, root: arr)
        result = arrays.resolve_array("foo.npy", "local", root=str(tmp_path))
        assert result is arr

    def test_unknown_kind_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            arrays.resolve_array("x", "weird")
        assert exc.value.status_code == 422


class TestResolveArrayTiled:
    def test_resolves_nested_path_and_descends(self, monkeypatch: pytest.MonkeyPatch):
        leaf = np.zeros((2, 2))
        stack = FakeContainer({"slice_0": leaf})
        root = FakeContainer({"browse": FakeContainer({"sample1": stack})})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        result = arrays.resolve_array("browse/sample1", "tiled")
        assert result is stack

    def test_missing_path_raises_404(self, monkeypatch: pytest.MonkeyPatch):
        root = FakeContainer({})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        with pytest.raises(HTTPException) as exc:
            arrays.resolve_array("does/not/exist", "tiled")
        assert exc.value.status_code == 404

    def test_result_is_cached_by_key(self, monkeypatch: pytest.MonkeyPatch):
        calls = []
        leaf = np.zeros((2, 2))
        root = FakeContainer({"a": leaf})

        def fake_get_client(uri, key):
            calls.append(uri)
            return root

        monkeypatch.setattr("arrays.get_tiled_client", fake_get_client)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        arrays.resolve_array("a", "tiled", server_uri="http://x")
        arrays.resolve_array("a", "tiled", server_uri="http://x")
        assert len(calls) == 1

    def test_different_server_uri_is_a_different_cache_key(self, monkeypatch: pytest.MonkeyPatch):
        calls = []
        leaf = np.zeros((2, 2))
        root = FakeContainer({"a": leaf})

        def fake_get_client(uri, key):
            calls.append(uri)
            return root

        monkeypatch.setattr("arrays.get_tiled_client", fake_get_client)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        arrays.resolve_array("a", "tiled", server_uri="http://x")
        arrays.resolve_array("a", "tiled", server_uri="http://y")
        assert len(calls) == 2


class TestResolveContainer:
    def test_non_tiled_kind_raises(self):
        with pytest.raises(HTTPException) as exc:
            arrays.resolve_container("x", "local")
        assert exc.value.status_code == 422

    def test_resolves_without_descending(self, monkeypatch: pytest.MonkeyPatch):
        stack = FakeContainer({"slice_0": np.zeros((2, 2))})
        root = FakeContainer({"browse": FakeContainer({"sample1": stack})})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        result = arrays.resolve_container("browse/sample1", "tiled")
        assert result is stack  # NOT descended into slice_0

    def test_missing_path_raises_404(self, monkeypatch: pytest.MonkeyPatch):
        root = FakeContainer({})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        with pytest.raises(HTTPException) as exc:
            arrays.resolve_container("nope", "tiled")
        assert exc.value.status_code == 404

    def test_blank_path_segments_are_skipped(self, monkeypatch: pytest.MonkeyPatch):
        leaf = FakeContainer({})
        root = FakeContainer({"a": leaf})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        result = arrays.resolve_container("/a//", "tiled")
        assert result is leaf


# ---------------------------------------------------------------------------
# pyramid_info
# ---------------------------------------------------------------------------

class TestPyramidInfo:
    def test_non_tiled_kind_returns_none(self):
        assert arrays.pyramid_info("x", "local") is None

    def test_path_not_addressing_a_scale_level_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        assert arrays.pyramid_info("browse/sample1", "tiled") is None

    def test_path_addressing_a_scale_level_directly(self, monkeypatch: pytest.MonkeyPatch):
        finest = np.zeros((100, 512, 512))
        coarse = np.zeros((25, 128, 128))
        volume = FakeContainer({"scale0": finest, "scale1": coarse})
        root = FakeContainer({"browse": FakeContainer({"sample1": volume})})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)

        info = arrays.pyramid_info("browse/sample1/scale1", "tiled")
        assert info is not None
        assert info["level_key"] == "scale1"
        assert info["level_index"] == 1
        assert info["level_count"] == 2
        assert info["full_shape"] == [100, 512, 512]
        assert info["z_downsample"] == 4.0

    def test_path_addressing_a_scale_levels_image_child(self, monkeypatch: pytest.MonkeyPatch):
        finest_arr = np.zeros((10, 20, 20))
        finest_level = FakeContainer({"image": finest_arr})
        coarse_arr = np.zeros((5, 10, 10))
        coarse_level = FakeContainer({"image": coarse_arr})
        volume = FakeContainer({"scale0": finest_level, "scale1": coarse_level})
        root = FakeContainer({"browse": FakeContainer({"sample1": volume})})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)

        info = arrays.pyramid_info("browse/sample1/scale1/image", "tiled")
        assert info is not None
        assert info["level_key"] == "scale1"

    def test_nonexistent_parent_path_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        root = FakeContainer({})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        assert arrays.pyramid_info("nope/scale0", "tiled") is None

    def test_parent_without_multiscale_levels_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        volume = FakeContainer({"scale0": np.zeros((2, 2, 2))})  # only one level
        root = FakeContainer({"sample1": volume})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        assert arrays.pyramid_info("sample1/scale0", "tiled") is None

    def test_non_3d_level_shape_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        volume = FakeContainer({"scale0": np.zeros((10, 10)), "scale1": np.zeros((5, 5))})
        root = FakeContainer({"sample1": volume})
        monkeypatch.setattr("arrays.get_tiled_client", lambda uri, key: root)
        monkeypatch.setattr("arrays.api_key_for_uri", lambda uri: None)
        assert arrays.pyramid_info("sample1/scale1", "tiled") is None
