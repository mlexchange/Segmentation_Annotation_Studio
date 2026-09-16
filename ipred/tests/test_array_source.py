import numpy as np

from ipred import array_source


class _FakeArray:
    def __init__(self, shape):
        self._data = np.zeros(shape, dtype="float32")
        self.structure_family = "array"
        self.shape = shape
        # Spies so tests can assert on WHICH access pattern was actually used —
        # the whole point of the O(N^2) regression fix is that a single-slice
        # read must index this node, not realize the whole thing via __array__.
        self.array_calls = 0
        self.getitem_calls: list[int] = []

    def __array__(self, dtype=None):
        self.array_calls += 1
        return self._data

    def __getitem__(self, idx):
        self.getitem_calls.append(idx)
        return self._data[idx]


class _FakeContainer(dict):
    structure_family = "container"


def _fake_pyramid():
    """Container mimicking a registered multiscale volume: scaleN -> {image: array}."""
    return _FakeContainer(
        {
            "scale0": _FakeContainer({"image": _FakeArray((9, 32, 32))}),
            "scale1": _FakeContainer({"image": _FakeArray((5, 16, 16))}),
            "scale2": _FakeContainer({"image": _FakeArray((3, 8, 8))}),
            "scale3": _FakeContainer({"image": _FakeArray((2, 4, 4))}),
            "scale4": _FakeContainer({"image": _FakeArray((1, 2, 2))}),
        }
    )


class TestDescendToArray:
    def test_descends_multiscale_pyramid_to_finest_level(self) -> None:
        node = array_source._descend_to_array(_fake_pyramid())
        assert not array_source._is_container_node(node)
        assert np.asarray(node).shape == (9, 32, 32)

    def test_descends_wrapper_container_to_array_stack(self) -> None:
        wrapper = _FakeContainer({"vol": _FakeContainer({"a": _FakeArray((4, 4))})})
        node = array_source._descend_to_array(wrapper)
        assert array_source._is_container_node(node)  # container of arrays == stack

    def test_leaves_bare_array_untouched(self) -> None:
        arr = _FakeArray((4, 4))
        assert array_source._descend_to_array(arr) is arr

    def test_five_level_pyramid_does_not_collapse_to_length_5_array(self) -> None:
        # Regression: a naive np.asarray(container) on an un-descended 5-child
        # pyramid container previously yielded shape (5,), tripping
        # "unsupported array shape" in _index_slice during train/infer.
        node = array_source._descend_to_array(_fake_pyramid())
        data = np.asarray(node)
        assert data.shape != (5,)
        sliced = array_source._index_slice(data, 0)
        assert sliced.shape == (32, 32)


class TestReadTiledLazySlice:
    """Regression: a batch-apply job over N slices used to call np.asarray(node)
    (materializing the WHOLE stack) on every single-slice request — an O(N^2)
    read that dominated "predict across all slices" wall-clock time. A real
    single-slice read must index the node BEFORE realizing it."""

    def _patch_from_uri(self, monkeypatch, client):
        import tiled.client

        monkeypatch.setattr(tiled.client, "from_uri", lambda *a, **k: client)

    def test_bare_nhw_stack_indexes_without_materializing_whole_array(self, monkeypatch) -> None:
        arr = _FakeArray((50, 16, 16))
        client = _FakeContainer({"ds": arr})
        self._patch_from_uri(monkeypatch, client)

        out = array_source.read_slice(kind="tiled", source="ds", slice_index=7, server_uri="http://x")

        assert out.shape == (16, 16)
        assert arr.getitem_calls == [7]
        assert arr.array_calls == 0  # the whole-stack path must never fire

    def test_container_of_per_slice_arrays_only_realizes_the_requested_one(self, monkeypatch) -> None:
        wanted = _FakeArray((16, 16))
        other = _FakeArray((16, 16))
        client = _FakeContainer({"ds": _FakeContainer({"0": other, "1": wanted})})
        self._patch_from_uri(monkeypatch, client)

        out = array_source.read_slice(kind="tiled", source="ds", slice_index=1, server_uri="http://x")

        assert out.shape == (16, 16)
        assert wanted.array_calls == 1
        assert other.array_calls == 0  # the sibling slice must never be touched

    def test_hwc_single_image_is_returned_whole(self, monkeypatch) -> None:
        arr = _FakeArray((64, 64, 3))
        client = _FakeContainer({"ds": arr})
        self._patch_from_uri(monkeypatch, client)

        out = array_source.read_slice(kind="tiled", source="ds", slice_index=0, server_uri="http://x")

        assert out.shape == (64, 64, 3)
        assert arr.getitem_calls == []  # a single HWC image has nothing to index into


class TestReadLocalTiffPage:
    """Regression: `_read_local` used to call `tifffile.imread` (decodes every
    page) for a single-slice request."""

    def test_reads_one_page_of_a_multipage_tiff(self, tmp_path, monkeypatch) -> None:
        import tifffile

        path = tmp_path / "stack.tif"
        pages = np.stack([np.full((8, 8), i, dtype=np.uint8) for i in range(5)])
        tifffile.imwrite(str(path), pages)

        # `tifffile.imread` decodes the whole stack up front — assert the fix
        # never calls it for a multi-page file.
        monkeypatch.setattr(
            tifffile, "imread", lambda *a, **k: (_ for _ in ()).throw(AssertionError("full-stack imread called"))
        )

        out = array_source._read_local("stack.tif", slice_index=3, root=str(tmp_path))
        assert out.shape == (8, 8)
        assert int(out[0, 0]) == 3

    def test_single_page_tiff_still_works(self, tmp_path) -> None:
        import tifffile

        path = tmp_path / "single.tif"
        tifffile.imwrite(str(path), np.full((8, 8), 9, dtype=np.uint8))

        out = array_source._read_local("single.tif", slice_index=0, root=str(tmp_path))
        assert out.shape == (8, 8)
        assert int(out[0, 0]) == 9
