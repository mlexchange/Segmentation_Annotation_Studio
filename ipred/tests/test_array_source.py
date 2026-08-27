import numpy as np

from ipred import array_source


class _FakeArray:
    def __init__(self, shape):
        self._data = np.zeros(shape, dtype="float32")
        self.structure_family = "array"

    def __array__(self, dtype=None):
        return self._data


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
