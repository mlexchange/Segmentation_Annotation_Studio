"""Regression tests for container-stack slice indexing and shape consistency.

A "stack" is a Tiled container whose children are individually-written array
nodes (one per slice) — the layout drag-and-drop ingest produces. These tests
use a plain dict-backed fake node so no real Tiled server is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from arrays import _stack_keys, _stack_shape_meta, is_sidecar_key, read_slice


class _FakeNode(dict):
    """Dict-backed stand-in for a Tiled container: ``list(node)`` -> keys."""


class _FakeContainerChild:
    """Stand-in for a Tiled sidecar container child (e.g. ``__v_thumbs``)."""

    structure_family = "container"


def _meta(keys: list[str], height: int = 4, width: int = 4) -> dict:
    return {"shape_kind": "STACK", "keys": keys, "height": height, "width": width}


def test_negative_index_returns_last_slice() -> None:
    """STACK indexing supports Python-style negative indices, like a real list."""
    node = _FakeNode({
        "0000": np.zeros((4, 4), dtype=np.uint8),
        "0001": np.ones((4, 4), dtype=np.uint8),
    })
    keys = ["0000", "0001"]

    result = read_slice(node, _meta(keys), -1)

    assert np.array_equal(result, node["0001"])


def test_out_of_range_index_raises_404_not_slice_zero() -> None:
    """An out-of-range index must fail loudly, not silently substitute slice 0."""
    node = _FakeNode({"0000": np.zeros((4, 4), dtype=np.uint8)})
    keys = ["0000"]

    with pytest.raises(HTTPException) as exc_info:
        read_slice(node, _meta(keys), 5)

    assert exc_info.value.status_code == 404


def test_mismatched_slice_shape_raises_422() -> None:
    """A later slice whose shape disagrees with the stack's declared shape must
    be rejected rather than silently reaching rendering/export with a mismatched
    array."""
    node = _FakeNode({
        "0000": np.zeros((4, 4), dtype=np.uint8),
        "0001": np.zeros((8, 8), dtype=np.uint8),  # shape mismatch
    })
    keys = ["0000", "0001"]

    with pytest.raises(HTTPException) as exc_info:
        read_slice(node, _meta(keys, height=4, width=4), 1)

    assert exc_info.value.status_code == 422


def test_consistent_slice_shape_is_returned_unchanged() -> None:
    """The common case — every slice matches the declared shape — is unaffected."""
    node = _FakeNode({
        "0000": np.zeros((4, 4), dtype=np.uint8),
        "0001": np.full((4, 4), 7, dtype=np.uint8),
    })
    keys = ["0000", "0001"]

    result = read_slice(node, _meta(keys), 1)

    assert np.array_equal(result, node["0001"])


@pytest.mark.parametrize("sidecar_key", ["sample__v_thumbs", "sample__masks"])
def test_stack_keys_excludes_sidecar_container(sidecar_key: str) -> None:
    """A per-sample container that wraps one array plus a sidecar container
    (version thumbnails, synced masks) must not count the sidecar as a slice —
    it has a different shape and isn't image data."""
    node = _FakeNode({
        "sample": np.zeros((4, 4), dtype=np.uint8),
        sidecar_key: _FakeContainerChild(),
    })

    assert _stack_keys(node) == ["sample"]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("sample__v_thumbs", True),
        ("sample__masks", True),
        ("sample", False),
        ("0001", False),
        ("masks", False),  # bare name, not the `__masks` suffix
        ("v_thumbs_sample", False),  # suffix must be at the end
    ],
)
def test_is_sidecar_key(key: str, expected: bool) -> None:
    """Shared by arrays' slice reading and the /api/browse/slices route, so the
    two always agree on what counts as a slice."""
    assert is_sidecar_key(key) is expected


def test_stack_shape_meta_ignores_sidecar_when_computing_n_slices() -> None:
    """End-to-end: n_slices for an annotated single-image container must be 1,
    not 2, once its version-thumbnails sidecar container exists alongside it."""
    node = _FakeNode({
        "sample": np.zeros((4, 4), dtype=np.uint8),
        "sample__v_thumbs": _FakeContainerChild(),
    })

    meta = _stack_shape_meta(node)

    assert meta["n_slices"] == 1
    assert meta["keys"] == ["sample"]
