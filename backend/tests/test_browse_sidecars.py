"""Sidecar containers must never surface as browsable samples or slices.

Saving annotations writes ``<name>__v_thumbs`` / ``<name>__masks`` containers
next to the array they describe (see :mod:`sidecars`). They are bookkeeping, so
the sample listing must skip them and the per-sample slice count must not count
them — a lone annotated image whose sidecar was counted would be reported as a
2-slice volume and sent down the Browse drill-in path instead of opening
directly.
"""

from __future__ import annotations

import numpy as np

from browse_helpers import tiled_search_items


class _FakeArray:
    structure_family = "array"

    def __init__(self, meta: dict | None = None) -> None:
        self.metadata = meta or {}
        self._data = np.zeros((4, 4), dtype=np.uint8)


class _FakeContainer(dict):
    """Dict-backed Tiled container: ``list(node)`` -> child keys."""

    structure_family = "container"

    def __init__(self, children: dict, meta: dict | None = None) -> None:
        super().__init__(children)
        self.metadata = meta or {}

    def search(self, *_args, **_kwargs):  # pragma: no cover - unfiltered paths only
        return self


def test_sidecar_container_is_not_listed_as_a_sample() -> None:
    root = _FakeContainer({
        "dataset": _FakeContainer({"img_0001": _FakeArray()}),
        "dataset__v_thumbs": _FakeContainer({"v0001": _FakeArray()}),
        "other__masks": _FakeContainer({"m": _FakeArray()}),
    })

    result = tiled_search_items(root)

    assert [it["sample"] for it in result["items"]] == ["dataset"]
    assert result["total"] == 1


def test_n_slices_excludes_sidecar_children() -> None:
    """A 2-image dataset that also holds one image's thumbnail sidecar has 2
    slices, not 3 — matching /api/browse/slices and arrays._stack_keys."""
    root = _FakeContainer({
        "dataset": _FakeContainer({
            "img_0001": _FakeArray(),
            "img_0001__v_thumbs": _FakeContainer({"v0001": _FakeArray()}),
            "img_0002": _FakeArray(),
        }),
    })

    result = tiled_search_items(root)

    assert result["items"][0]["n_slices"] == 2


def test_single_image_with_sidecar_is_not_reported_as_multi_slice() -> None:
    """The regression that matters for the UI: n_slices must stay 1 so the
    sample opens directly instead of drilling into a Slices column."""
    root = _FakeContainer({
        "sample": _FakeContainer({
            "sample": _FakeArray(),
            "sample__v_thumbs": _FakeContainer({"v0001": _FakeArray()}),
        }),
    })

    result = tiled_search_items(root)

    assert result["items"][0]["n_slices"] == 1


def test_grouped_upload_surfaces_each_sample_not_the_batch_container() -> None:
    """Prefix-grouped ingest writes browse/<batch>/<sample>/<arrays>. Browse must
    list the samples, not a single opaque row for the batch — otherwise a grouped
    upload looks like one empty dataset and its images are unreachable."""
    root = _FakeContainer({
        "dataset": _FakeContainer({
            "almond_control": _FakeContainer({f"c_{i:02d}": _FakeArray() for i in range(1, 4)}),
            "almond_drought": _FakeContainer({f"d_{i:02d}": _FakeArray() for i in range(1, 3)}),
        }),
    })

    result = tiled_search_items(root)

    by_sample = {it["sample"]: it for it in result["items"]}
    assert sorted(by_sample) == ["almond_control", "almond_drought"]
    assert by_sample["almond_control"]["n_slices"] == 3
    assert by_sample["almond_drought"]["n_slices"] == 2
    # Paths must stay fully qualified so the image routes can resolve them.
    assert by_sample["almond_control"]["path"] == "dataset/almond_control"


def test_nested_group_recurses_instead_of_being_one_opaque_sample() -> None:
    """Regression: expansion previously stopped one level deep. A group whose
    child is ITSELF an all-container group (e.g. an upload grouped, then
    reorganised into a sub-folder) was described as a single sample whose
    "first child" is a container, not image data — n_slices=1 for something
    that is actually a whole nested batch, and opening it 422s."""
    root = _FakeContainer({
        "dataset": _FakeContainer({
            "batch_a": _FakeContainer({
                "almond_control": _FakeContainer({f"c_{i:02d}": _FakeArray() for i in range(1, 4)}),
                "almond_drought": _FakeContainer({"d_01": _FakeArray()}),
            }),
        }),
    })

    result = tiled_search_items(root)

    by_sample = {it["sample"]: it for it in result["items"]}
    assert sorted(by_sample) == ["almond_control", "almond_drought"]
    assert by_sample["almond_control"]["n_slices"] == 3
    assert by_sample["almond_control"]["path"] == "dataset/batch_a/almond_control"


def test_group_expansion_stops_at_the_recursion_depth_cap() -> None:
    """Deeper than _MAX_GROUP_EXPANSION_DEPTH, a group is described as one
    sample rather than recursed into forever — a defensive bound, not a shape
    ingest itself produces."""
    import browse_helpers

    # One level beyond the cap: batch_a -> batch_b -> real sample containers.
    innermost = _FakeContainer({"leaf": _FakeArray()})
    for _ in range(browse_helpers._MAX_GROUP_EXPANSION_DEPTH + 1):
        innermost = _FakeContainer({"inner": innermost})
    root = _FakeContainer({"dataset": innermost})

    result = tiled_search_items(root)

    # Did not crash, and produced exactly one row (the capped description) —
    # the precise shape at the cap is an implementation detail; not looping
    # forever and not raising is what matters.
    assert len(result["items"]) == 1


def test_expansion_stops_once_the_limit_is_reached() -> None:
    """Regression: the top-level loop capped how many KEYS it visited to
    `limit`, but one grouping container can expand into many rows — so the
    result could overshoot `limit` by a large factor, and `total` (len of the
    already-overshot list) was meaningless as a page-size indicator."""
    root = _FakeContainer({
        "dataset": _FakeContainer({
            f"sample_{i:02d}": _FakeContainer({"only": _FakeArray()}) for i in range(20)
        }),
    })

    result = tiled_search_items(root, limit=5)

    assert len(result["items"]) == 5
    assert result["total"] == 5


def test_a_flat_stack_is_still_one_sample() -> None:
    """The ungrouped case must not start expanding: a container of arrays is one
    sample with N slices, exactly as before."""
    root = _FakeContainer({
        "stack": _FakeContainer({f"img_{i:04d}": _FakeArray() for i in range(1, 6)}),
    })

    result = tiled_search_items(root)

    assert len(result["items"]) == 1
    assert result["items"][0]["sample"] == "stack"
    assert result["items"][0]["n_slices"] == 5


def test_empty_container_is_not_treated_as_a_group() -> None:
    root = _FakeContainer({"empty": _FakeContainer({})})

    result = tiled_search_items(root)

    assert [it["sample"] for it in result["items"]] == ["empty"]
    assert result["items"][0]["n_slices"] == 1


def test_metadata_is_still_borrowed_from_the_first_real_child() -> None:
    """A metadata-less container borrows its first child's metadata. With a
    sidecar sorting first, that must still be the first *array*, not the
    sidecar."""
    root = _FakeContainer({
        "dataset": _FakeContainer({
            "aaa__masks": _FakeContainer({"m": _FakeArray()}),
            "img_0001": _FakeArray({"sample_name": "real"}),
        }),
    })

    result = tiled_search_items(root)

    assert result["items"][0]["metadata"].get("sample_name") == "real"
