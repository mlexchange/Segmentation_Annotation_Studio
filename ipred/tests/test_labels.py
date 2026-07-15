"""Label rasterization accepts studio shape wire formats."""

from __future__ import annotations

import numpy as np

from ipred.labels import build_label_map, shape_to_mask


def test_polygon_flat_xy_list() -> None:
    """Studio polygons store points as flat [x0,y0,x1,y1,...]."""
    shape = {
        "kind": "polygon",
        "classId": 6,
        "points": [10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0],
    }
    mask = shape_to_mask(shape, 64, 64)
    assert int(mask.sum()) > 100

    labels = build_label_map([shape], 64, 64)
    assert int((labels == 6).sum()) == int(mask.sum())


def test_polygon_nested_pairs_still_work() -> None:
    shape = {
        "kind": "polygon",
        "classId": 1,
        "points": [[10, 10], [40, 10], [40, 40], [10, 40]],
    }
    assert shape_to_mask(shape, 64, 64).sum() > 100


def test_brush_flat_points_and_radius() -> None:
    """Studio brush strokes use flat points + radius (not nested {x,y})."""
    shape = {
        "kind": "brush",
        "classId": 2,
        "strokes": [
            {"points": [8.0, 8.0, 20.0, 20.0, 32.0, 8.0], "radius": 4.0, "mode": "paint"},
        ],
    }
    mask = shape_to_mask(shape, 48, 48)
    assert int(mask.sum()) > 20
    labels = build_label_map([shape], 48, 48)
    assert np.any(labels == 2)
