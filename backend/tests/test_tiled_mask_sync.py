"""Unit tests for tiled_mask_sync.build_mask_volumes (pure rasterization → volumes)."""
import numpy as np

from schemas import AnnotationClass, ExportSourceItem
from tiled_mask_sync import build_mask_volumes, merge_mask_volumes

H = W = 32


def _classes():
    return [
        AnnotationClass(classId=10, label="Cell", color="#ff0000"),
        AnnotationClass(classId=20, label="Wall", color="#00ff00"),
    ]


def test_build_mask_volumes_stacks_and_labels():
    slices = {
        "5": [{"id": "b", "kind": "polygon", "classId": 20,
               "points": [10, 10, 20, 10, 20, 20, 10, 20]}],
        "2": [{"id": "a", "kind": "rectangle", "classId": 10,
               "x": 2, "y": 2, "w": 6, "h": 6}],
    }
    item = ExportSourceItem(kind="tiled", source="browse/ds/img", server_uri=None, slices=slices)
    vols = build_mask_volumes(item, _classes(), {"height": H, "width": W})

    assert vols is not None
    # Sorted numeric slice order.
    assert vols["slice_indices"] == [2, 5]
    assert vols["semantic"].shape == (2, H, W)
    assert vols["semantic"].dtype == np.uint8

    # Slice 0 (key "2") = the rectangle → class id 1 (Cell); slice 1 = polygon → id 2 (Wall).
    assert vols["semantic"][0].max() == 1
    assert vols["semantic"][1].max() == 2
    assert vols["semantic"][0, 4, 4] == 1   # inside the rect
    assert vols["semantic"][1, 15, 15] == 2  # inside the polygon

    # One binary volume per class, 0/255, present only on its slice.
    assert set(vols["class_vols"]) == {"Cell", "Wall"}
    cell, wall = vols["class_vols"]["Cell"], vols["class_vols"]["Wall"]
    assert cell.shape == (2, H, W) and wall.shape == (2, H, W)
    assert set(np.unique(cell)).issubset({0, 255})
    assert cell[0].sum() > 0 and cell[1].sum() == 0   # Cell only on slice 0
    assert wall[1].sum() > 0 and wall[0].sum() == 0   # Wall only on slice 1

    # Legend mirrors the zip's legend shape.
    assert vols["legend"] == [
        {"id": 1, "name": "Cell", "color": "#ff0000"},
        {"id": 2, "name": "Wall", "color": "#00ff00"},
    ]


def test_negative_slices_emitted_as_zero_frames():
    slices = {"1": [{"id": "a", "kind": "rectangle", "classId": 10, "x": 2, "y": 2, "w": 4, "h": 4}]}
    item = ExportSourceItem(kind="tiled", source="browse/ds/img", slices=slices, negative_slices=["3"])
    vols = build_mask_volumes(item, _classes(), {"height": H, "width": W})

    assert vols is not None
    assert vols["slice_indices"] == [1, 3]
    # Negative slice 3 (index 1 in the stack) is all background.
    assert vols["semantic"][1].sum() == 0


def test_returns_none_without_slices():
    item = ExportSourceItem(kind="tiled", source="browse/ds/img", slices={})
    assert build_mask_volumes(item, _classes(), {"height": H, "width": W}) is None


def _volumes(slices, negatives=None):
    item = ExportSourceItem(
        kind="tiled", source="browse/ds/img", slices=slices, negative_slices=negatives or [],
    )
    return build_mask_volumes(item, _classes(), {"height": H, "width": W})


def test_merge_updates_pushed_slice_and_keeps_others():
    # Existing container holds slices 2 (Cell) and 5 (Wall).
    existing_vols = _volumes({
        "2": [{"id": "a", "kind": "rectangle", "classId": 10, "x": 2, "y": 2, "w": 6, "h": 6}],
        "5": [{"id": "b", "kind": "rectangle", "classId": 20, "x": 2, "y": 2, "w": 6, "h": 6}],
    })
    existing = {
        "slice_indices": existing_vols["slice_indices"],
        "semantic": existing_vols["semantic"],
        "class_arrays": existing_vols["class_vols"],
        "legend": existing_vols["legend"],
    }

    # Re-push ONLY slice 2, now as Wall (class changed on that slice).
    new = _volumes({"2": [{"id": "c", "kind": "rectangle", "classId": 20, "x": 2, "y": 2, "w": 6, "h": 6}]})
    merged = merge_mask_volumes(existing, new)

    # Both slices survive; only slice 2 reported as updated.
    assert merged["slice_indices"] == [2, 5]
    assert merged["updated_indices"] == [2]
    # Slice 2 (index 0) is now Wall (id 2), slice 5 (index 1) still Wall.
    assert merged["semantic"][0, 4, 4] == 2   # updated slice → Wall
    assert merged["semantic"][1, 4, 4] == 2   # untouched slice preserved
    # Slice 2's Cell mask was replaced (now empty there); Wall present on both.
    assert merged["class_vols"]["Cell"][0].sum() == 0
    assert merged["class_vols"]["Wall"][0].sum() > 0


def test_merge_fresh_when_no_existing():
    new = _volumes({"3": [{"id": "a", "kind": "rectangle", "classId": 10, "x": 1, "y": 1, "w": 4, "h": 4}]})
    merged = merge_mask_volumes(None, new)
    assert merged["slice_indices"] == [3]
    assert merged["updated_indices"] == [3]
    assert merged["semantic"].shape == (1, H, W)
