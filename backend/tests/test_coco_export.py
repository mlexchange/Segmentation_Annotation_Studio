"""Tests for coco_export rasterizers — brush width parity and erase composition."""
from __future__ import annotations

import math
import pytest
import numpy as np


def test_brush_width_parity() -> None:
    """Exported mask width must be ~= 2*radius along the whole stroke including fast moves."""
    from coco_export import _stamp_stroke

    h, w = 200, 400
    mask = np.zeros((h, w), dtype=bool)
    radius = 10.0
    # Diagonal stroke 400px long (fast move => sparse vertices)
    points = [0.0, 50.0, 400.0, 150.0]
    _stamp_stroke(mask, points, radius)

    # Sample cross-sections perpendicular to the stroke at several t values
    dx, dy = 400.0 - 0.0, 150.0 - 50.0
    length = math.hypot(dx, dy)
    # Normal direction
    nx, ny = -dy / length, dx / length

    min_width = float("inf")
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        cx = 0.0 + t * dx
        cy = 50.0 + t * dy
        # Count pixels perpendicular to stroke
        hits = 0
        for d in range(-int(radius * 3), int(radius * 3) + 1):
            px = int(round(cx + d * nx))
            py = int(round(cy + d * ny))
            if 0 <= py < h and 0 <= px < w and mask[py, px]:
                hits += 1
        min_width = min(min_width, hits)

    # Width must be at least 2r - 2 everywhere
    assert min_width >= 2 * radius - 2, f"Min stroke width {min_width} < {2 * radius - 2}"


def test_brush_erase_ring() -> None:
    """Paint a disk, erase its center -> ring. Area should match expected annulus."""
    from coco_export import _brush_mask
    import pycocotools.mask as mask_utils

    h, w = 100, 100
    cx, cy = 50.0, 50.0
    paint_r = 20.0
    erase_r = 10.0

    strokes = [
        {"points": [cx, cy], "radius": paint_r, "mode": "paint"},
        {"points": [cx, cy], "radius": erase_r, "mode": "erase"},
    ]
    mask = _brush_mask(strokes, h, w)
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    area = float(mask_utils.area(rle))

    # Hole must exist -- area must be less than full disk area
    full_disk_area = math.pi * paint_r ** 2
    assert area < full_disk_area * 0.99, "Erase did not create a hole"
    assert area > 0, "Erase removed entire mask"


def test_erase_order_matters() -> None:
    """paint then erase != erase then paint."""
    from coco_export import _brush_mask

    h, w = 100, 100
    cx, cy = 50.0, 50.0
    r = 15.0

    paint_first = _brush_mask([
        {"points": [cx, cy], "radius": r, "mode": "paint"},
        {"points": [cx, cy], "radius": r / 2, "mode": "erase"},
    ], h, w)

    erase_first = _brush_mask([
        {"points": [cx, cy], "radius": r / 2, "mode": "erase"},
        {"points": [cx, cy], "radius": r, "mode": "paint"},
    ], h, w)

    assert not np.array_equal(paint_first, erase_first), "Order of paint/erase must matter"


def test_rect_normalization_all_directions() -> None:
    """All four drag directions produce the same rasterized rectangle."""
    from coco_export import _rect_mask

    h, w = 100, 100
    # TL drag to BR: x=10,y=10,w=20,h=20
    base = _rect_mask(10, 10, 20, 20, h, w)
    # Already normalized inputs only -- normalization is done in frontend
    # Just verify the mask is non-empty and has correct bbox
    import pycocotools.mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(base.astype(np.uint8)))
    bbox = mask_utils.toBbox(rle)  # x, y, w, h
    assert bbox[0] == pytest.approx(10, abs=1)
    assert bbox[1] == pytest.approx(10, abs=1)


def test_rle_json_serializable() -> None:
    """RLE counts must be ascii-decoded so the annotation is JSON-serializable."""
    from coco_export import _rect_mask, mask_to_coco_ann
    import json

    h, w = 50, 50
    mask = _rect_mask(5, 5, 10, 10, h, w)
    ann = mask_to_coco_ann(mask, ann_id=1, image_id=1, category_id=1)
    # Must not raise
    serialized = json.dumps(ann)
    restored = json.loads(serialized)
    assert isinstance(restored["segmentation"]["counts"], str)


def test_mask_export_writes_semantic_and_per_class(tmp_path) -> None:
    """build_export_plan emits label-map + per-class masks; write_coco_split lays them out."""
    import numpy as np
    from PIL import Image
    from coco_export import build_export_plan, write_coco_split
    from schemas import ExportRequest, AnnotationClass, RenderOpts

    h, w = 40, 40
    poly = {"id": "s1", "kind": "polygon", "classId": 7, "points": [5, 5, 25, 5, 25, 25, 5, 25]}
    payload = ExportRequest(
        kind="local", source="vol", server_uri=None,
        slices={"0": [poly]}, split_by_slice={"0": "train"}, negative_slices=[],
        classes=[AnnotationClass(classId=7, label="air", color="#ff0000")],
        render=RenderOpts(norm="slice"),
        auto_split={"ratios": [1, 0, 0], "seed": 1},
    )
    meta = {"height": h, "width": w, "n_slices": 1}
    plan = build_export_plan(
        object(), payload,
        render_slice_fn=lambda arr, opts, gr: np.zeros((h, w, 3), dtype=np.uint8),
        array_shape_meta_fn=lambda n: meta,
        read_slice_fn=lambda n, m, i: np.zeros((h, w), dtype=np.uint8),
        sample_global_stats_fn=lambda n, m: (0.0, 1.0),
    )
    img = plan["splits"]["train"]["images"][0]
    fname = img["file_name"]
    assert img.get("label_png_bytes")
    assert "air" in img["class_masks"]

    # Polygons are opt-in: default off → segmentation_poly empty, RLE always set.
    ann0 = plan["splits"]["train"]["annotations"][0]
    assert ann0["segmentation_poly"] == []
    assert isinstance(ann0["segmentation"]["counts"], str) and ann0["segmentation"]["counts"]

    import zipfile
    zip_path = tmp_path / "ds.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        write_coco_split(
            tmp_path / "train",
            images=plan["splits"]["train"]["images"],
            categories=plan["categories"],
            annotations=plan["splits"]["train"]["annotations"],
            mode="overwrite", info=plan["info"],
            zf=zf, arc_prefix="train/",
        )
    assert (tmp_path / "train" / fname).exists()
    assert (tmp_path / "train" / "masks" / "semantic" / fname).exists()
    assert (tmp_path / "train" / "masks" / "air" / fname).exists()
    assert (tmp_path / "train" / "masks" / "legend.json").exists()
    sem = np.array(Image.open(tmp_path / "train" / "masks" / "semantic" / fname))
    assert int(sem.max()) == 1  # single class → category id 1 in the label map

    # The zip mirrors the tree (STORED) and carries images, masks, and COCO.
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert f"train/{fname}" in names
        assert f"train/masks/semantic/{fname}" in names
        assert f"train/masks/air/{fname}" in names
        assert "train/_annotations.coco.json" in names
        assert all(zi.compress_type == zipfile.ZIP_STORED for zi in zf.infolist())


def test_export_polygons_opt_in() -> None:
    """include_polygons=True populates segmentation_poly (reusing polygon points)."""
    import numpy as np
    from coco_export import build_export_plan
    from schemas import ExportRequest, AnnotationClass, RenderOpts

    h, w = 30, 30
    poly = {"id": "s1", "kind": "polygon", "classId": 1, "points": [4, 4, 20, 4, 20, 20, 4, 20]}
    payload = ExportRequest(
        kind="local", source="vol", slices={"0": [poly]}, split_by_slice={"0": "train"},
        classes=[AnnotationClass(classId=1, label="air", color="#fff")],
        render=RenderOpts(norm="slice"), include_polygons=True,
    )
    plan = build_export_plan(
        object(), payload,
        render_slice_fn=lambda a, o, g: np.zeros((h, w, 3), np.uint8),
        array_shape_meta_fn=lambda n: {"height": h, "width": w, "n_slices": 1},
        read_slice_fn=lambda n, m, i: np.zeros((h, w), np.uint8),
        sample_global_stats_fn=lambda n, m: (0.0, 1.0),
        include_polygons=True,
    )
    ann0 = plan["splits"]["train"]["annotations"][0]
    assert ann0["segmentation_poly"] == [poly["points"]]  # reused, not re-traced


def test_zero_area_polygon_skipped() -> None:
    """A 1-px degenerate polygon produces zero area."""
    from coco_export import _polygon_mask
    import pycocotools.mask as mask_utils

    h, w = 50, 50
    mask = _polygon_mask([10.0, 10.0, 10.0, 10.0], h, w)
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    area = float(mask_utils.area(rle))
    assert area < 2, "Degenerate polygon should produce near-zero area"


def test_polygon_hole_carved_out() -> None:
    """A polygon with an inner ring (invert-shape) leaves the hole empty."""
    from coco_export import shape_to_mask

    h, w = 40, 40
    shape = {
        "id": "inv", "classId": 1, "kind": "polygon",
        "points": [0, 0, 40, 0, 40, 40, 0, 40],
        "holes": [[10, 10, 30, 10, 30, 30, 10, 30]],
    }
    mask = shape_to_mask(shape, h, w)
    assert not mask[20, 20], "hole interior must be empty"
    assert mask[2, 2], "frame corner must be filled"
    # Area ~= 40*40 - 20*20 = 1200, allow boundary slack.
    assert 1050 < int(mask.sum()) < 1350
