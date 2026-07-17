"""Tests for the DINOv3 / Lightly semantic-segmentation export writer."""
from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image as PILImage


def _mk_image(file_name: str, label: np.ndarray) -> dict:
    """A build_export_plan-style image dict with png + label-map payloads."""
    from coco_export import _encode_png

    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(rgb).save(buf, format="PNG")
    return {
        "file_name": file_name,
        "height": h,
        "width": w,
        "png_bytes": buf.getvalue(),
        "label_png_bytes": _encode_png(label.astype(np.uint8)),
    }


def test_write_lightly_split_matching_stems(tmp_path) -> None:
    """images/<stem>.png and masks/<stem>.png share the exact filename."""
    from coco_export import write_lightly_split

    label = np.zeros((16, 16), dtype=np.uint8)
    label[4:8, 4:8] = 1
    label[10:14, 10:14] = 2
    images = [_mk_image("sample_0000.png", label), _mk_image("sample_0001.png", label)]

    summary = write_lightly_split(tmp_path / "train", images)
    assert summary["n_images"] == 2

    img_dir = tmp_path / "train" / "images"
    mask_dir = tmp_path / "train" / "masks"
    imgs = sorted(p.name for p in img_dir.iterdir())
    masks = sorted(p.name for p in mask_dir.iterdir())
    assert imgs == masks == ["sample_0000.png", "sample_0001.png"]


def test_write_lightly_split_mask_is_index_labelmap(tmp_path) -> None:
    """Mask PNGs are single-channel with pixel value == class index."""
    from coco_export import write_lightly_split

    label = np.zeros((16, 16), dtype=np.uint8)
    label[4:8, 4:8] = 1
    label[10:14, 10:14] = 2
    write_lightly_split(tmp_path / "val", [_mk_image("s_0000.png", label)])

    m = PILImage.open(tmp_path / "val" / "masks" / "s_0000.png")
    assert m.mode == "L"  # single-channel integer
    arr = np.array(m)
    assert arr[0, 0] == 0        # background
    assert arr[5, 5] == 1        # class 1
    assert arr[12, 12] == 2      # class 2
    assert set(np.unique(arr)).issubset({0, 1, 2})


def test_lightly_classes_map(tmp_path) -> None:
    """classes.json maps index→name with 0=background and contiguous ids."""
    from coco_export import lightly_classes_map

    cats = [{"id": 1, "name": "cell"}, {"id": 2, "name": "wall"}]
    m = lightly_classes_map(cats)
    assert m == {"0": "background", "1": "cell", "2": "wall"}
    # round-trips as JSON (Lightly accepts a path to this)
    assert json.loads(json.dumps(m))["1"] == "cell"
