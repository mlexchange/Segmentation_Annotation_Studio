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


def test_write_lightly_split_mask_is_single_channel_passthrough(tmp_path) -> None:
    """write_lightly_split writes the given label PNG verbatim as a single-channel mask
    (the DINOv3 255/0-indexed convention is applied upstream in build_export_plan)."""
    from coco_export import write_lightly_split

    label = np.full((16, 16), 255, dtype=np.uint8)
    label[4:8, 4:8] = 0
    label[10:14, 10:14] = 1
    write_lightly_split(tmp_path / "val", [_mk_image("s_0000.png", label)])

    m = PILImage.open(tmp_path / "val" / "masks" / "s_0000.png")
    assert m.mode == "L"  # single-channel integer
    arr = np.array(m)
    assert arr[0, 0] == 255      # unannotated → ignore index
    assert arr[5, 5] == 0        # class 0 (0-indexed)
    assert arr[12, 12] == 1      # class 1
    assert set(np.unique(arr)).issubset({0, 1, 255})


def test_lightly_classes_map() -> None:
    """classes.json maps index→name, 0-indexed and contiguous, with NO background."""
    from coco_export import lightly_classes_map

    cats = [{"id": 1, "name": "cell"}, {"id": 2, "name": "wall"}]
    m = lightly_classes_map(cats)
    assert m == {"0": "cell", "1": "wall"}
    assert "background" not in m.values()
    # round-trips as JSON (Lightly accepts a path to this)
    assert json.loads(json.dumps(m))["0"] == "cell"


def test_build_export_plan_lightly_labelmap() -> None:
    """In lightly mode, build_export_plan's label map is 0-indexed with unannotated=255."""
    import numpy as np

    from coco_export import build_export_plan
    from schemas import AnnotationClass, ExportRequest, RenderOpts

    h, w = 40, 40
    # Two classes; a polygon of the SECOND class (classId=7) → internal cat id 2 → pixel 1.
    poly = {"id": "s1", "kind": "polygon", "classId": 7, "points": [5, 5, 25, 5, 25, 25, 5, 25]}
    payload = ExportRequest(
        kind="local", source="vol", server_uri=None,
        slices={"0": [poly]}, split_by_slice={"0": "train"}, negative_slices=[],
        classes=[AnnotationClass(classId=3, label="cell", color="#00ff00"),
                 AnnotationClass(classId=7, label="wall", color="#ff0000")],
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
        lightly=True,
    )
    img = plan["splits"]["train"]["images"][0]
    arr = np.array(PILImage.open(io.BytesIO(img["label_png_bytes"])))
    assert arr[0, 0] == 255                       # unannotated → ignore
    assert arr[15, 15] == 1                        # 'wall' is the 2nd class → pixel 1 (0-indexed)
    assert set(np.unique(arr)).issubset({0, 1, 255})


def test_fold_lightly_splits_merges_valid_and_test() -> None:
    """valid + test fold into a single 'val' bucket; train stays; images concatenated."""
    from coco_export import fold_lightly_splits

    merged = {
        "train": {"images": [{"file_name": "a"}]},
        "valid": {"images": [{"file_name": "b"}]},
        "test": {"images": [{"file_name": "c"}]},
    }
    folded = fold_lightly_splits(merged)
    assert set(folded) == {"train", "val"}
    assert [i["file_name"] for i in folded["train"]["images"]] == ["a"]
    assert sorted(i["file_name"] for i in folded["val"]["images"]) == ["b", "c"]
