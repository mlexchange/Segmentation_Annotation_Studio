"""COCO + sidecar import: round-trip back to annotation payload.

Prefers the _studio_shapes.json sidecar (lossless vector shapes).
Falls back to reconstructing PolygonShapes from segmentation_poly or RLE contours.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def import_dataset(dataset_dir: str) -> dict[str, Any]:
    """Load a COCO dataset directory back into editor payload format.

    Args:
        dataset_dir: Path to a split directory containing _annotations.coco.json.

    Returns:
        Dict with keys: classes, slices, split_by_slice, negative_slices, render.

    Raises:
        FileNotFoundError: If the annotations file does not exist.
        ValueError: If the annotations file is not valid JSON.
    """
    base = Path(dataset_dir)
    coco_path = base / "_annotations.coco.json"
    sidecar_path = base / "_studio_shapes.json"

    if not coco_path.exists():
        raise FileNotFoundError(f"No _annotations.coco.json in {dataset_dir!r}")

    coco = json.loads(coco_path.read_text())
    has_sidecar = sidecar_path.exists()

    if has_sidecar:
        return _import_with_sidecar(coco, json.loads(sidecar_path.read_text()))
    return _import_from_coco(coco)


def _import_with_sidecar(coco: dict[str, Any], sidecar: dict[str, Any]) -> dict[str, Any]:
    """Lossless import using _studio_shapes.json."""
    classes = sidecar.get("classes", [])
    render = sidecar.get("render", {})
    images_meta = sidecar.get("images", {})

    slices: dict[str, list[dict[str, Any]]] = {}
    split_by_slice: dict[str, str] = {}

    # Map file_name -> split from COCO split dirs
    for fname, img_meta in images_meta.items():
        source_key = img_meta.get("source_key", "")
        slice_index = str(img_meta.get("slice_index", 0))
        shapes = img_meta.get("shapes", [])

        if source_key not in slices:
            slices[source_key] = {}
        slices[source_key][slice_index] = shapes  # type: ignore[assignment]

    return {
        "classes": classes,
        "slices": slices,
        "split_by_slice": split_by_slice,
        "negative_slices": [],
        "render": render,
    }


def _import_from_coco(coco: dict[str, Any]) -> dict[str, Any]:
    """Best-effort import from COCO only (no sidecar).

    Brush shapes become polygons (stated limitation).
    """
    import pycocotools.mask as mask_utils
    from skimage import measure

    images_by_id = {img["id"]: img for img in coco.get("images", [])}

    classes = [
        {"classId": c["id"], "label": c["name"], "color": "#1f77b4", "isVisible": True}
        for c in coco.get("categories", [])
    ]

    # Group annotations by image
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    slices: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for img_id, img in images_by_id.items():
        source_key = img.get("source_key", img["file_name"])
        slice_index = str(img.get("slice_index", 0))
        shapes: list[dict[str, Any]] = []

        for ann in anns_by_image.get(img_id, []):
            class_id = ann.get("category_id", 1)
            seg = ann.get("segmentation_poly") or ann.get("segmentation")

            if isinstance(seg, list) and seg:
                for poly_pts in seg:
                    if isinstance(poly_pts, list) and len(poly_pts) >= 6:
                        shapes.append({
                            "id": f"imported_{ann['id']}",
                            "classId": class_id,
                            "kind": "polygon",
                            "points": poly_pts,
                        })
            elif isinstance(seg, dict):
                # RLE -> contours
                try:
                    h, w = seg["size"]
                    rle_obj = {"size": seg["size"], "counts": seg["counts"].encode("ascii") if isinstance(seg["counts"], str) else seg["counts"]}
                    mask = mask_utils.decode(rle_obj).astype(bool)
                    for contour in measure.find_contours(mask.astype(float), 0.5):
                        pts = np.flip(contour, axis=1).ravel().tolist()
                        if len(pts) >= 6:
                            shapes.append({
                                "id": f"imported_{ann['id']}",
                                "classId": class_id,
                                "kind": "polygon",
                                "points": [round(p, 2) for p in pts],
                            })
                except Exception as exc:
                    logger.warning("Failed to decode RLE for ann %d: %s", ann["id"], exc)

        if source_key not in slices:
            slices[source_key] = {}
        slices[source_key][slice_index] = shapes

    return {
        "classes": classes,
        "slices": slices,
        "split_by_slice": {},
        "negative_slices": [],
        "render": coco.get("info", {}).get("render", {}),
    }
