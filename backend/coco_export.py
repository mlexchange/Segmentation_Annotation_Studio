"""Shape rasterization and COCO dataset writing for SAM3 fine-tuning.

Ports mlex ShapeConversion rasterizers, replacing matplotlib contains_points
(O(H*W) per shape) with skimage.draw, and fixing the v1 brush and rectangle bugs.
"""
from __future__ import annotations

import io
import json
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image as PILImage
from skimage import draw, measure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shape -> binary mask
# ---------------------------------------------------------------------------

def _polygon_mask(points: list[float], h: int, w: int) -> np.ndarray:
    """Rasterize a closed polygon given flat [x0,y0,x1,y1,...] coords."""
    xs = np.asarray(points[0::2])
    ys = np.asarray(points[1::2])
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = draw.polygon(ys, xs, shape=(h, w))
    mask[rr, cc] = True
    return mask


def _rect_mask(x: float, y: float, rw: float, rh: float, h: int, w: int) -> np.ndarray:
    """Rasterize a normalized (w,h >= 0) rectangle via start/end like mlex."""
    x0 = int(np.clip(round(x), 0, w - 1))
    y0 = int(np.clip(round(y), 0, h - 1))
    x1 = int(np.clip(round(x + rw), 0, w - 1))
    y1 = int(np.clip(round(y + rh), 0, h - 1))
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = draw.rectangle(start=(y0, x0), end=(y1, x1))
    mask[rr.astype(int), cc.astype(int)] = True
    return mask


def _ellipse_mask(cx: float, cy: float, rx: float, ry: float, h: int, w: int) -> np.ndarray:
    """Rasterize an axis-aligned ellipse (radii pre-normalized >= 0)."""
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = draw.ellipse(cy, cx, max(ry, 0.5), max(rx, 0.5), shape=(h, w))
    mask[rr, cc] = True
    return mask


def _stamp_stroke(mask: np.ndarray, points: list[float], radius: float) -> None:
    """Stamp disks of radius along a polyline at <= radius/2 spacing (in-place).

    Fixes the v1 "beads on a hairline" bug: a disk per recorded vertex joined
    by 1-px lines exported strokes far thinner than the on-screen
    strokeWidth = 2 * radius rendering on fast mouse moves.
    """
    h, w = mask.shape
    xs = np.asarray(points[0::2], dtype=float)
    ys = np.asarray(points[1::2], dtype=float)
    r = max(1.0, float(radius))
    step = max(0.5, r / 2.0)
    for i in range(len(xs)):
        rr, cc = draw.disk((ys[i], xs[i]), r, shape=(h, w))
        mask[rr, cc] = True
    for i in range(len(xs) - 1):
        dist = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        if dist < 1e-9:
            continue
        for k in range(1, int(dist // step) + 1):
            t = k * step / dist
            cy = ys[i] + t * (ys[i + 1] - ys[i])
            cx = xs[i] + t * (xs[i + 1] - xs[i])
            rr, cc = draw.disk((cy, cx), r, shape=(h, w))
            mask[rr, cc] = True


def _brush_mask(strokes: list[dict[str, Any]], h: int, w: int) -> np.ndarray:
    """Compose ordered paint/erase strokes into one instance mask.

    Paint strokes OR pixels in; erase strokes AND them out.
    Order matters and matches the on-canvas destination-out rendering exactly.
    """
    mask = np.zeros((h, w), dtype=bool)
    for stroke in strokes:
        stamp = np.zeros((h, w), dtype=bool)
        _stamp_stroke(stamp, stroke["points"], stroke["radius"])
        if stroke.get("mode", "paint") == "erase":
            mask &= ~stamp
        else:
            mask |= stamp
    return mask


def _apply_erased(mask: np.ndarray, erased: list[dict[str, Any]] | None) -> np.ndarray:
    """Subtract erase carve-outs from a vector shape's mask (in-place-safe)."""
    if not erased:
        return mask
    for stroke in erased:
        stamp = np.zeros(mask.shape, dtype=bool)
        _stamp_stroke(stamp, stroke["points"], stroke["radius"])
        mask &= ~stamp
    return mask


def shape_to_mask(shape: dict[str, Any], h: int, w: int) -> np.ndarray:
    """Rasterize one Shape (image-pixel coords) to an (h, w) boolean mask."""
    kind = shape["kind"]
    if kind == "polygon":
        mask = _polygon_mask(shape["points"], h, w)
        # Carve inner rings (holes), e.g. from "invert shape", so exports match.
        for hole in shape.get("holes") or []:
            mask &= ~_polygon_mask(hole, h, w)
    elif kind == "rectangle":
        mask = _rect_mask(shape["x"], shape["y"], shape["w"], shape["h"], h, w)
    elif kind == "ellipse":
        mask = _ellipse_mask(shape["cx"], shape["cy"], shape["rx"], shape["ry"], h, w)
    elif kind == "brush":
        # Brush erase strokes are part of its own stroke list, not `erased`.
        return _brush_mask(shape["strokes"], h, w)
    else:
        raise ValueError(f"Unknown shape kind: {kind!r}")
    # Vector shapes can carry eraser carve-outs applied after rasterization.
    return _apply_erased(mask, shape.get("erased"))


# ---------------------------------------------------------------------------
# Mask -> COCO annotation
# ---------------------------------------------------------------------------

def _encode_png(arr: np.ndarray, compress_level: int = 0) -> bytes:
    """Encode a single-channel uint8 array as a grayscale PNG.

    Masks are tiny and pre-binarized; ``compress_level=0`` (store) is the fastest
    and the size cost is negligible.
    """
    buf = io.BytesIO()
    PILImage.fromarray(arr, mode="L").save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


def _safe_name(name: str) -> str:
    """Filesystem-safe folder name for a class label."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "class"


def _mask_to_polygons(mask: np.ndarray, min_pts: int = 6) -> list[list[float]]:
    """Outer+inner contours as COCO-style flat polygons (holes NOT encoded -- see RLE)."""
    polys: list[list[float]] = []
    for contour in measure.find_contours(mask.astype(float), 0.5):
        flat = np.flip(contour, axis=1).ravel().tolist()
        if len(flat) >= min_pts:
            polys.append([round(v, 2) for v in flat])
    return polys


def mask_to_coco_ann(
    mask: np.ndarray,
    ann_id: int,
    image_id: int,
    category_id: int,
    *,
    include_polygons: bool = False,
    poly_override: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Build a COCO annotation: RLE in segmentation, optional polygon copy.

    RLE is exact (holes, multiple components) and is what SAM3's pycocotools
    segm path consumes — always present. The polygon copy (``segmentation_poly``)
    is a convenience for external viewers; generating it via ``find_contours`` is
    costly, so it is opt-in. ``poly_override`` lets a polygon-kind shape reuse its
    own points instead of re-tracing the mask.
    """
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    if not include_polygons:
        seg_poly: list[list[float]] = []
    elif poly_override is not None:
        seg_poly = poly_override
    else:
        seg_poly = _mask_to_polygons(mask)
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
        "area": float(mask_utils.area(rle)),
        "bbox": [float(v) for v in mask_utils.toBbox(rle)],
        "segmentation": rle,
        "segmentation_poly": seg_poly,
    }


# ---------------------------------------------------------------------------
# Merge-safe dataset writer
# ---------------------------------------------------------------------------

class ExportConflict(Exception):
    """Raised when a merge would corrupt category ids or names."""


def _resolve_split(
    slice_keys: list[str],
    split_by_slice: dict[str, str],
    auto_split: dict[str, Any],
) -> dict[str, str]:
    """Resolve 'auto' entries using a seeded ratio split."""
    ratios = auto_split.get("ratios", [0.8, 0.1, 0.1])
    seed = auto_split.get("seed", 1234)
    splits_out = dict(split_by_slice)
    auto_keys = [k for k in slice_keys if splits_out.get(k, "auto") == "auto"]
    if auto_keys:
        rng = random.Random(seed)
        rng.shuffle(auto_keys)
        n = len(auto_keys)
        n_train = int(n * ratios[0])
        n_valid = int(n * ratios[1])
        labels = ["train"] * n_train + ["valid"] * n_valid + ["test"] * (n - n_train - n_valid)
        for k, lbl in zip(auto_keys, labels):
            splits_out[k] = lbl
    return splits_out


def write_coco_split(
    split_dir: Path,
    images: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    mode: str = "fail",
    info: dict[str, Any] | None = None,
    zf: Any = None,
    arc_prefix: str = "",
) -> dict[str, Any]:
    """Write/merge one split directory.

    If ``zf`` (an open ``zipfile.ZipFile``) is given, every file is also added to
    it under ``arc_prefix`` from the same in-memory bytes — a single pass with no
    disk re-read (used to build the download .zip cheaply).

    In merge mode existing image entries are matched by file_name:
    matched images are replaced (their old annotations dropped),
    new images get ids above the existing max; annotation ids likewise
    re-baselined. Categories are matched by name -- id collisions with
    different names abort with a clear error rather than silently corrupting.

    Args:
        split_dir: Destination directory (created if needed).
        images: List of image dicts with file_name, height, width, png_bytes.
        categories: List of category dicts with id, name.
        annotations: List of annotation dicts.
        mode: 'fail' | 'overwrite' | 'merge'.
        info: Optional COCO info block.

    Returns:
        Summary dict with n_images, n_annotations, written paths.

    Raises:
        FileExistsError: mode='fail' and annotations file already exists.
        ExportConflict: merge category name/id conflict.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    coco_path = split_dir / "_annotations.coco.json"

    def _emit(rel: str, data: bytes) -> None:
        """Mirror a just-written file into the download zip (if building one)."""
        if zf is not None:
            zf.writestr(arc_prefix + rel, data)

    existing_coco: dict[str, Any] = {"images": [], "annotations": [], "categories": []}
    if coco_path.exists():
        if mode == "fail":
            raise FileExistsError(f"{coco_path} already exists (use overwrite or merge)")
        if mode == "merge":
            try:
                existing_coco = json.loads(coco_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Existing COCO file corrupt -- treating as empty")

    # Category merge: match by name
    merged_cats = {c["name"]: c for c in existing_coco.get("categories", [])}
    for new_cat in categories:
        if new_cat["name"] in merged_cats:
            if merged_cats[new_cat["name"]]["id"] != new_cat["id"]:
                # Remap to existing id -- no conflict unless names differ
                pass
        else:
            # Assign id above current max
            max_id = max((c["id"] for c in merged_cats.values()), default=0)
            merged_cats[new_cat["name"]] = {**new_cat, "id": max_id + 1}
    final_cats = sorted(merged_cats.values(), key=lambda c: c["id"])

    # Build name->id mapping for annotation remapping
    name_to_id = {c["name"]: c["id"] for c in final_cats}
    old_cat_map = {c["id"]: c["name"] for c in categories}

    # Image merge
    existing_by_fname: dict[str, dict[str, Any]] = {
        img["file_name"]: img for img in existing_coco.get("images", [])
    }
    existing_anns: list[dict[str, Any]] = list(existing_coco.get("annotations", []))
    replaced_image_ids: set[int] = set()

    new_image_fnames = {img["file_name"] for img in images}
    for fname in new_image_fnames:
        if fname in existing_by_fname:
            replaced_image_ids.add(existing_by_fname[fname]["id"])

    kept_images = [img for img in existing_coco.get("images", []) if img["id"] not in replaced_image_ids]
    kept_anns = [a for a in existing_anns if a["image_id"] not in replaced_image_ids]

    max_img_id = max((img["id"] for img in kept_images), default=0)
    max_ann_id = max((a["id"] for a in kept_anns), default=0)

    out_images: list[dict[str, Any]] = list(kept_images)
    out_anns: list[dict[str, Any]] = list(kept_anns)

    for img in images:
        max_img_id += 1
        img_id = max_img_id
        png_bytes: bytes | None = img.pop("png_bytes", None)
        # Pop mask payloads so they don't leak into the COCO json.
        label_png: bytes | None = img.pop("label_png_bytes", None)
        class_masks: dict[str, bytes] = img.pop("class_masks", {}) or {}
        out_img = {**img, "id": img_id}
        out_images.append(out_img)

        fname = img["file_name"]
        # Write PNG (+ mirror into zip)
        if png_bytes is not None:
            (split_dir / fname).write_bytes(png_bytes)
            _emit(fname, png_bytes)

        # Write masks: masks/semantic/<file> (label map) + masks/<class>/<file>.
        if label_png is not None:
            sem_dir = split_dir / "masks" / "semantic"
            sem_dir.mkdir(parents=True, exist_ok=True)
            (sem_dir / fname).write_bytes(label_png)
            _emit(f"masks/semantic/{fname}", label_png)
        for cname, cbytes in class_masks.items():
            safe = _safe_name(cname)
            cls_dir = split_dir / "masks" / safe
            cls_dir.mkdir(parents=True, exist_ok=True)
            (cls_dir / fname).write_bytes(cbytes)
            _emit(f"masks/{safe}/{fname}", cbytes)

        # Write annotations for this image
        img_anns = [a for a in annotations if a.get("_image_file_name") == img["file_name"]]
        for ann in img_anns:
            max_ann_id += 1
            # Remap category id via name
            orig_cat_id = ann.get("category_id", 1)
            cat_name = old_cat_map.get(orig_cat_id, str(orig_cat_id))
            new_cat_id = name_to_id.get(cat_name, orig_cat_id)
            clean_ann = {k: v for k, v in ann.items() if not k.startswith("_")}
            out_anns.append({**clean_ann, "id": max_ann_id, "image_id": img_id, "category_id": new_cat_id})

    coco_doc = {
        "info": info or {
            "description": "SAM3 fine-tune dataset -- Segmentation Annotation Studio",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "images": out_images,
        "categories": final_cats,
        "annotations": out_anns,
    }
    coco_json = json.dumps(coco_doc, indent=2)
    coco_path.write_text(coco_json)
    _emit("_annotations.coco.json", coco_json.encode("utf-8"))

    # Legend mapping semantic label index -> class name/color (for mask viewers).
    masks_dir = split_dir / "masks"
    if masks_dir.exists():
        legend = [{"id": c["id"], "name": c["name"], "color": c.get("color")} for c in final_cats]
        legend_json = json.dumps(legend, indent=2)
        (masks_dir / "legend.json").write_text(legend_json)
        _emit("masks/legend.json", legend_json.encode("utf-8"))

    return {
        "n_images": len(images),
        "n_annotations": sum(1 for a in out_anns if a["image_id"] in {img["id"] for img in out_images[-len(images):]}),
        "path": str(coco_path),
    }


def write_lightly_split(
    split_dir: Path,
    images: list[dict[str, Any]],
    *,
    zf: Any = None,
    arc_prefix: str = "",
) -> dict[str, Any]:
    """Write one split in the DINOv3 / Lightly semantic-segmentation layout:
    ``<split>/images/<stem>.png`` (rendered frame) + ``<split>/masks/<stem>.png``
    (single-channel label map, pixel = class index, 0 = background) with MATCHING
    filename stems. Reuses the ``png_bytes`` / ``label_png_bytes`` already built by
    ``build_export_plan`` (the same rasterization as the COCO/semantic export).

    If ``zf`` is given, each file is mirrored into the download zip under
    ``arc_prefix`` from the same bytes.
    """
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    def _emit(rel: str, data: bytes) -> None:
        if zf is not None:
            zf.writestr(arc_prefix + rel, data)

    n = 0
    for img in images:
        fname = img["file_name"]  # e.g. "sample_0003.png"
        png_bytes = img.get("png_bytes")
        label_png = img.get("label_png_bytes")
        if png_bytes is not None:
            (img_dir / fname).write_bytes(png_bytes)
            _emit(f"images/{fname}", png_bytes)
        if label_png is not None:
            (mask_dir / fname).write_bytes(label_png)
            _emit(f"masks/{fname}", label_png)
        n += 1
    return {"n_images": n, "n_annotations": 0, "path": str(img_dir)}


# Lightly/DINOv3 uses train/val only: the app's 'valid' AND 'test' both fold into 'val'.
LIGHTLY_SPLIT_DIRS = {"train": "train", "valid": "val", "test": "val"}


def fold_lightly_splits(merged_splits: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map the app's train/valid/test splits onto Lightly's train/val directories,
    merging the image lists of any splits that map to the same directory (valid+test
    → val). Returns ``{dir_name: {"images": [...]}}``."""
    folded: dict[str, dict[str, Any]] = {}
    for split_name, split_data in merged_splits.items():
        dir_name = LIGHTLY_SPLIT_DIRS.get(split_name, split_name)
        folded.setdefault(dir_name, {"images": []})["images"].extend(split_data.get("images", []))
    return folded


def lightly_classes_map(categories: list[dict[str, Any]]) -> dict[str, str]:
    """Build the Lightly/DINOv3 ``classes`` mapping (index → name), 0-indexed and
    contiguous with NO background class. Internal category ids are 1-based (see
    build_export_plan), so shift by -1; unannotated pixels use 255 (ignore) instead
    of a background class."""
    return {str(int(c["id"]) - 1): str(c["name"]) for c in categories}


def build_export_plan(
    node: Any,
    payload: Any,
    render_slice_fn: Any,
    array_shape_meta_fn: Any,
    read_slice_fn: Any,
    sample_global_stats_fn: Any,
    progress_cb: Any = None,
    include_polygons: bool = False,
    lightly: bool = False,
) -> dict[str, Any]:
    """Build the full export plan (rasterize all shapes, render PNGs).

    Args:
        node: Lazily-sliceable array node.
        payload: ExportRequest pydantic model.
        render_slice_fn: images.render_slice callable.
        array_shape_meta_fn: arrays.array_shape_meta callable.
        read_slice_fn: arrays.read_slice callable.
        sample_global_stats_fn: images._sample_global_stats callable.

    Returns:
        Dict with splits (each has images, categories, annotations, info).
    """
    meta = array_shape_meta_fn(node)
    h, w = meta["height"], meta["width"]
    render_opts = payload.render.model_dump() if hasattr(payload.render, "model_dump") else dict(payload.render)
    render_opts_mapped = {
        "norm": render_opts.get("norm", "global"),
        "scale": render_opts.get("scale", "linear"),
        "vmin_pct": render_opts.get("vmin_pct", 1.0),
        "vmax_pct": render_opts.get("vmax_pct", 99.0),
        "cmap": render_opts.get("cmap", "gray"),
    }

    global_range = None
    if render_opts_mapped["norm"] == "global":
        global_range = sample_global_stats_fn(node, meta)

    cat_id_map: dict[int, int] = {}
    cat_id_to_name: dict[int, str] = {}
    categories: list[dict[str, Any]] = []
    for i, cls in enumerate(payload.classes, 1):
        # `color` is an extra key (COCO ignores it) used for the mask legend.
        categories.append({
            "id": i, "name": cls.label, "supercategory": "object",
            "color": getattr(cls, "color", None),
        })
        cat_id_map[cls.classId] = i
        cat_id_to_name[i] = cls.label

    all_slice_keys = list(payload.slices.keys())
    neg_keys = set(str(k) for k in payload.negative_slices)
    all_keys = sorted(set(all_slice_keys) | neg_keys, key=int)

    resolved_splits = _resolve_split(all_keys, {str(k): v for k, v in payload.split_by_slice.items()}, payload.auto_split)

    source_stem = str(payload.source).replace("/", "_").replace("\\", "_")[-30:]

    def _process_slice(slice_key: str) -> dict[str, Any]:
        """Read → render → PNG-encode → rasterize one slice. Pure & independent,
        so slices run concurrently — the read/encode I/O dominates wall time."""
        slice_idx = int(slice_key)
        arr = read_slice_fn(node, meta, slice_idx)
        rgb = render_slice_fn(arr, render_opts_mapped, global_range)

        buf = io.BytesIO()
        # compress_level=1: PNG encode of a large frame is a big chunk of export
        # time; level 1 is ~3x faster than the default for a small size cost.
        PILImage.fromarray(rgb).save(buf, format="PNG", compress_level=1)
        png_bytes = buf.getvalue()

        file_name = f"{source_stem}_{slice_idx:04d}.png"
        anns: list[dict[str, Any]] = []
        skipped = 0
        # Semantic label map + per-class binary masks, from the same rasterization
        # used for the COCO annotations. COCO: 0 = background, classes 1..N. Lightly/
        # DINOv3: unannotated = 255 (ignore), classes 0-indexed (cat_id - 1).
        label = np.full((h, w), 255, dtype=np.uint8) if lightly else np.zeros((h, w), dtype=np.uint8)
        class_acc: dict[str, np.ndarray] = {}
        for shape in payload.slices.get(slice_key, []):
            shape_dict = shape if isinstance(shape, dict) else shape.model_dump()
            mask = shape_to_mask(shape_dict, h, w)
            if float(mask.sum()) < 1:
                skipped += 1
                logger.warning("Zero-area shape %r skipped", shape_dict.get("id"))
                continue
            cat_id = cat_id_map.get(shape_dict.get("classId", 1), 1)
            # A polygon shape can reuse its own points instead of re-tracing.
            poly_override = (
                [shape_dict["points"]]
                if include_polygons and shape_dict.get("kind") == "polygon" and shape_dict.get("points")
                else None
            )
            ann = mask_to_coco_ann(
                mask, ann_id=0, image_id=0, category_id=cat_id,
                include_polygons=include_polygons, poly_override=poly_override,
            )
            ann["_image_file_name"] = file_name
            anns.append(ann)
            # Paint label map (last shape wins on overlap) + accumulate per class.
            # Lightly is 0-indexed (cat_id - 1); COCO keeps the 1-based id.
            label[mask] = (cat_id - 1) if lightly else cat_id
            cname = cat_id_to_name.get(cat_id, str(cat_id))
            if cname not in class_acc:
                class_acc[cname] = np.zeros((h, w), dtype=bool)
            class_acc[cname] |= mask

        # Semantic map is always emitted (zeros for negative slices) so every
        # exported image has a matching label; per-class only where present.
        class_masks = {name: _encode_png((m * 255).astype(np.uint8)) for name, m in class_acc.items()}

        result = {
            "split": resolved_splits.get(slice_key, "train"),
            "image": {
                "file_name": file_name,
                "height": h,
                "width": w,
                "source_key": str(payload.source),
                "slice_index": slice_idx,
                "png_bytes": png_bytes,
                "label_png_bytes": _encode_png(label),
                "class_masks": class_masks,
            },
            "annotations": anns,
            "skipped": skipped,
        }
        if progress_cb is not None:
            progress_cb(f"slice {slice_idx}: {len(anns)} object(s)")
        return result

    splits_data: dict[str, dict[str, Any]] = {}
    skipped_zero_area = 0

    # I/O-bound (Tiled reads) + partially GIL-releasing (numpy/PIL/pycocotools)
    # → thread pool. EXPORT_WORKERS tunes parallelism. ex.map preserves order.
    import os
    workers = min(int(os.getenv("EXPORT_WORKERS", "16")), max(1, len(all_keys)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_process_slice, all_keys))

    for r in results:
        split = r["split"]
        if split not in splits_data:
            splits_data[split] = {"images": [], "annotations": []}
        splits_data[split]["images"].append(r["image"])
        splits_data[split]["annotations"].extend(r["annotations"])
        skipped_zero_area += r["skipped"]

    info = {
        "description": "SAM3 fine-tune dataset -- Segmentation Annotation Studio",
        "date_created": datetime.now(timezone.utc).isoformat(),
        "render": render_opts_mapped,
    }

    return {
        "splits": splits_data,
        "categories": categories,
        "info": info,
        "skipped_zero_area": skipped_zero_area,
    }
