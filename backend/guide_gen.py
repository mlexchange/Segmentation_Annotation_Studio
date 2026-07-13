"""Generate an annotation guide from an existing annotation.

Given a set of annotated shapes (a draft or a saved version's payload), this
picks a few representative example crops per class — the largest instances,
cropped to their bounding box with a small margin, with the class color overlaid
on the labeled region — so a project lead gets a guide skeleton (label, color,
example images) with one click, then fills in the descriptions.

Performance: instances are ranked by *analytic* area and bounding box (no
rasterization), only the slices actually used are read, those are downsampled,
and only the selected top-N shapes are rasterized (at reduced resolution) for the
color overlay. Reuses the mask rasterizer from :mod:`coco_export` and the slice
reading / colour-mapping helpers from :mod:`arrays` / :mod:`thumbnails`.
"""

from __future__ import annotations

import base64
import logging
import math
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_DEFAULT_PER_CLASS = 4
_CROP_MAX_PX = 160
_MARGIN_FRAC = 0.15   # bbox padding as a fraction of the larger bbox side
_RENDER_MAX_DIM = 1024  # cap the working slice resolution for speed
_OVERLAY_ALPHA = 0.45   # class-color tint strength on the labeled region


def _parse_source_key(source_key: str) -> tuple[str, str, str | None]:
    from source_keys import parse_source_key

    parsed = parse_source_key(source_key)
    return parsed["kind"] or "local", parsed["path"] or "", parsed["server_uri"]


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    h = (hex_color or "#1f77b4").lstrip("#")
    if len(h) < 6:
        h = h.ljust(6, "0")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# --- analytic area + bbox (image px), no rasterization ----------------------

def _shoelace(pts: list[float]) -> float:
    n = len(pts) // 2
    if n < 3:
        return 0.0
    s = 0.0
    j = n - 1
    for i in range(n):
        s += pts[j * 2] * pts[i * 2 + 1] - pts[i * 2] * pts[j * 2 + 1]
        j = i
    return abs(s) / 2.0


def _analytic_area(shape: dict[str, Any]) -> float:
    kind = shape.get("kind")
    if kind == "rectangle":
        return abs(float(shape.get("w", 0)) * float(shape.get("h", 0)))
    if kind == "ellipse":
        return math.pi * abs(float(shape.get("rx", 0)) * float(shape.get("ry", 0)))
    if kind == "polygon":
        a = _shoelace(shape.get("points") or [])
        for hole in shape.get("holes") or []:
            a -= _shoelace(hole)
        return max(0.0, a)
    if kind == "brush":
        a = 0.0
        for st in shape.get("strokes") or []:
            if st.get("mode") != "paint":
                continue
            p = st.get("points") or []
            r = float(st.get("radius", 1))
            length = 0.0
            for i in range(0, len(p) - 3, 2):
                length += math.hypot(p[i + 2] - p[i], p[i + 3] - p[i + 1])
            a += length * 2 * r + math.pi * r * r
        return a
    return 0.0


def _analytic_bbox(shape: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """(x0, y0, x1, y1) in image px, or None."""
    kind = shape.get("kind")
    if kind == "rectangle":
        x, y = float(shape.get("x", 0)), float(shape.get("y", 0))
        w, h = float(shape.get("w", 0)), float(shape.get("h", 0))
        return min(x, x + w), min(y, y + h), max(x, x + w), max(y, y + h)
    if kind == "ellipse":
        cx, cy = float(shape.get("cx", 0)), float(shape.get("cy", 0))
        rx, ry = abs(float(shape.get("rx", 0))), abs(float(shape.get("ry", 0)))
        return cx - rx, cy - ry, cx + rx, cy + ry
    xs: list[float] = []
    ys: list[float] = []
    if kind == "polygon":
        p = shape.get("points") or []
        xs = p[0::2]
        ys = p[1::2]
    elif kind == "brush":
        for st in shape.get("strokes") or []:
            p = st.get("points") or []
            r = float(st.get("radius", 1))
            for i in range(0, len(p) - 1, 2):
                xs.extend([p[i] - r, p[i] + r])
                ys.extend([p[i + 1] - r, p[i + 1] + r])
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _scale_shape(shape: dict[str, Any], f: float) -> dict[str, Any]:
    """Return a copy of *shape* with all coordinates scaled by *f* (for rasterizing
    at a reduced resolution)."""
    s = dict(shape)
    k = s.get("kind")
    if k == "polygon":
        s["points"] = [v * f for v in s.get("points", [])]
        if s.get("holes"):
            s["holes"] = [[v * f for v in ring] for ring in s["holes"]]
    elif k == "rectangle":
        for key in ("x", "y", "w", "h"):
            s[key] = float(s.get(key, 0)) * f
    elif k == "ellipse":
        for key in ("cx", "cy", "rx", "ry"):
            s[key] = float(s.get(key, 0)) * f
    elif k == "brush":
        s["strokes"] = [
            {**st, "points": [v * f for v in st.get("points", [])], "radius": float(st.get("radius", 1)) * f}
            for st in s.get("strokes", [])
        ]
    if s.get("erased"):
        s["erased"] = [
            {**er, "points": [v * f for v in er.get("points", [])], "radius": float(er.get("radius", 1)) * f}
            for er in s["erased"]
        ]
    return s


def _boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Boolean boundary ring of *mask* (mask minus its 4-neighbour erosion),
    dilated to ~*thickness* pixels so it stays visible after downscaling."""
    m = mask
    eroded = m.copy()
    eroded[1:, :] &= m[:-1, :]
    eroded[:-1, :] &= m[1:, :]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    edge = m & ~eroded
    for _ in range(max(0, thickness - 1)):
        d = edge.copy()
        d[1:, :] |= edge[:-1, :]
        d[:-1, :] |= edge[1:, :]
        d[:, 1:] |= edge[:, :-1]
        d[:, :-1] |= edge[:, 1:]
        edge = d & m  # keep the thickened outline inside the region
    return edge


def _crop_with_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    max_px: int,
) -> str | None:
    """Crop *rgb* to the mask's bbox (+margin), tint the masked region with *color*,
    resize, and return a PNG data URL. Coordinates are in the (reduced) grid of both
    arrays. Returns None if the mask is empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    h, w = rgb.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin = int(round(max(x1 - x0, y1 - y0) * _MARGIN_FRAC)) + 1
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    cmask = mask[y0:y1, x0:x1]
    tint = np.array(color, dtype=np.float32)
    # Fill: blend the class color over the labeled region.
    crop[cmask] = crop[cmask] * (1.0 - _OVERLAY_ALPHA) + tint * _OVERLAY_ALPHA
    # Outline: draw a solid class-color boundary (thickened) so the region reads
    # clearly regardless of the underlying intensity.
    outline = _boundary(cmask, thickness=2)
    crop[outline] = tint
    img = PILImage.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).convert("RGB")

    scale = min(max_px / img.width, max_px / img.height, 1.0)
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), PILImage.Resampling.BILINEAR)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def generate_guide(
    source_key: str,
    payload: dict[str, Any],
    *,
    per_class: int = _DEFAULT_PER_CLASS,
    crop_px: int = _CROP_MAX_PX,
) -> dict[str, Any]:
    """Build a guide skeleton (per-class label/color + color-coded example crops).

    Descriptions are left blank for the lead to fill in.
    """
    import arrays as arrays_mod
    from coco_export import shape_to_mask
    from thumbnails import _prepare_intensity, _prepare_rgb

    classes = payload.get("classes") or []
    slices: dict[str, list[dict[str, Any]]] = payload.get("slices") or {}

    # Index shapes per class with their analytic area + slice, ranked largest-first.
    by_class: dict[int, list[tuple[float, int, dict[str, Any]]]] = {}
    for skey, shapes in slices.items():
        if not isinstance(shapes, list):
            continue
        try:
            sidx = int(skey)
        except (TypeError, ValueError):
            continue
        for shape in shapes:
            cid = int(shape.get("classId", -1))
            by_class.setdefault(cid, []).append((_analytic_area(shape), sidx, shape))
    for cid in by_class:
        by_class[cid].sort(key=lambda t: t[0], reverse=True)

    # Which slices do the selected top-N instances live on? Only render those.
    selected: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    needed_slices: set[int] = set()
    for cid, items in by_class.items():
        picks = [(sidx, shape) for area, sidx, shape in items[:per_class] if area > 0]
        selected[cid] = picks
        needed_slices.update(sidx for sidx, _ in picks)

    # Render each needed slice once, downsampled, recording the downsample step.
    node = meta = None
    if needed_slices:
        try:
            kind, source, server_uri = _parse_source_key(source_key)
            node = arrays_mod.resolve_array(source, kind, server_uri)
            meta = arrays_mod.array_shape_meta(node)
        except Exception as exc:
            logger.warning("guide_gen: cannot resolve array %s: %s", source_key, exc)
            node = meta = None

    rgb_cache: dict[int, tuple[np.ndarray, int] | None] = {}

    def _slice_rgb(sidx: int) -> tuple[np.ndarray, int] | None:
        """Downsampled RGB slice + integer downsample step, or None."""
        if sidx in rgb_cache:
            return rgb_cache[sidx]
        result: tuple[np.ndarray, int] | None = None
        if node is not None and meta is not None:
            try:
                sc = max(0, min(sidx, meta["n_slices"] - 1))
                arr = np.asarray(arrays_mod.read_slice(node, meta, sc))
                h, w = arr.shape[:2]
                step = max(1, int(math.ceil(max(h, w) / _RENDER_MAX_DIM)))
                arr = arr[::step, ::step, ...] if arr.ndim == 3 else arr[::step, ::step]
                if arr.ndim == 3 and arr.shape[2] in (3, 4):
                    rgb = _prepare_rgb(arr[:, :, :3])
                elif arr.ndim == 2:
                    rgb = _prepare_intensity(arr)
                else:
                    rgb = None
                if rgb is not None:
                    result = (rgb, step)
            except Exception as exc:
                logger.warning("guide_gen: cannot render slice %d: %s", sidx, exc)
        rgb_cache[sidx] = result
        return result

    color_by_class = {int(c.get("classId", -1)): _hex_rgb(str(c.get("color", "#1f77b4"))) for c in classes}

    out_classes: list[dict[str, Any]] = []
    for cls in classes:
        cid = int(cls.get("classId", -1))
        color = color_by_class.get(cid, (31, 119, 180))
        crops: list[str] = []
        for sidx, shape in selected.get(cid, []):
            got = _slice_rgb(sidx)
            if got is None:
                continue
            rgb, step = got
            gh, gw = rgb.shape[:2]
            # Rasterize this one shape at the reduced resolution for the overlay.
            try:
                mask = shape_to_mask(_scale_shape(shape, 1.0 / step), gh, gw).astype(bool)
            except Exception:
                continue
            url = _crop_with_overlay(rgb, mask, color, crop_px)
            if url:
                crops.append(url)

        out_classes.append(
            {
                "label": str(cls.get("label", "")),
                "color": str(cls.get("color", "#1f77b4")),
                "description": "",
                "exampleCrops": crops,
            }
        )

    return {"classes": out_classes, "notes": ""}
