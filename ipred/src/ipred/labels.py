"""Rasterize annotation shapes to a sparse label map (self-contained)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw


def build_label_map(shapes: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """Compose uint8 label map (0 = unlabeled); last shape wins."""
    labels = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        class_id = int(shape.get("classId") or shape.get("class_id") or 0)
        if class_id <= 0 or class_id > 255:
            continue
        mask = shape_to_mask(shape, height, width)
        labels[mask] = class_id
    return labels


def shape_to_mask(shape: dict[str, Any], h: int, w: int) -> np.ndarray:
    """Rasterize one shape to boolean mask."""
    kind = shape.get("kind")
    if kind == "rectangle":
        return _rect_mask(
            float(shape["x"]),
            float(shape["y"]),
            float(shape["w"]),
            float(shape["h"]),
            h,
            w,
        )
    if kind == "ellipse":
        return _ellipse_mask(
            float(shape["cx"]),
            float(shape["cy"]),
            float(shape["rx"]),
            float(shape["ry"]),
            h,
            w,
        )
    if kind == "polygon":
        return _polygon_mask(shape.get("points") or [], h, w, shape.get("holes"))
    if kind == "brush":
        return _brush_mask(shape.get("strokes") or [], h, w)
    raise ValueError(f"Unknown shape kind: {kind!r}")


def _rect_mask(x: float, y: float, ww: float, hh: float, h: int, w: int) -> np.ndarray:
    x0 = max(0, int(np.floor(min(x, x + ww))))
    x1 = min(w, int(np.ceil(max(x, x + ww))))
    y0 = max(0, int(np.floor(min(y, y + hh))))
    y1 = min(h, int(np.ceil(max(y, y + hh))))
    mask = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _ellipse_mask(
    cx: float, cy: float, rx: float, ry: float, h: int, w: int
) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    rx = max(rx, 1e-6)
    ry = max(ry, 1e-6)
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _xy_pairs(points: list[Any]) -> list[tuple[float, float]]:
    """Normalize studio point payloads to ``(x, y)`` pairs.

    Studio wire format for polygons/brushes is a flat ``[x0, y0, x1, y1, ...]``
    float list (same as coco_export). Nested ``[[x,y], ...]`` and ``{x,y}``
    dicts are also accepted.
    """
    if not points:
        return []
    if all(isinstance(p, (int, float)) for p in points):
        if len(points) < 2:
            return []
        return [
            (float(points[i]), float(points[i + 1]))
            for i in range(0, len(points) - 1, 2)
        ]
    xy: list[tuple[float, float]] = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            xy.append((float(p[0]), float(p[1])))
        elif isinstance(p, dict):
            xy.append((float(p["x"]), float(p["y"])))
    return xy


def _polygon_mask(
    points: list[Any], h: int, w: int, holes: list[Any] | None = None
) -> np.ndarray:
    """Rasterize the outer ring, then carve out each hole ring.

    Holes come from the studio's clip-to-other-classes tool (see
    ``clipToClasses.ts``), which represents "this region minus already-labeled
    neighbor classes" as a polygon with holes rather than reshaping the outer
    ring. Ignoring them would paint the full outer extent — including pixels
    that visually belong to other classes on the canvas.
    """
    xy = _xy_pairs(points)
    if len(xy) < 3:
        return np.zeros((h, w), dtype=bool)
    img = PILImage.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(xy, outline=1, fill=1)
    for hole in holes or []:
        hxy = _xy_pairs(hole)
        if len(hxy) >= 3:
            draw.polygon(hxy, outline=0, fill=0)
    return np.asarray(img, dtype=np.uint8) > 0


def _brush_mask(strokes: list[dict[str, Any]], h: int, w: int) -> np.ndarray:
    """Paint OR / erase AND-NOT along stroke points."""
    mask = np.zeros((h, w), dtype=bool)
    for stroke in strokes:
        pts = stroke.get("points") or []
        xy = _xy_pairs(pts)
        if not xy:
            continue
        # Studio stamps use radius directly; legacy payloads may send diameter as size.
        if stroke.get("radius") is not None:
            radius = max(1, int(round(float(stroke["radius"]))))
        else:
            radius = max(1, int(round(float(stroke.get("size") or 3) / 2)))
        mode = str(stroke.get("mode") or stroke.get("type") or "paint")
        erase = mode in ("erase", "eraser")
        layer = PILImage.new("L", (w, h), 0)
        ld = ImageDraw.Draw(layer)
        if len(xy) == 1:
            x, y = xy[0]
            ld.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        else:
            ld.line(xy, fill=1, width=max(1, radius * 2))
            for x, y in xy:
                ld.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        layer_m = np.asarray(layer, dtype=np.uint8) > 0
        if erase:
            mask &= ~layer_m
        else:
            mask |= layer_m
    return mask
