"""Annotated thumbnail generation for version history previews.

Renders the most-annotated slice from the source image, overlays the
annotation shapes in their class colours, and returns PNG bytes.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image as PILImage, ImageDraw

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = 256
# Downsample large source arrays before colour-mapping / overlay work.
_READ_MAX_DIM = 512
_ANNOTATION_ALPHA = 160  # 0-255 overlay opacity


def _parse_source_key(source_key: str) -> tuple[str, str, str | None]:
    """Return (kind, source, server_uri) from a canonical source key."""
    from source_keys import parse_source_key

    parsed = parse_source_key(source_key)
    return parsed["kind"] or "local", parsed["path"] or "", parsed["server_uri"]


def decode_thumbnail_base64(data: str) -> bytes | None:
    """Decode a base64 PNG from the save modal preview, or return None."""
    if not data:
        return None
    try:
        raw = data.strip()
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        return base64.b64decode(raw)
    except Exception as exc:
        logger.warning("annotation_thumbnails: invalid thumbnail_base64: %s", exc)
        return None


def _stride_downsample(arr: np.ndarray, max_dim: int) -> np.ndarray:
    """Fast nearest-neighbour downsample so huge volumes don't stall thumbnails."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_dim:
        return arr
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    if arr.ndim == 2:
        return arr[::step, ::step]
    return arr[::step, ::step, ...]


def _hex_to_rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) < 6:
        h = h.ljust(6, "0")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


def _draw_shapes(
    draw: ImageDraw.ImageDraw,
    shapes: list[dict[str, Any]],
    class_colors: dict[int, str],
    scale: float,
    alpha: int = _ANNOTATION_ALPHA,
) -> None:
    """Draw annotation shapes onto *draw* at thumbnail scale."""
    for shape in shapes:
        color_hex = class_colors.get(int(shape.get("classId", -1)), "#4090ff")
        rgba = _hex_to_rgba(color_hex, alpha)
        kind = shape.get("kind")

        if kind == "polygon":
            pts = shape.get("points") or []
            if len(pts) >= 6:
                scaled = [(pts[i] * scale, pts[i + 1] * scale) for i in range(0, len(pts) - 1, 2)]
                try:
                    draw.polygon(scaled, fill=rgba)
                except Exception:
                    pass

        elif kind == "rectangle":
            x = float(shape.get("x", 0))
            y = float(shape.get("y", 0))
            w = float(shape.get("w", 0))
            h = float(shape.get("h", 0))
            draw.rectangle(
                [(x * scale, y * scale), ((x + w) * scale, (y + h) * scale)],
                fill=rgba,
            )

        elif kind == "ellipse":
            cx = float(shape.get("cx", 0))
            cy = float(shape.get("cy", 0))
            rx = float(shape.get("rx", 0))
            ry = float(shape.get("ry", 0))
            draw.ellipse(
                [((cx - rx) * scale, (cy - ry) * scale), ((cx + rx) * scale, (cy + ry) * scale)],
                fill=rgba,
            )

        elif kind == "brush":
            for stroke in shape.get("strokes") or []:
                if stroke.get("mode") == "erase":
                    continue
                pts = stroke.get("points") or []
                if len(pts) < 2:
                    continue
                radius = max(1, int(float(stroke.get("radius", 5)) * scale))
                stroke_pts = [
                    (pts[i] * scale, pts[i + 1] * scale) for i in range(0, len(pts) - 1, 2)
                ]
                if len(stroke_pts) >= 2:
                    # Round caps on the line replace per-point disk stamps (much faster).
                    draw.line(stroke_pts, fill=rgba, width=radius * 2)


def render_annotated_thumbnail(
    source_key: str,
    payload: dict[str, Any],
    size: int = THUMBNAIL_SIZE,
) -> bytes | None:
    """Render a PNG thumbnail of the most-annotated slice with shapes overlaid.

    Args:
        source_key: Canonical source key (``tiled:…`` or ``local:…``).
        payload: DraftPayload dict (``classes``, ``slices``, …).
        size: Maximum thumbnail dimension in pixels.

    Returns:
        PNG bytes, or ``None`` if rendering failed.
    """
    try:
        import arrays as arrays_mod
        from thumbnails import _prepare_rgb, _prepare_intensity
    except ImportError as exc:
        logger.warning("annotation_thumbnails: missing module: %s", exc)
        return None

    kind, source, server_uri = _parse_source_key(source_key)

    try:
        node = arrays_mod.resolve_array(source, kind, server_uri)
        meta = arrays_mod.array_shape_meta(node)
    except Exception as exc:
        logger.warning("annotation_thumbnails: cannot resolve array %s: %s", source_key, exc)
        return None

    # Choose the slice with the most annotations
    slices: dict[str, list[dict[str, Any]]] = payload.get("slices") or {}
    best_slice = 0
    best_count = 0
    for slice_key, shapes in slices.items():
        n = len(shapes) if isinstance(shapes, list) else 0
        if n > best_count:
            best_count = n
            try:
                best_slice = int(slice_key)
            except ValueError:
                pass
    best_slice = max(0, min(best_slice, meta["n_slices"] - 1))

    # Read slice, downsample early, then colour-map the smaller array.
    try:
        arr = arrays_mod.read_slice(node, meta, best_slice)
        arr = _stride_downsample(np.asarray(arr), _READ_MAX_DIM)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            rgb = _prepare_rgb(arr[:, :, :3])
        elif arr.ndim == 2:
            rgb = _prepare_intensity(arr)
        else:
            logger.warning("annotation_thumbnails: unsupported array shape %s", arr.shape)
            return None
    except Exception as exc:
        logger.warning("annotation_thumbnails: cannot read/render slice: %s", exc)
        return None

    img_h, img_w = rgb.shape[:2]
    scale = min(size / img_h, size / img_w, 1.0)
    nw = max(1, int(img_w * scale))
    nh = max(1, int(img_h * scale))

    base = PILImage.fromarray(rgb).resize((nw, nh), PILImage.Resampling.BILINEAR).convert("RGBA")
    overlay = PILImage.new("RGBA", (nw, nh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    classes = payload.get("classes") or []
    class_colors: dict[int, str] = {
        int(c.get("classId", -1)): str(c.get("color", "#4090ff")) for c in classes
    }
    slice_shapes: list[dict[str, Any]] = slices.get(str(best_slice)) or []
    if slice_shapes:
        _draw_shapes(draw, slice_shapes, class_colors, scale)

    result = PILImage.alpha_composite(base, overlay).convert("RGB")
    buf = BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def upload_thumbnail_to_tiled(
    source_key: str,
    version: int,
    saved_at: str,
    png_bytes: bytes,
) -> None:
    """Write the annotated thumbnail to a Tiled sibling node (best-effort).

    Creates ``…/<source_stem>__v_thumbs/v{version:04d}`` alongside the source
    array.  Silently skips local sources or any Tiled write errors.
    """
    try:
        from tiled_clients import api_key_for_uri, get_tiled_client
    except ImportError:
        return

    kind, source, server_uri = _parse_source_key(source_key)
    if kind != "tiled":
        return

    try:
        api_key = api_key_for_uri(server_uri)
        client = get_tiled_client(server_uri, api_key)

        parts = source.strip("/").split("/")
        source_stem = parts[-1]
        parent_parts = parts[:-1]

        parent: Any = client
        for part in parent_parts:
            parent = parent[part]

        container_key = f"{source_stem}__v_thumbs"
        try:
            thumb_container = parent[container_key]
        except KeyError:
            thumb_container = parent.create_container(
                key=container_key,
                metadata={"studio_type": "version_thumbnails", "source": source},
            )

        arr = np.array(PILImage.open(BytesIO(png_bytes)).convert("RGB"))
        thumb_container.write_array(
            arr,
            key=f"v{version:04d}",
            metadata={"version": version, "saved_at": saved_at, "studio_type": "version_thumbnail"},
        )
        logger.info("annotation_thumbnails: wrote v%d thumbnail to Tiled at %s/%s", version, source, container_key)
    except Exception as exc:
        logger.debug("annotation_thumbnails: Tiled thumbnail write skipped (%s)", exc)
