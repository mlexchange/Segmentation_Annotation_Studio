"""Array node resolution, shape dispatch, and slice reading.

This module provides a unified interface for accessing array data from two
source kinds:

* ``"tiled"`` — a Tiled server node reachable via the configured clients.
* ``"local"`` — a file on the local filesystem under ``LOCAL_DATA_ROOT``.

Resolved nodes are cached with a 5-minute TTL so repeated requests for the
same source do not pay connection / file-open overhead.

Shape dispatch
--------------
The module recognises four array layouts:

+----------+------------+--------+----------+
| Layout   | Shape      | Slices | RGB?     |
+==========+============+========+==========+
| ``HW``   | (H, W)     | 1      | No       |
+----------+------------+--------+----------+
| ``HWC``  | (H, W, C)  | 1      | Yes      |
+----------+------------+--------+----------+
| ``NHW``  | (N, H, W)  | N      | No       |
+----------+------------+--------+----------+
| ``NHWC`` | (N, H, W, C)| N     | Yes      |
+----------+------------+--------+----------+
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import HTTPException

import local_fs
from cache import TTLCache
from tiled_clients import api_key_for_uri, get_tiled_client

logger = logging.getLogger(__name__)
_node_cache: TTLCache = TTLCache(ttl_seconds=300.0, max_entries=32)


def resolve_array(
    source: str,
    kind: str,
    server_uri: str | None = None,
    root: str | None = None,
) -> Any:
    """Return a lazily-sliceable array node for *source*.

    Results are cached by ``(kind, source, server_uri, root)`` for 5 minutes.

    Args:
        source: Tiled path (slash-separated) or local relative path.
        kind: ``"tiled"`` or ``"local"``.
        server_uri: Tiled server URI; only used when ``kind == "tiled"``.
        root: Granted absolute local root; only used when ``kind == "local"``.

    Returns:
        A Tiled array node or a NumPy-compatible array.

    Raises:
        HTTPException: 404 if the path does not exist; 422 for unknown kind.
    """
    key = (kind, source, server_uri or "", root or "")
    cached = _node_cache.get(key)
    if cached is not None:
        return cached

    if kind == "tiled":
        api_key = api_key_for_uri(server_uri)
        client = get_tiled_client(server_uri, api_key)
        node: Any = client
        for part in source.strip("/").split("/"):
            try:
                node = node[part]
            except KeyError as exc:
                raise HTTPException(404, f"Tiled path not found: {source!r}") from exc
        # Drag-and-drop ingest nests datasets as browse/<dataset>/<array(s)>, so
        # a Browse selection resolves to the wrapping container — descend to the
        # array or slice-stack it represents. Direct array paths are unchanged.
        node = _descend_to_stack(node)
    elif kind == "local":
        node = local_fs.open_array(source, root)
    else:
        raise HTTPException(422, f"Unknown source kind: {kind!r}")

    _node_cache.set(key, node)
    return node


def _is_container_node(node: Any) -> bool:
    """True if *node* is a Tiled container (vs. an array/leaf) node."""
    sf = getattr(node, "structure_family", None)
    return str(getattr(sf, "value", sf)) == "container"


def multiscale_levels(node: Any) -> list[str] | None:
    """Ordered ``scale*`` child keys of a multiscale (pyramid) container.

    Registered Zarr volumes are OME-NGFF groups: ``scale0/image``,
    ``scale1/image``, … Returns the level keys finest-first, or None when *node*
    is not a pyramid.
    """
    if not _is_container_node(node):
        return None
    try:
        keys = [k for k in node if str(k).startswith("scale")]
    except Exception:  # noqa: BLE001 — not enumerable → not a pyramid
        return None
    if len(keys) < 2:
        return None

    def _index(key: str) -> int:
        digits = "".join(ch for ch in str(key) if ch.isdigit())
        return int(digits) if digits else 0

    return sorted(keys, key=_index)


def _level_array(node: Any, level_key: str) -> Any | None:
    """The array inside one pyramid level (``scaleN`` wraps a single array)."""
    try:
        level = node[level_key]
    except Exception:  # noqa: BLE001
        return None
    if not _is_container_node(level):
        return level
    try:
        first = next(iter(level))
    except StopIteration:
        return None
    return level[first]


def _descend_to_stack(node: Any, max_depth: int = 8) -> Any:
    """Resolve a Browse selection to the array/stack it should open.

    Drag-and-drop ingest nests datasets as ``browse/<dataset>/<array(s)>``. This
    descends through *wrapper* containers (a container whose only/first child is
    itself a container) but STOPS at a container whose children are arrays —
    returning that container so it can be treated as a slice stack (one array
    node per slice). Array nodes (and non-Tiled inputs) are returned unchanged.

    A multiscale Zarr volume is handled first and explicitly. The generic walk
    below would descend into ``scale0``, find its single ``image`` child, and
    return ``scale0`` as a one-element "stack" — presenting a whole 3-D volume as
    a single slice. Selecting such a dataset resolves to the FINEST level's
    array, which is the full-resolution volume the user expects to annotate.
    """
    levels = multiscale_levels(node)
    if levels:
        array = _level_array(node, levels[0])
        if array is not None:
            return array

    depth = 0
    while _is_container_node(node) and depth < max_depth:
        try:
            first = next(iter(node))
        except StopIteration:
            return node  # empty container — nothing to descend into
        child = node[first]
        if not _is_container_node(child):
            return node  # container of arrays → the stack itself
        node = child
        depth += 1
    return node


def _stack_keys(node: Any) -> list[str]:
    """Sorted child keys of a container-stack (ingest zero-pads, so lexical
    order == slice order)."""
    return sorted(node)


def node_keywords(node: Any) -> list[str]:
    """Return the ``keywords`` tag list stored on *node* (or its first child).

    Ingest writes ``keywords`` on both the dataset container and each array
    child. For a stack container we read the container metadata first, then fall
    back to the first slice. Non-Tiled inputs have no metadata → empty list.

    Args:
        node: A resolved Tiled node (array or container-stack) or NumPy array.

    Returns:
        List of tag strings; empty when none are present.
    """
    def _from_meta(meta: Any) -> list[str] | None:
        try:
            value = (meta or {}).get("keywords")
        except AttributeError:
            return None
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return None

    tags = _from_meta(getattr(node, "metadata", None))
    if tags:
        return tags
    if _is_container_node(node):
        try:
            keys = _stack_keys(node)
            if keys:
                child_tags = _from_meta(getattr(node[keys[0]], "metadata", None))
                if child_tags:
                    return child_tags
        except Exception:  # noqa: BLE001 — metadata is best-effort
            pass
    return []


def pyramid_info(
    source: str,
    kind: str,
    server_uri: str | None = None,
    root: str | None = None,
) -> dict[str, Any] | None:
    """Describe the pyramid *source* belongs to, if it addresses one level.

    Annotations are stored in FULL-RESOLUTION coordinates regardless of which
    level is being viewed, so the caller needs the finest level's shape even when
    a coarse level is open. Returns ``None`` for anything that is not a level of
    a multiscale volume.

    Returns:
        ``{"level_key", "level_index", "level_count", "full_shape",
        "z_downsample"}`` where ``z_downsample`` is finest-z / this-level-z — the
        factor mapping a full-resolution slice index onto this level.
    """
    if kind != "tiled":
        return None
    parts = [p for p in source.strip("/").split("/") if p]
    # A level is addressed as <volume>/scaleN or <volume>/scaleN/<array>.
    for trim in (1, 2):
        if len(parts) <= trim:
            continue
        level_key = parts[-trim]
        if not str(level_key).startswith("scale"):
            continue
        try:
            parent = resolve_container(("/".join(parts[:-trim])), kind, server_uri, root)
        except HTTPException:
            return None
        levels = multiscale_levels(parent)
        if not levels or level_key not in levels:
            return None
        finest = _level_array(parent, levels[0])
        current = _level_array(parent, level_key)
        if finest is None or current is None:
            return None
        full_shape = [int(v) for v in finest.shape]
        cur_shape = [int(v) for v in current.shape]
        if len(full_shape) != 3 or len(cur_shape) != 3:
            return None
        return {
            "level_key": level_key,
            "level_index": levels.index(level_key),
            "level_count": len(levels),
            "full_shape": full_shape,
            # Ratio, not an integer factor: real pyramids are not always clean
            # powers of two in z (690 -> 172 is 4.0116), so rounding a fixed
            # factor would drift by whole slices at the end of the volume.
            "z_downsample": (full_shape[0] / cur_shape[0]) if cur_shape[0] else 1.0,
        }
    return None


def resolve_container(
    source: str,
    kind: str,
    server_uri: str | None = None,
    root: str | None = None,
) -> Any:
    """Resolve *source* to its node WITHOUT descending to an array/stack.

    :func:`resolve_array` deliberately descends (a Browse selection should open
    the data); this is for callers that need the container itself, such as
    reading a volume's pyramid structure.
    """
    if kind != "tiled":
        raise HTTPException(422, "Only tiled sources have containers")
    client = get_tiled_client(server_uri, api_key_for_uri(server_uri))
    node: Any = client
    for part in source.strip("/").split("/"):
        if not part:
            continue
        try:
            node = node[part]
        except KeyError as exc:
            raise HTTPException(404, f"Tiled path not found: {source!r}") from exc
    return node


def array_shape_meta(node: Any, pyramid: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return shape-dispatch metadata for *node*.

    Args:
        node: A Tiled array node or NumPy array.
        pyramid: Optional :func:`pyramid_info` result. When given, the reported
            ``height``/``width``/``n_slices`` describe the FINEST level rather
            than *node* — so annotation coordinates are full-resolution whichever
            level is displayed — and ``z_downsample`` tells :func:`read_slice`
            how to map a full-resolution slice index onto this level.

    Returns:
        Dict with keys: ``n_slices``, ``height``, ``width``, ``dtype``,
        ``is_rgb``, ``shape_kind``.

    Raises:
        HTTPException: 422 for unsupported array shapes (e.g. 1-D or 5-D).
    """
    if _is_container_node(node):
        return _stack_shape_meta(node)

    raw = (
        np.asarray(node)
        if hasattr(node, "__array__") and not hasattr(node, "shape")
        else node
    )
    shape = tuple(raw.shape)
    dtype = str(raw.dtype)

    if len(shape) == 2:
        h, w = shape
        return {
            "n_slices": 1,
            "height": h,
            "width": w,
            "dtype": dtype,
            "is_rgb": False,
            "shape_kind": "HW",
        }
    if len(shape) == 3 and shape[2] in (3, 4):
        h, w = shape[:2]
        return {
            "n_slices": 1,
            "height": h,
            "width": w,
            "dtype": dtype,
            "is_rgb": True,
            "shape_kind": "HWC",
        }
    if len(shape) == 3:
        n, h, w = shape
        meta = {
            "n_slices": n,
            "height": h,
            "width": w,
            "dtype": dtype,
            "is_rgb": False,
            "shape_kind": "NHW",
        }
        if pyramid:
            full_z, full_h, full_w = pyramid["full_shape"]
            # Report the finest level's geometry. The canvas draws the (smaller)
            # level image at these dimensions, so every annotation coordinate is
            # full-resolution by construction — no rescaling on save or load.
            meta.update(
                n_slices=full_z,
                height=full_h,
                width=full_w,
                z_downsample=pyramid["z_downsample"],
                level_key=pyramid["level_key"],
                level_index=pyramid["level_index"],
                level_count=pyramid["level_count"],
                level_height=h,
                level_width=w,
                level_n_slices=n,
            )
        return meta
    if len(shape) == 4 and shape[3] in (3, 4):
        n, h, w = shape[:3]
        return {
            "n_slices": n,
            "height": h,
            "width": w,
            "dtype": dtype,
            "is_rgb": True,
            "shape_kind": "NHWC",
        }
    raise HTTPException(422, f"Unsupported array shape: {shape}")


def _stack_shape_meta(node: Any) -> dict[str, Any]:
    """Shape-dispatch metadata for a container-of-arrays treated as a stack.

    Each child is one slice; ``n_slices`` is the child count and the per-slice
    H/W/dtype/is_rgb come from the first child. The sorted child keys are stored
    under ``"keys"`` so :func:`read_slice` maps a slice index to its node.

    Raises:
        HTTPException: 422 for an empty container or unsupported slice shape.
    """
    keys = _stack_keys(node)
    if not keys:
        raise HTTPException(422, "Container has no array slices")
    first = node[keys[0]]
    fshape = tuple(first.shape)
    dtype = str(first.dtype)

    if len(fshape) == 2:
        h, w, is_rgb = fshape[0], fshape[1], False
    elif len(fshape) == 3 and fshape[2] in (3, 4):
        h, w, is_rgb = fshape[0], fshape[1], True
    else:
        raise HTTPException(422, f"Unsupported slice shape in stack: {fshape}")

    return {
        "n_slices": len(keys),
        "height": h,
        "width": w,
        "dtype": dtype,
        "is_rgb": is_rgb,
        "shape_kind": "STACK",
        "keys": keys,
    }


def read_slice(node: Any, meta: dict[str, Any], idx: int) -> np.ndarray:
    """Read one slice from *node* and return it as a NumPy array.

    Args:
        node: Array node (Tiled or NumPy-compatible).
        meta: Shape-dispatch dict returned by :func:`array_shape_meta`.
        idx: Zero-based slice index (ignored for 2-D / RGB images).

    Returns:
        2-D or H×W×C NumPy array for the requested slice.

    Raises:
        HTTPException: 422 if the shape kind is unrecognised.
    """
    kind = meta["shape_kind"]
    if kind == "HW":
        return np.asarray(node)
    if kind == "HWC":
        return np.asarray(node)
    if kind in ("NHW", "NHWC"):
        # `idx` is a FULL-RESOLUTION slice index when a pyramid level is open
        # (see `array_shape_meta`); map it onto this level's own z range.
        z_down = float(meta.get("z_downsample") or 1.0)
        if z_down != 1.0:
            limit = int(meta.get("level_n_slices") or 0)
            idx = int(round(idx / z_down))
            if limit:
                idx = max(0, min(limit - 1, idx))
        return np.asarray(node[idx])
    if kind == "STACK":
        keys = meta.get("keys") or _stack_keys(node)
        if not keys:
            raise HTTPException(422, "Container has no array slices")
        key = keys[idx] if 0 <= idx < len(keys) else keys[0]
        return np.asarray(node[key])
    raise HTTPException(422, f"Cannot slice shape kind: {kind}")
