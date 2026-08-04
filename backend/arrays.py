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
from sidecars import SIDECAR_SUFFIXES, is_sidecar_key  # noqa: F401 — re-exported for callers
from tiled_clients import get_tiled_client

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
        root: Server-configured local root; only used when ``kind == "local"``.

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
        client = get_tiled_client(server_uri)
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


def _descend_to_stack(node: Any, max_depth: int = 8) -> Any:
    """Resolve a Browse selection to the array/stack it should open.

    Drag-and-drop ingest nests datasets as ``browse/<dataset>/<array(s)>``. This
    descends through *wrapper* containers — a container with exactly ONE child
    that is itself a container — but STOPS at a container whose (first) child is
    an array, returning that container so it can be treated as a slice stack
    (one array node per slice). Array nodes (and non-Tiled inputs) are returned
    unchanged.

    Raises:
        HTTPException: 422 if a container has more than one child and the first
            is itself a container — descending into "the first" would be a
            guess among several equally-plausible datasets, not an unambiguous
            passthrough wrapper.
    """
    depth = 0
    while _is_container_node(node) and depth < max_depth:
        try:
            keys = list(node)
        except TypeError:
            return node
        if not keys:
            return node  # empty container — nothing to descend into
        child = node[keys[0]]
        if not _is_container_node(child):
            return node  # container of arrays → the stack itself
        if len(keys) > 1:
            raise HTTPException(
                422,
                "Ambiguous container: multiple children found where a single "
                "wrapper or array stack was expected — open a more specific path",
            )
        node = child
        depth += 1
    return node


def _stack_keys(node: Any) -> list[str]:
    """Sorted child keys of a container-stack.

    Node keys are the raw uploaded filename stems (ingest zero-pads only the
    ``image_number`` metadata, not the key itself — see ``ingest.py``), so
    lexical order only matches numeric slice order when the source filenames
    were already zero-padded on disk.

    Excludes sidecar containers (see :data:`SIDECAR_SUFFIXES`) — they are not
    slices and don't share the stack's array shape.
    """
    return sorted(k for k in node if not is_sidecar_key(k))


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


def array_shape_meta(node: Any) -> dict[str, Any]:
    """Return shape-dispatch metadata for *node*.

    Args:
        node: A Tiled array node or NumPy array.

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
        return {
            "n_slices": n,
            "height": h,
            "width": w,
            "dtype": dtype,
            "is_rgb": False,
            "shape_kind": "NHW",
        }
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
    if kind == "NHW":
        return np.asarray(node[idx])
    if kind == "NHWC":
        return np.asarray(node[idx])
    if kind == "STACK":
        keys = meta.get("keys") or _stack_keys(node)
        if not keys:
            raise HTTPException(422, "Container has no array slices")
        try:
            # A plain list index: supports Python-style negative indexing (-1 =
            # last slice) and raises for anything truly out of range, instead of
            # the previous silent fallback to slice 0 for any bad index.
            key = keys[idx]
        except IndexError as exc:
            raise HTTPException(
                404, f"Slice index {idx} out of range (0..{len(keys) - 1})"
            ) from exc
        arr = np.asarray(node[key])
        # `meta`'s height/width/dtype came from the FIRST slice only (see
        # _stack_shape_meta) — cheap, since checking every slice upfront would
        # mean one network round-trip per slice for a stack that may have
        # thousands. Catch a mismatched slice here, at the point of use, rather
        # than letting a wrong-shaped array silently reach rendering/export.
        expected_hw = (meta.get("height"), meta.get("width"))
        if arr.shape[:2] != expected_hw:
            raise HTTPException(
                422,
                f"Slice {idx} shape {arr.shape[:2]} does not match the stack's "
                f"declared shape {expected_hw}",
            )
        return arr
    raise HTTPException(422, f"Cannot slice shape kind: {kind}")
