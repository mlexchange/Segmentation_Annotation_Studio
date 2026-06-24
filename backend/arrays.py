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

from cache import TTLCache
import local_fs
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
    elif kind == "local":
        node = local_fs.open_array(source, root)
    else:
        raise HTTPException(422, f"Unknown source kind: {kind!r}")

    _node_cache.set(key, node)
    return node


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
    raise HTTPException(422, f"Cannot slice shape kind: {kind}")
