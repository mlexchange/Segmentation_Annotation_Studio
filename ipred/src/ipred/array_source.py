"""Minimal local / Tiled slice reader (no annotate-backend imports)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def _local_root(root: str | None) -> Path:
    if root:
        return Path(root).expanduser().resolve()
    return Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve()


def read_slice(
    *,
    kind: str,
    source: str,
    slice_index: int = 0,
    server_uri: str | None = None,
    root: str | None = None,
) -> np.ndarray:
    """Load one 2D/ HxWxC slice array."""
    if kind == "local":
        return _read_local(source, slice_index=slice_index, root=root)
    if kind == "tiled":
        return _read_tiled(
            source, slice_index=slice_index, server_uri=server_uri
        )
    raise ValueError(f"unknown kind {kind!r}")


def _read_local(
    source: str,
    *,
    slice_index: int,
    root: str | None,
) -> np.ndarray:
    base = _local_root(root)
    path = (base / source).resolve()
    if not str(path).startswith(str(base)):
        raise PermissionError("local path escapes LOCAL_DATA_ROOT")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        from PIL import Image as PILImage

        arr = np.asarray(PILImage.open(path))
    return _index_slice(np.asarray(arr), slice_index)


def _read_tiled(
    source: str,
    *,
    slice_index: int,
    server_uri: str | None,
) -> np.ndarray:
    from tiled.client import from_uri

    uri = (server_uri or os.getenv("TILED_URI") or "http://127.0.0.1:8010").rstrip(
        "/"
    )
    api_key = os.getenv("TILED_API_KEY")
    client: Any = from_uri(uri, api_key=api_key) if api_key else from_uri(uri)
    node: Any = client
    for part in source.strip("/").split("/"):
        if not part:
            continue
        node = node[part]
    # Descend one level if container with a single array child
    sf = getattr(node, "structure_family", None)
    if str(getattr(sf, "value", sf)) == "container":
        keys = list(node)
        if len(keys) == 1:
            node = node[keys[0]]
    data = np.asarray(node)
    return _index_slice(data, slice_index)


def _index_slice(arr: np.ndarray, slice_index: int) -> np.ndarray:
    """Pick HW or HWC frame from NHW / NHWC / HW / HWC."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # HWC vs NHW
        if a.shape[-1] in (1, 3, 4) and a.shape[0] > 8:
            return a  # HWC single
        if a.shape[-1] in (1, 3, 4) and a.shape[0] <= 8:
            # ambiguous small N — treat as HWC if last dim channel-like and square-ish
            if a.shape[0] == a.shape[1]:
                return a
        return a[int(slice_index)]
    if a.ndim == 4:
        return a[int(slice_index)]
    raise ValueError(f"unsupported array shape {a.shape}")
