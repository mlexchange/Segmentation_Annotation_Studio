"""Sandboxed local filesystem access for the SAM3 Annotation Studio.

All paths are resolved relative to ``LOCAL_DATA_ROOT`` (environment variable).
Any attempt to escape the root via path traversal (e.g. ``../../etc/passwd``)
is rejected with HTTP 403.

Supported array formats
-----------------------
* ``.tif`` / ``.tiff``  — memory-mapped via ``tifffile``
* ``.npy``              — memory-mapped via ``numpy`` (no pickle)
* ``.png`` / ``.jpg`` / ``.jpeg`` — loaded fully via ``PIL``
"""

from __future__ import annotations

IMAGE_EXTS: frozenset[str] = frozenset({".tif", ".tiff", ".npy", ".png", ".jpg", ".jpeg"})

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)

_ROOT: Path = Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve()


def _safe(rel: str) -> Path:
    """Resolve *rel* under ``LOCAL_DATA_ROOT``; raise 403 on traversal.

    Args:
        rel: Relative path string supplied by the caller.

    Returns:
        Absolute :class:`pathlib.Path` guaranteed to be inside ``_ROOT``.

    Raises:
        HTTPException: 403 if the resolved path escapes ``_ROOT``.
    """
    resolved = (_ROOT / rel).resolve()
    if not str(resolved).startswith(str(_ROOT)):
        logger.warning("Path traversal attempt: %r", rel)
        raise HTTPException(403, "Path traversal not allowed")
    return resolved


def list_dir(rel: str = "") -> list[dict[str, Any]]:
    """List directory entries under ``LOCAL_DATA_ROOT``.

    Args:
        rel: Relative path to the directory to list (empty → root).

    Returns:
        List of dicts with keys ``name``, ``path``, ``is_dir``, ``size``.

    Raises:
        HTTPException: 404 if path does not exist; 400 if not a directory.
    """
    path = _safe(rel)
    if not path.exists():
        # A missing root directory is treated as empty rather than an error,
        # so the file browser can still render (and the user can fix the
        # LOCAL_DATA_ROOT configuration) instead of seeing a 404.
        if rel in ("", "."):
            logger.warning("LOCAL_DATA_ROOT does not exist: %s", _ROOT)
            return []
        raise HTTPException(404, f"Path not found: {rel!r}")
    if not path.is_dir():
        raise HTTPException(400, f"Not a directory: {rel!r}")
    entries: list[dict[str, Any]] = []
    for child in sorted(path.iterdir()):
        entries.append(
            {
                "name": child.name,
                "path": str(child.relative_to(_ROOT)),
                "is_dir": child.is_dir(),
                "size": child.stat().st_size if child.is_file() else None,
            }
        )
    return entries


def count_image_files(rel: str) -> int:
    """Return a recursive count of image files under ``LOCAL_DATA_ROOT/rel``.

    Args:
        rel: Relative path to a directory under ``LOCAL_DATA_ROOT``.

    Returns:
        Number of files with a supported image extension.
    """
    path = _safe(rel)
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def list_image_files(rel: str) -> list[dict[str, Any]]:
    """Return a flat, sorted list of image files under ``LOCAL_DATA_ROOT/rel``.

    Args:
        rel: Relative path to a directory under ``LOCAL_DATA_ROOT``.

    Returns:
        List of ``{"name", "path"}`` dicts sorted by path.

    Raises:
        HTTPException: 404 if path does not exist; 400 if not a directory.
    """
    path = _safe(rel)
    if not path.exists():
        raise HTTPException(404, f"Path not found: {rel!r}")
    if not path.is_dir():
        raise HTTPException(400, f"Not a directory: {rel!r}")
    entries = sorted(
        (
            {
                "name": f.name,
                "path": str(f.relative_to(_ROOT)),
            }
            for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda e: e["path"],
    )
    return entries


def open_array(rel: str) -> Any:
    """Open a local array file and return a lazily-sliceable array.

    Supported extensions: ``.tif``, ``.tiff``, ``.npy``, ``.png``,
    ``.jpg``, ``.jpeg``.

    Args:
        rel: Relative path to the file under ``LOCAL_DATA_ROOT``.

    Returns:
        A memory-mapped or fully-loaded NumPy array.

    Raises:
        HTTPException: 404 if file not found; 422 for unsupported type;
            500 on read error.
    """
    import tifffile
    from PIL import Image as PILImage

    path = _safe(rel)
    if not path.exists():
        raise HTTPException(404, f"File not found: {rel!r}")
    suffix = path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            return tifffile.memmap(str(path))
        if suffix == ".npy":
            return np.load(str(path), mmap_mode="r", allow_pickle=False)
        if suffix in (".png", ".jpg", ".jpeg"):
            arr = np.asarray(PILImage.open(str(path)))
            return arr
    except Exception as exc:
        logger.error("Failed to open %r: %s", rel, exc)
        raise HTTPException(500, f"Failed to open file: {exc}") from exc
    raise HTTPException(422, f"Unsupported file type: {suffix!r}")
