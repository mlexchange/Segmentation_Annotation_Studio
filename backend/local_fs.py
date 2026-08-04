"""Sandboxed local filesystem access for the Segmentation Annotation Studio.

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

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import HTTPException

IMAGE_EXTS: frozenset[str] = frozenset(
    {".tif", ".tiff", ".npy", ".png", ".jpg", ".jpeg"}
)

logger = logging.getLogger(__name__)

load_dotenv()

_DEFAULT_ROOT: Path = (
    Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve()
)


def _load_allowed_roots() -> tuple[Path, ...]:
    """Load the server-owned local filesystem allowlist.

    ``LOCAL_DATA_ROOT`` is always included. Operators may add roots with the
    platform path separator in ``LOCAL_DATA_ROOTS`` (``:`` on Unix, ``;`` on
    Windows). Request parameters can select a root or descendant from this
    list, but can never grant the process new filesystem authority.
    """
    roots = [_DEFAULT_ROOT]
    for raw in (os.getenv("LOCAL_DATA_ROOTS") or "").split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw.strip()).expanduser().resolve())
    return tuple(dict.fromkeys(roots))


_ALLOWED_ROOTS: tuple[Path, ...] = _load_allowed_roots()


def _resolve_root(root: str | None) -> Path:
    """Return an allowlisted browse root as an absolute, resolved Path.

    Args:
        root: Requested root under a server-configured root, or ``None`` to
            use ``LOCAL_DATA_ROOT``.

    Returns:
        Absolute :class:`pathlib.Path`.

    Raises:
        HTTPException: 403 when the requested root is not allowlisted.
    """
    selected = Path(root).expanduser().resolve() if root else _DEFAULT_ROOT
    if not any(_within(allowed, selected) for allowed in _ALLOWED_ROOTS):
        logger.warning("Rejected unconfigured local data root: %s", selected)
        raise HTTPException(403, "Local data root is not configured")
    return selected


def _within(base: Path, resolved: Path) -> bool:
    """Return True if *resolved* is *base* itself or a descendant of it.

    Parent containment avoids the string-prefix footgun where ``/data2`` would
    slip past a ``/data`` root.
    """
    return base == resolved or base in resolved.parents


def _safe(rel: str, root: str | None = None) -> Path:
    """Resolve *rel* to an absolute path with traversal protection.

    Three cases, all constrained by the server-owned root allowlist:

    * *root* given → validate the selected root, join under it, and enforce
      containment (the sandboxed Browse flow).
    * *root* omitted and *rel* absolute → allow re-opening a chosen file only
      when the resolved path remains under a configured root.
    * *root* omitted and *rel* relative → legacy behaviour under
      ``LOCAL_DATA_ROOT`` with containment enforced.

    Raises:
        HTTPException: 403 if a sandboxed path escapes its root.
    """
    if root:
        base = _resolve_root(root)
        resolved = (base / rel).resolve()
        if not _within(base, resolved):
            logger.warning("Path traversal attempt: rel=%r root=%r", rel, root)
            raise HTTPException(403, "Path traversal not allowed")
        return resolved

    candidate = Path(rel).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not any(_within(allowed, resolved) for allowed in _ALLOWED_ROOTS):
            logger.warning(
                "Rejected absolute path outside configured roots: %s", resolved
            )
            raise HTTPException(403, "Path is outside configured local data roots")
        return resolved

    base = _DEFAULT_ROOT
    resolved = (base / rel).resolve()
    if not _within(base, resolved):
        logger.warning("Path traversal attempt: rel=%r", rel)
        raise HTTPException(403, "Path traversal not allowed")
    return resolved


def list_dir(rel: str = "", root: str | None = None) -> list[dict[str, Any]]:
    """List directory entries under the granted *root*.

    Args:
        rel: Relative path to the directory to list (empty → root).
        root: Server-configured root (defaults to ``LOCAL_DATA_ROOT``).

    Returns:
        List of dicts with keys ``name``, ``path``, ``is_dir``, ``size``.

    Raises:
        HTTPException: 404 if path does not exist; 400 if not a directory.
    """
    base = _resolve_root(root)
    path = _safe(rel, root)
    if not path.exists():
        # A missing root directory is treated as empty rather than an error,
        # so the file browser can still render (and the user can fix the
        # granted path) instead of seeing a 404.
        if rel in ("", "."):
            logger.warning("Browse root does not exist: %s", base)
            return []
        raise HTTPException(404, f"Path not found: {rel!r}")
    if not path.is_dir():
        raise HTTPException(400, f"Not a directory: {rel!r}")
    entries: list[dict[str, Any]] = []
    for child in sorted(path.iterdir()):
        entries.append(
            {
                "name": child.name,
                "path": str(child.relative_to(base)),
                "is_dir": child.is_dir(),
                "size": child.stat().st_size if child.is_file() else None,
            }
        )
    return entries


def count_image_files(rel: str, root: str | None = None) -> int:
    """Return a recursive count of image files under ``<root>/rel``.

    Args:
        rel: Relative path to a directory under the configured root.
        root: Server-configured root (defaults to ``LOCAL_DATA_ROOT``).

    Returns:
        Number of files with a supported image extension.
    """
    path = _safe(rel, root)
    if not path.exists() or not path.is_dir():
        return 0
    return sum(
        1 for f in path.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


def list_image_files(rel: str, root: str | None = None) -> list[dict[str, Any]]:
    """Return a flat, sorted list of image files under ``<root>/rel``.

    Args:
        rel: Relative path to a directory under the configured root.
        root: Server-configured root (defaults to ``LOCAL_DATA_ROOT``).

    Returns:
        List of ``{"name", "path"}`` dicts sorted by path.

    Raises:
        HTTPException: 404 if path does not exist; 400 if not a directory.
    """
    base = _resolve_root(root)
    path = _safe(rel, root)
    if not path.exists():
        raise HTTPException(404, f"Path not found: {rel!r}")
    if not path.is_dir():
        raise HTTPException(400, f"Not a directory: {rel!r}")
    entries = sorted(
        (
            {
                "name": f.name,
                "path": str(f.relative_to(base)),
            }
            for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda e: e["path"],
    )
    return entries


def open_array(rel: str, root: str | None = None) -> Any:
    """Open a local array file and return a lazily-sliceable array.

    Supported extensions: ``.tif``, ``.tiff``, ``.npy``, ``.png``,
    ``.jpg``, ``.jpeg``.

    Args:
        rel: Relative path to the file under the configured root.
        root: Server-configured root (defaults to ``LOCAL_DATA_ROOT``).

    Returns:
        A memory-mapped or fully-loaded NumPy array.

    Raises:
        HTTPException: 404 if file not found; 422 for unsupported type;
            500 on read error.
    """
    import tifffile
    from PIL import Image as PILImage

    path = _safe(rel, root)
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
        raise HTTPException(500, "Failed to open local image") from exc
    raise HTTPException(422, f"Unsupported file type: {suffix!r}")
