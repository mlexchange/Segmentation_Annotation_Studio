"""Session-draft persistence and version history for the SAM3 Annotation Studio.

Drafts are stored as JSON files under ``$LOCAL_DATA_ROOT/.drafts/``.
Each draft is keyed by an arbitrary *source_key* string (typically the Tiled
path or local file path of the opened image).  The key is hashed to a short
hex digest to produce a safe filename.

Writes are atomic (write-to-temp + rename) so a crash mid-save does not
produce a corrupt draft.

Versions are stored in a ``<digest>.versions/`` subdirectory alongside the
draft.  Each version is an immutable JSON file named ``v<NNNN>.json``.
Versions are created only by an explicit user Save action, not by autosave.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_DRAFT_DIR = (
    Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve() / ".drafts"
)


def _draft_path(source_key: str) -> Path:
    """Return the JSON file path for *source_key*.

    Args:
        source_key: Arbitrary string identifying the image source.

    Returns:
        Absolute path to the draft JSON file.
    """
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]
    return _DRAFT_DIR / f"{digest}.json"


def _versions_dir(source_key: str) -> Path:
    """Return the versions directory path for *source_key*."""
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]
    return _DRAFT_DIR / f"{digest}.versions"


def save_draft(source_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a session draft for *source_key* (crash-recovery autosave).

    Does NOT sync metadata to Tiled — that is reserved for explicit saves via
    :func:`save_version`.  Writes atomically (temp file → rename).

    Args:
        source_key: Arbitrary string identifying the image source.
        payload: Session state dict (classes, slices, …).

    Returns:
        Dict with ``saved_at`` (ISO-8601 timestamp) and ``path`` (str).
    """
    _DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "source_key": source_key,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    path = _draft_path(source_key)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)
    return {"saved_at": doc["saved_at"], "path": str(path)}


def load_draft(source_key: str) -> dict[str, Any] | None:
    """Return the saved draft for *source_key*, or ``None``.

    Args:
        source_key: Arbitrary string identifying the image source.

    Returns:
        Parsed draft document, or ``None`` if no draft exists.
    """
    path = _draft_path(source_key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Corrupt draft %s: %s", path, exc)
        return None


def list_drafts() -> list[dict[str, Any]]:
    """List all saved drafts with summary metadata.

    Returns:
        List of dicts with ``source_key``, ``saved_at``, and ``file`` keys,
        sorted by filename (stable across runs).
    """
    if not _DRAFT_DIR.exists():
        return []
    results: list[dict[str, Any]] = []
    for f in sorted(_DRAFT_DIR.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
            payload = doc.get("payload") or {}
            slices = payload.get("slices") or {}
            has_annotations = any(
                isinstance(shapes, list) and len(shapes) > 0 for shapes in slices.values()
            )
            results.append(
                {
                    "source_key": doc.get("source_key"),
                    "saved_at": doc.get("saved_at"),
                    "file": f.name,
                    "has_annotations": has_annotations,
                }
            )
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Version history (explicit user saves)
# ---------------------------------------------------------------------------


def save_version(
    source_key: str,
    payload: dict[str, Any],
    *,
    annotated_by: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Create a new immutable version snapshot for *source_key*.

    Also updates the draft file so the crash-recovery draft stays in sync.

    Args:
        source_key: Arbitrary string identifying the image source.
        payload: Full session state dict (classes, slices, …).
        annotated_by: Optional name of the annotator for this version.
        notes: Optional free-text notes for this version.

    Returns:
        Dict with ``version`` (int), ``saved_at`` (ISO-8601), and ``path`` (str).
    """
    _DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    vdir = _versions_dir(source_key)
    vdir.mkdir(parents=True, exist_ok=True)

    existing = sorted(vdir.glob("v*.json"))
    n = len(existing) + 1
    saved_at = datetime.now(timezone.utc).isoformat()

    # Count shapes for the summary
    slices = payload.get("slices") or {}
    shape_count = sum(
        len(shapes) for shapes in slices.values() if isinstance(shapes, list)
    )
    class_count = len(payload.get("classes") or [])

    doc: dict[str, Any] = {
        "source_key": source_key,
        "version": n,
        "saved_at": saved_at,
        "shape_count": shape_count,
        "class_count": class_count,
        "annotated_by": annotated_by.strip() or None,
        "notes": notes.strip() or None,
        "payload": payload,
    }
    vpath = vdir / f"v{n:04d}.json"
    tmp = vpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(vpath)

    # Keep the crash-recovery draft in sync with the explicit save
    save_draft(source_key, payload)

    return {"version": n, "saved_at": saved_at, "shape_count": shape_count, "path": str(vpath)}


def list_versions(source_key: str) -> list[dict[str, Any]]:
    """List version metadata (without payload) sorted oldest-first.

    Args:
        source_key: Arbitrary string identifying the image source.

    Returns:
        List of dicts with ``version``, ``saved_at``, ``shape_count``,
        ``class_count``, and ``file``.
    """
    vdir = _versions_dir(source_key)
    if not vdir.exists():
        return []
    results: list[dict[str, Any]] = []
    for f in sorted(vdir.glob("v*.json")):
        try:
            doc = json.loads(f.read_text())
            version_num = doc.get("version")
            thumb_exists = (vdir / f"v{version_num:04d}.thumb.png").exists() if version_num else False
            results.append(
                {
                    "version": version_num,
                    "saved_at": doc.get("saved_at"),
                    "shape_count": doc.get("shape_count", 0),
                    "class_count": doc.get("class_count", 0),
                    "annotated_by": doc.get("annotated_by"),
                    "notes": doc.get("notes"),
                    "has_thumbnail": thumb_exists,
                    "file": f.name,
                }
            )
        except Exception:
            continue
    return results


def save_version_thumbnail(source_key: str, version: int, png_bytes: bytes) -> None:
    """Persist an annotated thumbnail PNG for *version*.

    Stored alongside the version JSON as ``v{version:04d}.thumb.png``.

    Args:
        source_key: Arbitrary string identifying the image source.
        version: 1-based version number.
        png_bytes: PNG-encoded thumbnail bytes.
    """
    vdir = _versions_dir(source_key)
    vdir.mkdir(parents=True, exist_ok=True)
    thumb_path = vdir / f"v{version:04d}.thumb.png"
    tmp = thumb_path.with_suffix(".png.tmp")
    tmp.write_bytes(png_bytes)
    tmp.replace(thumb_path)


def get_version_thumbnail(source_key: str, version: int) -> bytes | None:
    """Return the thumbnail PNG bytes for *version*, or ``None``.

    Args:
        source_key: Arbitrary string identifying the image source.
        version: 1-based version number.

    Returns:
        PNG bytes, or ``None`` if no thumbnail exists.
    """
    thumb_path = _versions_dir(source_key) / f"v{version:04d}.thumb.png"
    if not thumb_path.exists():
        return None
    return thumb_path.read_bytes()


def get_version(source_key: str, version: int) -> dict[str, Any] | None:
    """Return the full document (including payload) for a specific version.

    Args:
        source_key: Arbitrary string identifying the image source.
        version: 1-based version number.

    Returns:
        Parsed version document, or ``None`` if it does not exist.
    """
    vdir = _versions_dir(source_key)
    vpath = vdir / f"v{version:04d}.json"
    if not vpath.exists():
        return None
    try:
        return json.loads(vpath.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Corrupt version %s: %s", vpath, exc)
        return None
