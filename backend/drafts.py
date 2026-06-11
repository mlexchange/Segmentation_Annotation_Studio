"""Session-draft persistence for the SAM3 Annotation Studio.

Drafts are stored as JSON files under ``$LOCAL_DATA_ROOT/.drafts/``.
Each draft is keyed by an arbitrary *source_key* string (typically the Tiled
path or local file path of the opened image).  The key is hashed to a short
hex digest to produce a safe filename.

Writes are atomic (write-to-temp + rename) so a crash mid-save does not
produce a corrupt draft.
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


def save_draft(source_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a session draft for *source_key*.

    Writes atomically (temp file → rename).

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
            results.append(
                {
                    "source_key": doc.get("source_key"),
                    "saved_at": doc.get("saved_at"),
                    "file": f.name,
                }
            )
        except Exception:
            continue
    return results
