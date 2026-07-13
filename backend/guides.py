"""Annotation-guide persistence for the SAM3 Annotation Studio.

A *guide* is a project lead's curated description of each class — label, color,
a free-text description of what the class is and how it looks, and a few example
image crops.  Annotators see the guide in the Reference tab and get its classes
offered as one-click suggestions, so everyone labels the same thing the same way.

Guides are dataset-scoped: stored as JSON under ``$LOCAL_DATA_ROOT/.drafts/``
keyed by the same *source_key* used for drafts, so opening a dataset surfaces its
guide automatically.  A guide can additionally be exported as a shareable bundle
(see ``coco_export``) for hand-off to another machine.

Writes are atomic (write-to-temp + rename), matching :mod:`drafts`.
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


def _guide_path(source_key: str) -> Path:
    """Return the JSON file path for *source_key*'s guide."""
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]
    return _DRAFT_DIR / f"{digest}.guide.json"


def save_guide(source_key: str, guide: dict[str, Any]) -> dict[str, Any]:
    """Persist the annotation guide for *source_key*.

    Args:
        source_key: Arbitrary string identifying the image source.
        guide: Guide document (``{classes: [...], notes?}``).

    Returns:
        Dict with ``saved_at`` (ISO-8601 timestamp) and ``path`` (str).
    """
    _DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "source_key": source_key,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "guide": guide,
    }
    path = _guide_path(source_key)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)
    return {"saved_at": doc["saved_at"], "path": str(path)}


def load_guide(source_key: str) -> dict[str, Any] | None:
    """Return the saved guide document for *source_key*, or ``None``."""
    path = _guide_path(source_key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Corrupt guide %s: %s", path, exc)
        return None
