"""# REMOVE THIS AND USE YOUR OWN STUFF

Reusable annotation label sets (class id + label + color).

TEMPORARY scaffold storage under ``$LOCAL_DATA_ROOT/.label_sets/<id>/meta.json``
for Connect/Browse. Replace with your own label taxonomy backend, then delete
this module and the ``/api/label-sets`` routes.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ##############################################################################
# # REMOVE THIS AND USE YOUR OWN STUFF — disk label-set catalog
# ##############################################################################


def _root() -> Path:
    return Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve() / ".label_sets"


@dataclass(frozen=True)
class LabelSetMeta:
    """Indexed label-set summary."""

    id: str
    name: str
    n_classes: int
    created_at: str


def _dir_for(set_id: str) -> Path:
    return _root() / set_id


def _validate_classes(classes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and validate class entries."""
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in classes:
        if not isinstance(raw, dict):
            raise ValueError("each class must be an object")
        cid = int(raw.get("classId") or raw.get("class_id") or 0)
        if cid <= 0:
            raise ValueError("classId must be a positive integer")
        if cid in seen:
            raise ValueError(f"duplicate classId {cid}")
        seen.add(cid)
        label = str(raw.get("label") or f"class_{cid}").strip() or f"class_{cid}"
        color = str(raw.get("color") or "#1f77b4")
        visible = bool(raw.get("isVisible", raw.get("is_visible", True)))
        out.append(
            {
                "classId": cid,
                "label": label,
                "color": color,
                "isVisible": visible,
            }
        )
    if not out:
        raise ValueError("label set must include at least one class")
    return out


def save_label_set(*, name: str, classes: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist a named label set; return full record."""
    cleaned = _validate_classes(classes)
    set_id = uuid.uuid4().hex
    dest = _dir_for(set_id)
    dest.mkdir(parents=True, exist_ok=True)
    record = {
        "id": set_id,
        "name": (name or "Untitled labels").strip() or "Untitled labels",
        "classes": cleaned,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (dest / "meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Saved label set %s (%s, n=%d)", set_id, record["name"], len(cleaned))
    return record


def list_label_sets() -> list[dict[str, Any]]:
    """Return summaries for all saved label sets (newest first)."""
    root = _root()
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for child in root.iterdir():
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": data["id"],
                    "name": data.get("name", data["id"]),
                    "n_classes": len(data.get("classes") or []),
                    "created_at": data.get("created_at", ""),
                    "classes": data.get("classes") or [],
                }
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Skipping corrupt label set %s: %s", child, exc)
    items.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return items


def get_label_set(set_id: str) -> dict[str, Any] | None:
    """Load one label set by id."""
    meta_path = _dir_for(set_id) / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def delete_label_set(set_id: str) -> bool:
    """Delete a label set directory. Returns False if missing."""
    import shutil

    dest = _dir_for(set_id)
    if not dest.is_dir():
        return False
    shutil.rmtree(dest)
    return True
