"""Sync annotation session state into Tiled node metadata for Browse discovery."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import arrays as arrays_mod
from source_keys import parse_source_key

logger = logging.getLogger(__name__)

# Metadata keys written onto Tiled array nodes (searchable via Browse facets).
STUDIO_ANNOTATED = "studio_annotated"  # "yes" | "no"
STUDIO_SHAPE_COUNT = "studio_shape_count"
STUDIO_CLASS_COUNT = "studio_class_count"
STUDIO_UPDATED_AT = "studio_updated_at"


def annotation_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Build Tiled-safe metadata fields from a draft/export payload."""
    slices = payload.get("slices") or {}
    shape_count = sum(
        len(shapes) for shapes in slices.values() if isinstance(shapes, list)
    )
    classes = payload.get("classes") or []
    return {
        STUDIO_ANNOTATED: "yes" if shape_count > 0 else "no",
        STUDIO_SHAPE_COUNT: int(shape_count),
        STUDIO_CLASS_COUNT: int(len(classes)) if isinstance(classes, list) else 0,
        STUDIO_UPDATED_AT: datetime.now(timezone.utc).isoformat(),
    }


def sync_annotation_metadata(source_key: str, payload: dict[str, Any]) -> None:
    """Merge annotation summary metadata onto a Tiled node.

    No-op for local sources. Logs and re-raises on Tiled write failure so the
    caller can decide whether to fail the request or continue.

    Args:
        source_key: Canonical source key (``tiled:…`` or ``local:…``).
        payload: Draft or export payload with ``slices`` and ``classes``.
    """
    parsed = parse_source_key(source_key)
    if parsed["kind"] != "tiled" or not parsed["path"]:
        return

    node = arrays_mod.resolve_array(parsed["path"], "tiled", parsed["server_uri"])
    meta = annotation_metadata(payload)

    if not hasattr(node, "update_metadata"):
        logger.warning("Tiled node at %r has no update_metadata; skipping sync", parsed["path"])
        return

    node.update_metadata(meta)
    logger.info(
        "Synced annotation metadata to Tiled %r (%s, %d shapes)",
        parsed["path"],
        meta[STUDIO_ANNOTATED],
        meta[STUDIO_SHAPE_COUNT],
    )
