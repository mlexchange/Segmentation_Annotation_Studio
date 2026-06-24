"""Add browsable metadata to each TIFF sample node.

The Browse UI is metadata-driven: it can only list/filter samples that have a
metadata field with >=2 distinct values. Freshly-registered TIFF nodes have
empty metadata ({}), so the browser shows nothing to add as a column.

This script writes per node:
    image_number : frame index parsed from the trailing digits of the key,
                   zero-padded to match the filenames (e.g. "00000".."00689").
    size         : pixel dimensions from the node's array structure,
                   e.g. "2560 x 2560".

Padding keeps the browser column in natural order (lexical sort == numeric).

Run from the REPO ROOT:
    python backend/scripts/add_sample_metadata.py
Refresh the Browse UI (or restart Tiled) afterwards.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DB = REPO_ROOT / ".tiled" / "catalog.db"

# Trailing digits of the node key, e.g. "..._petiole22_00042" -> "00042".
_FRAME_RE = re.compile(r"(\d+)$")


def main(dry_run: bool = False) -> None:
    if not CATALOG_DB.exists():
        print(f"ERROR: catalog.db not found at {CATALOG_DB}")
        sys.exit(1)

    conn = sqlite3.connect(str(CATALOG_DB))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Sample nodes live directly under the root (parent=0). Join through the
    # data_source to the structure so we can read each node's pixel shape.
    rows = c.execute(
        """
        SELECT n.id, n.key, n.metadata, s.structure
        FROM nodes n
        LEFT JOIN data_sources ds ON ds.node_id = n.id
        LEFT JOIN structures s ON s.id = ds.structure_id
        WHERE n.structure_family = 'array' AND n.parent = 0
        """
    ).fetchall()

    print(f"Found {len(rows)} sample nodes.")
    if not rows:
        print("Nothing to update.")
        conn.close()
        return

    # Use a consistent pad width across the whole set.
    width = max(
        (len(m.group(1)) for m in (_FRAME_RE.search(r["key"]) for r in rows) if m),
        default=1,
    )

    updated = 0
    for row in rows:
        m = _FRAME_RE.search(row["key"])
        if not m:
            continue
        new_fields = {"image_number": m.group(1).zfill(width)}

        # Pixel dimensions from the array structure, e.g. "2560 x 2560".
        if row["structure"]:
            try:
                shape = json.loads(row["structure"]).get("shape") or []
                if shape:
                    new_fields["size"] = " x ".join(str(d) for d in shape)
            except (TypeError, json.JSONDecodeError):
                pass

        try:
            existing = json.loads(row["metadata"]) if row["metadata"] else {}
        except (TypeError, json.JSONDecodeError):
            existing = {}

        merged = {**existing, **new_fields}
        if merged == existing:
            continue

        if dry_run:
            if updated < 3:
                print(f"  {row['key']} -> {new_fields}")
        else:
            c.execute(
                "UPDATE nodes SET metadata = ? WHERE id = ?",
                (json.dumps(merged), row["id"]),
            )
        updated += 1

    if dry_run:
        print(f"Would update {updated} node(s) (dry run).")
    else:
        conn.commit()
        print(f"Set image_number on {updated} node(s).")
        print("Refresh the Browse UI (or restart Tiled) to see the 'image_number' column.")
    conn.close()


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
