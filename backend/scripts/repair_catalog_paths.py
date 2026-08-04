"""Re-anchor catalog.db asset paths to the current repo location.

Tiled stores absolute file:// URIs in the catalog database. If the repo is
moved or cloned to a different path the stored URIs point at the old location
and Tiled refuses to serve the files. This script rewrites all asset URIs so
they point to the .tiled/data directory relative to the repo root.

Usage (run from the repo root):
    python backend/scripts/repair_catalog_paths.py

start_all.sh calls this automatically before starting Tiled.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DB = REPO_ROOT / ".tiled" / "catalog.db"
DATA_DIR = REPO_ROOT / ".tiled" / "data"


def repair(dry_run: bool = False) -> int:
    if not CATALOG_DB.exists():
        print(f"catalog.db not found at {CATALOG_DB} — nothing to do.")
        return 0

    conn = sqlite3.connect(str(CATALOG_DB))
    c = conn.cursor()

    # Only repair managed assets (those stored under .tiled/data/).
    # External assets (management='external') live at user-supplied paths and
    # should never be rewritten here — they're intentionally outside the repo.
    managed_asset_ids: set[int] = set(
        row[0]
        for row in c.execute(
            """
            SELECT DISTINCT a.id
            FROM assets a
            JOIN data_source_asset_association dsaa ON dsaa.asset_id = a.id
            JOIN data_sources ds ON ds.id = dsaa.data_source_id
            WHERE ds.management != 'external'
            """
        ).fetchall()
    )

    rows = c.execute("SELECT id, data_uri FROM assets").fetchall()

    updated = 0
    for row_id, uri in rows:
        if row_id not in managed_asset_ids:
            continue  # external asset — leave it alone
        if not uri.startswith("file://localhost"):
            continue
        # Extract the path portion after the scheme+host
        file_path = Path(uri[len("file://localhost"):])

        # Find .tiled/data/ anchor anywhere in the path
        parts = file_path.parts
        try:
            idx = next(
                i for i, p in enumerate(parts)
                if p == "data" and i > 0 and parts[i - 1] == ".tiled"
            )
            rel_parts = parts[idx + 1:]
        except StopIteration:
            print(f"  [{row_id}] cannot parse managed URI, skipping: {uri}")
            continue

        new_path = DATA_DIR.joinpath(*rel_parts)
        new_uri = f"file://localhost{new_path}"

        if new_uri == uri:
            continue

        print(f"  [{row_id}] {uri}")
        print(f"       -> {new_uri}")
        if not dry_run:
            c.execute("UPDATE assets SET data_uri=? WHERE id=?", (new_uri, row_id))
        updated += 1

    if updated and not dry_run:
        conn.commit()
        print(f"Updated {updated} asset URI(s) in {CATALOG_DB}.")
    elif updated and dry_run:
        print(f"Would update {updated} asset URI(s) (dry run).")
    else:
        print("All asset URIs already point to the correct location.")

    conn.close()
    return updated


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    repair(dry_run=dry_run)
