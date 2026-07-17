"""Re-ingest stacked TIFF array as 690 individual sample nodes.

The 690 TIFFs were previously registered as one stacked multi-chunk array.
This script splits them back into individual nodes so the Browse UI can
treat each TIFF as a separate sample.

Run from the REPO ROOT (not from backend/scripts):
    python backend/scripts/reingest_tiffs_as_individual.py

Tiled must NOT be running when you run this (it modifies the catalog DB
directly). Restart Tiled after running.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

try:
    import canonicaljson
except ImportError:
    print("ERROR: canonicaljson not installed. Run: pip install canonicaljson")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DB = REPO_ROOT / ".tiled" / "catalog.db"

STACKED_KEY = "20260221_135217_petiole22_"

# Structure for a single 2560×2560 float32 TIFF frame.
SINGLE_TIFF_STRUCTURE = {
    "data_type": {"endianness": "little", "itemsize": 4, "kind": "f", "dt_units": None},
    "chunks": [[2560], [2560]],
    "shape": [2560, 2560],
    "dims": None,
    "resizable": False,
}


def compute_structure_id(structure: dict) -> str:
    canonical = canonicaljson.encode_canonical_json(structure)
    return hashlib.md5(canonical).hexdigest()


def main(dry_run: bool = False) -> None:
    if not CATALOG_DB.exists():
        print(f"ERROR: catalog.db not found at {CATALOG_DB}")
        sys.exit(1)

    conn = sqlite3.connect(str(CATALOG_DB))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # --- find the stacked node ---
    c.execute("SELECT id, parent FROM nodes WHERE key = ?", (STACKED_KEY,))
    stacked_row = c.fetchone()
    if stacked_row is None:
        print(f"No node with key '{STACKED_KEY}' found — nothing to do.")
        conn.close()
        return

    stacked_node_id = stacked_row["id"]
    root_node_id = stacked_row["parent"]
    print(f"Found stacked node id={stacked_node_id}, parent={root_node_id}")

    # --- collect assets (TIFFs) in order ---
    c.execute(
        """
        SELECT a.id AS asset_id, a.data_uri, dsaa.num
        FROM data_source_asset_association dsaa
        JOIN data_sources ds ON dsaa.data_source_id = ds.id
        JOIN assets a ON dsaa.asset_id = a.id
        WHERE ds.node_id = ?
        ORDER BY dsaa.num
        """,
        (stacked_node_id,),
    )
    tiff_rows = c.fetchall()
    print(f"Found {len(tiff_rows)} TIFF assets to re-register.")

    if len(tiff_rows) == 0:
        print("No assets found for the stacked node. Aborting.")
        conn.close()
        return

    if dry_run:
        for r in tiff_rows[:5]:
            print(f"  [{r['num']}] {r['data_uri']}")
        print("  ... (dry run, not modifying DB)")
        conn.close()
        return

    struct_id = compute_structure_id(SINGLE_TIFF_STRUCTURE)
    struct_json = json.dumps(SINGLE_TIFF_STRUCTURE)

    # --- delete stacked node (cascades to data_sources and associations) ---
    print(f"Deleting stacked node id={stacked_node_id} ...")
    c.execute("DELETE FROM nodes WHERE id = ?", (stacked_node_id,))

    # --- ensure single-TIFF structure exists ---
    c.execute("SELECT id FROM structures WHERE id = ?", (struct_id,))
    if c.fetchone() is None:
        c.execute("INSERT INTO structures (id, structure) VALUES (?, ?)", (struct_id, struct_json))
        print(f"Inserted structure id={struct_id}")

    # --- insert 690 individual nodes ---
    print("Inserting individual nodes ...")
    inserted = 0
    for row in tiff_rows:
        uri: str = row["data_uri"]
        asset_id: int = row["asset_id"]

        # Derive key from the filename stem
        fname = uri.split("/")[-1]
        stem = fname.rsplit(".", 1)[0] if "." in fname else fname
        node_key = stem

        # Insert node
        c.execute(
            """
            INSERT INTO nodes (parent, key, structure_family, metadata, specs, access_blob)
            VALUES (?, ?, 'array', '{}', '[]', '{}')
            """,
            (root_node_id, node_key),
        )
        node_id = c.lastrowid

        # Insert data_source
        c.execute(
            """
            INSERT INTO data_sources
                (node_id, structure_id, mimetype, parameters, properties, management, structure_family)
            VALUES (?, ?, 'image/tiff', '{}', '{}', 'external', 'array')
            """,
            (node_id, struct_id),
        )
        ds_id = c.lastrowid

        # Insert association (single file → parameter=data_uri, num=NULL)
        c.execute(
            """
            INSERT INTO data_source_asset_association
                (data_source_id, asset_id, parameter, num)
            VALUES (?, ?, 'data_uri', NULL)
            """,
            (ds_id, asset_id),
        )

        inserted += 1
        if inserted % 100 == 0:
            print(f"  {inserted}/{len(tiff_rows)} ...")

    conn.commit()
    print(f"Done. Inserted {inserted} individual sample nodes.")
    print("Restart Tiled to pick up the changes.")
    conn.close()


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
