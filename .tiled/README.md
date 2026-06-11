# Bundled Tiled catalog (committed to git)

This directory is **version-controlled** so a fresh `git clone` includes a working
SQLite catalog and sample array data (~1 MB).

- **`catalog.db`** — Tiled catalog metadata (paths, structure, metadata keys).
- **`data/`** — Writable storage (Zarr chunks) for `browse/generated_data/*` sample datasets.

After clone, run `./start_all.sh` from the repo root. The script still creates
`backend/.env` from `.env.example` and may generate a `TILED_API_KEY` if missing;
that key is only for authenticating to the local server, not stored inside these files.

To replace this with an empty catalog, delete `.tiled/` and run `tiled catalog init`
(see `start_all.sh`). To add more datasets, use `backend/scripts/seed_generated_data_to_tiled.py`.
