# Bundled Tiled catalog (committed to git)

This directory is **version-controlled** so a fresh `git clone` includes a working
SQLite catalog and sample array data (~1 MB).

- **`catalog.db`** — Tiled catalog metadata (paths, structure, metadata keys).
- **`data/`** — Writable storage (Zarr chunks) for `browse/generated_data/*` sample datasets.

After clone, run `./start_all.sh` from the repo root. The script creates
`backend/.env` from `.env.example` and **generates a strong `TILED_API_KEY`** into it
when the key is missing (or still the old committed value, which it auto-rotates).
`tiled/config.yml` holds no literal key — the script passes the generated key to
Tiled at launch via `--api-key` (robust across Tiled versions). The key lives only
in the gitignored `backend/.env`, never inside these catalog files and never sent
to the frontend.

To replace this with an empty catalog, delete `.tiled/` and run `tiled catalog init`
(see `start_all.sh`). To add more datasets, use `backend/scripts/seed_generated_data_to_tiled.py`.

## Security note
Earlier commits of `tiled/config.yml` contained a hardcoded `single_user_api_key`
(`3b1d23cd…`). It has been removed from the working tree and is auto-rotated on the
next `./start_all.sh`, but it **remains in git history**. It only guards a
local-only (127.0.0.1) server with anonymous read enabled, so exposure is low. To
purge it from history, use `git filter-repo` (or BFG) to replace the blob, then
force-push — destructive; coordinate with anyone who has cloned the repo.
