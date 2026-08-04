# Local Tiled runtime directory

Only this README is version-controlled. A fresh clone does not include a catalog
or sample arrays. `start_all.sh` or `windows/start_all.ps1` creates the SQLite
catalog on first launch, and Tiled writes local arrays beneath this directory.

- **`catalog.db`** — generated Tiled catalog metadata.
- **`data/`** — generated writable storage (Zarr chunks).

After clone, run `./start_all.sh` from the repo root. The script creates
`backend/.env` from `.env.example` and **generates a strong `TILED_API_KEY`** into it
when the key is missing (or still the old committed value, which it auto-rotates).
`tiled/config.yml` holds no literal key — the script passes the generated key to
Tiled at launch via `--api-key` (robust across Tiled versions). The key lives only
in the gitignored `backend/.env`, never inside these catalog files and never sent
to the frontend.

To reset the catalog, stop the services, remove the generated catalog and data,
and let the launcher initialize them again. This is destructive, so back up any
locally ingested data first. To create demo datasets, use
`backend/scripts/seed_generated_data_to_tiled.py`.

## Security note
Earlier commits of `tiled/config.yml` contained a hardcoded `single_user_api_key`.
It has been removed from the working tree and is auto-rotated on the next local
launch, but it **remains in git history**. It only guards a
local-only (127.0.0.1) server with anonymous read enabled, so exposure is low. To
purge it from history, use `git filter-repo` (or BFG) to replace the blob, then
force-push — destructive; coordinate with anyone who has cloned the repo.
