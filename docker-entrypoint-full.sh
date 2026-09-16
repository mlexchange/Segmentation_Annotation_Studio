#!/bin/sh
# Entrypoint for the app-full Docker image (Dockerfile's app-full stage):
# starts Tiled and ipred as background processes, then execs the backend in
# the foreground as the container's main process. All three talk over
# 127.0.0.1 since they share one container — no Docker networking needed.
#
# Known limitation, deliberate for this "batteries included, single command"
# image: Tiled/ipred are plain backgrounded processes, not supervised — if
# either crashes after startup, the container keeps running (its main
# process is the backend) but that service silently stays down until the
# whole container is restarted. Fine for local/demo use; a production
# deployment that needs real service supervision should run Tiled and ipred
# as separate containers instead (see docker-compose.yml + an external Tiled,
# or split ipred out the same way if this ever needs to be hardened further).
set -e

# Generate a Tiled API key if one wasn't provided, mirroring start_all.sh's
# own local-dev behavior — without one, Tiled's `allow_anonymous_access: true`
# only permits reads (see tiled/config.docker.yml), so writes (ingest, mask
# sync, volume registration) would fail with no explanation on first run.
if [ -z "${TILED_API_KEY:-}" ]; then
  export TILED_API_KEY="$(python3 -c 'import secrets, string; print("".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32)))')"
  echo "Generated a Tiled API key for this container (not persisted — set TILED_API_KEY yourself to keep one across restarts)."
fi

mkdir -p /data/.tiled/data /data/.tiled/volumes /data/processed

tiled serve config /app/tiled/config.docker.yml --host 0.0.0.0 --port 8010 --api-key "$TILED_API_KEY" &
uvicorn ipred.api:app --host 0.0.0.0 --port 8003 &

# Wait for Tiled to actually accept connections before starting the backend —
# without this, the backend's first Tiled-dependent request can race a still-
# initializing catalog and fail. python3 (not curl, which python:3.12-slim
# doesn't ship) is guaranteed present in this image already.
python3 -c "
import socket, time
for _ in range(60):
    try:
        socket.create_connection(('127.0.0.1', 8010), timeout=1).close()
        break
    except OSError:
        time.sleep(0.5)
else:
    print('Tiled did not become reachable on port 8010 in time — starting the backend anyway.')
"

export TILED_URI="http://127.0.0.1:8010"
export IPRED_URL="http://127.0.0.1:8003"
# Keep the mask/volume pyramid cache under the same persisted /data volume,
# in the exact path tiled/config.docker.yml's readable_storage expects it —
# same "keep in step" rule tiled/config.yml documents for local dev.
export VOLUME_CACHE_DIR="/data/.tiled/volumes"

# Auto-register any Zarr stores AND image-slice folders already sitting under
# the bind-mounted LOCAL_SOURCE_DIR (see docker-compose.full.yml/docker-
# compose.local.yml) — without this, a container brought up against a folder
# of pre-existing data shows nothing in Browse until someone manually scans or
# registers each one through the UI. Safe on every restart: both scans skip
# anything already registered, so this never re-registers or duplicates
# existing entries. The image-stack scan deliberately does NOT build a 3-D
# pyramid here — that's comparatively expensive (reads every slice) and is
# left to the on-demand "Build 3D volume" button on the 3D page, so startup
# stays fast regardless of how large a folder of TIFFs/PNGs is. Sequential,
# not concurrent: both scans call _ensure_container against the same target
# container, and running them in parallel would race on its creation.
# Non-fatal on failure — a scan problem must never block the app from
# starting; it just leaves auto-discovery for that run to be retried
# manually via the "Scan folder for datasets" button in the Zarr loader
# (POST /api/scan-datasets).
python3 -c "
import sys
sys.path.insert(0, '/app')
import ingest
import zarr_source

def report(label, result):
    print(
        f\"{label}: {len(result['registered'])} new, \"
        f\"{len(result['skipped'])} already present, \"
        f\"{len(result['shadowed'])} shadowed, {len(result['errors'])} failed.\"
    )
    for s in result['shadowed']:
        print(
            f\"  - {s['name']}: same name already registered as {s['existing_kind']!r} — \"
            f\"retry via the Zarr loader's Scan button with a different key \"
            f\"(e.g. {s['suggested_key']!r}) if you want both.\"
        )
    for err in result['errors']:
        print(f\"  - {err['name']}: {err['error']}\")

try:
    report('Zarr auto-registration', zarr_source.scan_and_register_zarrs('http://127.0.0.1:8010', '/data/processed', 'browse'))
except Exception as exc:
    print(f'Zarr auto-registration scan failed (non-fatal): {exc}')

try:
    report('Image-stack auto-registration', ingest.scan_and_register_image_stacks('http://127.0.0.1:8010', '/data/processed', 'browse'))
except Exception as exc:
    print(f'Image-stack auto-registration scan failed (non-fatal): {exc}')
" || true

exec uvicorn annotation_server:app --host 0.0.0.0 --port 8002
