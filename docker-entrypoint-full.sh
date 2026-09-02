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

mkdir -p /data/.tiled/data /data/.tiled/volumes /data/raw

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

exec uvicorn annotation_server:app --host 0.0.0.0 --port 8002
