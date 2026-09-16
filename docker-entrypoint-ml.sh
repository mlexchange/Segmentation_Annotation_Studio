#!/bin/sh
# Entrypoint for the app-ml Docker image (Dockerfile's app-ml stage): starts
# ipred as a background process, then execs the backend in the foreground as
# the container's main process. Unlike app-full's entrypoint, this does NOT
# start (or need) a Tiled process at all — TILED_URI must be set at `docker
# run` time to point at the deployment's own external, already-running Tiled
# (e.g. ALS's production Tiled for the `:als` tag).
#
# Known limitation, same as app-full's entrypoint: ipred is a plain
# backgrounded process, not supervised — if it crashes after startup, the
# container keeps running (its main process is the backend) but iPred/Train
# silently stay down until the whole container is restarted.
set -e

mkdir -p /data

uvicorn ipred.api:app --host 0.0.0.0 --port 8003 &

export IPRED_URL="http://127.0.0.1:8003"

exec uvicorn annotation_server:app --host 0.0.0.0 --port 8002
