# syntax=docker/dockerfile:1
# Three images from one file, selected via `--target`:
#   app       (default) — frontend + backend only, Tiled external
#                          (docker build . / docker compose up, unchanged).
#   app-ml              — the same, PLUS the `ml` extra (torch/dlsia) and
#                          ipred bundled, Tiled still external — for a
#                          deployment with its own production Tiled that
#                          still wants Train/iPred to work out of the box
#                          (see docker-compose.ml.yml, and the `:als` tag).
#   app-full            — app-ml PLUS a bundled Tiled server in the same
#                          container (all three talk over 127.0.0.1), for a
#                          single `docker run` that needs nothing external —
#                          see docker-compose.full.yml, and the `:local` tag.
# Local dev still uses start_all.sh, which runs the same three services as
# separate local processes instead of inside a container.

# --- Stage: build the frontend (same-origin: VITE_API_BASE left empty) ---
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# The WebGPU volume renderer is a git submodule under frontend/vendor/. Docker
# copies the working tree as-is, so an uninitialised submodule arrives as an
# empty directory and the build fails deep inside Vite with an unresolved
# import. Fail here instead, with the fix in the message.
RUN test -f vendor/view_tomography_recon_app/src/zarr-viewer/src/ome-zarr-viewer.ts \
    || (echo "ERROR: submodule frontend/vendor/view_tomography_recon_app is missing." \
        && echo "Run: git submodule update --init --recursive" && exit 1)
# Read at build time (see vite.config.ts's `base` / src/config.ts's API_BASE) —
# empty/unset produces the exact same root-hosted output as before this arg
# existed. Set for a subpath deployment, e.g. --build-arg VITE_BASE_PATH=/bl832/seg_studio/
ARG VITE_BASE_PATH=""
ENV VITE_BASE_PATH=${VITE_BASE_PATH}
RUN npm run build   # → /web/dist

# --- Stage: lightweight production image (frontend + backend only) ---
FROM python:3.12-slim AS app
WORKDIR /app

# Install Python deps first (cached until pyproject changes). py-modules=[] means
# this installs dependencies only; the app code runs from the copied source below.
# pyproject pins tiled[client] (not [all]/[server]), so the Tiled server is NOT
# installed into this image — the app only connects to an external Tiled.
# If a wheel is unavailable for pycocotools/imagecodecs on this platform, add:
#   RUN apt-get update && apt-get install -y --no-install-recommends build-essential
COPY backend/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir .

# App source + the built SPA (served from ./static by annotation_server.py).
COPY backend/ ./
COPY --from=web /web/dist ./static

# Drafts/versions/exports persist here — mount a volume in production.
ENV LOCAL_DATA_ROOT=/data
VOLUME ["/data"]

EXPOSE 8002
CMD ["uvicorn", "annotation_server:app", "--host", "0.0.0.0", "--port", "8002"]

# --- Stage: backend + ipred + ml, Tiled still external ---
# For a deployment that already has its own production Tiled (so bundling a
# second, empty one would be actively wrong) but still wants the Train tab
# and the fast pixel classifier to work without standing up a separate ipred
# service just for this app — see the `:als` tag and docker-compose.ml.yml.
FROM app AS app-ml

# The `ml` extra (torch/dlsia/qlty) — kept out of the lean `app` image above.
# Pulls in the full CUDA toolkit (nvidia-cudnn-cu12, nvidia-cublas-cu12, etc.)
# as torch dependencies — several GB — needed for real GPU support when this
# container is run with `--gpus all` on a CUDA host; a build here can fail
# with an I/O error mid-write on a disk-constrained host, which is a local
# Docker disk-space problem to fix (see docker system df / prune, or grow
# Docker Desktop's disk allocation), not a reason to drop CUDA support.
RUN pip install --no-cache-dir ".[ml]"

# ipred is a sibling package with its own pyproject, not a backend dependency.
COPY ipred/ /ipred/
RUN pip install --no-cache-dir /ipred

COPY docker-entrypoint-ml.sh /usr/local/bin/docker-entrypoint-ml.sh
RUN chmod +x /usr/local/bin/docker-entrypoint-ml.sh

# 8002 backend; 8003 ipred, exposed so it can be reached directly if wanted.
# No Tiled port here — this stage never starts one; TILED_URI at `docker run`
# time points at the deployment's own external Tiled.
EXPOSE 8002 8003
ENTRYPOINT ["/usr/local/bin/docker-entrypoint-ml.sh"]

# --- Stage: batteries-included image (+ Tiled server too) ---
# Bundles all three backend services into one container over loopback — the
# CI/Docker coverage gap this stage fixes: previously NOTHING packaged ipred
# or a Tiled server at all, so the app image alone could never run iPred or
# dlsia, and there was no single-command way to try the full stack without
# start_all.sh's separate local processes.
FROM app-ml AS app-full

# tiled[all] pulls in the actual server (catalog, array/table adapters) —
# `app`'s pyproject only pins tiled[client], which has no server component.
RUN pip install --no-cache-dir "tiled[all]"

# Portable Tiled config (see tiled/config.docker.yml's own doc comment for
# why this isn't just tiled/config.yml — that one hardcodes a local dev
# machine's absolute data path).
COPY tiled/config.docker.yml /app/tiled/config.docker.yml

COPY docker-entrypoint-full.sh /usr/local/bin/docker-entrypoint-full.sh
RUN chmod +x /usr/local/bin/docker-entrypoint-full.sh

# 8002 backend (the only port most deployments need — same-origin SPA+API);
# 8003/8010 exposed too so ipred/Tiled can be reached directly if wanted.
EXPOSE 8002 8003 8010
ENTRYPOINT ["/usr/local/bin/docker-entrypoint-full.sh"]
