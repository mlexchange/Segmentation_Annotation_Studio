# syntax=docker/dockerfile:1
# Lightweight production image: builds the SPA and serves it from the FastAPI
# backend (single container, one port). Tiled is NOT included — point the app at
# an external Tiled via TILED_URI/TILED_API_KEY. Local dev still uses start_all.sh.

# --- Stage 1: build the frontend (same-origin: VITE_API_BASE left empty) ---
# Keep in sync with .nvmrc / .github/workflows/ci.yml (Node 20) — README documents
# the same version for local dev, so a build here matches CI and local installs.
FROM node:20-alpine AS web
RUN mkdir -p /web && chown node:node /web
WORKDIR /web
USER node
COPY --chown=node:node frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY --chown=node:node frontend/ ./
ARG VITE_DOCS_URL=""
ENV VITE_DOCS_URL=${VITE_DOCS_URL}
RUN npm run build   # → /web/dist

# --- Stage 2: backend + built SPA ---
FROM python:3.12-slim AS app
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python deps first (cached until pyproject changes). py-modules=[] means
# this installs dependencies only; the app code runs from the copied source below.
# pyproject pins tiled[client] (not [all]/[server]), so the Tiled server is NOT
# installed into this image — the app only connects to an external Tiled.
# If a wheel is unavailable for pycocotools/imagecodecs on this platform, add:
#   RUN apt-get update && apt-get install -y --no-install-recommends build-essential
COPY backend/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir .

# Use a stable, unprivileged identity for the API and its persisted data.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app \
    && mkdir -p /data \
    && chown app:app /app /data

# App source + the built SPA (served from ./static by annotation_server.py).
COPY --chown=app:app backend/ ./
COPY --chown=app:app --from=web /web/dist ./static
# Licensing/copyright notices, so they ship with the artifact that carries them.
COPY --chown=app:app LICENSE.txt COPYRIGHT.txt ./

# Drafts/versions/exports persist here — mount a volume in production.
ENV LOCAL_DATA_ROOT=/data
VOLUME ["/data"]

# The app user has no home directory (--no-create-home); point matplotlib's
# cache somewhere writable so it doesn't warn/fall back to /tmp on every boot.
ENV MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8002
# The container listens on its network namespace; Compose publishes it only on
# host loopback. Do not publish this port on all host interfaces without adding
# authentication and TLS.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=2).read()"]
USER app
CMD ["uvicorn", "annotation_server:app", "--host", "0.0.0.0", "--port", "8002"]
