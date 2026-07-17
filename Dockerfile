# syntax=docker/dockerfile:1
# Lightweight production image: builds the SPA and serves it from the FastAPI
# backend (single container, one port). Tiled is NOT included — point the app at
# an external Tiled via TILED_URI/TILED_API_KEY. Local dev still uses start_all.sh.

# --- Stage 1: build the frontend (same-origin: VITE_API_BASE left empty) ---
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build   # → /web/dist

# --- Stage 2: backend + built SPA ---
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
