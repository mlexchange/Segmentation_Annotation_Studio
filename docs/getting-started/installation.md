# Installation

There are two ways to run Segmentation Annotation Studio:

- **Local development** — one command starts everything (recommended for annotators and evaluation).
- **Docker** — a prebuilt or locally-built container image; pick the shape that matches
  whether you already have a Tiled server and whether you want iPred/Train available.

For hosting a shared, ALS-style deployment behind a reverse proxy (rather than running
it yourself), see [Production deployment](../reference/deployment.md) instead.

---

## Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| **Python** | 3.11 or newer | 3.12 is used by the local launcher. |
| **Node.js / npm** | 18 or newer | Needed to build and run the frontend. |
| **git** | any recent | To clone the repository. |
| `curl`, `openssl`, `lsof` | system tools | Used by the launcher for health checks, key generation, and port cleanup. |

The launcher will automatically install [`uv`](https://github.com/astral-sh/uv)
(a fast Python package manager) if it is not already on your `PATH`.

!!! note "macOS / Linux"
    The one-command launcher (`start_all.sh`) is a bash script and targets
    macOS and Linux. On Windows, use WSL or follow the
    [manual setup](#manual-setup) below.

---

## Option 1 — One-command local start (recommended)

From the repository root:

```bash
./start_all.sh
```

That's it. The script bootstraps the entire stack and opens three services:

| Service | URL |
| --- | --- |
| **Frontend** (open this in your browser) | <http://127.0.0.1:5173> |
| **Backend API** | <http://127.0.0.1:8002> |
| **Tiled** (data catalog) | <http://127.0.0.1:8010> |

Press ++ctrl+c++ in the terminal to stop all three services.

??? info "What `start_all.sh` does for you"
    On first run the launcher:

    1. Creates a Python 3.12 virtual environment in `.venv` (via `uv venv`).
    2. Installs the backend dependencies, including the full Tiled server (`tiled[all]`).
    3. Copies `backend/.env.example` → `backend/.env` if it doesn't exist yet.
    4. Generates a strong `TILED_API_KEY` and writes it into `backend/.env`.
    5. Initializes/repairs the local Tiled catalog at `.tiled/catalog.db`.
    6. Starts Tiled, the backend (`uvicorn annotation_server:app`), and the frontend (`npm run dev`).
    7. Runs `npm install` if `frontend/node_modules` is missing.
    8. Downloads the SlimSAM model in the background so the **Smart (AI)** tool works offline.

### Changing the ports

If a default port is busy, the launcher automatically scans upward for a free
one. To pin specific ports, set environment variables before launching:

```bash
FRONTEND_PORT=5200 BACKEND_PORT=8100 TILED_PORT=8110 ./start_all.sh
```

---

## Option 2 — Docker

Three Dockerfile targets/compose files cover different needs — pick the one that
matches what you already have running and whether you want iPred/Train available.
None replaces the others; `app-ml` and `app-full` each build on the previous stage
rather than duplicating install steps.

| Compose file | Image target | Tiled | iPred / Train | Use when |
| --- | --- | --- | --- | --- |
| `docker-compose.yml` | `app` | External (you provide one) | Not included | You already have a Tiled server and only need Connect/Browse/Annotate/Export. |
| `docker-compose.ml.yml` | `app-ml` | External (you provide one) | **Bundled** | You already have a Tiled server but also want Train/iPred to work, without standing up a separate iPred service. |
| `docker-compose.full.yml` | `app-full` | **Bundled** | **Bundled** | You want the whole stack (Tiled + backend + iPred) with nothing external to set up — the simplest way to try everything. |

### Lean (`app`) — bring your own Tiled

```bash
TILED_URI=https://tiled.example.com \
TILED_API_KEY=your-key \
docker compose up --build
```

Then open <http://localhost:8002>.

### With iPred/Train, external Tiled (`app-ml`)

```bash
TILED_URI=https://tiled.example.com \
TILED_API_KEY=your-key \
docker compose -f docker-compose.ml.yml up --build
```

Then open <http://localhost:8002>. iPred is also reachable directly on `:8003` if needed.

### Fully bundled (`app-full`) — nothing external required

```bash
docker compose -f docker-compose.full.yml up --build
```

Then open <http://localhost:8002>. Tiled (`:8010`) and iPred (`:8003`) are also
reachable directly if you want to hit them outside the app.

!!! warning "Persisting your data"
    Every compose file above mounts `LOCAL_DATA_ROOT` (`/data`) as a named volume,
    so annotation drafts, versions, and exports already survive a container
    restart. For `app-full`, bind-mount your own source datasets to `/data/raw`
    (see the compose file's own comment) so Tiled can ingest them.

---

## Manual setup

If you prefer to run each service yourself (for example on Windows, or in
separate terminals), install and start the backend and frontend independently.
Tiled must be running and reachable at `TILED_URI` first — the easiest way to
get a local Tiled instance is still `./start_all.sh`.

### Backend

```bash
cd backend
pip install -e ".[dev,test]"
uvicorn annotation_server:app --host 127.0.0.1 --port 8002
```

Run the backend test suite with:

```bash
pytest
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on <http://127.0.0.1:5173> and proxies all `/api`
requests to the backend on port 8002. Other useful scripts:

```bash
npm run build      # type-check and build the production bundle
npm run typecheck  # type-check only
npm test           # run the Vitest unit tests
```

---

## Environment variables

Backend configuration lives in `backend/.env` (created from
`backend/.env.example` on first launch) when running via `start_all.sh` /
directly with `uvicorn`. **Running via `docker compose` instead**, copy the
repo root's own `.env.example` to `.env` — `docker compose` loads that
automatically for every `docker-compose*.yml` file, so this is where
`TILED_URI`/`TILED_API_KEY`/`TILED_BROWSE_PATH` (and, for `app-ml`,
`VITE_BASE_PATH`) actually get substituted in. See
[Production deployment](../reference/deployment.md#setting-these-for-docker-compose)
for details. The most relevant settings either way:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TILED_URI` | Address of the Tiled server | `http://127.0.0.1:8010` |
| `TILED_API_KEY` | Auth key for Tiled writes/ingest | *(generated locally)* |
| `LOCAL_DATA_ROOT` | Where drafts, versions, and exports are stored | `~/data` |
| `EXPORT_ROOT` | Override output folder for exports | `~/data/exports` |
| `BROWSE_CACHE_TTL_SECONDS` | Cache lifetime for Tiled browse listings | `300` |
| `BROWSE_ALLOWED_ORIGINS` | CORS origins (only needed for split frontend/backend hosting) | *(empty)* |
| `TILED_BROWSE_PATH` | Path into the Tiled tree Browse treats as its root | *(unset — this repo's own ingest root)* |

`VITE_BASE_PATH` is a **frontend build-time** setting (a Docker build-arg, not a
runtime env var — see [Production deployment](../reference/deployment.md)) for
hosting under a URL prefix rather than at the domain root; leave it unset for
everything on this page.

!!! danger "Never commit secrets"
    `backend/.env` is git-ignored. Never commit it, and never expose
    `TILED_API_KEY` to the frontend — all Tiled access goes through the backend.

---

## Verify it's working

1. Open the frontend at <http://127.0.0.1:5173>.
2. You should land on the **Connect** tab with the ALS logo in the header.
3. Check the backend health endpoint:

    ```bash
    curl http://127.0.0.1:8002/health
    ```

Once the app loads, continue to the [Quick start](quick-start.md).
