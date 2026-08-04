# Installation

There are two ways to run Segmentation Annotation Studio:

- **Local development** — one command starts everything (recommended for annotators and evaluation).
- **Docker** — a single production container that serves the app against an external Tiled server.

---

## Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| **Python** | 3.11 or newer | 3.12 is used by the local launcher. |
| **Node.js / npm** | 20 or newer | Needed to build and run the frontend. |
| **git** | any recent | To clone the repository. |
| `curl`, `openssl`, `lsof` | system tools | Used by the launcher for health checks, key generation, and port cleanup. |

The launcher will automatically install [`uv`](https://github.com/astral-sh/uv)
(a fast Python package manager) if it is not already on your `PATH`.

!!! note "macOS / Linux"
    The one-command launcher (`start_all.sh`) is a bash script and targets
    macOS and Linux. On Windows, use the native launcher by double-clicking
    `windows\start_all.cmd`, or run
    `powershell -ExecutionPolicy Bypass -File .\windows\start_all.ps1`.

---

## Option 1 — One-command local start (recommended)

From the repository root:

```bash
./start_all.sh
```

That's it. The script bootstraps the entire stack and opens four local services
(the documentation service is best-effort):

| Service | URL |
| --- | --- |
| **Frontend** (open this in your browser) | <http://127.0.0.1:5173> |
| **Backend API** | <http://127.0.0.1:8002> |
| **Tiled** (data catalog) | <http://127.0.0.1:8010> |
| **Documentation** | <http://127.0.0.1:8000> |

Press ++ctrl+c++ in the terminal to stop all services.

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

## Option 2 — Docker (production)

Docker runs a **single container** that serves the built frontend and the API
together on port **8002**. Tiled is **not** included — you point the container
at an existing Tiled server.

```bash
docker compose up --build
```

Then open <http://127.0.0.1:8002>.

The Compose port mapping is intentionally restricted to host loopback because
the application is designed as a single-user local tool and does not yet have
network-user authentication. Do not change it to `8002:8002` or otherwise expose
it to a LAN/public network without adding authentication and TLS.

Configure the connection to your external Tiled through environment variables
(see [Environment variables](#environment-variables)):

```bash
TILED_URI=https://tiled.example.com \
TILED_API_KEY=your-key \
docker compose up --build
```

To show the Docs button in the container build, point it at a deployed MkDocs
site. If omitted, the button is hidden:

```bash
VITE_DOCS_URL=https://docs.example.com docker compose up --build
```

!!! warning "Persisting your data"
    Mount `LOCAL_DATA_ROOT` as a volume so annotation drafts, versions, and
    exports survive container restarts.

---

## Manual setup

If you prefer to run each service yourself in separate terminals, install and
start the backend and frontend independently.
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
`backend/.env.example` on first launch). The most relevant settings:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TILED_URI` | Address of the Tiled server | `http://127.0.0.1:8010` |
| `TILED_API_KEY` | Auth key for Tiled writes/ingest | *(generated locally)* |
| `LOCAL_DATA_ROOT` | Where drafts, versions, and exports are stored | `~/data` |
| `EXPORT_ROOT` | Override output folder for exports | `~/data/exports` |
| `BROWSE_CACHE_TTL_SECONDS` | Cache lifetime for Tiled browse listings | `300` |
| `BROWSE_ALLOWED_ORIGINS` | CORS origins (only needed for split frontend/backend hosting) | *(empty)* |

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
