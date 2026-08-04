# Segmentation Annotation Studio

A local, browser-based tool for drawing segmentation masks on scientific images and
exporting training-ready datasets. Load data from a [Tiled](https://blueskyproject.io/tiled/)
server or a local folder, annotate with a full set of drawing tools (including an
in-browser AI "Magic" wand), and export to **COCO** (for SAM3 fine-tuning) or
**DINOv3 / Lightly** semantic-segmentation format.

Everything runs on `127.0.0.1` — no data leaves your machine for image processing,
annotation, or export. Two things are opt-in exceptions: the in-app "Feedback"
button, which shows you exactly what it would send (including the open sample's
path) and asks before submitting it to an external Google Form; and the SAM "Magic"
wand's model download, which can fetch its weights from Hugging Face on first use.

## Quick start

You need **Node.js 20+** (`npm`) and **`curl`** on your PATH. That's it — everything
else is bootstrapped for you.

### Node.js

You must have `node.js` installed on your machine. Follow the official instructions here for your operating system:
https://nodejs.org/en/download

### Mac/Linux
```bash
chmod +x start_all.sh
./start_all.sh
```

On first run this will automatically:

- install [`uv`](https://docs.astral.sh/uv/) if it's missing,
- create a `.venv` with Python 3.12 and install the backend dependencies,
- generate a strong Tiled API key into `backend/.env` (gitignored, never sent to the browser),
- vendor the SlimSAM model in the background so the AI Magic tool works offline,
- start Tiled, the backend API, the frontend dev server, and the docs site.

When it's ready, open the **Frontend** URL it prints. Press **Ctrl+C** to stop everything.

| Service   | Default URL             | Notes                                        |
| --------- | ----------------------- | -------------------------------------------- |
| Frontend  | http://127.0.0.1:5173   | The app (Vite dev server)                    |
| Backend   | http://127.0.0.1:8002   | FastAPI — `/api/*`                           |
| Tiled     | http://127.0.0.1:8010   | Data server (anonymous access is read-only)  |
| Docs      | http://127.0.0.1:8000   | MkDocs (best-effort; the in-app Docs button) |

Ports are just defaults — if one is busy, `start_all.sh` automatically picks the next
free port and wires the services together. You can also override them, e.g.
`BACKEND_PORT=9002 ./start_all.sh`.

### Windows

On Windows, use the native PowerShell launcher instead (same behavior, no WSL/Git Bash needed).
Double-click `windows\start_all.cmd`, or from a terminal:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\start_all.ps1
```

`$env:PROD = "1"` before running does the production build; `$env:BACKEND_PORT` etc. override
ports. See [windows/README.md](windows/README.md) for details. You need **Node.js 20+** and
**PowerShell 5.1+** (built into Windows 10/11); everything else is bootstrapped for you.

### Production build (single origin)

To serve the optimized SPA directly from the backend (one origin, gzip, no Vite dev
server) instead of the dev setup:

```bash
PROD=1 ./start_all.sh
```

The frontend is built to `backend/static/` and served by FastAPI. The whole app is then
available at the **Backend** URL (http://127.0.0.1:8002).

## Workflow

The app is organized into four tabs:

1. **Connect** — choose a Tiled server + dataset, or point at a local folder of images
   (`.tif/.tiff`, `.npy`, `.png/.jpg`).
2. **Browse** — explore and filter Tiled datasets by metadata, and open a sample into Annotate.
3. **Reference** — author a per-dataset annotation guide: for each class a label, color,
   a written description, and example crops. Guide classes surface as one-click
   suggestions in Annotate, keeping annotators consistent.
4. **Annotate** — draw and edit masks, manage classes, navigate slices, and export.

### Annotation tools

Polygon, magnetic lasso (live-wire), rectangle, ellipse, brush, fill, and a **Magic**
wand backed by an in-browser SAM model (with a classic intensity wand as fallback). An
eraser and a select/transform tool round out editing, with boolean clip/merge so new
strokes don't overlap existing classes. A CLAHE adaptive-contrast display filter plus
brightness/contrast/gamma controls help with low-contrast scientific data — display-only,
never affecting exported pixels.

## Export formats

Export runs from within the Annotate tab (the download action). Two targets:

- **COCO (SAM3)** — COCO JSON where `segmentation` is compressed RLE and
  `categories[].name` is the SAM3 concept phrase, alongside the rendered images. Default.
- **DINOv3 / Lightly** — a semantic-segmentation layout: `images/` + `masks/` with matching
  filename stems (each mask a single-channel integer PNG, pixel = class id, 0 = background)
  plus a `classes.json` index. For training non-SAM3 models (e.g. via LightlyTrain).

## Development

Backend (FastAPI, Python 3.11+):

```bash
cd backend
pip install -e ".[dev,test]"                                  # or: uv pip install -e ".[dev,test]"
flake8 . --max-line-length=120 --extend-ignore=E501,W503      # lint (isort enforced)
pytest                                                        # tests
```

Frontend (React + TypeScript + Vite):

```bash
cd frontend
npm install
npm run typecheck    # tsc -b
npm run test         # vitest
npm run build        # production bundle
npm run dev          # dev server (start_all.sh runs this for you)
```

These are the same checks CI runs (see `.github/workflows/ci.yml`).

## Security & data

- The local launchers explicitly bind every service to `127.0.0.1`. Docker Compose
  likewise publishes the container API on `127.0.0.1` only. Do not broaden either
  binding without first adding network-user authentication and TLS.
- Tiled anonymous access is **read-only**; writes (e.g. ingest) require the API key that
  `start_all.sh` generates into `backend/.env`. The key is resolved server-side and is
  **never** exposed to the frontend.
- Annotation coordinates stay in image pixels throughout.

## Project layout

```
backend/     FastAPI API, COCO/Lightly export, Tiled client, local-folder access
frontend/    React SPA (Konva canvas, Zustand stores, in-browser SAM)
tiled/        Local Tiled server config
docs/         MkDocs Material documentation site
start_all.sh  One-command launcher for the full stack
```
