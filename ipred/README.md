# ipred — architecture and operations guide

`ipred` is the **iterative prediction** compute service for SAM3 Annotation Studio. It owns feature compositions, preprocessing / feature banks, CatBoost training, Mondrian conformal prediction, and Suggest Labels (manifold). The Vite UI never talks to it directly: the Annotate FastAPI backend (`:8002`) proxies all GUI calls.

This document describes how the service is laid out, where compute runs, how data reaches it, and how compositions / models are discovered.

---

## 1. Big picture

```text
┌──────────────┐   HTTP    ┌────────────────────┐   HTTP    ┌─────────────────┐
│  Vite UI     │ ───────▶  │ Annotate backend   │ ───────▶  │ ipred FastAPI   │
│  :5173       │ /api/ipred│ 127.0.0.1:8002     │ IPRED_URL │ 127.0.0.1:8003  │
└──────────────┘           └────────────────────┘           └────────┬────────┘
                                                                     │
                    ┌────────────────────────────────────────────────┤
                    │                                                │
                    ▼                                                ▼
         LOCAL_DATA_ROOT/ipred/                          Data plane:
         catalog.db, projects/*/                         • local files under LOCAL_DATA_ROOT
                                                         • Tiled over HTTP (:8010)
                                                         • optional uploaded array blobs
```

| Role | Process | Bind | Responsibility |
|------|---------|------|----------------|
| UI | Vite | `127.0.0.1:5173` | Composition editor, Draw / Train, Layers |
| Annotate | FastAPI | `127.0.0.1:8002` | Tiled/browse/auth; **proxies** `/api/ipred/*` |
| **ipred** | FastAPI (`ipred.api:app`) | `127.0.0.1:8003` | Features, train, infer, manifold |
| Tiled | catalog + API | `127.0.0.1:8010` | Array storage / browse |

All local binds stay on **loopback** (`127.0.0.1`), never `0.0.0.0`.

---

## 2. Where compute lives

### Process

- Package: `ipred/` (editable install into the shared repo `.venv`).
- Entry module: `ipred.api:app` in `ipred/src/ipred/api.py`.
- Launch (via `start_all.sh`):

  ```bash
  cd ipred
  PYTHONPATH=src uvicorn ipred.api:app --host 127.0.0.1 --port "${IPRED_PORT:-8003}"
  ```

- The `ipred` console script (`ipred.cli:main`) is an **offline CLI** against the same SQLite/FS layout; it does **not** start uvicorn.

### Engine filesystem root

Everything durable for a running engine lives under:

```text
$LOCAL_DATA_ROOT/ipred/          # default LOCAL_DATA_ROOT=~/data
  catalog.db                     # projects, sessions, banks, models, runs
  projects/<project_id>/
    features/<feature_id>/       # FeatureBank blobs
    models/<model_id>/           # CatBoost + conformal calibration
    runs/<run_id>/               # proba + conformal maps
    arrays/<array_ref>/          # optional uploaded tensors
```

Implemented in `ipred/src/ipred/paths.py`:

| Helper | Path |
|--------|------|
| `local_data_root()` | `$LOCAL_DATA_ROOT` (default `~/data`) |
| `engine_root()` | `$LOCAL_DATA_ROOT/ipred` |
| `catalog_db_path()` | `…/ipred/catalog.db` |
| `project_blob_dir(id)` | `…/ipred/projects/<id>/` |
| `feature_models_root()` | `$LOCAL_DATA_ROOT/.feature_models` (compositions + legacy setups) |

**Compute affinity:** all CPU/GPU work for encode / CatBoost / PCA / manifold runs **inside the ipred process**. The Annotate backend only forwards HTTP.

### Startup order (`start_all.sh`)

1. Shared `.venv` (+ `uv pip install -e ipred`)
2. Load `backend/.env`
3. **Tiled** → wait ready  
4. **ipred** → wait `GET http://127.0.0.1:8003/health`  
5. **Annotate backend** (receives `IPRED_URL`) → wait `/health`  
6. **Frontend**

On API lifespan, ipred opens the catalog and seeds default feature setups + compositions.

---

## 3. How the UI reaches compute (proxy boundary)

### Rule

The browser **never** calls `:8003`. It calls Annotate:

```text
frontend  →  ${API_BASE}/api/ipred/...  →  backend/ipred_client.py  →  ${IPRED_URL}/...
```

- Client library: `frontend/src/lib/ipredApi.ts`
- Server client: `backend/ipred_client.py` (httpx; **no** Python import of the `ipred` package)
- Env: `IPRED_URL` (default `http://127.0.0.1:8003`; legacy alias `CLF_ENGINE_URL`)

If ipred is down, Annotate returns **503** (`ipred unreachable at …`).

### Representative route mirror

| UI / Annotate | ipred |
|---------------|-------|
| `GET /api/ipred/health` | `GET /health` |
| `POST /api/ipred/sessions` | `POST /sessions` |
| `GET /api/ipred/modules` | `GET /modules` |
| `GET/POST /api/ipred/compositions` | `/compositions` |
| `POST /api/ipred/preprocess` | `POST /preprocess` |
| `POST /api/ipred/train` / `infer` / `rethreshold` | same |
| `GET /api/ipred/runs/{id}/proba/{i}.png` | same |
| `POST /api/ipred/runs/{id}/threshold-class` | same |
| `POST /api/ipred/manifold/sample` | same |

---

## 4. Sessions, projects, and identity

A **project** is a content-addressed identity of *which array source* you are working on. A **session** is a UUID workspace pointer into that project (current feature / model / run ids).

`POST /sessions` body:

```json
{
  "kind": "local" | "tiled",
  "source": "<path or tiled container path>",
  "server_uri": "<optional tiled URI>",
  "root": "<optional local root override>"
}
```

- `project_id` = first 16 hex of SHA1 over `{kind, source, server_uri, root}`
- `session_id` = fresh `uuid4` each open (many sessions can share one project)

Stored in SQLite `catalog.db`. Blob trees hang off `projects/<project_id>/`.

Typical open path: Browse / Connect → `useOpenInAnnotate` → `openIpredSession` → `connectionStore.ipredSessionId` + `ipredProjectId`.

---

## 5. Data plane — how pixels reach the engine

Three mutually exclusive ways to get a 2‑D slice into preprocess:

### A. Local files (default for local Browse)

- Read: `array_source.read_slice(kind="local", …)`
- Path: `(root or LOCAL_DATA_ROOT) / source`, with a containment check
- Formats: TIFF (tifffile) or PIL-readable; then z/slice index

**Requirement:** the ipred host must see that filesystem (same machine, NFS, etc.).

### B. Tiled over HTTP

- Read: `tiled.client.from_uri(server_uri or TILED_URI)` + optional `TILED_API_KEY`
- Walk `source` path; pull the slice array over the network

**Requirement:** network reachability to Tiled. Compute does **not** need the Tiled data directory on disk.

### C. Uploaded array blobs (compute ≠ data host)

For when Annotate and ipred do not share `LOCAL_DATA_ROOT`:

1. `POST /sessions/{id}/arrays` with base64 float32 payload → content-addressed  
   `$LOCAL_DATA_ROOT/ipred/projects/<pid>/arrays/<array_ref>/{array.npy, meta.json}`
2. `POST /preprocess` with `array_ref` loads that blob

When `array_ref` is set, **preprocess cache hits are skipped** (identity is upload content + composition, not just slice index on a stable URI).

### Environment checklist

| Variable | Purpose |
|----------|---------|
| `LOCAL_DATA_ROOT` | Engine + local file root |
| `TILED_URI` | Default tiled server (`http://127.0.0.1:8010`) |
| `TILED_API_KEY` | Optional tiled auth **only on server side** |
| `IPRED_URL` | Annotate → ipred base URL |

---

## 6. Discovering modules and compositions

### Module catalog (runtime capability discovery)

`GET /modules` → `ipred.modules.list_module_catalog()`.

Registered producers (`ipred/src/ipred/modules/`):

| Module id | Typical runtime | Output |
|-----------|-----------------|--------|
| `skimage_multiscale` | numpy / skimage | Multi-channel float stack |
| `clahe` | numpy | Optional CLAHE grayscale channel |
| `slimsam` | **onnx** | Dense embedding |
| `tomojepa` | **onnx** if matching `.onnx` present, else **torch** | Dense embedding |
| `pca` | numpy / sklearn | Embedding → PCA channels |

Each entry reports `id`, `params_schema`, `runtime`, `ready`, and whether it accepts `input_from` / produces channels vs embeddings.

UI: left column of **CompositionPanel** on the Ipred page.

### Composition documents (feature graphs)

A composition is an ordered graph of module instances. The feature bank is the **channel-wise concatenation** of nodes listed in `outputs`.

Stored under:

```text
$LOCAL_DATA_ROOT/.feature_models/_compositions/<id>/meta.json
```

Schema (v1):

```json
{
  "id": "comp-skimage-mark11",
  "kind": "composition",
  "name": "…",
  "nodes": [
    {"id": "n1", "module": "skimage_multiscale", "params": {…}},
    {"id": "n2", "module": "clahe", "params": {…}},
    {"id": "n3", "module": "tomojepa", "params": {"weights_id": "mark11", "input_size": 512}, "input_from": "n2"},
    {"id": "n4", "module": "pca", "params": {"dims": 64}, "input_from": "n3"}
  ],
  "outputs": ["n1", "n4"]
}
```

- `input_from` wires grayscale (or prior emb) into the next module — this is how “CLAHE → Mark11” is made explicit.
- `content_hash` = short SHA1 of `{kind, nodes, outputs}` — part of the preprocess cache key.
- API: `GET/POST /compositions`, `GET /compositions/{id}`, `POST /compositions/preview` (concat channel labels without running encode).

**Builtins** (auto-seeded):  
`comp-skimage`, `comp-skimage-slimsam`, `comp-slimsam-clahe`, `comp-skimage-mark25`, `comp-mark25-clahe`, `comp-skimage-mark11`, `comp-mark11-clahe`.

Default preference in the UI (`connectionStore`):

- `preferredCompositionId = "comp-skimage-slimsam"`
- also mirrored to legacy `preferredFeatureSetupId` for one release

### Legacy Feature Setups (migration)

Older “procedure × encoder” combos still exist as disk shelves under `$LOCAL_DATA_ROOT/.feature_models/<setup_id>/` (`feature_setups.py`).  
`preprocess` accepts either `composition_id` or `feature_setup_id`; legacy ids map via `_LEGACY_SETUP_MAP` (e.g. `default-skimage-slimsam` → `comp-skimage-slimsam`).

Prefer compositions for new work; setups are a compatibility layer.

---

## 7. Preprocess → FeatureBank

`POST /preprocess`:

```json
{
  "session_id": "…",
  "composition_id": "comp-skimage-slimsam",
  "slice_index": 0,
  "array_ref": null
}
```

Pipeline:

1. Resolve composition (or legacy setup → composition).
2. Load slice (local / tiled / array_ref).
3. Cache lookup on `(project_id, setup_id, content_hash, slice_index)` when `array_ref` is null.
4. Else `compose_run.run_composition(doc)` → concat outputs.
5. Write FeatureBank under `projects/<pid>/features/<feature_id>/`:

| Artifact | Meaning |
|----------|---------|
| `float_stack.npy` | Training features (float16 HxWxC) |
| `uint8_stack.npy` | Display-oriented stack |
| `labels.json` | Channel names (concat order) |
| `channels/NNNN.png` | Browseable per-channel previews |
| `sam_emb.npy` + `sam_meta.json` | Dense emb when an encoder ran (unless fully baked via PCA) |

UI hook: `useFeatureChannels` (Preprocess stage) calls proxy preprocess and exposes channel navigation.

---

## 8. Train, predict, probability maps

### Trainer discovery

`GET /trainers` → plugin list. Today only **`catboost`** (`ipred.trainers.catboost_trainer.CatBoostTrainer`).  
UI preference: `connectionStore.preferredTrainerId` (+ depth / trees / LR).

### Train (`POST /train`)

- Rasterize sparse Draw shapes → label map.
- Stratified train / calibration split.
- Fit CatBoost → write:

  ```text
  projects/<pid>/models/<model_id>/
    model.cbm
    meta.json
    cal_scores.json
    (optional sam_pca.npz)
  ```

- Register row in catalog `models`; set session `current_model_id`.

### Infer (`POST /infer`)

- Full-image `predict_proba` → `proba.npy`.
- Mondrian split-conformal (`conformal.py`) using calibration scores + α → membership / commit / status.
- Run directory:

  ```text
  projects/<pid>/runs/<run_id>/
    proba.npy
    commit.npy / status.npy / membership.npy
    commit.png / status.png
    meta.json
  ```

Status codes: abstain `0`, singleton `1`, multi `2`.

### Probability overlays

- `GET /runs/{run_id}/proba/{class_index}.png` — grayscale softmax channel.
- Frontend colorizes with **viridis**, clipping below the per-class threshold (transparent).
- `POST /runs/{run_id}/threshold-class` `{class_id, threshold}` → dense label map for mask-set cache.

Layers panel keeps Image / Features / Probability / Predictions / Annotations independently toggleable.

### Two “model” concepts (do not confuse)

| Store | Path | Owner | Purpose |
|-------|------|-------|---------|
| **ipred project models** | `$LOCAL_DATA_ROOT/ipred/projects/<pid>/models/` | ipred | Session train artifacts |
| **Annotate clf shelf** | `$LOCAL_DATA_ROOT/.clf_models/` | Annotate (`backend/clf_shelf.py`) | Named reusable classifiers from Connect/Browse scaffold UI |

The shelf is separately flagged `# REMOVE THIS AND USE YOUR OWN STUFF` in the UI.

---

## 9. Encoder backends (ONNX vs torch)

| Encoder | Preferred | Fallback / notes |
|---------|-----------|------------------|
| **SlimSAM** | ONNX (`FEATURE_ENCODER_ONNX` or repo SlimSAM `vision_encoder.onnx`) | No torch path |
| **TomoJEPA Mark25/11** | ONNX if `TOMOJEPA[_11]_ONNX` or `ipred/models/tomojepa{25,11}.onnx` **and** spatial size matches requested `input_size` | Else torch `.pth` (`TOMOJEPA[_11]_WEIGHTS` or `ipred/models/*.pth`; needs `ipred[torch]`) |

Weights / ONNX are **gitignored**. Export:

```bash
python -m ipred.scripts.export_tomojepa_onnx --input-size 512
```

`GET /modules` reports `runtime` and `ready` so the Composition UI can grey out broken modules.

---

## 10. Manifold / Suggest Labels

Lightweight placement suggestions on an existing FeatureBank:

1. `POST /manifold/sample` — PCA-whitened variance windows + exclusion; optional ROI from selected shapes.
2. `GET /manifold/{sample_id}/heatmap.png` — residual interestingness.

Implementation: `manifold.py` + `manifold_jobs.py`; results TTL-cached in process. Wired from Preprocess via `ManifoldSuggestPanel` / `useFeatureManifold` (still through `/api/ipred/…`).

---

## 11. Frontend map

| Surface | Path | Talks to |
|---------|------|----------|
| Ipred hub page | `frontend/src/app/pages/IpredPage.tsx` | Composition + trainer prefs |
| Composition window | `frontend/src/components/CompositionPanel/` | `/modules`, `/compositions` |
| Prefer composition | `connectionStore.preferredCompositionId` | Used by preprocess / train hooks |
| Preprocess / Draw / Train | `AnnotateWorkspace` stages | features, classifier, layers |
| Layers | `LayersPanel` + `layerVisibilityStore` | Canvas only (no ipred) |
| API wrapper | `frontend/src/lib/ipredApi.ts` | Annotate proxy only |

---

## 12. Package map (`ipred/src/ipred/`)

| Module | Responsibility |
|--------|----------------|
| `api.py` | FastAPI routes |
| `cli.py` | Offline JSON CLI |
| `catalog.py` | SQLite schema + CRUD |
| `paths.py` | Roots under `LOCAL_DATA_ROOT` |
| `array_source.py` | local / tiled slice load |
| `array_blobs.py` | uploaded content-addressed arrays |
| `compositions.py` | Composition docs + migration |
| `compose_run.py` | Execute module graph |
| `modules/*` | Feature module registry |
| `preprocess.py` | Cache + FeatureBank write |
| `feature_setups.py` | Legacy setup shelf |
| `features.py` | Skimage / PNG helpers |
| `sam_embed.py` / `tomojepa_*.py` | Encoders |
| `train_infer.py` | Train / infer / rethreshold / proba |
| `trainers/*` | Trainer plugins |
| `conformal.py` | Mondrian conformal maps |
| `labels.py` | Shapes → label raster |
| `manifold*.py` | Suggest Labels |
| `cache.py` | In-memory TTL (manifold) |

Tests live in `ipred/tests/` (pytest). Optional deps: base wheel includes **onnxruntime**; torch/timm via `ipred[torch]`.

---

## 13. Mental model (one paragraph)

**ipred is a loopback compute worker** whose durable state sits under `$LOCAL_DATA_ROOT/ipred`. The UI discovers **modules** (`GET /modules`) and builds **compositions** (graphs). Preprocess runs that graph against a session’s data identity (local path, Tiled URI, or uploaded blob) and caches a **FeatureBank**. Train writes a **project model**; infer writes a **run** with softmax + conformal maps. Annotate is only a security/data façade: it never exposes Tiled keys or `:8003` to the browser, and optional array upload lets compute run without sharing the data filesystem.

---

## 14. Quick ops

```bash
# health
curl -s http://127.0.0.1:8003/health

# modules
curl -s http://127.0.0.1:8003/modules | python -m json.tool

# full stack
./start_all.sh
# ipred:    http://127.0.0.1:8003
# annotate: http://127.0.0.1:8002  (proxies /api/ipred)
```

Set secrets and URLs in `backend/.env` (see `backend/.env.example`): `LOCAL_DATA_ROOT`, `IPRED_URL`, `TILED_*`, `TOMOJEPA_*`, `TOMOJEPA_ONNX`, etc.
