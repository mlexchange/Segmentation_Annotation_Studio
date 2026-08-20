"""FastAPI entry point for the Segmentation Annotation Studio API.

Endpoints
---------
This file has grown well past any list that's worth hand-maintaining here —
grep for ``@app.get``/``@app.post`` for the current, authoritative route list;
each route's own docstring covers what it does.

Run with
--------
    uvicorn annotation_server:app --host 127.0.0.1 --port 8002
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi import APIRouter
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware

import annotation_thumbnails
import arrays as arrays_mod
import denoise as denoise_mod
import denoise_bake as denoise_bake_mod
import drafts as drafts_mod
import export_jobs
import guides as guides_mod
import images as images_mod
import ingest as ingest_mod
import local_fs
import tiled_clients as tiled_clients_mod
import volumes as volumes_mod
from browse_helpers import (
    _SINGLE_VALUE_FACET_RAW_KEYS,
    FieldMapping,
    build_field_mapping,
    distinct_from_rows,
    scoped_metadata_rows,
    tiled_distinct_values,
    tiled_search_items,
)
from cache import TTLCache
from coco_export import (
    build_export_plan,
    fold_lightly_splits,
    lightly_classes_map,
    shape_to_mask,
    write_coco_split,
    write_lightly_split,
)
from schemas import (
    BatchProbeRequest,
    DenoiseBakeRequest,
    DraftPayload,
    ExportRequest,
    ExportSourceItem,
    GuidePayload,
    ImageMeta,
    InferRequest,
    IngestPreflightRequest,
    MasksFromTiledRequest,
    MeasureRequest,
    SaveVersionRequest,
    TrainRequest,
)
from sidecars import is_sidecar_key
from source_keys import parse_source_key
from thumbnails import render_thumbnail
from tiled_clients import (
    get_browse_container_for,
    get_tiled_client,
)
from tiled_config import get_tiled_servers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("annotation-server")

app = FastAPI(
    title="Segmentation Annotation Studio API",
    description="Annotation API for SAM3 fine-tuning dataset generation",
    version="0.1.0",
)

# Allowed dev origins. Credentials + wildcard is rejected by browsers, so the
# origin list is explicit and credentials stays off.
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "BROWSE_ALLOWED_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
    # X-Volume-Meta carries /api/image/volume's dims JSON; browsers hide any
    # response header from JS that isn't explicitly exposed, CORS or not.
    expose_headers=["X-Volume-Meta"],
)

# Compress text responses (SPA JS/CSS, JSON). Matters for the production path where
# FastAPI serves the built SPA + API from one origin; PNGs are already compressed so
# the ~500-byte floor skips tiny/binary payloads. (Dev uses the Vite server instead.)
app.add_middleware(GZipMiddleware, minimum_size=500)

# Reject an oversized body before ASGI/Starlette spools it to disk, purely from
# the declared Content-Length header. This is defense-in-depth only — a client
# that omits Content-Length or lies about it still hits the per-file/aggregate
# streaming quotas enforced in ingest_upload(); this middleware just avoids
# paying the parsing cost for an obviously oversized request.
_MAX_REQUEST_BODY_BYTES = int(
    os.getenv("MAX_REQUEST_BODY_BYTES", str(32 * 1024 * 1024 * 1024))
)


@app.middleware("http")
async def _limit_body_size(request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_REQUEST_BODY_BYTES:
                return Response(status_code=413, content="Request body too large")
        except ValueError:
            pass
    return await call_next(request)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    name: str
    uri: str
    has_api_key: bool
    # api_key field intentionally omitted — never expose keys to frontend


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_CACHE_TTL = float(os.getenv("BROWSE_CACHE_TTL_SECONDS", "300"))
_FIELD_MAPPING_TTL = float(os.getenv("BROWSE_FIELD_MAPPING_TTL_SECONDS", "300"))

# Per-endpoint TTL caches. Keys are tuples of normalised query parameters.
_column_cache: TTLCache = TTLCache(ttl_seconds=_CACHE_TTL, max_entries=256)
_items_cache: TTLCache = TTLCache(ttl_seconds=_CACHE_TTL, max_entries=128)
_field_mapping_cache: TTLCache = TTLCache(ttl_seconds=_FIELD_MAPPING_TTL, max_entries=32)
# An entry is now bounded by volumes.MAX_VOLUME_VOXELS (192 MB) rather than the
# old 384**3 (~56.6 MB), since the quality ladder goes to 1024 for thin stacks.
# Entries dropped 4 -> 2 to keep the worst case in the same ballpark (~384 MB)
# instead of quadrupling it.
_volume_cache: TTLCache = TTLCache(ttl_seconds=300.0, max_entries=2)
# Denoised slice PNGs only (the un-denoised path stays uncached, as before, and
# is already cheap). Entries are compressed PNGs — a few MB each at 3232² — so
# 32 covers scrubbing a stack back and forth without unbounded growth. Without
# this, every parameter tweak or revisit re-pays the full filter cost, which for
# NLM/TV is seconds, not milliseconds.
_slice_cache: TTLCache = TTLCache(ttl_seconds=300.0, max_entries=32)


def _require_tiled_server(server_uri: str | None) -> str:
    """Return an allowlisted server URI or raise a redacted client error."""
    try:
        uri, _ = tiled_clients_mod.resolve_tiled_server(server_uri)
    except tiled_clients_mod.TiledClientConfigurationError as exc:
        raise HTTPException(403, "Tiled server is not configured") from exc
    return uri


def _configured_export_root() -> Path:
    """Return the resolved server-owned root for every export/import operation."""
    configured = os.getenv("EXPORT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    local_root = Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve()
    return (local_root / "exports").resolve()


def _resolve_export_dataset(dataset_dir: str) -> Path:
    """Resolve an import path and enforce containment beneath ``EXPORT_ROOT``."""
    export_root = _configured_export_root()
    supplied = Path(dataset_dir).expanduser()
    candidate = supplied.resolve() if supplied.is_absolute() else (export_root / supplied).resolve()
    if not candidate.is_relative_to(export_root):
        raise HTTPException(403, "COCO dataset is outside the configured export root")
    return candidate


def _resolve_field_mapping(
    container: object, server_uri: str, technique: str, container_path: str = ""
) -> FieldMapping:
    """Return a cached :class:`FieldMapping` for the given container."""
    key = (server_uri, technique, container_path or "")
    cached = _field_mapping_cache.get(key)
    if cached is not None:
        return cached
    mapping = build_field_mapping(container)
    _field_mapping_cache.set(key, mapping)
    return mapping


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/config/servers", response_model=list[ServerConfig])
async def get_servers() -> list[ServerConfig]:
    """List configured Tiled servers, with the local (8010) entry first."""
    items = [
        ServerConfig(
            name=name,
            uri=cfg["uri"],
            has_api_key=bool(cfg.get("api_key")),
        )
        for name, cfg in get_tiled_servers().items()
    ]
    items.sort(key=lambda s: (0 if ":8010" in s.uri else 1, s.name))
    return items


@app.get("/api/browse/facets")
async def browse_facets(
    server_uri: Optional[str] = None,
    technique: str = Query("GIWAXS"),
    container_path: Optional[str] = Query(None, description="Tiled container to browse"),
    refresh: bool = Query(False),  # noqa: ARG001 — kept for client API compat
) -> dict[str, list[str]]:
    """Return ordered list of browsable metadata fields, discovered live.

    For each key in the field mapping, the field is kept if at least two
    distinct non-null values exist. Live (no cache) so newly-seeded metadata
    appears in the UI without a restart.
    """
    configured_uri = _require_tiled_server(server_uri)

    def _discover() -> dict[str, list[str]]:
        client = get_tiled_client(configured_uri)
        container, _ = get_browse_container_for(client, container_path)
        mapping = _resolve_field_mapping(container, configured_uri, technique, container_path or "")

        # `container.distinct()` is catalog-global; when browsing a specific
        # container, read its children once and compute values scoped to it.
        # A sample is enough to detect which fields have >=2 distinct values;
        # exact value lists are computed per-field (scoped) by /api/browse/column.
        scoped = bool(container_path)
        scoped_rows = scoped_metadata_rows(container, limit=300) if scoped else []

        def _facet_for_key(disp_key: str) -> tuple[list[str], list[str]]:
            raw_key = mapping.display_to_raw.get(disp_key, disp_key)
            min_values = 1 if raw_key in _SINGLE_VALUE_FACET_RAW_KEYS else 2
            if scoped:
                raw_values = distinct_from_rows(scoped_rows, raw_key)
            else:
                try:
                    result = container.distinct(raw_key, counts=True)
                except Exception:
                    return [], []
                raw_values = result.get("metadata", {}).get(raw_key, [])
            non_null = [
                v for v in raw_values
                if v.get("value") is not None
                and str(v["value"]).strip() not in ("", "None", "NaN", "nan")
            ]
            facet_names: list[str] = []
            if len(non_null) >= min_values:
                facet_names.append(disp_key)
            tech_extra: list[str] = []
            if disp_key in ("technique", "scan_type"):
                for v in non_null:
                    sv = str(v["value"]).strip()
                    if sv:
                        tech_extra.append(sv)
            return facet_names, tech_extra

        keys = mapping.all_display_keys
        max_workers = min(16, max(1, len(keys)))
        facets: list[str] = []
        techniques_seen: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_facet_for_key, dk) for dk in keys]
            for fut in as_completed(futures):
                facet_names, tech_extra = fut.result()
                facets.extend(facet_names)
                for sv in tech_extra:
                    if sv not in techniques_seen:
                        techniques_seen.append(sv)

        return {"facets": sorted(facets), "techniques": techniques_seen}

    try:
        return await asyncio.to_thread(_discover)
    except Exception as exc:
        logger.warning("browse_facets failed: %s", exc)
        return {"facets": [], "techniques": []}


@app.get("/api/browse/column")
async def browse_column(
    server_uri: Optional[str] = None,
    technique: str = Query("GIWAXS"),
    field: str = Query(..., description="Display-key metadata field to group by"),
    filters: str = Query("{}", description="JSON dict of upstream display_key=value selections"),
    container_path: Optional[str] = Query(None, description="Tiled container to browse"),
    limit: int = Query(500, ge=1, le=5000),
    refresh: bool = Query(False),
) -> dict:
    """Return distinct values (+ counts) for *field* via Tiled ``distinct()``."""
    filter_dict = _parse_json_filters(filters)

    configured_uri = _require_tiled_server(server_uri)
    cache_key = ("column", configured_uri, technique, container_path or "", field, filters, limit)
    if not refresh:
        cached = _column_cache.get(cache_key)
        if cached is not None:
            return cached

    def _build() -> dict:
        client = get_tiled_client(configured_uri)
        container, _ = get_browse_container_for(client, container_path)
        mapping = _resolve_field_mapping(container, configured_uri, technique, container_path or "")
        raw_key = mapping.display_to_raw.get(field, field)
        return tiled_distinct_values(
            container,
            raw_key,
            filters=filter_dict,
            field_mapping=mapping,
            limit=limit,
            scoped=bool(container_path),
        )

    try:
        result = await asyncio.to_thread(_build)
    except Exception as exc:
        logger.warning("browse_column failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to browse Tiled column") from exc

    _column_cache.set(cache_key, result)
    return result


@app.get("/api/browse/items")
async def browse_items(
    server_uri: Optional[str] = None,
    technique: str = Query("GIWAXS"),
    filters: str = Query("{}", description="JSON dict of display_key=value selections"),
    container_path: Optional[str] = Query(None, description="Tiled container to browse"),
    limit: int = Query(500, ge=1, le=2000),
    refresh: bool = Query(False),
) -> dict:
    """Return sample records (path + metadata) matching *filters* via ``search()``."""
    filter_dict = _parse_json_filters(filters)

    configured_uri = _require_tiled_server(server_uri)
    cache_key = ("items", configured_uri, technique, container_path or "", filters, limit)
    if not refresh:
        cached = _items_cache.get(cache_key)
        if cached is not None:
            return cached

    def _build() -> dict:
        client = get_tiled_client(configured_uri)
        container, prefix = get_browse_container_for(client, container_path)
        mapping = _resolve_field_mapping(container, configured_uri, technique, container_path or "")
        return tiled_search_items(
            container,
            filters=filter_dict,
            field_mapping=mapping,
            limit=limit,
            container_path_prefix=prefix,
        )

    try:
        result = await asyncio.to_thread(_build)
    except Exception as exc:
        logger.warning("browse_items failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to browse Tiled items") from exc

    _items_cache.set(cache_key, result)
    return result


@app.get("/api/browse/slices")
async def browse_slices(
    path: str = Query(..., description="Tiled container path of a multi-slice dataset"),
    server_uri: Optional[str] = None,
    limit: int = Query(2000, ge=1, le=10000),
    refresh: bool = Query(False),
) -> dict:
    """List a dataset container's array children as individually-openable slices.

    Used by Browse drill-in: each returned record is ``{path, sample, metadata}``
    where ``path`` points at a single array node that opens as a 2-D image.
    """
    configured_uri = _require_tiled_server(server_uri)
    cache_key = ("slices", configured_uri, path, limit)
    if not refresh:
        cached = _items_cache.get(cache_key)
        if cached is not None:
            return cached

    def _build() -> dict:
        client = get_tiled_client(configured_uri)
        container, prefix = get_browse_container_for(client, path)
        result = tiled_search_items(container, limit=limit, container_path_prefix=prefix)
        # Drop sidecar containers so this list matches arrays._stack_keys — the
        # backend's slice ordering. Callers map a row's position to a slice index
        # (Browse "open at slice N"), so a phantom row would shift every index
        # after it.
        result["items"] = [it for it in result["items"] if not is_sidecar_key(it["sample"])]
        # Order slices by key. Ingest writes the node's key as the raw filename
        # stem (only the `image_number` METADATA is zero-padded, not the key
        # itself — see ingest.py), so this only matches numeric slice order for
        # filenames that were already zero-padded on disk.
        result["items"].sort(key=lambda it: it["sample"])
        result["total"] = len(result["items"])
        return result

    try:
        result = await asyncio.to_thread(_build)
    except Exception as exc:
        logger.warning("browse_slices failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to list Tiled slices") from exc

    _items_cache.set(cache_key, result)
    return result


@app.get("/api/browse/thumbnail")
async def browse_thumbnail(
    tiled_path: str = Query(..., description="Slash-separated Tiled path"),
    server_uri: Optional[str] = None,
    size: int = Query(256, ge=32, le=512),
) -> Response:
    """Return a PNG thumbnail for the array at *tiled_path*."""
    configured_uri = _require_tiled_server(server_uri)

    def _build() -> bytes | None:
        client = get_tiled_client(configured_uri)
        node = client
        for part in tiled_path.strip("/").split("/"):
            node = node[part]
        return render_thumbnail(node, size=size)

    try:
        png_bytes = await asyncio.to_thread(_build)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Tiled path not found") from exc
    except Exception as exc:
        logger.warning("browse_thumbnail failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to load Tiled thumbnail") from exc

    if png_bytes is None:
        raise HTTPException(status_code=404, detail="No array data found at the given path")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


def _drafts_for_server(configured_uri: str) -> list[dict]:
    """Local drafts whose source_key belongs to *configured_uri* — the exact
    prefix ``buildSourceKey('tiled', path, serverUri)`` produces client-side,
    so this only ever matches drafts for THIS Tiled server, never a local-file
    draft or one for a different configured server."""
    import drafts as drafts_mod

    prefix = f"tiled:{configured_uri}:"
    return [d for d in drafts_mod.list_drafts() if (d.get("source_key") or "").startswith(prefix)]


@app.get("/api/browse/reset-tiled/preview")
async def browse_reset_tiled_preview(server_uri: Optional[str] = None) -> dict:
    """Local annotation drafts/versions a reset would also clear (with
    ``clear_drafts=True``) — feeds the confirmation dialog's live count, so
    the user sees the real number before committing rather than after."""
    import drafts as drafts_mod

    configured_uri = _require_tiled_server(server_uri)

    def _count() -> tuple[int, int]:
        matching = _drafts_for_server(configured_uri)
        version_count = sum(len(drafts_mod.list_versions(d["source_key"])) for d in matching)
        return len(matching), version_count

    draft_count, version_count = await asyncio.to_thread(_count)
    return {"draft_count": draft_count, "version_count": version_count}


@app.post("/api/browse/reset-tiled")
async def browse_reset_tiled(server_uri: Optional[str] = None, clear_drafts: bool = Query(True)) -> dict:
    """Permanently delete every top-level container on this Tiled server.

    A deliberately blunt "empty the catalog" — everything ingested/browsable
    lives under the root, so wiping every root key is the same end state as
    dropping and recreating the catalog, without needing to stop the Tiled
    process or touch its database file directly. Never touches saved
    training runs (always kept — those live entirely outside Tiled and
    aren't reproducible from it).

    ``clear_drafts`` (default True) additionally removes local annotation
    drafts/versions for this server: a draft is keyed by path, not by what's
    actually in Tiled, so leaving one behind after wiping the data it
    describes means it silently reattaches — full annotation overlays and
    all — to whatever unrelated data happens to land at that same path next.
    Set it to False to keep old annotations around for reference even though
    the Tiled data they were made on is gone.

    Irreversible: there is no undo once this returns. The frontend is
    responsible for the "are you sure" step; this endpoint does exactly what
    it's asked the moment it's called.
    """
    configured_uri = _require_tiled_server(server_uri)

    def _reset() -> tuple[list[str], list[dict[str, str]]]:
        client = get_tiled_client(configured_uri)
        deleted: list[str] = []
        errors: list[dict[str, str]] = []
        for key in list(client.keys()):
            try:
                client.delete_contents(key, recursive=True, external_only=False)
                deleted.append(key)
            except Exception as exc:  # noqa: BLE001 — one bad key must not abort the rest
                logger.warning("reset-tiled: could not delete %r: %s", key, exc)
                errors.append({"key": key, "message": str(exc)})
        return deleted, errors

    def _reset_drafts() -> int:
        import drafts as drafts_mod

        return sum(1 for d in _drafts_for_server(configured_uri) if drafts_mod.delete_draft(d["source_key"]))

    try:
        deleted, errors = await asyncio.to_thread(_reset)
        drafts_deleted = await asyncio.to_thread(_reset_drafts) if clear_drafts else 0
    except Exception as exc:
        logger.error("reset-tiled failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to reset the Tiled server") from exc

    # Every cached listing/mapping was built from data that may no longer
    # exist — same invalidation ingest and save-version already do after
    # their own writes.
    _field_mapping_cache.clear()
    _column_cache.clear()
    _items_cache.clear()

    return {"deleted_keys": deleted, "errors": errors, "drafts_deleted": drafts_deleted}


@app.get("/api/local/list")
async def local_list(
    rel: str = Query("", description="Relative path under the granted root"),
    root: Optional[str] = Query(None, description="Server-configured local root"),
) -> list[dict]:
    """List directory entries under the granted local root."""
    return await asyncio.to_thread(local_fs.list_dir, rel, root)


@app.get("/api/local/samples")
async def local_samples(
    rel: str = Query(..., description="Relative path to a folder under the granted root"),
    root: Optional[str] = Query(None, description="Server-configured local root"),
) -> dict:
    """Return all image files under a local folder (used by the Browse tab).

    Args:
        rel: Relative folder path under the granted root.
        root: Granted absolute browse root (defaults to ``LOCAL_DATA_ROOT``).

    Returns:
        ``{"items": [{"name", "path"}], "total": int}``
    """
    items = await asyncio.to_thread(local_fs.list_image_files, rel, root)
    return {"items": items, "total": len(items)}


@app.get("/api/connect/summary")
async def connect_summary(
    kind: str = Query(..., description="'tiled' or 'local'"),
    server_uri: Optional[str] = None,
    rel: str = Query("", description="Local folder path (kind=local only)"),
    root: Optional[str] = Query(None, description="Server-configured root (kind=local)"),
    container_path: Optional[str] = Query(None, description="Tiled container to browse (kind=tiled)"),
) -> dict:
    """Return a connection summary: sample count and display label.

    For Tiled sources, counts catalog items via the browse API.
    For local sources, counts image files recursively under the folder.

    Returns:
        ``{"kind", "label", "sample_count", "server_uri"}``
    """
    if kind == "local":
        count = await asyncio.to_thread(local_fs.count_image_files, rel, root)
        label = (root or "Local Data Root") + (f"/{rel}" if rel else "")
        return {
            "kind": "local",
            "label": label,
            "sample_count": count,
            "server_uri": None,
            "local_root": root,
        }

    if kind == "tiled":
        configured_uri = _require_tiled_server(server_uri)

        def _count() -> int:
            client = get_tiled_client(configured_uri)
            container, _ = get_browse_container_for(client, container_path)
            # An unfiltered count is just the container size — a single request.
            # Avoid iterating every child and building per-item metadata dicts,
            # which is O(N) HTTP round trips and stalls the connect UI.
            try:
                return int(len(container))
            except Exception:
                result = tiled_search_items(container, filters={}, limit=10_000)
                return int(result.get("total", 0))

        try:
            count = await asyncio.to_thread(_count)
        except Exception as exc:
            logger.warning("connect_summary tiled count failed: %s", exc)
            count = 0

        servers = get_tiled_servers()
        label = next(
            (cfg.get("name", name) for name, cfg in servers.items()
             if (cfg.get("uri") or "").rstrip("/") == configured_uri.rstrip("/")),
            configured_uri,
        )
        if container_path:
            label = f"{label} · {container_path}"
        return {
            "kind": "tiled",
            "label": label,
            "sample_count": count,
            "server_uri": configured_uri,
            "container_path": container_path,
        }

    raise HTTPException(400, f"Unknown kind: {kind!r}; must be 'tiled' or 'local'")


@app.get("/api/tiled/list")
async def tiled_list(
    path: str = Query("", description="Slash-separated Tiled node path ('' = root)"),
    server_uri: Optional[str] = None,
) -> list[dict]:
    """List children of a Tiled node, classifying containers vs. arrays.

    Args:
        path: Slash-separated path into the Tiled tree (empty → root).
        server_uri: Tiled server URI; falls back to the default server.

    Returns:
        A list of ``{"name", "path", "is_dir", "is_array"}`` entries sorted
        with containers first, then arrays, both alphabetically.

    Raises:
        HTTPException: 404 if the path does not exist, 500 on read failure.
    """
    configured_uri = _require_tiled_server(server_uri)

    def _run() -> list[dict]:
        client = get_tiled_client(configured_uri)
        node = client
        for key in [k for k in path.split("/") if k]:
            try:
                node = node[key]
            except (KeyError, TypeError) as exc:
                raise HTTPException(404, f"Tiled path not found: {path!r}") from exc

        entries: list[dict] = []
        try:
            keys = list(node.keys())
        except Exception as exc:  # leaf node (array) has no children
            raise HTTPException(400, f"Not a container: {path!r}") from exc

        for key in keys:
            child = node[key]
            family = getattr(getattr(child, "structure_family", None), "value", None) or str(
                getattr(child, "structure_family", "")
            )
            is_array = family == "array"
            is_container = family == "container"
            entries.append(
                {
                    "name": key,
                    "path": f"{path}/{key}".strip("/"),
                    "is_dir": is_container,
                    "is_array": is_array,
                }
            )
        entries.sort(key=lambda e: (0 if e["is_dir"] else 1, e["name"]))
        return entries

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("tiled_list failed: %s", exc)
        raise HTTPException(500, "Failed to list Tiled path") from exc


@app.get("/api/image/meta", response_model=ImageMeta)
async def image_meta(
    source: str = Query(...),
    kind: str = Query(...),
    server_uri: Optional[str] = None,
    root: Optional[str] = Query(None, description="Server-configured root (kind=local)"),
) -> ImageMeta:
    """Return shape / dtype metadata for an image source."""
    if kind == "tiled":
        server_uri = _require_tiled_server(server_uri)

    def _run() -> ImageMeta:
        node = arrays_mod.resolve_array(source, kind, server_uri, root)
        meta = arrays_mod.array_shape_meta(node)
        sl = arrays_mod.read_slice(node, meta, 0)
        flat = sl.ravel().astype(float)
        return ImageMeta(
            n_slices=meta["n_slices"],
            height=meta["height"],
            width=meta["width"],
            dtype=meta["dtype"],
            is_rgb=meta["is_rgb"],
            value_range=[float(flat.min()), float(flat.max())],
            keywords=arrays_mod.node_keywords(node),
        )

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("image_meta failed: %s", exc)
        raise HTTPException(500, "Failed to read image metadata") from exc


def _model_denoised_png(
    source: str,
    kind: str,
    server_uri: Optional[str],
    root: Optional[str],
    slice_index: int,
    run_id: str,
    crop: int,
) -> bytes:
    """Render one slice through a trained Noise2Noise/Noise2Void run.

    Preprocessing is delegated to ``denoise_train``'s own helpers rather than
    reimplemented here: the network was trained on the display-mapped uint8
    grayscale those produce (volume-global bounds, the run's saved render
    options), so any divergence would feed it data unlike anything it saw in
    training. Reusing the exact functions is what keeps the two halves of that
    contract from drifting.

    The output is therefore ALREADY in display space, which is why it goes
    through ``images.apply_colormap`` instead of ``render_slice`` — normalizing
    it again would apply the intensity mapping twice.

    Takes ``ML_LOCK``: unlike the classical filters this is GPU work, and must
    not run concurrently with a training job.
    """
    import denoise_train
    import tiling
    import train_common

    config = train_common.load_run_config(run_id)
    if config.get("model_family") != "dlsia_denoiser" or config.get("task") != "denoising":
        raise HTTPException(
            422,
            f"Run {run_id!r} is not a denoiser — pick a trained denoiser run.",
        )
    # Which network this run is; defaults to TUNet for runs saved before the
    # field existed. Only the TUNet architecture needs dlsia.
    try:
        fam = train_common.denoiser_runtime_for(config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if train_common.denoiser_needs_dlsia(config) and not train_common.dlsia_available():
        raise HTTPException(503, "dlsia is not installed on this server")
    if not tiling.qlty_available():
        raise HTTPException(503, "Applying a denoiser needs the 'qlty' package, which is missing")

    device = train_common.pick_device()
    if device is None:
        raise HTTPException(503, "torch is not installed on this server")

    node = arrays_mod.resolve_array(source, kind, server_uri, root)
    meta = arrays_mod.array_shape_meta(node)
    opts = denoise_train._render_opts(config.get("render") or {})
    global_range = images_mod._sample_global_stats(node, meta)
    gray = denoise_train._slice_to_gray_uint8(node, meta, slice_index, opts, global_range)
    gray = _centre_crop(gray, crop)

    # Non-blocking: a preview must report "busy" rather than queue behind a
    # multi-minute training run holding the lock.
    if not train_common.ML_LOCK.acquire(blocking=False):
        raise HTTPException(409, "The device is busy with another job — try again when it finishes.")
    try:
        import torch

        state = train_common.load_adapter_state(run_id)
        model = fam.load_model(state, device)
        model.eval()
        with torch.no_grad():
            out = tiling.denoise_image_tiled(
                gray,
                forward_fn=fam.make_forward_fn(model),
                to_tensor_fn=fam.make_to_tensor_fn(),
                window=int(config["image_size"]),
                device=device,
            )
    finally:
        train_common.ML_LOCK.release()

    if out is None:
        raise HTTPException(500, "Denoising was interrupted")
    unit = np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)
    return images_mod.encode_png(images_mod.apply_colormap(unit, opts.get("cmap", "gray")))


def _centre_crop(arr: "np.ndarray", size: int) -> "np.ndarray":
    """Centred square crop of *size*, or *arr* unchanged if it already fits."""
    h, w = arr.shape[:2]
    if size <= 0 or (h <= size and w <= size):
        return arr
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return arr[top : top + min(size, h), left : left + min(size, w)]


def _denoised_slice(
    node: object,
    meta: dict,
    slice_index: int,
    method: str,
    strength: float,
    crop: int = 0,
) -> "np.ndarray":
    """Read *slice_index* and denoise it, pulling z-neighbours when the method
    is a 3-D one.

    The 3-D filters are the training-free way to exploit slice-to-slice
    correlation (adjacent tomographic slices share structure, their noise is
    independent), which is why this reads a window rather than one slice. The
    window is clamped to the volume, and the target slice's position inside the
    returned stack is tracked explicitly — it is NOT always the centre, since
    the window is truncated at the first and last slice.

    When *crop* > 0 the crop is taken BEFORE filtering (that's the whole point —
    filtering 10.4 MP is what's slow). Consequence worth knowing: results near
    the crop border, and NLM's patch search in particular, differ slightly from
    the full-slice result, so a crop is a tuning aid rather than a byte-exact
    preview of the bake.
    """
    radius = denoise_mod.z_radius_for(method)
    if radius == 0:
        sl = _centre_crop(np.asarray(arrays_mod.read_slice(node, meta, slice_index)), crop)
        return denoise_mod.denoise_slice(sl, method, strength)

    n_slices = int(meta["n_slices"])
    lo = max(0, slice_index - radius)
    hi = min(n_slices - 1, slice_index + radius)
    frames = []
    for idx in range(lo, hi + 1):
        try:
            frames.append(_centre_crop(np.asarray(arrays_mod.read_slice(node, meta, idx)), crop))
        except Exception as exc:  # noqa: BLE001 — a bad neighbour must not fail the view
            logger.warning("denoise: skipping unreadable neighbour slice %d: %s", idx, exc)
            if idx == slice_index:
                raise
    if len(frames) < 2:
        # Not enough usable z-context (single-slice source, or neighbours
        # unreadable) — fall back to the 2-D sibling rather than erroring out.
        fallback = "gaussian" if method == "gaussian3d" else "median"
        sl = _centre_crop(np.asarray(arrays_mod.read_slice(node, meta, slice_index)), crop)
        return denoise_mod.denoise_slice(sl, fallback, strength)

    target_pos = min(slice_index - lo, len(frames) - 1)
    stack = np.stack(frames, axis=0)
    return denoise_mod.denoise_stack(stack, method, strength)[target_pos]


@app.get("/api/image/slice")
async def image_slice(
    source: str = Query(...),
    kind: str = Query(...),
    slice_index: int = Query(0),
    server_uri: Optional[str] = None,
    root: Optional[str] = Query(None, description="Server-configured root (kind=local)"),
    norm: str = Query("global"),
    scale: str = Query("linear"),
    vmin_pct: float = Query(1.0),
    vmax_pct: float = Query(99.0),
    cmap: str = Query("gray"),
    denoise_method: str = Query("none", description="Classical denoise filter (see denoise.ALL_METHODS)"),
    denoise_strength: float = Query(0.5, ge=0.0, le=1.0),
    denoise_run_id: Optional[str] = Query(
        None,
        description="Saved denoiser run to apply when denoise_method='model' (a trained "
                    "Noise2Noise/Noise2Void run, not a classical filter).",
    ),
    denoise_crop: int = Query(
        0, ge=0, le=2048,
        description="If >0, denoise and return only a centred square crop of this size at 1:1 "
                    "(fast tuning preview — measured 7.5s full-slice vs ~0.4s cropped for bilateral "
                    "at 3232²). Denoising cannot be judged on a downscaled image, since downscaling "
                    "is itself a denoiser, so this crops rather than resizes.",
    ),
) -> Response:
    """Render one slice of an image source as a PNG.

    Denoising, when requested, runs on the RAW slice before normalization —
    noise statistics live in the source's own intensity units, not the 8-bit
    display range. It is deliberately applied HERE rather than inside
    ``images.render_slice`` for two reasons: ``render_slice`` early-returns for
    RGB inputs (so a hook at its normalize call would silently skip colour
    sources), and ``volumes.build_volume`` normalizes independently — denoising
    inside the shared helper would silently desynchronize the 3D tab from the
    2D canvas. Keeping it in this route makes the 3D tab showing raw data an
    explicit choice rather than an accident.

    ``_sample_global_stats`` intentionally still samples RAW slices, so display
    contrast doesn't jump when denoising is toggled on and off.
    """
    if kind == "tiled":
        server_uri = _require_tiled_server(server_uri)

    # "model" is not a classical filter — it applies a trained denoiser run and
    # takes an entirely different path (see _model_denoised_png).
    if denoise_method == "model":
        if not denoise_run_id:
            raise HTTPException(422, "denoise_method='model' needs a denoise_run_id")
        try:
            png = await asyncio.to_thread(
                _model_denoised_png, source, kind, server_uri, root,
                slice_index, denoise_run_id, denoise_crop,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("model denoise preview failed: %s", exc)
            raise HTTPException(500, "Failed to apply the denoiser to this slice") from exc
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=300"},
        )

    if denoise_method not in denoise_mod.ALL_METHODS:
        raise HTTPException(422, f"Unknown denoise method: {denoise_method}")
    if denoise_method not in denoise_mod.available_methods():
        raise HTTPException(
            422,
            f"Denoise method {denoise_method!r} is unavailable on this server "
            "(missing optional dependency)",
        )

    opts = {
        "norm": norm,
        "scale": scale,
        "vmin_pct": vmin_pct,
        "vmax_pct": vmax_pct,
        "cmap": cmap,
    }
    # Full-parameter cache key. Deliberately NOT keyed on id(node) the way
    # images._stats_cache is — that's a CPython object address, which can miss
    # after a node is re-resolved and (worse) be reused by a different node
    # after GC. Denoising is the expensive part of this route, and the route had
    # no server-side cache at all before, so re-viewing a slice used to pay the
    # full cost again.
    cache_key = (
        "slice", kind, source, server_uri or "", root or "", slice_index,
        norm, scale, round(vmin_pct, 3), round(vmax_pct, 3), cmap,
        denoise_method, round(denoise_strength, 3), denoise_crop,
    )

    def _run() -> bytes:
        if denoise_method != "none":
            cached = _slice_cache.get(cache_key)
            if cached is not None:
                return cached

        node = arrays_mod.resolve_array(source, kind, server_uri, root)
        meta = arrays_mod.array_shape_meta(node)
        global_range = None
        if norm == "global":
            # Deliberately sampled from RAW slices, and from the WHOLE volume
            # even for a crop, so the crop preview's brightness matches the main
            # canvas instead of auto-levelling to whatever is inside the crop.
            global_range = images_mod._sample_global_stats(node, meta)

        if denoise_method == "none":
            sl = arrays_mod.read_slice(node, meta, slice_index)
        else:
            sl = _denoised_slice(
                node, meta, slice_index, denoise_method, denoise_strength, denoise_crop
            )

        rgb = images_mod.render_slice(sl, opts, global_range)
        png = images_mod.encode_png(rgb)
        if denoise_method != "none":
            _slice_cache.set(cache_key, png)
        return png

    try:
        png = await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except ValueError as exc:  # denoise rejected the request (bad method/params)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.error("image_slice failed: %s", exc)
        raise HTTPException(500, "Failed to render image slice") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


@app.get("/api/denoise/auto")
async def denoise_auto(
    source: str = Query(...),
    kind: str = Query(...),
    method: str = Query(...),
    slice_index: int = Query(0),
    server_uri: Optional[str] = None,
    root: Optional[str] = Query(None, description="Server-configured root (kind=local)"),
) -> dict:
    """Suggest a denoise strength for one slice from its own measured noise.

    Backs the panel's "Auto" button. Uses the same estimator the bake job does,
    so an auto-picked strength previews and bakes identically. Cheap
    (one convolution) and deliberately never takes ``ML_LOCK``.
    """
    if kind == "tiled":
        server_uri = _require_tiled_server(server_uri)
    if method not in denoise_mod.ALL_METHODS:
        raise HTTPException(422, f"Unknown denoise method: {method}")

    def _run() -> dict:
        node = arrays_mod.resolve_array(source, kind, server_uri, root)
        meta = arrays_mod.array_shape_meta(node)
        sl = np.asarray(arrays_mod.read_slice(node, meta, slice_index))
        return {
            "method": method,
            "strength": denoise_mod.auto_strength(sl, method),
            "noise_sigma": denoise_mod.estimate_noise_sigma(sl),
        }

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("denoise_auto failed: %s", exc)
        raise HTTPException(500, "Failed to estimate a denoise strength") from exc


@app.post("/api/denoise/bake")
async def denoise_bake(payload: DenoiseBakeRequest) -> dict:
    """Denoise a whole volume and save it as a new, annotatable Tiled dataset.

    The Annotate-tab preview is display-only; this is how a denoised volume
    becomes real data you can annotate, train on and export. Runs on a
    background thread (no ``ML_LOCK`` — classical filters are CPU-only and must
    not contend with training); poll ``/api/export/status/{job_id}``.
    """
    server_uri = _require_tiled_server(payload.server_uri)
    if payload.method == "none":
        raise HTTPException(422, "Pick a denoise method before saving a denoised copy.")
    # "model" applies a trained denoiser run; it is validated against the run
    # registry inside the job, not against the classical-filter menu.
    if payload.method == "model":
        if not payload.run_id:
            raise HTTPException(422, "Applying a trained denoiser needs a run_id.")
    else:
        if payload.method not in denoise_mod.ALL_METHODS:
            raise HTTPException(422, f"Unknown denoise method: {payload.method}")
        if payload.method not in denoise_mod.available_methods():
            raise HTTPException(
                422,
                f"Denoise method {payload.method!r} is unavailable on this server "
                "(missing optional dependency)",
            )

    request = payload.model_copy(update={"server_uri": server_uri})
    jid = export_jobs.new_job(f"denoise:{payload.source}")
    threading.Thread(
        target=denoise_bake_mod.run_denoise_bake_job,
        args=(jid, request),
        daemon=True,
    ).start()
    return {"job_id": jid}


@app.get("/api/image/volume")
async def image_volume(
    source: str = Query(...),
    kind: str = Query(...),
    server_uri: Optional[str] = None,
    root: Optional[str] = Query(None, description="Server-configured root (kind=local)"),
    max_dim: int = Query(256, ge=32, le=1024),
    norm: str = Query("global"),
    scale: str = Query("linear"),
    vmin_pct: float = Query(1.0),
    vmax_pct: float = Query(99.0),
) -> Response:
    """Return a downsampled uint8 intensity volume for the 3D tab.

    Body is raw C-order ``(nz, ny, nx)`` bytes (``application/octet-stream``);
    dimensions and any skipped slices are carried in the ``X-Volume-Meta``
    JSON response header rather than the body, so the frontend can validate
    ``byteLength == nz*ny*nx`` before touching the buffer. See
    ``volumes.volume_dims`` for the exact grid-size contract the frontend's
    label-volume rasterizer mirrors.
    """
    if kind == "tiled":
        server_uri = _require_tiled_server(server_uri)

    opts = {"norm": norm, "scale": scale, "vmin_pct": vmin_pct, "vmax_pct": vmax_pct}
    cache_key = (
        "volume", kind, source, server_uri or "", root or "",
        max_dim, norm, scale, round(vmin_pct, 3), round(vmax_pct, 3),
    )

    def _run() -> tuple[bytes, dict]:
        cached = _volume_cache.get(cache_key)
        if cached is not None:
            return cached
        node = arrays_mod.resolve_array(source, kind, server_uri, root)
        meta = arrays_mod.array_shape_meta(node)
        global_range = None
        if norm == "global":
            global_range = images_mod._sample_global_stats(node, meta)
        # Clamp to what fits the voxel budget. The frontend applies the same
        # ladder before rasterizing its label volume, so both land on the same
        # grid; doing it here too means a hand-made request can't blow memory.
        effective = volumes_mod.effective_max_dim(
            int(meta["n_slices"]), int(meta["height"]), int(meta["width"]), max_dim
        )
        if effective != max_dim:
            logger.info("volume quality %d reduced to %d to fit the voxel budget", max_dim, effective)
        result = volumes_mod.build_volume(node, meta, opts, effective, global_range)
        _volume_cache.set(cache_key, result)
        return result

    try:
        payload, vol_meta = await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("image_volume failed: %s", exc)
        raise HTTPException(500, "Failed to build image volume") from exc

    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "private, max-age=300",
            "X-Volume-Meta": json.dumps(vol_meta, separators=(",", ":")),
        },
    )


@app.get("/api/annotations/draft")
async def get_draft(source_key: str = Query(...)) -> dict:
    """Return the saved draft for source_key, or 404."""
    result = await asyncio.to_thread(drafts_mod.load_draft, source_key)
    if result is None:
        raise HTTPException(404, "No draft found")
    return result


@app.put("/api/annotations/draft")
async def put_draft(
    source_key: str = Query(...),
    payload: DraftPayload = ...,
) -> dict:
    """Persist a crash-recovery draft for source_key (autosave only).

    Does NOT sync to Tiled — use POST /api/annotations/save for that.
    """
    body = payload.model_dump()
    return await asyncio.to_thread(drafts_mod.save_draft, source_key, body)


@app.get("/api/annotations/drafts")
async def list_drafts_route() -> list[dict]:
    """List all saved drafts with summary metadata."""
    return await asyncio.to_thread(drafts_mod.list_drafts)


@app.get("/api/guide")
async def get_guide(source_key: str = Query(...)) -> dict:
    """Return the annotation guide for source_key, or 404 if none exists."""
    result = await asyncio.to_thread(guides_mod.load_guide, source_key)
    if result is None:
        raise HTTPException(404, "No guide found")
    return result


@app.put("/api/guide")
async def put_guide(
    source_key: str = Query(...),
    guide: GuidePayload = ...,
) -> dict:
    """Persist the annotation guide for source_key (dataset-scoped)."""
    return await asyncio.to_thread(guides_mod.save_guide, source_key, guide.model_dump())


@app.post("/api/measure")
async def measure_region(
    source_key: str = Query(...),
    body: MeasureRequest = ...,
) -> dict:
    """Return raw-intensity statistics inside the union of the given shapes on a
    slice: min/max/mean/std and pixel count. Uses the raw array values (not the
    display-rendered image), so results are meaningful for scientific data.
    """
    def _run() -> dict:
        parsed = parse_source_key(source_key)
        kind = parsed["kind"] or "local"
        node = arrays_mod.resolve_array(parsed["path"] or "", kind, parsed["server_uri"])
        meta = arrays_mod.array_shape_meta(node)
        sidx = max(0, min(int(body.slice_index), meta["n_slices"] - 1))
        arr = np.asarray(arrays_mod.read_slice(node, meta, sidx))
        # Collapse RGB to luminance so intensity stats are single-channel.
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
        h, w = arr.shape[:2]

        union = np.zeros((h, w), dtype=bool)
        for shape in body.shapes:
            try:
                union |= shape_to_mask(shape, h, w)
            except Exception:
                continue

        vals = arr[union]
        if vals.size == 0:
            return {"pixel_count": 0, "min": None, "max": None, "mean": None, "std": None}
        return {
            "pixel_count": int(vals.size),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return await asyncio.to_thread(_run)


@app.post("/api/guide/generate")
async def generate_guide_route(
    source_key: str = Query(...),
    payload: DraftPayload = ...,
) -> dict:
    """Build a guide skeleton (per-class label/color + example crops) from an
    annotation payload (the current draft or a fetched version).

    Descriptions are returned blank for the lead to fill in. Does not persist —
    the client merges the result into the guide and saves via PUT /api/guide.
    """
    def _run() -> dict:
        import guide_gen
        return guide_gen.generate_guide(source_key, payload.model_dump())

    return await asyncio.to_thread(_run)


@app.post("/api/annotations/preview-thumbnail")
async def preview_annotation_thumbnail(
    source_key: str = Query(...),
    payload: DraftPayload = ...,
) -> Response:
    """Render a PNG preview of the annotated thumbnail without saving a version."""
    def _run() -> bytes:
        import annotation_thumbnails
        png = annotation_thumbnails.render_annotated_thumbnail(source_key, payload.model_dump())
        if not png:
            raise HTTPException(500, "Could not render preview thumbnail")
        return png

    try:
        png_bytes = await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Preview thumbnail failed for %s: %s", source_key, exc)
        raise HTTPException(500, "Preview thumbnail failed") from exc

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/annotations/save")
async def save_annotation_version(
    source_key: str = Query(...),
    body: SaveVersionRequest = ...,
) -> dict:
    """Create a new immutable version, generate an annotated thumbnail, and sync
    summary metadata to Tiled.

    Returns the new version number, timestamp, shape count, and whether a
    thumbnail was generated.
    """
    def _run() -> dict:
        payload = body.payload.model_dump()
        # Persist version JSON first so the critical data lands quickly.
        result = drafts_mod.save_version(
            source_key,
            payload,
            annotated_by=body.annotated_by,
            notes=body.notes,
        )
        version = result["version"]
        saved_at = result["saved_at"]

        # Thumbnail: reuse the modal preview when provided (avoids a second render).
        has_thumbnail = False
        png: bytes | None = None
        if body.thumbnail_base64:
            png = annotation_thumbnails.decode_thumbnail_base64(body.thumbnail_base64)
        if not png:
            try:
                png = annotation_thumbnails.render_annotated_thumbnail(source_key, payload)
            except Exception as exc:
                logger.warning("Thumbnail generation failed for %s v%d: %s", source_key, version, exc)

        if png:
            try:
                drafts_mod.save_version_thumbnail(source_key, version, png)
                has_thumbnail = True
                # Tiled upload is slow — run in background, don't block the save response.
                threading.Thread(
                    target=annotation_thumbnails.upload_thumbnail_to_tiled,
                    args=(source_key, version, saved_at, png),
                    daemon=True,
                ).start()
            except Exception as exc:
                logger.warning("Thumbnail persist failed for %s v%d: %s", source_key, version, exc)
        result["has_thumbnail"] = has_thumbnail

        try:
            import tiled_annotation_sync
            tiled_annotation_sync.sync_annotation_metadata(source_key, payload)
            _field_mapping_cache.clear()
            _column_cache.clear()
            _items_cache.clear()
        except Exception as exc:
            logger.warning("Tiled metadata sync failed for %s: %s", source_key, exc)
        return result

    return await asyncio.to_thread(_run)


@app.get("/api/annotations/versions/{version}/thumbnail")
async def get_version_thumbnail(version: int, source_key: str = Query(...)) -> Response:
    """Return the annotated PNG thumbnail for a specific version, or 404."""
    png = await asyncio.to_thread(drafts_mod.get_version_thumbnail, source_key, version)
    if png is None:
        raise HTTPException(404, "No thumbnail for this version")
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/annotations/versions")
async def list_versions_route(source_key: str = Query(...)) -> list[dict]:
    """List saved versions (metadata only, no payload) oldest-first."""
    return await asyncio.to_thread(drafts_mod.list_versions, source_key)


@app.get("/api/annotations/versions/{version}")
async def get_version_route(version: int, source_key: str = Query(...)) -> dict:
    """Return the full payload for a specific version number."""
    doc = await asyncio.to_thread(drafts_mod.get_version, source_key, version)
    if doc is None:
        raise HTTPException(404, f"Version {version} not found for this source")
    return doc


@app.post("/api/export/coco")
async def export_coco(payload: ExportRequest) -> dict:
    """Write the COCO dataset for one or more annotated samples.

    Supports two modes:
    - **Single-source** (legacy): ``kind``/``source``/``slices`` on the payload.
    - **Multi-source**: ``sources`` list, each with its own kind/source/slices.

    The output directory is derived automatically from ``dataset_name`` or the
    first source path + a timestamp; always written under ``EXPORT_ROOT``.
    After writing, the dataset is registered as a Tiled node.
    """
    from datetime import datetime, timezone

    export_root = _configured_export_root()

    # Auto-derive a dataset folder name.
    if payload.dataset_name:
        folder_name = payload.dataset_name
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        first_source = (payload.sources[0].source if payload.sources else payload.source) or "dataset"
        stem = first_source.replace("/", "_").replace("\\", "_")[-30:].strip("_") or "dataset"
        folder_name = f"{stem}_{ts}"

    # Stamp the annotator into the folder so gathered downloads are self-identifying.
    annotator = (payload.annotator or "").strip()
    if annotator:
        from coco_export import _safe_name
        folder_name = f"{_safe_name(annotator)}__{folder_name}"

    out_root = (export_root / folder_name).resolve()
    if not out_root.is_relative_to(export_root):
        raise HTTPException(403, "Derived output path escapes EXPORT_ROOT")

    # Build a normalised list of source items from either multi or single mode.
    if payload.sources:
        source_items: list[ExportSourceItem] = payload.sources
    else:
        source_items = [ExportSourceItem(
            kind=payload.kind,  # type: ignore[arg-type]
            source=payload.source,
            server_uri=payload.server_uri,
            slices=payload.slices,
            split_by_slice=payload.split_by_slice,
            negative_slices=payload.negative_slices,
        )]

    # Dry run = counts only. Resolve splits from the payload without touching
    # Tiled, reading slices, sampling stats, or rasterizing — so the preview is
    # instant. Returned synchronously (no job).
    if payload.dry_run:
        from coco_export import _resolve_split
        summary: dict = {"skipped_zero_area": 0, "splits": {}}
        for item in source_items:
            neg_keys = {str(k) for k in item.negative_slices}
            all_keys = list(set(item.slices.keys()) | neg_keys)
            resolved = _resolve_split(
                all_keys,
                {str(k): v for k, v in item.split_by_slice.items()},
                payload.auto_split,
            )
            for k in all_keys:
                split = resolved.get(k, "train")
                bucket = summary["splits"].setdefault(split, {"n_images": 0, "n_annotations": 0})
                bucket["n_images"] += 1
                bucket["n_annotations"] += len(item.slices.get(k, []))
        return summary

    # Real export runs on a background thread; the UI polls /api/export/status
    # for phase/progress/log lines and downloads the .zip when done.
    jid = export_jobs.new_job(str(out_root))
    threading.Thread(
        target=_run_export_job,
        args=(jid, source_items, payload, out_root),
        daemon=True,
    ).start()
    return {"job_id": jid, "dataset_path": str(out_root)}


def _run_export_job(
    jid: str,
    source_items: "list[ExportSourceItem]",
    payload: "ExportRequest",
    out_root: Path,
) -> None:
    """Background worker: render+rasterize all sources, write the dataset tree
    (images + masks + COCO), zip it for download, then sync Tiled metadata."""
    lightly = getattr(payload, "format", "coco_sam3") == "lightly_dinov3"

    try:
        export_jobs.update(jid, state="running", phase="reading")
        total = sum(
            len(set(item.slices.keys()) | {str(k) for k in item.negative_slices})
            for item in source_items
        )
        export_jobs.set_total(jid, total)

        merged_splits: dict[str, dict] = {}
        skipped_total = 0
        merged_categories: list[dict] = []
        merged_info: dict = {}

        for item in source_items:
            export_jobs.log(jid, f"Reading {item.source} …")
            node = arrays_mod.resolve_array(item.source, item.kind, item.server_uri)
            tmp = ExportRequest(
                kind=item.kind,
                source=item.source,
                server_uri=item.server_uri,
                slices=item.slices,
                split_by_slice=item.split_by_slice,
                negative_slices=item.negative_slices,
                classes=payload.classes,
                render=payload.render,
                auto_split=payload.auto_split,
            )

            def _cb(message: str, _jid: str = jid) -> None:
                export_jobs.bump(_jid, 1)
                export_jobs.log(_jid, message)

            plan = build_export_plan(
                node, tmp,
                render_slice_fn=images_mod.render_slice,
                array_shape_meta_fn=arrays_mod.array_shape_meta,
                read_slice_fn=arrays_mod.read_slice,
                sample_global_stats_fn=images_mod._sample_global_stats,
                progress_cb=_cb,
                include_polygons=payload.include_polygons,
                lightly=lightly,
            )
            skipped_total += plan["skipped_zero_area"]
            if not merged_categories:
                merged_categories = plan["categories"]
                merged_info = plan["info"]
            for split_name, split_data in plan["splits"].items():
                bucket = merged_splits.setdefault(split_name, {"images": [], "annotations": []})
                bucket["images"].extend(split_data["images"])
                bucket["annotations"].extend(split_data["annotations"])

        # Stamp the annotator so downloads are self-identifying for external
        # inter-annotator-agreement analysis (folder name + COCO info + manifest).
        annotator = (payload.annotator or "").strip()
        exported_at = datetime.now(timezone.utc).isoformat()
        if annotator:
            merged_info = {**(merged_info or {}), "annotator": annotator}
        source_keys = [
            (f"tiled:{item.server_uri or ''}:{item.source}" if item.kind == "tiled"
             else f"local:{item.source}")
            for item in source_items
        ]
        manifest = {
            "annotator": annotator,
            "exported_at": exported_at,
            "source_keys": source_keys,
            "classes": [
                (c.model_dump() if hasattr(c, "model_dump") else dict(c)) for c in payload.classes
            ],
        }
        if lightly:
            # Document the DINOv3/Lightly convention: masks are 0-indexed class ids
            # with unannotated pixels set to this ignore index.
            manifest["ignore_index"] = 255

        # Write files AND build the download .zip in one pass. ZIP_STORED: the
        # PNGs are already compressed, so re-deflating them is wasted CPU.
        import zipfile
        export_jobs.update(jid, phase="writing")
        zip_path = f"{out_root}.zip"
        written: dict = {}
        # Create the dataset dir (and its parent EXPORT_ROOT, e.g. ~/data/exports)
        # BEFORE opening the zip — the zip lives at {out_root}.zip, so its parent
        # must exist or ZipFile("w") raises FileNotFoundError on a fresh install.
        out_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            manifest_json = json.dumps(manifest, indent=2)
            (out_root / "manifest.json").write_text(manifest_json)
            zf.writestr("manifest.json", manifest_json)

            if lightly:
                # DINOv3 / Lightly: classes.json (index→name, 0-indexed, no background)
                # at the dataset root; each split as images/ + masks/ with matching
                # stems. Unannotated pixels are 255 (the ignore index).
                classes_json = json.dumps(lightly_classes_map(merged_categories), indent=2)
                (out_root / "classes.json").write_text(classes_json)
                zf.writestr("classes.json", classes_json.encode("utf-8"))
                # Lightly uses train/val only: 'valid' AND 'test' fold into 'val'.
                for dir_name, split_data in fold_lightly_splits(merged_splits).items():
                    export_jobs.log(jid, f"Writing split '{dir_name}' ({len(split_data['images'])} images + masks)…")
                    written[dir_name] = write_lightly_split(
                        out_root / dir_name,
                        images=split_data["images"],
                        zf=zf,
                        arc_prefix=f"{dir_name}/",
                    )
            else:
                for split_name, split_data in merged_splits.items():
                    export_jobs.log(jid, f"Writing split '{split_name}' ({len(split_data['images'])} images + masks)…")
                    written[split_name] = write_coco_split(
                        out_root / split_name,
                        images=split_data["images"],
                        categories=merged_categories,
                        annotations=split_data["annotations"],
                        mode=payload.mode,
                        info=merged_info,
                        zf=zf,
                        arc_prefix=f"{split_name}/",
                    )

        export_jobs.update(jid, phase="syncing")
        import tiled_annotation_sync
        for item in source_items:
            if item.kind != "tiled":
                continue
            sk = f"tiled:{item.server_uri or ''}:{item.source}"
            try:
                tiled_annotation_sync.sync_annotation_metadata(
                    sk, {"classes": payload.classes, "slices": item.slices},
                )
            except Exception as sync_exc:
                logger.warning("Export Tiled sync failed for %s: %s", sk, sync_exc)

        result = {
            "skipped_zero_area": skipped_total,
            "written": written,
            "dataset_path": str(out_root),
            "zip_available": True,
            "splits": {
                k: {"n_images": len(v["images"]), "n_annotations": len(v["annotations"])}
                for k, v in merged_splits.items()
            },
        }
        export_jobs.update(jid, zip_path=zip_path, result=result, phase="done", state="done")
        export_jobs.log(jid, "Export complete.")
    except FileExistsError as exc:
        export_jobs.update(jid, state="error", phase="error", error=f"{exc} (use overwrite or merge)")
    except Exception as exc:  # noqa: BLE001
        logger.error("Export job failed: %s", exc)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))


@app.get("/api/export/status/{job_id}")
async def export_status(job_id: str) -> dict:
    """Poll an export job's progress (state, phase, done/total, log, result)."""
    job = export_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return job


@app.get("/api/export/download/{job_id}")
async def export_download(job_id: str) -> Response:
    """Stream the finished export .zip (browser save dialog picks the location)."""
    zp = export_jobs.zip_path(job_id)
    if not zp or not Path(zp).exists():
        raise HTTPException(404, "Export zip not ready")
    return FileResponse(zp, media_type="application/zip", filename=Path(zp).name)


@app.post("/api/masks/to-tiled")
async def masks_to_tiled(payload: ExportRequest) -> dict:
    """Write rasterized masks into Tiled as stacked volumes (standalone action).

    Reuses the export payload (classes/slices/negative_slices per source) but,
    instead of building a training zip, rasterizes each Tiled source's shapes and
    writes a ``<source_stem>__masks`` sibling container. Runs on a background
    thread; poll ``/api/export/status/{job_id}`` for progress.
    """
    import tiled_mask_sync

    if payload.sources:
        source_items: list[ExportSourceItem] = payload.sources
    else:
        source_items = [ExportSourceItem(
            kind=payload.kind,  # type: ignore[arg-type]
            source=payload.source,
            server_uri=payload.server_uri,
            slices=payload.slices,
            split_by_slice=payload.split_by_slice,
            negative_slices=payload.negative_slices,
        )]

    jid = export_jobs.new_job("")
    threading.Thread(
        target=tiled_mask_sync.run_mask_sync_job,
        args=(jid, source_items, payload),
        daemon=True,
    ).start()
    return {"job_id": jid}


@app.get("/api/masks/from-tiled/preview")
async def masks_from_tiled_preview(source: str = Query(...), server_uri: Optional[str] = None) -> dict:
    """Metadata-only check for saved masks on this sample, before offering to
    load them — powers the "no saved masks yet" disabled state in Annotate's
    Load-saved-masks panel without launching a job destined to fail."""
    import tiled_mask_sync

    return await asyncio.to_thread(tiled_mask_sync.read_masks_summary, source, server_uri)


@app.post("/api/masks/from-tiled")
async def masks_from_tiled(payload: MasksFromTiledRequest) -> dict:
    """Vectorize a previously-written ``<stem>__masks`` container back into
    annotation shapes, for the Annotate tab's "Load saved masks" action.

    Runs on a background thread (no ML_LOCK — pure I/O + skimage, not a model
    forward pass); poll ``/api/export/status/{job_id}`` for progress. The
    result is shaped exactly like an inference job's ({classes, slices,
    n_shapes}), so the frontend's existing import machinery is reused as-is.
    """
    import tiled_mask_sync

    summary = await asyncio.to_thread(tiled_mask_sync.read_masks_summary, payload.source, payload.server_uri)
    if not summary.get("available"):
        raise HTTPException(
            404,
            "No saved masks found for this sample — run inference in the Train tab "
            "and use 'Write masks to Tiled' first.",
        )

    jid = export_jobs.new_job("")
    threading.Thread(
        target=tiled_mask_sync.run_masks_readback_job,
        args=(jid, payload),
        daemon=True,
    ).start()
    return {"job_id": jid}


# ---------------------------------------------------------------------------
# Train tab (DINOv3 + LoRA / dlsia TUNet fine-tuning and inference)
#
# torch (and dlsia, for the second model family) are optional dependencies —
# every route here degrades to a clear error instead of a crash when they
# aren't installed. Progress for /start and /infer is polled via the existing
# /api/export/status/{job_id}, same as export and mask-sync jobs.
# ---------------------------------------------------------------------------

@app.get("/api/train/capability")
async def train_capability() -> dict:
    """Report Train-tab readiness: torch/dlsia availability, device, and the
    checkpoints/runs discovered on disk. Never fails — see train_common.capability."""
    import train_common

    return await asyncio.to_thread(train_common.capability)


@app.get("/api/train/runs")
async def train_runs() -> list[dict]:
    """List saved fine-tuned runs (both model families), newest first."""
    import train_common

    return await asyncio.to_thread(train_common.list_runs)


@app.delete("/api/train/runs/{run_id}")
async def train_delete_run(run_id: str) -> dict:
    """Permanently delete a saved fine-tuned run (config, metrics, and weights)."""
    import train_common

    await asyncio.to_thread(train_common.delete_run, run_id)
    return {"ok": True}


@app.post("/api/train/start")
async def train_start(payload: TrainRequest) -> dict:
    """Start a background fine-tuning job. Returns immediately with a job_id
    and the new run_id; poll /api/export/status/{job_id} for progress.

    With ``resume_from_run_id`` set, continues fine-tuning from that run's saved
    weights instead of starting fresh, and always writes a new run.
    """
    import train_common
    import train_jobs

    if not train_common.torch_available():
        raise HTTPException(503, "Training is unavailable: torch is not installed on this server")
    if payload.model.model_family == "dlsia_tunet" and not train_common.dlsia_available():
        raise HTTPException(503, "Training is unavailable: dlsia is not installed on this server")

    # Validate a resume synchronously: an unknown run or an incompatible class
    # list should be an immediate, actionable 404/400 rather than a job that
    # starts, appears to work, and then errors out in the progress log.
    if payload.resume_from_run_id:
        parent_config = await asyncio.to_thread(train_common.load_run_config, payload.resume_from_run_id)
        try:
            train_jobs.check_resume_compatible(parent_config, payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    if train_common.ML_LOCK.locked():
        raise HTTPException(409, "Another training or inference job is already running")

    run_id = train_jobs.new_run_id(payload.model.model_family)
    jid = export_jobs.new_job(f"run:{run_id}")
    threading.Thread(
        target=train_jobs.run_train_job,
        args=(jid, payload, run_id),
        daemon=True,
    ).start()
    return {"job_id": jid, "run_id": run_id}


@app.post("/api/train/estimate-batch")
async def train_estimate_batch(payload: BatchProbeRequest) -> dict:
    """Measure the largest batch size that fits, for this exact model config.

    Runs real forward+backward steps at increasing batch sizes in the background —
    see :mod:`batch_probe`. Needs no sources: the probe feeds synthetic tensors of
    the shape training would. Poll ``/api/export/status/{job_id}``; the result
    carries ``suggested_batch_size``.
    """
    import batch_probe
    import train_common

    if not train_common.torch_available():
        raise HTTPException(503, "Estimating is unavailable: torch is not installed on this server")
    if payload.model.model_family == "dlsia_tunet" and not train_common.dlsia_available():
        raise HTTPException(503, "Estimating is unavailable: dlsia is not installed on this server")
    if train_common.ML_LOCK.locked():
        raise HTTPException(409, "Another training or inference job is already running")

    jid = export_jobs.new_job("batch-size probe")
    threading.Thread(
        target=batch_probe.run_probe_job, args=(jid, payload), daemon=True
    ).start()
    return {"job_id": jid}


@app.post("/api/train/cancel/{job_id}")
async def train_cancel(job_id: str) -> dict:
    """Cooperatively request cancellation of a running train/infer job.

    The worker checks this flag between batches/slices — cancellation is not
    immediate, and a partial run/result may still be saved.
    """
    if export_jobs.get_job(job_id) is None:
        raise HTTPException(404, "Unknown job_id")
    export_jobs.update(job_id, cancel_requested=True)
    return {"ok": True}


@app.post("/api/train/infer")
async def train_infer(payload: InferRequest) -> dict:
    """Start a background inference job for a saved run over requested slices."""
    import infer_jobs
    import train_common

    if not train_common.torch_available():
        raise HTTPException(503, "Inference is unavailable: torch is not installed on this server")
    # Fail fast on an unknown run before spawning a thread (load_run_config raises 404/500).
    await asyncio.to_thread(train_common.load_run_config, payload.run_id)
    if train_common.ML_LOCK.locked():
        raise HTTPException(409, "Another training or inference job is already running")

    jid = export_jobs.new_job(f"infer:{payload.run_id}")
    threading.Thread(
        target=infer_jobs.run_infer_job,
        args=(jid, payload),
        daemon=True,
    ).start()
    return {"job_id": jid}


@app.get("/api/train/infer/preview/{job_id}/{slice_index}")
async def train_infer_preview(job_id: str, slice_index: int) -> Response:
    """Colourised RGBA overlay PNG for one predicted slice of an inference job."""
    import infer_jobs

    try:
        png = await asyncio.to_thread(infer_jobs.preview_png, job_id, slice_index)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("infer preview failed for %s/%s: %s", job_id, slice_index, exc)
        raise HTTPException(500, "Failed to render prediction preview") from exc
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, no-store"})


@app.post("/api/train/infer/write-tiled/{job_id}")
async def train_infer_write_tiled(job_id: str) -> dict:
    """Push a completed inference job's predicted masks into Tiled (background job)."""
    import infer_jobs

    source_job = export_jobs.get_job(job_id)
    if source_job is None:
        raise HTTPException(404, "Unknown job_id")

    new_jid = export_jobs.new_job("")
    threading.Thread(
        target=infer_jobs.run_write_tiled_job,
        args=(new_jid, job_id),
        daemon=True,
    ).start()
    return {"job_id": new_jid}


@app.post("/api/import/coco")
async def import_coco(dataset_dir: str = Query(...)) -> dict:
    """Import a COCO dataset directory back into editor payload."""
    safe_dataset_dir = _resolve_export_dataset(dataset_dir)

    def _run() -> dict:
        from coco_import import import_dataset

        return import_dataset(str(safe_dataset_dir))

    try:
        return await asyncio.to_thread(_run)
    except FileNotFoundError as exc:
        raise HTTPException(404, "COCO dataset was not found") from exc
    except Exception as exc:
        logger.error("Import failed for %s: %s", dataset_dir, exc)
        raise HTTPException(500, "COCO import failed") from exc


@app.post("/api/ingest/preflight")
async def ingest_preflight(req: IngestPreflightRequest) -> dict:
    """Report which of ``req.names`` already exist in ``req.container_path``.

    Called before the upload so the user can resolve collisions (replace / skip /
    new dataset / browse the existing one) instead of uploading files that would
    each fail with a 409. POST (not GET) because a dropped folder easily carries
    hundreds of filenames — see :class:`schemas.IngestPreflightRequest`.

    Raises:
        HTTPException: 502 if the Tiled server could not be read.
    """
    try:
        return await asyncio.to_thread(
            ingest_mod.preflight, req.server_uri, req.container_path, req.names, req.grouping
        )
    except Exception as exc:
        logger.warning("ingest preflight failed for %s: %s", req.container_path, exc)
        raise HTTPException(502, "Could not check the destination on the Tiled server") from exc


# Multipart upload quotas, enforced before any bytes are decoded. A single
# dropped folder is normally a few hundred files; these defaults give generous
# headroom while still bounding disk/thread usage for an unauthenticated-by-
# default local API. Kept as module globals (not nested in ingest.py) so tests
# can monkeypatch them directly against this route.
MAX_INGEST_FILES = int(os.getenv("MAX_INGEST_FILES", "5000"))
MAX_INGEST_FILE_BYTES = int(os.getenv("MAX_INGEST_FILE_BYTES", str(512 * 1024 * 1024)))
MAX_INGEST_TOTAL_BYTES = int(os.getenv("MAX_INGEST_TOTAL_BYTES", str(16 * 1024 * 1024 * 1024)))


# Non-file form fields on the upload route (container_path, description,
# on_conflict, grouping). max_fields counts these alongside files, so budgeting
# the file quota alone would trip a batch of exactly MAX_INGEST_FILES.
_INGEST_FORM_FIELDS = 16


class _LargeUploadRoute(APIRoute):
    """Route class that parses multipart bodies with OUR file-count quota.

    Starlette's ``Request.form()`` defaults to ``max_files=1000`` and rejects
    anything larger with "Too many files. Maximum number of files is 1000." —
    so ``MAX_INGEST_FILES`` (5000) was unreachable, and the real ceiling was a
    framework default nobody chose, reported in wording unlike this API's other
    limits.

    This has to be a route class rather than a dependency: FastAPI parses the
    body at ``routing.py``'s ``body = await request.form()`` BEFORE it calls
    ``solve_dependencies``, so a dependency always loses the race (verified —
    the first attempt at this fix was a dependency and it changed nothing).
    Pre-parsing here, before delegating to the normal handler, wins because
    ``Request._get_form`` caches into ``request._form`` and returns that cache
    on FastAPI's own later call.

    Deliberately does NOT reject oversized batches itself: ``ingest_upload``
    already does, with a clearer message and after closing the uploads it
    accepted.
    """

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            if request.headers.get("content-type", "").startswith("multipart/form-data"):
                headroom = MAX_INGEST_FILES + 1  # +1 so OUR check reports the overflow
                await request.form(
                    max_files=headroom,
                    max_fields=headroom + _INGEST_FORM_FIELDS,
                )
            return await original(request)

        return handler


# A one-route router purely to attach the route class above — `route_class` is
# an APIRouter option, and `@app.post()` has no equivalent. Included at the
# bottom of this module.
_upload_router = APIRouter(route_class=_LargeUploadRoute)


@_upload_router.post("/api/ingest/upload")
async def ingest_upload(
    server_uri: Optional[str] = Query(None, description="Target Tiled server URI"),
    container_path: str = Form(..., description="Target container, e.g. 'browse/myset'"),
    description: str = Form("", description="Optional keyword(s) stored on every ingested node"),
    on_conflict: str = Form("fail", description="'fail', 'replace' or 'skip' for existing keys"),
    grouping: str = Form(
        ingest_mod.DEFAULT_GROUPING,
        description="How images map onto samples: 'prefix' (split on '__'), 'per_image' or 'single'",
    ),
    files: list[UploadFile] = File(..., description="Image files to copy into Tiled"),
) -> dict:
    """Stream uploaded files to temp storage and start a background ingest job.

    Images are grouped into samples per *grouping* beneath *container_path* on the
    connected Tiled server. Returns a ``job_id`` to poll for progress.
    """
    if on_conflict not in ingest_mod.ON_CONFLICT_MODES:
        raise HTTPException(
            400, f"on_conflict must be one of {sorted(ingest_mod.ON_CONFLICT_MODES)}"
        )
    try:
        grouping = ingest_mod.validate_grouping(grouping)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if len(files) > MAX_INGEST_FILES:
        for upload in files:
            await upload.close()
        raise HTTPException(413, f"Too many files in one upload (max {MAX_INGEST_FILES})")

    tmp_dir = Path(tempfile.mkdtemp(prefix="ingest_"))
    saved: list[tuple[str, Path]] = []
    batch_bytes = 0
    try:
        for index, upload in enumerate(files):
            ext = Path(upload.filename or "").suffix.lower()
            if ext not in ingest_mod.IMAGE_EXTS:
                await upload.close()
                continue
            dest = tmp_dir / f"{index:06d}{ext}"
            file_bytes = 0
            # Stream in 1MB chunks — files can be 26MB+, never read() whole into memory.
            # Both per-file and whole-batch byte quotas are enforced while streaming, so
            # a single oversized upload can't exhaust disk before decode-time validation.
            with dest.open("wb") as out:
                while chunk := await upload.read(1024 * 1024):
                    file_bytes += len(chunk)
                    batch_bytes += len(chunk)
                    if file_bytes > MAX_INGEST_FILE_BYTES or batch_bytes > MAX_INGEST_TOTAL_BYTES:
                        await upload.close()
                        raise HTTPException(413, "Upload exceeds the configured size limit")
                    out.write(chunk)
            await upload.close()
            saved.append((upload.filename or dest.name, dest))
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        for upload in files:
            await upload.close()
        raise

    if not saved:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, "No supported image files in upload")

    jid = ingest_mod.new_job(len(saved), server_uri, container_path)

    def _run_and_invalidate() -> None:
        try:
            ingest_mod.run_ingest_job(
                jid, server_uri, container_path, saved, description, on_conflict, grouping
            )
        finally:
            # The ingest may have created/removed samples or changed container
            # metadata (see run_ingest_job) — Browse's cached listings and facet
            # mappings must not keep serving what existed before the upload, the
            # same invalidation save_annotation_version does after its own write.
            _field_mapping_cache.clear()
            _column_cache.clear()
            _items_cache.clear()

    threading.Thread(target=_run_and_invalidate, daemon=True).start()
    return {"job_id": jid, "total": len(saved), "container_path": container_path}


# Registered here rather than via @app.post so the upload gets _LargeUploadRoute
# (see its docstring — Starlette's 1000-file multipart default would otherwise
# reject the batch before ingest_upload's own MAX_INGEST_FILES check runs).
app.include_router(_upload_router)


@app.get("/api/ingest/status/{job_id}")
async def ingest_status(job_id: str) -> dict:
    """Return progress for an ingest job started by ``/api/ingest/upload``."""
    job = ingest_mod.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return job


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_filters(raw: str) -> dict:
    """Parse the ``filters`` query param, rejecting anything that isn't a JSON object.

    Silently falling back to ``{}`` here would mean a malformed/truncated filter
    string quietly widens the result set instead of failing — indistinguishable
    from an intentional "no filter" request. Fail closed with 422 instead.
    """
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "filters must be a JSON object") from exc
    if not isinstance(value, dict):
        raise HTTPException(422, "filters must be a JSON object")
    return value


# ---------------------------------------------------------------------------
# Static SPA (production container only)
# ---------------------------------------------------------------------------
# In the Docker image the built frontend is copied to backend/static/, and
# FastAPI serves it so the app is a single same-origin service. In local dev
# this directory is absent (Vite serves the SPA), so the mount is skipped.
# Registered AFTER all /api routes so the catch-all never shadows them.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> FileResponse:
        """Serve a real static file when it exists, else index.html (SPA routing)."""
        candidate = _STATIC_DIR / full_path
        if full_path and candidate.is_file() and _STATIC_DIR in candidate.resolve().parents:
            return FileResponse(str(candidate))
        return FileResponse(str(_STATIC_DIR / "index.html"))


if __name__ == "__main__":  # pragma: no cover — convenience entry point
    import uvicorn

    uvicorn.run("annotation_server:app", host="127.0.0.1", port=8002)
