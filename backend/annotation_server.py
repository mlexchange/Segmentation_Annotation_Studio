"""FastAPI entry point for the Segmentation Annotation Studio API.

Endpoints
---------
* ``GET /api/config/servers``   — list configured Tiled servers
* ``GET /api/browse/facets``    — metadata fields available for browsing
* ``GET /api/browse/column``    — distinct values + counts for one field
* ``GET /api/browse/items``     — sample records matching a filter set
* ``GET /api/browse/thumbnail`` — PNG thumbnail for a Tiled array path
* ``GET /health``               — liveness check
* ``/api/ipred/*``              — proxy to the standalone iPred service (see ``ipred_routes.py``)

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
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware

import annotation_thumbnails
import arrays as arrays_mod
import batch_probe
import denoise as denoise_mod
import denoise_bake as denoise_bake_mod
import drafts as drafts_mod
import export_jobs
import guides as guides_mod
import images as images_mod
import infer_jobs
import ingest as ingest_mod
import ipred_routes
import local_fs
import tiff_stack_source
import train_common
import train_jobs
import volume_build
import volume_nodes
import zarr_source
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
    MeasureRequest,
    SaveVersionRequest,
    TiffStackRegisterRequest,
    TrainRequest,
    VolumeBuildRequest,
    ZarrRegisterRequest,
)
from source_keys import parse_source_key
from thumbnails import render_thumbnail
from tiled_clients import (
    api_key_for_uri,
    get_browse_container_for,
    get_tiled_client,
)
from tiled_config import get_tiled_api_key, get_tiled_servers

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
)

# Compress text responses (SPA JS/CSS, JSON). Matters for the production path where
# FastAPI serves the built SPA + API from one origin; PNGs are already compressed so
# the ~500-byte floor skips tiny/binary payloads. (Dev uses the Vite server instead.)
app.add_middleware(GZipMiddleware, minimum_size=500)

# iPred (interactive segmentation) proxy — see ipred_routes.py. iPred is an
# optional, separately-run service (port 8003 by default); a down/missing
# service surfaces as 503 from these routes rather than breaking the app.
app.include_router(ipred_routes.router)


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
# Denoised slice PNGs only — the plain render path stays uncached because it is
# already cheap. Entries are a few MB each at full resolution, so 32 covers
# scrubbing a stack back and forth without unbounded growth.
_denoised_slice_cache: TTLCache = TTLCache(ttl_seconds=300.0, max_entries=32)


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
    server_api_key: Optional[str] = None,
    technique: str = Query("GIWAXS"),
    container_path: Optional[str] = Query(None, description="Tiled container to browse"),
    refresh: bool = Query(False),  # noqa: ARG001 — kept for client API compat
) -> dict[str, list[str]]:
    """Return ordered list of browsable metadata fields, discovered live.

    For each key in the field mapping, the field is kept if at least two
    distinct non-null values exist. Live (no cache) so newly-seeded metadata
    appears in the UI without a restart.
    """
    def _discover() -> dict[str, list[str]]:
        client = get_tiled_client(server_uri, server_api_key)
        container, _ = get_browse_container_for(client, container_path)
        mapping = _resolve_field_mapping(container, server_uri or "", technique, container_path or "")

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
    server_api_key: Optional[str] = None,
    technique: str = Query("GIWAXS"),
    field: str = Query(..., description="Display-key metadata field to group by"),
    filters: str = Query("{}", description="JSON dict of upstream display_key=value selections"),
    container_path: Optional[str] = Query(None, description="Tiled container to browse"),
    limit: int = Query(500, ge=1, le=5000),
    refresh: bool = Query(False),
) -> dict:
    """Return distinct values (+ counts) for *field* via Tiled ``distinct()``."""
    filter_dict = _parse_json_filters(filters)

    cache_key = ("column", server_uri or "", technique, container_path or "", field, filters, limit)
    if not refresh:
        cached = _column_cache.get(cache_key)
        if cached is not None:
            return cached

    def _build() -> dict:
        client = get_tiled_client(server_uri, server_api_key)
        container, _ = get_browse_container_for(client, container_path)
        mapping = _resolve_field_mapping(container, server_uri or "", technique, container_path or "")
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
        raise HTTPException(status_code=502, detail=f"Failed to browse column: {exc}") from exc

    _column_cache.set(cache_key, result)
    return result


@app.get("/api/browse/items")
async def browse_items(
    server_uri: Optional[str] = None,
    server_api_key: Optional[str] = None,
    technique: str = Query("GIWAXS"),
    filters: str = Query("{}", description="JSON dict of display_key=value selections"),
    container_path: Optional[str] = Query(None, description="Tiled container to browse"),
    limit: int = Query(500, ge=1, le=2000),
    refresh: bool = Query(False),
) -> dict:
    """Return sample records (path + metadata) matching *filters* via ``search()``."""
    filter_dict = _parse_json_filters(filters)

    cache_key = ("items", server_uri or "", technique, container_path or "", filters, limit)
    if not refresh:
        cached = _items_cache.get(cache_key)
        if cached is not None:
            return cached

    def _build() -> dict:
        client = get_tiled_client(server_uri, server_api_key)
        container, prefix = get_browse_container_for(client, container_path)
        mapping = _resolve_field_mapping(container, server_uri or "", technique, container_path or "")
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
        raise HTTPException(status_code=502, detail=f"Failed to browse items: {exc}") from exc

    _items_cache.set(cache_key, result)
    return result


@app.get("/api/browse/slices")
async def browse_slices(
    path: str = Query(..., description="Tiled container path of a multi-slice dataset"),
    server_uri: Optional[str] = None,
    server_api_key: Optional[str] = None,
    limit: int = Query(2000, ge=1, le=10000),
) -> dict:
    """List a dataset container's array children as individually-openable slices.

    Used by Browse drill-in: each returned record is ``{path, sample, metadata}``
    where ``path`` points at a single array node that opens as a 2-D image.
    """
    cache_key = ("slices", server_uri or "", path, limit)
    cached = _items_cache.get(cache_key)
    if cached is not None:
        return cached

    def _build() -> dict:
        client = get_tiled_client(server_uri, server_api_key)
        container, prefix = get_browse_container_for(client, path)
        result = tiled_search_items(container, limit=limit, container_path_prefix=prefix)
        # Order slices by key (ingest zero-pads, so lexical == slice order).
        result["items"].sort(key=lambda it: it["sample"])
        return result

    try:
        result = await asyncio.to_thread(_build)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to list slices: {exc}") from exc

    _items_cache.set(cache_key, result)
    return result


@app.get("/api/browse/thumbnail")
async def browse_thumbnail(
    tiled_path: str = Query(..., description="Slash-separated Tiled path"),
    server_uri: Optional[str] = None,
    size: int = Query(256, ge=32, le=512),
) -> Response:
    """Return a PNG thumbnail for the array at *tiled_path*."""
    def _build() -> bytes | None:
        api_key = api_key_for_uri(server_uri) or get_tiled_api_key()  # server-side only
        client = get_tiled_client(server_uri, api_key)
        node = client
        for part in tiled_path.strip("/").split("/"):
            node = node[part]
        return render_thumbnail(node, size=size)

    try:
        png_bytes = await asyncio.to_thread(_build)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Tiled path not found: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load array from Tiled: {exc}") from exc

    if png_bytes is None:
        raise HTTPException(status_code=404, detail="No array data found at the given path")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/api/local/list")
async def local_list(
    rel: str = Query("", description="Relative path under the granted root"),
    root: Optional[str] = Query(None, description="Granted absolute browse root"),
) -> list[dict]:
    """List directory entries under the granted local root."""
    return await asyncio.to_thread(local_fs.list_dir, rel, root)


@app.get("/api/local/samples")
async def local_samples(
    rel: str = Query(..., description="Relative path to a folder under the granted root"),
    root: Optional[str] = Query(None, description="Granted absolute browse root"),
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
    root: Optional[str] = Query(None, description="Granted absolute browse root (kind=local)"),
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
        def _count() -> int:
            client = get_tiled_client(server_uri)
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
             if (cfg.get("uri") or "").rstrip("/") == (server_uri or "").rstrip("/")),
            server_uri or "Tiled Server",
        )
        if container_path:
            label = f"{label} · {container_path}"
        return {
            "kind": "tiled",
            "label": label,
            "sample_count": count,
            "server_uri": server_uri,
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
    def _run() -> list[dict]:
        client = get_tiled_client(server_uri)
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
        raise HTTPException(500, f"Failed to list Tiled path: {exc}") from exc


@app.get("/api/image/meta", response_model=ImageMeta)
async def image_meta(
    source: str = Query(...),
    kind: str = Query(...),
    server_uri: Optional[str] = None,
    root: Optional[str] = Query(None, description="Granted absolute root (kind=local)"),
) -> ImageMeta:
    """Return shape / dtype metadata for an image source."""
    def _run() -> ImageMeta:
        node = arrays_mod.resolve_array(source, kind, server_uri, root)
        pyramid = arrays_mod.pyramid_info(source, kind, server_uri, root)
        meta = arrays_mod.array_shape_meta(node, pyramid)
        sl = arrays_mod.read_slice(node, meta, 0)
        flat = sl.ravel().astype(float)
        global_range = images_mod._sample_global_stats(node, meta) if not meta["is_rgb"] else None
        return ImageMeta(
            n_slices=meta["n_slices"],
            height=meta["height"],
            width=meta["width"],
            dtype=meta["dtype"],
            is_rgb=meta["is_rgb"],
            value_range=[float(flat.min()), float(flat.max())],
            global_value_range=list(global_range) if global_range is not None else None,
            keywords=arrays_mod.node_keywords(node),
            level_key=meta.get("level_key"),
            level_index=meta.get("level_index"),
            level_count=meta.get("level_count"),
            level_height=meta.get("level_height"),
            level_width=meta.get("level_width"),
            level_n_slices=meta.get("level_n_slices"),
            z_downsample=meta.get("z_downsample"),
        )

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("image_meta failed: %s", exc)
        raise HTTPException(500, f"Failed to read image meta: {exc}") from exc


@app.get("/api/image/slice")
async def image_slice(
    source: str = Query(...),
    kind: str = Query(...),
    slice_index: int = Query(0),
    server_uri: Optional[str] = None,
    root: Optional[str] = Query(None, description="Granted absolute root (kind=local)"),
    norm: str = Query("global"),
    scale: str = Query("linear"),
    vmin_pct: float = Query(1.0),
    vmax_pct: float = Query(99.0),
    cmap: str = Query("gray"),
    denoise_method: str = Query("none", description="Classical denoise filter (see denoise.ALL_METHODS)"),
    denoise_strength: float = Query(0.5, ge=0.0, le=1.0),
    denoise_crop: int = Query(
        0,
        ge=0,
        description="If >0, denoise and return only a centred square crop of this size at 1:1. "
                    "For tuning: filtering a full slice costs seconds for NLM/TV.",
    ),
) -> Response:
    """Render one slice of an image source as a PNG."""
    opts = {
        "norm": norm,
        "scale": scale,
        "vmin_pct": vmin_pct,
        "vmax_pct": vmax_pct,
        "cmap": cmap,
    }
    if denoise_method not in denoise_mod.ALL_METHODS:
        raise HTTPException(400, f"Unknown denoise method {denoise_method!r}")
    denoising = denoise_method != "none"

    # Denoised renders are cached; the plain path stays uncached because it is
    # already cheap. Without this, every slider tweak or revisit re-pays the full
    # filter cost — seconds, not milliseconds, for NLM and TV.
    cache_key = (
        source, kind, slice_index, server_uri, root, norm, scale, vmin_pct, vmax_pct, cmap,
        denoise_method, round(denoise_strength, 4), denoise_crop,
    )
    if denoising:
        cached = _denoised_slice_cache.get(cache_key)
        if cached is not None:
            return Response(content=cached, media_type="image/png")

    def _run() -> bytes:
        node = arrays_mod.resolve_array(source, kind, server_uri, root)
        pyramid = arrays_mod.pyramid_info(source, kind, server_uri, root)
        meta = arrays_mod.array_shape_meta(node, pyramid)
        # Denoise the RAW slice, before normalization: noise statistics live in
        # the source's own intensity units, not in the 8-bit display range.
        if denoising:
            sl = _denoised_slice(node, meta, slice_index, denoise_method, denoise_strength, denoise_crop)
        else:
            sl = arrays_mod.read_slice(node, meta, slice_index)
        global_range = None
        if norm == "global":
            # NB: deriving this from the pyramid's coarsest level was tried and
            # reverted. It is ~8x faster, but those levels are built by AVERAGING,
            # which pulls the extremes in hard — on the reference volume the range
            # came back (-20.7, 18.0) against (-73.0, 71.3) at full resolution.
            # Since this range IS the contrast window, the cheap version visibly
            # clips the image. The full-resolution sampler decimates instead of
            # averaging, so it keeps the extremes; it costs ~3s once per volume
            # and is then cached.
            global_range = images_mod._sample_global_stats(node, meta)
        rgb = images_mod.render_slice(sl, opts, global_range)
        return images_mod.encode_png(rgb)

    try:
        png = await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("image_slice failed: %s", exc)
        raise HTTPException(500, f"Failed to render slice: {exc}") from exc

    if denoising:
        _denoised_slice_cache.set(cache_key, png)

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=300"},
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
        headers={"Cache-Control": "public, max-age=3600"},
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
    import os
    from datetime import datetime, timezone

    export_root_env = os.getenv("EXPORT_ROOT", "")
    if export_root_env:
        export_root = Path(export_root_env).expanduser().resolve()
    else:
        local_root = Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve()
        export_root = local_root / "exports"

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
    if not str(out_root).startswith(str(export_root)):
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


@app.post("/api/export/cancel/{job_id}")
async def export_cancel(job_id: str) -> dict:
    """Ask a running job to stop at its next clean boundary.

    Cooperative rather than immediate: a job that stops mid-write would leave a
    partial dataset that looks complete. Jobs that honour it discard their
    partial output; those that do not simply run to completion.
    """
    if not export_jobs.request_cancel(job_id):
        raise HTTPException(404, "Unknown job_id")
    return {"cancelled": True}


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


@app.post("/api/import/coco")
async def import_coco(dataset_dir: str = Query(...)) -> dict:
    """Import a COCO dataset directory back into editor payload."""
    def _run() -> dict:
        from coco_import import import_dataset
        return import_dataset(dataset_dir)

    try:
        return await asyncio.to_thread(_run)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.error("Import failed for %s: %s", dataset_dir, exc)
        raise HTTPException(500, f"Import failed: {exc}") from exc


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
            ingest_mod.preflight, req.server_uri, req.container_path, req.names
        )
    except Exception as exc:
        logger.warning("ingest preflight failed for %s: %s", req.container_path, exc)
        raise HTTPException(502, "Could not check the destination on the Tiled server") from exc


@app.post("/api/ingest/upload")
async def ingest_upload(
    server_uri: Optional[str] = Query(None, description="Target Tiled server URI"),
    container_path: str = Form(..., description="Target container, e.g. 'browse/myset'"),
    description: str = Form("", description="Optional keyword(s) stored on every ingested node"),
    on_conflict: str = Form("fail", description="'fail', 'replace' or 'skip' for existing keys"),
    files: list[UploadFile] = File(..., description="Image files to copy into Tiled"),
) -> dict:
    """Stream uploaded files to temp storage and start a background ingest job.

    Each supported image becomes its own browsable node in *container_path* on
    the connected Tiled server. Returns a ``job_id`` to poll for progress.
    """
    if on_conflict not in ingest_mod.ON_CONFLICT_MODES:
        raise HTTPException(
            400, f"on_conflict must be one of {sorted(ingest_mod.ON_CONFLICT_MODES)}"
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="ingest_"))
    saved: list[tuple[str, Path]] = []
    for index, upload in enumerate(files):
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in ingest_mod.IMAGE_EXTS:
            await upload.close()
            continue
        dest = tmp_dir / f"{index:06d}{ext}"
        # Stream in 1MB chunks — files can be 26MB+, never read() whole into memory.
        with dest.open("wb") as out:
            while chunk := await upload.read(1024 * 1024):
                out.write(chunk)
        await upload.close()
        saved.append((upload.filename or dest.name, dest))

    if not saved:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, "No supported image files in upload")

    jid = ingest_mod.new_job(len(saved), server_uri, container_path)
    threading.Thread(
        target=ingest_mod.run_ingest_job,
        args=(jid, server_uri, container_path, saved, description, on_conflict),
        daemon=True,
    ).start()
    return {"job_id": jid, "total": len(saved), "container_path": container_path}


@app.get("/api/ingest/status/{job_id}")
async def ingest_status(job_id: str) -> dict:
    """Return progress for an ingest job started by ``/api/ingest/upload``."""
    job = ingest_mod.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return job


@app.get("/api/zarr/inspect")
async def zarr_inspect(
    path: str = Query(..., description="Absolute path to a .zarr directory on the server"),
) -> dict:
    """Describe a Zarr store's resolution pyramid without registering it.

    Lets the Connect page show what was found — dimensions, dtype, voxel size and
    the available levels — before the user commits to loading it.

    Raises:
        HTTPException: 4xx from :func:`zarr_source.inspect_zarr` with a
            user-facing message (missing path, zipped archive, empty group…).
    """
    return await asyncio.to_thread(zarr_source.inspect_zarr, path)


@app.post("/api/zarr/preflight")
async def zarr_preflight(req: ZarrRegisterRequest) -> dict:
    """Report whether loading this Zarr would collide with an existing node.

    Distinguishes a previous Zarr registration (safe to replace — only catalog
    rows are dropped) from internally-managed data such as an uploaded image
    stack, where replacing would delete the files themselves.
    """
    return await asyncio.to_thread(
        zarr_source.preflight_zarr, req.server_uri, req.path, req.container_path
    )


@app.post("/api/zarr/register")
async def zarr_register(req: ZarrRegisterRequest) -> dict:
    """Register an on-disk Zarr volume with Tiled, copying no data.

    Unlike ``/api/ingest/upload`` this needs no background job: registration
    writes catalog rows, not pixels, so it returns in well under a second even
    for a 56 GB store.
    """
    if req.on_conflict not in ingest_mod.ON_CONFLICT_MODES:
        raise HTTPException(
            400, f"on_conflict must be one of {sorted(ingest_mod.ON_CONFLICT_MODES)}"
        )
    return await asyncio.to_thread(
        zarr_source.register_zarr,
        req.server_uri,
        req.path,
        req.container_path,
        req.description,
        req.on_conflict,
    )


def _centre_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Centred square crop of *size*, or *arr* unchanged if it already fits."""
    h, w = arr.shape[:2]
    if size <= 0 or (h <= size and w <= size):
        return arr
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return arr[top: top + min(size, h), left: left + min(size, w)]


def _denoised_slice(
    node: Any,
    meta: dict,
    slice_index: int,
    method: str,
    strength: float,
    crop: int = 0,
) -> np.ndarray:
    """Read *slice_index* and denoise it, pulling z-neighbours for 3-D methods.

    The 3-D filters are the training-free way to exploit slice-to-slice
    correlation — adjacent tomographic slices share structure while their noise
    is independent — which is why this reads a window rather than one slice. The
    window is clamped to the volume, and the target's position inside the stack
    is tracked explicitly: it is NOT always the centre, since the window is
    truncated at the first and last slice.

    When *crop* > 0 the crop is taken BEFORE filtering — filtering a full 6.5 MP
    slice is what's slow. Worth knowing: results near the crop border, and NLM's
    patch search in particular, differ slightly from the full-slice result, so a
    crop is a tuning aid rather than a byte-exact preview of the bake.
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
        # Not enough usable z-context (single-slice source, or unreadable
        # neighbours) — fall back to the 2-D sibling rather than erroring out.
        fallback = "gaussian" if method == "gaussian3d" else "median"
        sl = _centre_crop(np.asarray(arrays_mod.read_slice(node, meta, slice_index)), crop)
        return denoise_mod.denoise_slice(sl, fallback, strength)

    target_pos = min(slice_index - lo, len(frames) - 1)
    return denoise_mod.denoise_stack(np.stack(frames, axis=0), method, strength)[target_pos]


@app.get("/api/denoise/methods")
async def denoise_methods() -> dict:
    """Denoise filters available in this environment, with cost hints.

    ``available`` is probed rather than assumed: ``denoise_wavelet`` imports
    fine without PyWavelets and only fails when called.
    """
    return {"methods": denoise_mod.describe_methods()}


@app.get("/api/denoise/auto")
async def denoise_auto(
    source: str = Query(...),
    kind: str = Query(...),
    slice_index: int = Query(0),
    method: str = Query("tv"),
    server_uri: Optional[str] = None,
    root: Optional[str] = Query(None),
) -> dict:
    """Suggest a strength for this slice, from its own measured noise level."""
    if method not in denoise_mod.ALL_METHODS:
        raise HTTPException(400, f"Unknown denoise method {method!r}")

    def _run() -> dict:
        node = arrays_mod.resolve_array(source, kind, server_uri, root)
        pyramid = arrays_mod.pyramid_info(source, kind, server_uri, root)
        meta = arrays_mod.array_shape_meta(node, pyramid)
        sl = np.asarray(arrays_mod.read_slice(node, meta, slice_index))
        unit, _, span = denoise_mod._to_unit(sl)
        return {
            "strength": denoise_mod.auto_strength(sl, method),
            # Noise as a fraction of the slice's own dynamic range, so the UI can
            # say how noisy this is rather than only what to do about it.
            "noise_sigma": denoise_mod.estimate_noise_sigma(unit) if span > 0 else 0.0,
        }

    return await asyncio.to_thread(_run)


@app.post("/api/denoise/bake")
async def denoise_bake(payload: DenoiseBakeRequest) -> dict:
    """Denoise a whole volume and save it as a new, annotatable Tiled dataset.

    The Annotate preview is display-only; this is how a denoised volume becomes
    real data you can annotate and export. Runs on a background thread; poll
    ``GET /api/export/status/{job_id}``.
    """
    if payload.method == "none":
        raise HTTPException(422, "Pick a denoise method before saving a denoised copy.")
    if payload.method not in denoise_mod.ALL_METHODS:
        raise HTTPException(422, f"Unknown denoise method: {payload.method}")
    if payload.method not in denoise_mod.available_methods():
        raise HTTPException(
            422,
            f"Denoise method {payload.method!r} is unavailable on this server "
            "(missing optional dependency).",
        )

    target = payload.target_path or denoise_bake_mod.default_target_path(payload.source)
    try:
        ingest_mod.validate_container_path(target)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    jid = export_jobs.new_job(target)
    threading.Thread(
        target=denoise_bake_mod.run_denoise_bake_job, args=(jid, payload), daemon=True
    ).start()
    return {"job_id": jid, "target_path": target}


@app.get("/api/train/capability")
async def train_capability() -> dict:
    """Best-effort snapshot of Train-tab readiness (torch/dlsia/tiling/device)."""
    return train_common.capability()


@app.get("/api/train/runs")
async def train_list_runs() -> dict:
    """List saved fine-tune runs (both dlsia_tunet and dlsia_denoiser), newest first."""
    return {"runs": train_common.list_runs()}


@app.delete("/api/train/runs/{run_id}")
async def train_delete_run(run_id: str) -> dict:
    """Permanently remove a saved run's directory (config, metrics, weights)."""
    train_common.delete_run(run_id)
    return {"deleted": run_id}


@app.post("/api/train/start")
async def train_start(payload: TrainRequest) -> dict:
    """Start a fine-tuning job. Runs on a background thread; poll
    ``GET /api/export/status/{job_id}``; cancel via
    ``POST /api/export/cancel/{job_id}`` (same shared registry every
    background job in this app already uses).
    """
    if not train_common.torch_available():
        raise HTTPException(503, "torch is not installed on this server — see the ml extra in pyproject.toml")
    needs_dlsia = payload.model.model_family == "dlsia_tunet" or (
        payload.model.model_family == "dlsia_denoiser" and payload.model.architecture == "tunet"
    )
    if needs_dlsia and not train_common.dlsia_available():
        raise HTTPException(503, "dlsia is not installed on this server")
    if payload.model.hyperparams.tiling:
        import tiling

        if not tiling.qlty_available():
            raise HTTPException(503, "Tiling requires the 'qlty' package, which is not installed on this server")

    run_id = train_jobs.new_run_id(payload.model.model_family)
    if payload.resume_from_run_id:
        try:
            parent_config = train_common.load_run_config(payload.resume_from_run_id)
            train_jobs.check_resume_compatible(parent_config, payload)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    if not train_common.ML_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another training or inference job is already running")
    train_common.ML_LOCK.release()  # run_train_job re-acquires it itself; this was just a pre-check

    jid = export_jobs.new_job(run_id)
    threading.Thread(target=train_jobs.run_train_job, args=(jid, payload, run_id), daemon=True).start()
    return {"job_id": jid, "run_id": run_id}


@app.post("/api/train/estimate-batch")
async def train_estimate_batch(payload: BatchProbeRequest) -> dict:
    """Measure the largest batch size that fits in device memory for *payload.model*.

    Runs on a background thread; poll ``GET /api/export/status/{job_id}``.
    """
    if not train_common.torch_available():
        raise HTTPException(503, "torch is not installed on this server")
    jid = export_jobs.new_job(f"batch-probe:{payload.model.model_family}")
    threading.Thread(target=batch_probe.run_probe_job, args=(jid, payload), daemon=True).start()
    return {"job_id": jid}


@app.post("/api/train/infer")
async def train_infer(payload: InferRequest) -> dict:
    """Run a saved fine-tuned run over the requested slices. Runs on a
    background thread; poll ``GET /api/export/status/{job_id}``.
    """
    if not train_common.torch_available():
        raise HTTPException(503, "torch is not installed on this server")
    try:
        train_common.load_run_config(payload.run_id)
    except HTTPException:
        raise
    jid = export_jobs.new_job(payload.run_id)
    threading.Thread(target=infer_jobs.run_infer_job, args=(jid, payload), daemon=True).start()
    return {"job_id": jid}


@app.get("/api/train/infer/preview/{job_id}/{slice_index}")
async def train_infer_preview(job_id: str, slice_index: int) -> Response:
    """Colourised RGBA overlay PNG for one predicted slice of a cached inference job."""
    return Response(content=infer_jobs.preview_png(job_id, slice_index), media_type="image/png")


@app.post("/api/train/infer/write-tiled/{job_id}")
async def train_infer_write_tiled(job_id: str) -> dict:
    """Push a completed inference job's label maps into Tiled. Runs on a
    background thread; poll ``GET /api/export/status/{job_id}`` with the
    RETURNED job id (distinct from *job_id*, the inference job being written).
    """
    write_jid = export_jobs.new_job(f"write-tiled:{job_id}")
    threading.Thread(target=infer_jobs.run_write_tiled_job, args=(write_jid, job_id), daemon=True).start()
    return {"job_id": write_jid}


@app.get("/api/volume/resolve")
async def volume_resolve(
    source: str = Query(..., description="Tiled path of the open dataset"),
    server_uri: Optional[str] = Query(None),
) -> dict:
    """Locate the renderable 3-D volume for the open dataset.

    Which node holds it depends on how the dataset was catalogued — a registered
    Zarr volume is one already, a TIFF stack's lives in its ``__volume`` sidecar,
    and a stack nobody has built one for has none. The frontend cannot tell these
    apart from the path, and guessing produces
    ``missing multiscales in root .zattrs`` at the viewer instead of an answer.
    """
    return await asyncio.to_thread(volume_nodes.resolve_volume, server_uri, source)


@app.get("/api/volume/build/inspect")
async def volume_build_inspect(
    source: str = Query(..., description="Tiled path of the per-slice dataset"),
    kind: str = Query("tiled"),
    server_uri: Optional[str] = Query(None),
) -> dict:
    """Describe the 3-D volume that would be built for this dataset."""
    return await asyncio.to_thread(
        volume_build.inspect_volume_build, source, kind, server_uri
    )


@app.post("/api/volume/build")
async def volume_build_start(req: VolumeBuildRequest) -> dict:
    """Build a 3-D volume from a slice stack already in the catalog.

    Needs nothing but the open dataset: the slices are already in Tiled, so
    asking for a source directory would be asking the user to re-supply data the
    app has. Returns a ``job_id``; poll ``GET /api/export/status/{job_id}``.
    """
    info = await asyncio.to_thread(
        volume_build.inspect_volume_build, req.source, req.kind, req.server_uri
    )
    jid = export_jobs.new_job(req.source)
    export_jobs.set_total(jid, max(info["slices_to_read"], 1))

    def _run() -> None:
        try:
            export_jobs.update(jid, state="running", phase="building")

            def _progress(message: str, done: int, total: int) -> None:
                export_jobs.update(jid, phase=message, done=done, total=max(total, 1))

            result = volume_build.build_volume(
                req.source, req.kind, req.server_uri, req.container_path, progress=_progress
            )
            export_jobs.update(jid, state="done", phase="done", result=result)
            export_jobs.log(jid, f"Built 3-D volume {result['key']!r}.")
        except HTTPException as exc:
            export_jobs.update(jid, state="error", phase="error", error=str(exc.detail))
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI via the job
            logger.exception("volume build failed")
            export_jobs.update(jid, state="error", phase="error", error=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": jid, **info}


@app.get("/api/tiff-stack/inspect")
async def tiff_stack_inspect(
    path: str = Query(..., description="Absolute path to a directory of TIFF slices"),
) -> dict:
    """Describe a TIFF directory and the pyramid that would be built for it.

    Reads only the first file, so this is cheap enough to call while the user is
    still typing a path. ``slices_to_read`` lets the UI say up front how much
    work registration will be, rather than appearing to hang.

    Raises:
        HTTPException: 4xx from :func:`tiff_stack_source.inspect_tiff_stack` with
            a user-facing message (missing path, no TIFFs, inconsistent
            numbering…).
    """
    return await asyncio.to_thread(tiff_stack_source.inspect_tiff_stack, path)


@app.post("/api/tiff-stack/preflight")
async def tiff_stack_preflight(req: TiffStackRegisterRequest) -> dict:
    """Report whether registering this TIFF stack would collide, changing nothing."""
    return await asyncio.to_thread(
        tiff_stack_source.preflight_tiff_stack, req.server_uri, req.path, req.container_path
    )


@app.post("/api/tiff-stack/register")
async def tiff_stack_register(req: TiffStackRegisterRequest) -> dict:
    """Register a TIFF directory as a 3-D multiscale volume, copying no slices.

    Returns a ``job_id`` immediately; poll ``GET /api/export/status/{job_id}``.
    A job rather than a straight call because — unlike Zarr registration, which
    only writes catalog rows — the downsampled levels the 3-D viewer renders have
    to be computed, and that means reading every source slice once.

    The full-resolution slices themselves are registered in place: no pixels are
    copied, and the existing per-slice nodes the 2-D canvas reads are untouched.
    """
    if req.on_conflict not in ingest_mod.ON_CONFLICT_MODES:
        raise HTTPException(
            400, f"on_conflict must be one of {sorted(ingest_mod.ON_CONFLICT_MODES)}"
        )
    # Validate before returning a job id, so a bad path is a 4xx the user sees
    # immediately rather than a job that fails a second later.
    info = await asyncio.to_thread(tiff_stack_source.inspect_tiff_stack, req.path)

    jid = export_jobs.new_job(req.path)
    # Every source slice is read exactly once: the finest generated level comes
    # from the TIFFs, the coarser ones cascade from it in memory.
    export_jobs.set_total(jid, max(info["slices_to_read"], 1))

    def _run() -> None:
        try:
            export_jobs.update(jid, state="running", phase="registering")

            def _progress(message: str, done: int, total: int) -> None:
                export_jobs.update(jid, phase=message, done=done, total=max(total, 1))

            result = tiff_stack_source.register_tiff_stack(
                req.server_uri,
                req.path,
                req.container_path,
                req.description,
                req.on_conflict,
                progress=_progress,
            )
            export_jobs.update(jid, state="done", phase="done", result=result)
            export_jobs.log(jid, f"Registered {result['key']!r} as a 3-D volume.")
        except HTTPException as exc:
            export_jobs.update(jid, state="error", phase="error", error=str(exc.detail))
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI via the job
            logger.exception("tiff stack registration failed")
            export_jobs.update(jid, state="error", phase="error", error=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": jid, **info}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_filters(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


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
