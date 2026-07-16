"""FastAPI entry point for the Segmentation Annotation Studio API.

Endpoints
---------
* ``GET /api/config/servers``   — list configured Tiled servers
* ``GET /api/browse/facets``    — metadata fields available for browsing
* ``GET /api/browse/column``    — distinct values + counts for one field
* ``GET /api/browse/items``     — sample records matching a filter set
* ``GET /api/browse/thumbnail`` — PNG thumbnail for a Tiled array path
* ``GET /health``               — liveness check

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
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import arrays as arrays_mod
import drafts as drafts_mod
import export_jobs
import guides as guides_mod
import images as images_mod
import ingest as ingest_mod
import local_fs
from browse_helpers import (
    FieldMapping,
    build_field_mapping,
    distinct_from_rows,
    scoped_metadata_rows,
    tiled_distinct_values,
    tiled_search_items,
    _SINGLE_VALUE_FACET_RAW_KEYS,
)
from cache import TTLCache
from schemas import DraftPayload, ExportRequest, ExportSourceItem, GuidePayload, ImageMeta, MeasureRequest, SaveVersionRequest
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
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402
app.add_middleware(GZipMiddleware, minimum_size=500)


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
) -> Response:
    """Render one slice of an image source as a PNG."""
    opts = {
        "norm": norm,
        "scale": scale,
        "vmin_pct": vmin_pct,
        "vmax_pct": vmax_pct,
        "cmap": cmap,
    }

    def _run() -> bytes:
        node = arrays_mod.resolve_array(source, kind, server_uri, root)
        meta = arrays_mod.array_shape_meta(node)
        sl = arrays_mod.read_slice(node, meta, slice_index)
        global_range = None
        if norm == "global":
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
        import numpy as np
        import arrays as arrays_mod
        from coco_export import shape_to_mask
        from source_keys import parse_source_key

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
        import annotation_thumbnails
        import threading

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
    from coco_export import build_export_plan, write_coco_split, write_lightly_split, lightly_classes_map
    import images as images_mod_local
    import arrays as arrays_mod_local
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
            node = arrays_mod_local.resolve_array(item.source, item.kind, item.server_uri)
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
                render_slice_fn=images_mod_local.render_slice,
                array_shape_meta_fn=arrays_mod_local.array_shape_meta,
                read_slice_fn=arrays_mod_local.read_slice,
                sample_global_stats_fn=images_mod_local._sample_global_stats,
                progress_cb=_cb,
                include_polygons=payload.include_polygons,
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
        from datetime import datetime as _dt, timezone as _tz
        annotator = (payload.annotator or "").strip()
        exported_at = _dt.now(_tz.utc).isoformat()
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
                # DINOv3 / Lightly: classes.json (index→name, 0=bg) at the dataset
                # root; each split as images/ + masks/ with matching stems.
                classes_json = json.dumps(lightly_classes_map(merged_categories), indent=2)
                (out_root / "classes.json").write_text(classes_json)
                zf.writestr("classes.json", classes_json.encode("utf-8"))
                for split_name, split_data in merged_splits.items():
                    # Lightly convention: 'valid' → 'val'; 'train'/'test' unchanged.
                    dir_name = "val" if split_name == "valid" else split_name
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


@app.post("/api/ingest/upload")
async def ingest_upload(
    server_uri: Optional[str] = Query(None, description="Target Tiled server URI"),
    container_path: str = Form(..., description="Target container, e.g. 'browse/myset'"),
    description: str = Form("", description="Optional keyword(s) stored on every ingested node"),
    files: list[UploadFile] = File(..., description="Image files to copy into Tiled"),
) -> dict:
    """Stream uploaded files to temp storage and start a background ingest job.

    Each supported image becomes its own browsable node in *container_path* on
    the connected Tiled server. Returns a ``job_id`` to poll for progress.
    """
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
        args=(jid, server_uri, container_path, saved, description),
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
