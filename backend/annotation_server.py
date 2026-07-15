"""FastAPI entry point for the SAM3 Annotation Studio API.

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
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import arrays as arrays_mod
import drafts as drafts_mod
import export_jobs
import images as images_mod
import ingest as ingest_mod
import local_fs
import clf_shelf as clf_shelf_mod
import label_sets as label_sets_mod
import feature_manifold as manifold_mod
import multiscale_features as features_mod
import pixel_clf as pixel_clf_mod
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
from schemas import DraftPayload, ExportRequest, ExportSourceItem, ImageMeta, SaveVersionRequest
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
    title="SAM3 Annotation Studio API",
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
        def _count() -> tuple[int, str]:
            client = get_tiled_client(server_uri)
            container, prefix = get_browse_container_for(client, container_path)
            # An unfiltered count is just the container size — a single request.
            # Avoid iterating every child and building per-item metadata dicts,
            # which is O(N) HTTP round trips and stalls the connect UI.
            try:
                n = int(len(container))
            except Exception:
                result = tiled_search_items(container, filters={}, limit=10_000)
                n = int(result.get("total", 0))
            return n, prefix

        try:
            count, discovered_prefix = await asyncio.to_thread(_count)
        except Exception as exc:
            logger.warning("connect_summary tiled count failed: %s", exc)
            count, discovered_prefix = 0, ""

        servers = get_tiled_servers()
        label = next(
            (cfg.get("name", name) for name, cfg in servers.items()
             if (cfg.get("uri") or "").rstrip("/") == (server_uri or "").rstrip("/")),
            server_uri or "Tiled Server",
        )
        resolved_path = (container_path or discovered_prefix or "").strip("/") or None
        if resolved_path:
            label = f"{label} · {resolved_path}"
        return {
            "kind": "tiled",
            "label": label,
            "sample_count": count,
            "server_uri": server_uri,
            "container_path": resolved_path,
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


class FeatureComputeRequest(BaseModel):
    """Parameters for multiscale feature computation on one slice."""

    source: str
    kind: str
    slice_index: int = 0
    server_uri: Optional[str] = None
    root: Optional[str] = None
    sigma_min: float = 1.0
    sigma_max: float = 8.0
    intensity: bool = True
    edges: bool = True
    texture: bool = True
    clahe: bool = True
    include_sam: bool = False


@app.get("/api/image/features/sam-status")
async def features_sam_status() -> dict:
    """Whether SlimSAM vision encoder ONNX is available for feature concat."""
    import sam_embed as sam_mod

    return {"available": sam_mod.sam_available()}


@app.post("/api/image/features")
async def compute_image_features(body: FeatureComputeRequest) -> dict:
    """Compute multiscale features for one slice; cache stack; return channel list."""

    def _run() -> dict:
        node = arrays_mod.resolve_array(body.source, body.kind, body.server_uri, body.root)
        meta = arrays_mod.array_shape_meta(node)
        sl = arrays_mod.read_slice(node, meta, body.slice_index)
        job = features_mod.compute_and_store(
            sl,
            sigma_min=body.sigma_min,
            sigma_max=body.sigma_max,
            intensity=body.intensity,
            edges=body.edges,
            texture=body.texture,
            clahe=body.clahe,
            include_sam=body.include_sam,
        )
        h, w, _ = job.stack.shape
        return {
            "job_id": job.job_id,
            "width": w,
            "height": h,
            "channels": [{"index": i, "label": lab} for i, lab in enumerate(job.labels)],
            "has_sam": job.sam_emb is not None,
            "sam_shape": list(job.sam_emb.shape) if job.sam_emb is not None else None,
        }

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error("compute_image_features failed: %s", exc)
        raise HTTPException(500, f"Failed to compute features: {exc}") from exc


@app.get("/api/image/features/{job_id}/{index}")
async def get_feature_channel(job_id: str, index: int) -> Response:
    """Return one cached feature channel as a grayscale PNG."""

    def _run() -> bytes:
        job = features_mod.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Feature job not found or expired")
        try:
            return features_mod.encode_channel_png(job.stack, index)
        except IndexError as exc:
            raise HTTPException(404, str(exc)) from exc

    try:
        png = await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_feature_channel failed: %s", exc)
        raise HTTPException(500, f"Failed to encode feature channel: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=600"},
    )


class ManifoldSampleRequest(BaseModel):
    """Greedy feature-variance inducing boxes on a feature job."""

    job_id: str
    k: int = 24
    """Full square box side length in image pixels."""
    box_size: int | None = None
    stride: int | None = None
    pca_dims: int = 16
    """Optional placement-region shapes (union mask); omit for full image."""
    shapes: list[dict] | None = None


@app.post("/api/image/features/manifold/sample")
async def manifold_sample(body: ManifoldSampleRequest) -> dict:
    """Sample diverse annotation boxes via variance + exclusion."""

    def _run() -> dict:
        job = features_mod.get_job(body.job_id)
        if job is None:
            raise HTTPException(404, "Feature job not found or expired")
        place_mask = None
        if body.shapes:
            from coco_export import shape_to_mask

            h, w = job.float_stack.shape[:2]
            place_mask = np.zeros((h, w), dtype=bool)
            for shape in body.shapes:
                try:
                    place_mask |= shape_to_mask(shape, h, w)
                except (KeyError, ValueError, TypeError) as exc:
                    raise HTTPException(400, f"invalid placement shape: {exc}") from exc
            if not bool(place_mask.any()):
                raise HTTPException(400, "placement mask is empty")
        try:
            result = manifold_mod.sample_inducing_points(
                job,
                k=body.k,
                box_size=body.box_size,
                stride=body.stride,
                pca_dims=body.pca_dims,
                mask=place_mask,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        cached = manifold_mod.store_sample(result)
        return {
            "sample_id": cached.sample_id,
            "points": cached.points,
            "k": result.meta["k"],
            "n_picked": result.meta["n_picked"],
            "n_subsample": result.meta["n_subsample"],
            "pca_dims": result.meta["pca_dims"],
            "explained_variance": result.meta["explained_variance"],
            "stride": result.meta["stride"],
            "radius": result.meta["radius"],
            "box_size": result.meta["box_size"],
            "mask_pixels": result.meta.get("mask_pixels", 0),
            "n_windows_in_mask": result.meta.get("n_windows_in_mask"),
            "has_mask": result.meta.get("has_mask", False),
        }

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("manifold_sample failed: %s", exc)
        raise HTTPException(500, f"Failed to sample manifold: {exc}") from exc


@app.get("/api/image/features/manifold/{sample_id}/heatmap.png")
async def manifold_heatmap_png(sample_id: str) -> Response:
    """Coverage heatmap PNG (grayscale 0–255)."""

    def _run() -> bytes:
        cached = manifold_mod.get_sample(sample_id)
        if cached is None:
            raise HTTPException(404, "Manifold sample not found or expired")
        return cached.heatmap_png

    try:
        png = await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


class ClfTrainRequest(BaseModel):
    """Train CatBoost on sparse annotations + a feature job's float stack."""

    job_id: str
    shapes: list[dict]
    iterations: int = 200
    depth: int = 6
    learning_rate: float = 0.1
    max_samples: int = 200_000


class ClfPredictRequest(BaseModel):
    """Mondrian conformal predict at misfire level *alpha*."""

    model_id: str
    job_id: str
    alpha: float = 0.05
    # When set, annotated pixels are zeroed (keep originals on commit).
    shapes: list[dict] | None = None


@app.post("/api/image/features/clf/train")
async def clf_train(body: ClfTrainRequest) -> dict:
    """Fit CatBoost on labeled pixels from *shapes* and the feature job float stack."""

    def _run() -> dict:
        job = features_mod.get_job(body.job_id)
        if job is None:
            raise HTTPException(404, "Feature job not found or expired")
        try:
            model = pixel_clf_mod.train_classifier(
                job,
                body.shapes,
                iterations=body.iterations,
                depth=body.depth,
                learning_rate=body.learning_rate,
                max_samples=body.max_samples,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return pixel_clf_mod.train_result_dict(model)

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("clf_train failed: %s", exc)
        raise HTTPException(500, f"Failed to train classifier: {exc}") from exc


@app.get("/api/image/features/clf/{model_id}/tree/{tree_index}")
async def clf_tree(model_id: str, tree_index: int = 0) -> dict:
    """Return one CatBoost oblivious tree (splits + leaf values) for the UI."""

    def _run() -> dict:
        model = pixel_clf_mod.get_model(model_id)
        if model is None:
            raise HTTPException(404, "Classifier model not found or expired")
        try:
            return pixel_clf_mod.tree_view(model, tree_index)
        except IndexError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("clf_tree failed: %s", exc)
        raise HTTPException(500, f"Failed to read tree: {exc}") from exc


@app.post("/api/image/features/clf/predict")
async def clf_predict(body: ClfPredictRequest) -> dict:
    """Conformal predict; returns meta + pred_id for commit/status PNG GETs."""

    def _run() -> dict:
        model = pixel_clf_mod.get_model(body.model_id)
        if model is None:
            raise HTTPException(404, "Classifier model not found or expired")
        job = features_mod.get_job(body.job_id)
        if job is None:
            raise HTTPException(404, "Feature job not found or expired")
        try:
            result = pixel_clf_mod.predict_conformal(
                model,
                job,
                alpha=body.alpha,
                preserve_shapes=body.shapes,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        cached = pixel_clf_mod.store_prediction(result)
        return cached.meta

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("clf_predict failed: %s", exc)
        raise HTTPException(500, f"Failed to predict: {exc}") from exc


@app.get("/api/image/features/clf/predict/{pred_id}/commit.png")
async def clf_predict_commit_png(pred_id: str) -> Response:
    """Singleton commit map (classId or 0)."""

    def _run() -> bytes:
        cached = pixel_clf_mod.get_prediction(pred_id)
        if cached is None:
            raise HTTPException(404, "Prediction not found or expired")
        return cached.commit_png

    try:
        png = await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})


@app.get("/api/image/features/clf/predict/{pred_id}/status.png")
async def clf_predict_status_png(pred_id: str) -> Response:
    """Status map: 0 abstain, 1 singleton, 2 multi."""

    def _run() -> bytes:
        cached = pixel_clf_mod.get_prediction(pred_id)
        if cached is None:
            raise HTTPException(404, "Prediction not found or expired")
        return cached.status_png

    try:
        png = await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})


class ClfShelfSaveRequest(BaseModel):
    """Persist a trained in-memory model + feature recipe to the shelf."""

    model_id: str
    name: str
    feature_recipe: dict


class ClfShelfPredictRequest(BaseModel):
    """Recompute features with the shelf recipe and run conformal predict."""

    source: str
    kind: str
    slice_index: int = 0
    server_uri: Optional[str] = None
    root: Optional[str] = None
    alpha: float = 0.05
    shapes: list[dict] | None = None


# ##############################################################################
# # REMOVE THIS AND USE YOUR OWN STUFF
# Scaffold CatBoost model-shelf + reusable label-set APIs. Replace with your
# own model registry / label taxonomy endpoints, then delete these routes and
# ``clf_shelf.py`` / ``label_sets.py``.
# ##############################################################################


@app.post("/api/clf/models/save")
async def clf_shelf_save(body: ClfShelfSaveRequest) -> dict:
    """# REMOVE THIS AND USE YOUR OWN STUFF — save shelf model."""

    def _run() -> dict:
        model = pixel_clf_mod.get_model(body.model_id)
        if model is None:
            raise HTTPException(404, "Classifier model not found or expired — train again")
        try:
            meta = clf_shelf_mod.save_model(
                model, name=body.name.strip() or "unnamed", feature_recipe=body.feature_recipe
            )
        except Exception as exc:
            raise HTTPException(500, f"Failed to save model: {exc}") from exc
        return {
            "id": meta.id,
            "name": meta.name,
            "class_ids": meta.class_ids,
            "n_train": meta.n_train,
            "n_cal": meta.n_cal,
            "train_accuracy": meta.train_accuracy,
            "uses_sam": meta.uses_sam,
            "feature_recipe": meta.feature_recipe,
            "created_at": meta.created_at,
            "n_features": meta.n_features,
        }

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise


@app.get("/api/clf/models")
async def clf_shelf_list() -> list[dict]:
    """List persisted classifier shelf models."""
    return await asyncio.to_thread(clf_shelf_mod.list_models)


class LabelSetSaveRequest(BaseModel):
    """Create a reusable annotation label set (classes)."""

    name: str
    classes: list[dict]


@app.get("/api/label-sets")
async def label_sets_list() -> dict:
    """# REMOVE THIS AND USE YOUR OWN STUFF — list scaffold label sets."""
    sets = await asyncio.to_thread(label_sets_mod.list_label_sets)
    return {"sets": sets}


@app.post("/api/label-sets")
async def label_sets_save(body: LabelSetSaveRequest) -> dict:
    """# REMOVE THIS AND USE YOUR OWN STUFF — save scaffold label set."""

    def _run() -> dict:
        try:
            return label_sets_mod.save_label_set(name=body.name, classes=body.classes)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise


@app.get("/api/label-sets/{set_id}")
async def label_sets_get(set_id: str) -> dict:
    """Load one label set."""
    data = await asyncio.to_thread(label_sets_mod.get_label_set, set_id)
    if data is None:
        raise HTTPException(404, "Label set not found")
    return data


@app.delete("/api/label-sets/{set_id}")
async def label_sets_delete(set_id: str) -> dict:
    """Delete a label set."""

    def _run() -> dict:
        ok = label_sets_mod.delete_label_set(set_id)
        if not ok:
            raise HTTPException(404, "Label set not found")
        return {"ok": True}

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise


@app.delete("/api/clf/models/{shelf_id}")
async def clf_shelf_delete(shelf_id: str) -> dict:
    """Delete a shelf model."""

    def _run() -> dict:
        ok = clf_shelf_mod.delete_model(shelf_id)
        if not ok:
            raise HTTPException(404, "Shelf model not found")
        return {"ok": True}

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise


@app.post("/api/clf/models/{shelf_id}/predict")
async def clf_shelf_predict(shelf_id: str, body: ClfShelfPredictRequest) -> dict:
    """Apply a shelf model to an image: recompute features + conformal predict."""

    def _run() -> dict:
        try:
            meta = clf_shelf_mod.get_meta(shelf_id)
            model = clf_shelf_mod.load_into_cache(shelf_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        recipe = meta.get("feature_recipe") or {}
        node = arrays_mod.resolve_array(body.source, body.kind, body.server_uri, body.root)
        arr_meta = arrays_mod.array_shape_meta(node)
        sl = arrays_mod.read_slice(node, arr_meta, body.slice_index)
        job = features_mod.compute_and_store(
            sl,
            sigma_min=float(recipe.get("sigma_min", 1.0)),
            sigma_max=float(recipe.get("sigma_max", 8.0)),
            intensity=bool(recipe.get("intensity", True)),
            edges=bool(recipe.get("edges", True)),
            texture=bool(recipe.get("texture", True)),
            clahe=bool(recipe.get("clahe", True)),
            include_sam=bool(recipe.get("include_sam", False)),
        )
        try:
            result = pixel_clf_mod.predict_conformal(
                model,
                job,
                alpha=body.alpha,
                preserve_shapes=body.shapes,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        cached = pixel_clf_mod.store_prediction(result)
        return {
            **cached.meta,
            "job_id": job.job_id,
            "shelf_id": shelf_id,
            "width": int(job.float_stack.shape[1]),
            "height": int(job.float_stack.shape[0]),
        }

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("clf_shelf_predict failed: %s", exc)
        raise HTTPException(500, f"Shelf predict failed: {exc}") from exc


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
    from coco_export import build_export_plan, write_coco_split
    import images as images_mod_local
    import arrays as arrays_mod_local

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

        # Write files AND build the download .zip in one pass. ZIP_STORED: the
        # PNGs are already compressed, so re-deflating them is wasted CPU.
        import zipfile
        export_jobs.update(jid, phase="writing")
        zip_path = f"{out_root}.zip"
        written: dict = {}
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
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
# ipred proxy (GUI never calls port 8003 directly)
# ---------------------------------------------------------------------------


class IpredSessionRequest(BaseModel):
    """Open an ipred session for one image/stack project."""

    kind: str
    source: str
    server_uri: Optional[str] = None
    root: Optional[str] = None


class IpredPreprocessRequest(BaseModel):
    """Proxy featurize request."""

    session_id: str
    feature_setup_id: Optional[str] = None
    composition_id: Optional[str] = None
    slice_index: int = 0
    array_ref: Optional[str] = None


class IpredCompositionUpsertRequest(BaseModel):
    """Create/update composition on ipred."""

    name: str
    nodes: list[dict[str, Any]]
    outputs: list[str]
    composition_id: Optional[str] = None
    builtin: bool = False


class IpredArrayUploadRequest(BaseModel):
    """Upload float array for remote ipred preprocess."""

    session_id: str
    shape: list[int]
    dtype: str = "float32"
    data_b64: str
    array_ref: Optional[str] = None


class IpredTrainRequest(BaseModel):
    """Proxy train request."""

    session_id: str
    shapes: list[dict[str, Any]]
    feature_id: Optional[str] = None
    trainer_id: str = "catboost"
    config: Optional[dict[str, Any]] = None


class IpredInferRequest(BaseModel):
    """Proxy infer request."""

    session_id: str
    model_id: Optional[str] = None
    feature_id: Optional[str] = None
    alpha: float = 0.05


class IpredRethresholdRequest(BaseModel):
    """Proxy rethreshold request."""

    session_id: str
    alpha: float
    run_id: Optional[str] = None


class IpredSetupUpsertRequest(BaseModel):
    """Create or update a Feature Setup on ipred."""

    name: str
    kind: str
    procedure_id: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    encoder_setup_id: Optional[str] = None
    weights_path: Optional[str] = None
    weights_format: Optional[str] = None
    inference: Optional[dict[str, Any]] = None
    setup_id: Optional[str] = None


class IpredManifoldSampleRequest(BaseModel):
    """Suggest Labels via ipred feature bank."""

    feature_id: str
    k: int = 24
    box_size: Optional[int] = None
    stride: Optional[int] = None
    pca_dims: int = 16
    shapes: Optional[list[dict[str, Any]]] = None


def _ipred_http_error(exc: Exception) -> HTTPException:
    import httpx

    import ipred_client as ipred_client_mod

    if isinstance(exc, httpx.HTTPStatusError):
        detail = exc.response.text
        try:
            detail = exc.response.json()
        except Exception:
            pass
        return HTTPException(exc.response.status_code, detail)
    if isinstance(exc, httpx.ConnectError):
        return HTTPException(
            503,
            f"ipred unreachable at {ipred_client_mod.ipred_url()}",
        )
    return HTTPException(500, str(exc))


@app.get("/api/ipred/health")
async def ipred_health() -> dict:
    """Liveness of the disjoint ipred."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.health()
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/sessions")
async def ipred_open_session(body: IpredSessionRequest) -> dict:
    """Open/create an engine session for a project."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.open_session(
            kind=body.kind,
            source=body.source,
            server_uri=body.server_uri,
            root=body.root,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/setups")
async def ipred_list_setups() -> dict:
    """List Feature Setups from the ipred."""
    import ipred_client as ipred_client_mod

    try:
        return {"setups": ipred_client_mod.list_setups()}
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/setups/{setup_id}")
async def ipred_get_setup(setup_id: str) -> dict:
    """Get one Feature Setup from ipred."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.get_setup(setup_id)
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/setups")
async def ipred_upsert_setup(body: IpredSetupUpsertRequest) -> dict:
    """Create or update a Feature Setup on ipred."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.upsert_setup(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/trainers")
async def ipred_list_trainers() -> dict:
    """List trainer plugin ids from ipred."""
    import ipred_client as ipred_client_mod

    try:
        return {"trainers": ipred_client_mod.list_trainers()}
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/modules")
async def ipred_list_modules() -> dict:
    """List composable feature modules."""
    import ipred_client as ipred_client_mod

    try:
        return {"modules": ipred_client_mod.list_modules()}
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/compositions")
async def ipred_list_compositions() -> dict:
    """List feature compositions."""
    import ipred_client as ipred_client_mod

    try:
        return {"compositions": ipred_client_mod.list_compositions()}
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/compositions/{composition_id}")
async def ipred_get_composition(composition_id: str) -> dict:
    """Get one composition."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.get_composition(composition_id)
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/compositions")
async def ipred_upsert_composition(body: IpredCompositionUpsertRequest) -> dict:
    """Create or update a composition."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.upsert_composition(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/compositions/preview")
async def ipred_preview_composition(body: IpredCompositionUpsertRequest) -> dict:
    """Preview concat labels for a composition draft."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.preview_composition(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/sessions/{session_id}/arrays")
async def ipred_upload_array(session_id: str, body: IpredArrayUploadRequest) -> dict:
    """Upload an array blob to ipred for remote preprocess."""
    import ipred_client as ipred_client_mod

    try:
        payload = body.model_dump(exclude_none=True)
        payload["session_id"] = session_id
        return ipred_client_mod.upload_session_array(session_id, payload)
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/manifold/sample")
async def ipred_manifold_sample(body: IpredManifoldSampleRequest) -> dict:
    """Suggest Labels on an ipred feature bank."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.manifold_sample(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/manifold/{sample_id}/heatmap.png")
async def ipred_manifold_heatmap(sample_id: str) -> Response:
    """Proxy manifold residual heatmap PNG."""
    import ipred_client as ipred_client_mod

    try:
        return Response(
            content=ipred_client_mod.manifold_heatmap_png(sample_id),
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=300"},
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/preprocess")
async def ipred_preprocess(body: IpredPreprocessRequest) -> dict:
    """Cache-aware featurize via ipred."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.preprocess(
            session_id=body.session_id,
            feature_setup_id=body.feature_setup_id,
            composition_id=body.composition_id,
            slice_index=body.slice_index,
            array_ref=body.array_ref,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/features/{feature_id}/channels/{index}")
async def ipred_feature_channel(feature_id: str, index: int) -> Response:
    """Proxy a feature-channel PNG from the ipred."""
    import ipred_client as ipred_client_mod

    try:
        data = ipred_client_mod.feature_channel_bytes(feature_id, index)
        return Response(content=data, media_type="image/png")
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/train")
async def ipred_train(body: IpredTrainRequest) -> dict:
    """Train via ipred trainer plugin."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.train(
            session_id=body.session_id,
            shapes=body.shapes,
            feature_id=body.feature_id,
            trainer_id=body.trainer_id,
            config=body.config,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/infer")
async def ipred_infer(body: IpredInferRequest) -> dict:
    """Infer + conformal products via ipred."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.infer(
            session_id=body.session_id,
            model_id=body.model_id,
            feature_id=body.feature_id,
            alpha=body.alpha,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.post("/api/ipred/rethreshold")
async def ipred_rethreshold(body: IpredRethresholdRequest) -> dict:
    """Rethreshold from cached proba via ipred."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.rethreshold(
            session_id=body.session_id,
            alpha=body.alpha,
            run_id=body.run_id,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/runs/{run_id}/commit.png")
async def ipred_run_commit(run_id: str) -> Response:
    """Proxy commit map PNG."""
    import ipred_client as ipred_client_mod

    try:
        return Response(
            content=ipred_client_mod.run_commit_png(run_id),
            media_type="image/png",
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/runs/{run_id}/status.png")
async def ipred_run_status(run_id: str) -> Response:
    """Proxy status map PNG."""
    import ipred_client as ipred_client_mod

    try:
        return Response(
            content=ipred_client_mod.run_status_png(run_id),
            media_type="image/png",
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@app.get("/api/ipred/runs/{run_id}/proba/{class_index}.png")
async def ipred_run_proba(run_id: str, class_index: int) -> Response:
    """Proxy softmax class heatmap PNG from ipred."""
    import ipred_client as ipred_client_mod

    try:
        return Response(
            content=ipred_client_mod.run_proba_png(run_id, class_index),
            media_type="image/png",
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


class IpredThresholdClassRequest(BaseModel):
    class_id: int
    threshold: float = 0.5


@app.post("/api/ipred/runs/{run_id}/threshold-class")
async def ipred_threshold_class(
    run_id: str, body: IpredThresholdClassRequest
) -> dict:
    """Threshold one softmax class into a dense label map via ipred."""
    import ipred_client as ipred_client_mod

    try:
        return ipred_client_mod.threshold_class_map(
            run_id,
            class_id=body.class_id,
            threshold=body.threshold,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


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
