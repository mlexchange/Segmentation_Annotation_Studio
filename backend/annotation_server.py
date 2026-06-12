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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import arrays as arrays_mod
import drafts as drafts_mod
import images as images_mod
import local_fs
from browse_helpers import FieldMapping, build_field_mapping, tiled_distinct_values, tiled_search_items, _STUDIO_RAW_KEYS
from cache import TTLCache
from schemas import DraftPayload, ExportRequest, ExportSourceItem, ImageMeta, RenderOpts, SaveVersionRequest
from thumbnails import render_thumbnail
from tiled_clients import api_key_for_uri, get_browse_container, get_tiled_client
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


def _resolve_field_mapping(container: object, server_uri: str, technique: str) -> FieldMapping:
    """Return a cached :class:`FieldMapping` for the given container."""
    key = (server_uri, technique)
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
    refresh: bool = Query(False),  # noqa: ARG001 — kept for client API compat
) -> dict[str, list[str]]:
    """Return ordered list of browsable metadata fields, discovered live.

    For each key in the field mapping, the field is kept if at least two
    distinct non-null values exist. Live (no cache) so newly-seeded metadata
    appears in the UI without a restart.
    """
    def _discover() -> dict[str, list[str]]:
        client = get_tiled_client(server_uri, server_api_key)
        container, _ = get_browse_container(client)
        mapping = _resolve_field_mapping(container, server_uri or "", technique)

        def _facet_for_key(disp_key: str) -> tuple[list[str], list[str]]:
            raw_key = mapping.display_to_raw.get(disp_key, disp_key)
            min_values = 1 if raw_key in _STUDIO_RAW_KEYS else 2
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
    limit: int = Query(500, ge=1, le=5000),
    refresh: bool = Query(False),
) -> dict:
    """Return distinct values (+ counts) for *field* via Tiled ``distinct()``."""
    filter_dict = _parse_json_filters(filters)

    cache_key = ("column", server_uri or "", technique, field, filters, limit)
    if not refresh:
        cached = _column_cache.get(cache_key)
        if cached is not None:
            return cached

    def _build() -> dict:
        client = get_tiled_client(server_uri, server_api_key)
        container, _ = get_browse_container(client)
        mapping = _resolve_field_mapping(container, server_uri or "", technique)
        raw_key = mapping.display_to_raw.get(field, field)
        return tiled_distinct_values(
            container,
            raw_key,
            filters=filter_dict,
            field_mapping=mapping,
            limit=limit,
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
    limit: int = Query(500, ge=1, le=2000),
    refresh: bool = Query(False),
) -> dict:
    """Return sample records (path + metadata) matching *filters* via ``search()``."""
    filter_dict = _parse_json_filters(filters)

    cache_key = ("items", server_uri or "", technique, filters, limit)
    if not refresh:
        cached = _items_cache.get(cache_key)
        if cached is not None:
            return cached

    def _build() -> dict:
        client = get_tiled_client(server_uri, server_api_key)
        container, prefix = get_browse_container(client)
        mapping = _resolve_field_mapping(container, server_uri or "", technique)
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
    rel: str = Query("", description="Relative path under LOCAL_DATA_ROOT"),
) -> list[dict]:
    """List directory entries under LOCAL_DATA_ROOT."""
    return await asyncio.to_thread(local_fs.list_dir, rel)


@app.get("/api/local/samples")
async def local_samples(
    rel: str = Query(..., description="Relative path to a folder under LOCAL_DATA_ROOT"),
) -> dict:
    """Return all image files under a local folder (used by the Browse tab).

    Args:
        rel: Relative folder path under ``LOCAL_DATA_ROOT``.

    Returns:
        ``{"items": [{"name", "path"}], "total": int}``
    """
    items = await asyncio.to_thread(local_fs.list_image_files, rel)
    return {"items": items, "total": len(items)}


@app.get("/api/connect/summary")
async def connect_summary(
    kind: str = Query(..., description="'tiled' or 'local'"),
    server_uri: Optional[str] = None,
    rel: str = Query("", description="Local folder path (kind=local only)"),
) -> dict:
    """Return a connection summary: sample count and display label.

    For Tiled sources, counts catalog items via the browse API.
    For local sources, counts image files recursively under the folder.

    Returns:
        ``{"kind", "label", "sample_count", "server_uri"}``
    """
    if kind == "local":
        count = await asyncio.to_thread(local_fs.count_image_files, rel)
        label = rel or "Local Data Root"
        return {"kind": "local", "label": label, "sample_count": count, "server_uri": None}

    if kind == "tiled":
        def _count() -> int:
            client = get_tiled_client(server_uri)
            container, _ = get_browse_container(client)
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
        return {"kind": "tiled", "label": label, "sample_count": count, "server_uri": server_uri}

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
) -> ImageMeta:
    """Return shape / dtype metadata for an image source."""
    def _run() -> ImageMeta:
        node = arrays_mod.resolve_array(source, kind, server_uri)
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
        node = arrays_mod.resolve_array(source, kind, server_uri)
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

    def _run() -> dict:
        from coco_export import build_export_plan, write_coco_split
        import images as images_mod_local
        import arrays as arrays_mod_local

        # Merged splits accumulator across all sources.
        merged_splits: dict[str, dict] = {}
        skipped_total = 0
        merged_categories: list[dict] = []
        merged_info: dict = {}

        for item in source_items:
            node = arrays_mod_local.resolve_array(item.source, item.kind, item.server_uri)
            # Build a temporary single-source payload object for reuse of build_export_plan.
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
            plan = build_export_plan(
                node, tmp,
                render_slice_fn=images_mod_local.render_slice,
                array_shape_meta_fn=arrays_mod_local.array_shape_meta,
                read_slice_fn=arrays_mod_local.read_slice,
                sample_global_stats_fn=images_mod_local._sample_global_stats,
            )
            skipped_total += plan["skipped_zero_area"]
            if not merged_categories:
                merged_categories = plan["categories"]
                merged_info = plan["info"]
            for split_name, split_data in plan["splits"].items():
                if split_name not in merged_splits:
                    merged_splits[split_name] = {"images": [], "annotations": []}
                merged_splits[split_name]["images"].extend(split_data["images"])
                merged_splits[split_name]["annotations"].extend(split_data["annotations"])

        summary: dict = {"skipped_zero_area": skipped_total, "splits": {}}
        if payload.dry_run:
            for split_name, split_data in merged_splits.items():
                summary["splits"][split_name] = {
                    "n_images": len(split_data["images"]),
                    "n_annotations": len(split_data["annotations"]),
                }
            return summary

        written: dict = {}
        for split_name, split_data in merged_splits.items():
            split_dir = out_root / split_name
            result = write_coco_split(
                split_dir,
                images=split_data["images"],
                categories=merged_categories,
                annotations=split_data["annotations"],
                mode=payload.mode,
                info=merged_info,
            )
            written[split_name] = result
        summary["written"] = written
        summary["dataset_path"] = str(out_root)

        # Sync annotation flags back onto source Tiled nodes for Browse discovery.
        import tiled_annotation_sync
        from source_keys import parse_source_key

        for item in source_items:
            if item.kind != "tiled":
                continue
            sk = f"tiled:{item.server_uri or ''}:{item.source}"
            try:
                tiled_annotation_sync.sync_annotation_metadata(
                    sk,
                    {
                        "classes": payload.classes,
                        "slices": item.slices,
                    },
                )
            except Exception as sync_exc:
                logger.warning("Export Tiled sync failed for %s: %s", sk, sync_exc)

        return summary

    try:
        result = await asyncio.to_thread(_run)
        return result
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        logger.error("Export failed: %s", exc)
        raise HTTPException(500, f"Export failed: {exc}") from exc


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


if __name__ == "__main__":  # pragma: no cover — convenience entry point
    import uvicorn

    uvicorn.run("annotation_server:app", host="127.0.0.1", port=8002)
