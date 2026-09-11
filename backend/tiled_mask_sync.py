"""Write rasterized annotation masks into Tiled as stacked volumes.

Standalone counterpart to the COCO .zip export: instead of writing per-slice PNG
files to disk, this rasterizes the same shapes (via ``coco_export.shape_to_mask``)
and stores them as compact **stacked uint8 arrays** in a sibling Tiled container
``<source_stem>__masks`` next to the source dataset — so a downstream app
(SAM3 / DINOv3 fine-tuning) can grab every mask in one request.

Layout (all annotated + explicit negative slices, stacked in sorted order):

    <source_stem>__masks/          (container; metadata carries legend + slice_indices)
      semantic   uint8 (n, H, W)   class-index per pixel, 0 = background
      <class>    uint8 (n, H, W)   0/255 binary volume, one per class

Tiled serves these back as PNG/TIFF via format negotiation. Only Tiled sources
are handled; local sources are skipped (nothing to write back to).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

import arrays as arrays_mod
import export_jobs
import ipred_client
import mask_pyramid
from coco_export import _safe_name, shape_to_mask
from tiled_clients import api_key_for_uri, get_tiled_client

logger = logging.getLogger(__name__)


def _read_predicted_label_map(run_id: str) -> np.ndarray:
    """Fetch an ipred run's commit.png and decode it into the same ``(H, W)``
    uint8 array shape a rasterized shape would produce — pixel value is the
    raw frontend classId directly (ipred's own convention, matching the
    frontend's ``labelMapToPolygonShapes``), NOT yet remapped to this
    export's legend ids; the caller does that remap the same way it already
    does for real shapes.
    """
    import io

    from PIL import Image

    png_bytes = ipred_client.run_commit_png(run_id)
    return np.asarray(Image.open(io.BytesIO(png_bytes)))


def _category_maps(classes: list[Any]) -> tuple[dict[int, int], dict[int, str], list[dict[str, Any]]]:
    """1-based COCO ids + legend, matching ``build_export_plan``'s convention."""
    cat_id_map: dict[int, int] = {}
    cat_id_to_name: dict[int, str] = {}
    legend: list[dict[str, Any]] = []
    for i, cls in enumerate(classes, 1):
        d = cls if isinstance(cls, dict) else cls.model_dump()
        cat_id_map[int(d["classId"])] = i
        cat_id_to_name[i] = d["label"]
        legend.append({"id": i, "name": d["label"], "color": d.get("color")})
    return cat_id_map, cat_id_to_name, legend


def build_mask_volumes(
    item: Any,
    classes: list[Any],
    meta: dict[str, Any],
    progress_cb: Any = None,
) -> dict[str, Any] | None:
    """Rasterize an item's shapes into stacked mask volumes.

    Returns ``{semantic, class_vols, slice_indices, legend}`` or ``None`` when
    there are no slices to write. ``semantic`` is ``(n,H,W)`` uint8 (class index,
    0=bg); ``class_vols`` maps class label → ``(n,H,W)`` uint8 (0/255). Slices are
    every annotated key plus any ``negative_slices`` (emitted as all-zero frames
    for hard negatives) plus any ``predicted_slices`` key not already covered by
    real shapes, in sorted numeric order.

    ``predicted_slices`` (see :class:`schemas.PredictedSlicePointer`) lets an
    un-vectorized iPred volume-apply result go straight into the mask volume —
    fetching its commit.png from the ipred service and rasterizing THAT,
    instead of requiring the frontend to first trace it into polygon shapes
    just so this function can immediately rasterize them back into a mask.
    A slice with real shapes always wins over its predicted pointer (matches
    the frontend's own precedence in ``handleCommitVolumeApply``/
    `usePredictedRasterStore`).
    """
    h, w = int(meta["height"]), int(meta["width"])
    cat_id_map, cat_id_to_name, legend = _category_maps(classes)

    slices: dict[str, list[dict[str, Any]]] = item.slices or {}
    neg = {str(k) for k in (item.negative_slices or [])}
    predicted: dict[str, Any] = getattr(item, "predicted_slices", None) or {}
    keys = sorted(
        {k for k, shapes in slices.items() if shapes} | neg | {k for k, p in predicted.items() if p},
        key=lambda k: int(k),
    )
    if not keys:
        return None

    class_names = [cat_id_to_name[i] for i in sorted(cat_id_to_name)]
    sem_list: list[np.ndarray] = []
    class_lists: dict[str, list[np.ndarray]] = {name: [] for name in class_names}

    for key in keys:
        label = np.zeros((h, w), dtype=np.uint8)
        acc: dict[str, np.ndarray] = {name: np.zeros((h, w), dtype=bool) for name in class_names}
        shapes_here = slices.get(key, [])
        if shapes_here:
            for shape in shapes_here:
                shape_dict = shape if isinstance(shape, dict) else shape.model_dump()
                mask = shape_to_mask(shape_dict, h, w)
                if float(mask.sum()) < 1:
                    continue
                cat_id = cat_id_map.get(int(shape_dict.get("classId", 1)), 1)
                label[mask] = cat_id
                acc[cat_id_to_name.get(cat_id, "")] |= mask
        elif key in predicted and predicted[key]:
            pointer = predicted[key]
            run_id = pointer["run_id"] if isinstance(pointer, dict) else pointer.run_id
            raw = _read_predicted_label_map(run_id)
            for raw_id in np.unique(raw):
                if raw_id == 0:
                    continue
                cat_id = cat_id_map.get(int(raw_id), 1)
                mask = raw == raw_id
                label[mask] = cat_id
                acc[cat_id_to_name.get(cat_id, "")] |= mask
        sem_list.append(label)
        for name in class_names:
            class_lists[name].append((acc[name] * 255).astype(np.uint8))
        if progress_cb is not None:
            progress_cb(f"slice {key}: rasterized")

    return {
        "semantic": np.stack(sem_list, axis=0),
        "class_vols": {name: np.stack(lst, axis=0) for name, lst in class_lists.items()},
        "slice_indices": [int(k) for k in keys],
        "legend": legend,
    }


def _legend_id_to_name(legend: list[dict[str, Any]]) -> dict[int, str]:
    return {int(e["id"]): e["name"] for e in legend}


def _remap_semantic(frame: np.ndarray, id_to_name: dict[int, str], name_to_uid: dict[str, int]) -> np.ndarray:
    """Recode a semantic frame from one legend's class ids to unified ids (by name)."""
    out = np.zeros_like(frame)
    for old_id in np.unique(frame):
        if old_id == 0:
            continue
        name = id_to_name.get(int(old_id))
        if name is None:
            continue
        uid = name_to_uid.get(name)
        if uid:
            out[frame == old_id] = uid
    return out


def merge_mask_volumes(existing: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Merge freshly-rasterized ``new`` volumes onto ``existing`` ones, per slice.

    Slices present in ``new`` overwrite the same index; other existing slices are
    kept. Classes are unioned by NAME and given a single unified id scheme, so the
    stored ``semantic`` label map stays consistent even if the class set changed
    between pushes. Pure (no I/O) so it can be unit-tested.

    ``existing`` is ``None`` (fresh) or ``{slice_indices, semantic (k,H,W),
    class_arrays {name:(k,H,W)}, legend}``. Returns
    ``{semantic, class_vols, slice_indices, legend, updated_indices}``.
    """
    old_legend = (existing or {}).get("legend") or []
    old_indices = [int(i) for i in (existing or {}).get("slice_indices", [])]
    old_sem = (existing or {}).get("semantic")
    old_cls = (existing or {}).get("class_arrays", {}) or {}
    new_indices = [int(i) for i in new["slice_indices"]]

    # Unified class list: existing names first (stable ids), then new-only names.
    unified_names: list[str] = [e["name"] for e in old_legend]
    for e in new["legend"]:
        if e["name"] not in unified_names:
            unified_names.append(e["name"])
    name_to_uid = {name: i + 1 for i, name in enumerate(unified_names)}
    color_by_name: dict[str, Any] = {}
    for e in [*old_legend, *new["legend"]]:  # new overrides old for colour
        color_by_name[e["name"]] = e.get("color")

    old_id_to_name = _legend_id_to_name(old_legend)
    new_id_to_name = _legend_id_to_name(new["legend"])

    merged_sem: dict[int, np.ndarray] = {}
    merged_cls: dict[str, dict[int, np.ndarray]] = {name: {} for name in unified_names}

    if old_sem is not None:
        for pos, idx in enumerate(old_indices):
            merged_sem[idx] = _remap_semantic(old_sem[pos], old_id_to_name, name_to_uid)
        for name, arr in old_cls.items():
            for pos, idx in enumerate(old_indices):
                merged_cls.setdefault(name, {})[idx] = arr[pos]

    for i, idx in enumerate(new_indices):  # new overrides same index
        merged_sem[idx] = _remap_semantic(new["semantic"][i], new_id_to_name, name_to_uid)
        for name, vol in new["class_vols"].items():
            merged_cls.setdefault(name, {})[idx] = vol[i]

    all_indices = sorted(merged_sem)
    h, w = new["semantic"].shape[1], new["semantic"].shape[2]
    zero = np.zeros((h, w), dtype=np.uint8)
    semantic = np.stack([merged_sem[i] for i in all_indices], axis=0).astype(np.uint8)
    class_vols = {
        name: np.stack([merged_cls.get(name, {}).get(i, zero) for i in all_indices], axis=0).astype(np.uint8)
        for name in unified_names
    }
    legend = [{"id": name_to_uid[n], "name": n, "color": color_by_name.get(n)} for n in unified_names]
    return {
        "semantic": semantic,
        "class_vols": class_vols,
        "slice_indices": all_indices,
        "legend": legend,
        "updated_indices": sorted(new_indices),
    }


def _read_existing_masks(container: Any) -> dict[str, Any] | None:
    """Read a prior masks container into ``merge_mask_volumes`` shape, or None."""
    try:
        meta = dict(container.metadata)
        legend = meta.get("legend") or meta.get("classes") or []
        indices = [int(i) for i in (meta.get("slice_indices") or [])]
        # "semantic" is now a registered multiscale node (mask_pyramid.py), not
        # a flat array — the merge logic always operates on native resolution,
        # never the downsampled viewer-only preview levels.
        semantic = mask_pyramid.read_mask_scale0(container, "semantic")
        safe_to_name = {_safe_name(e["name"]): e["name"] for e in legend}
        class_arrays: dict[str, np.ndarray] = {}
        for key in list(container):
            if key == "semantic":
                continue
            name = safe_to_name.get(key)
            if name is not None:
                class_arrays[name] = np.asarray(container[key][...])
        return {
            "slice_indices": indices,
            "semantic": semantic,
            "class_arrays": class_arrays,
            "legend": legend,
            "slice_updated_at": dict(meta.get("slice_updated_at") or {}),
        }
    except Exception as exc:  # noqa: BLE001 — unreadable/legacy → treat as fresh
        logger.warning("mask merge: could not read existing masks (%s) — replacing", exc)
        return None


def write_masks_to_tiled(
    source: str,
    server_uri: str | None,
    volumes: dict[str, Any],
    classes: list[Any],
    container_suffix: str = "",
) -> dict[str, Any]:
    """Merge stacked mask volumes into a ``<source_stem>__masks<container_suffix>``
    sibling container.

    Slices in this push overwrite the same index; previously-pushed slices are
    kept (merge). Metadata records ``updated_at`` and a per-slice
    ``slice_updated_at`` map plus ``last_updated_slices`` so the latest version of
    each slice is explicit. Returns ``{path, n_slices, updated, n_classes}``.

    ``container_suffix`` keeps independent producers of masks for the same
    source from silently merging into one blob: the manual "sync masks to
    Tiled" action and a dlsia run's "write to Tiled" (``infer_jobs.py``) both
    call this function, and without a suffix they'd write the exact same
    ``<stem>__masks`` container — a later push from one would merge onto
    (and, on overlapping slices, overwrite) the other's, making it impossible
    to keep both results around to compare, e.g. side-by-side in the 3-D
    viewer's two independent mask layers. Empty by default (the manual sync
    action's own container, unsuffixed, for backward compatibility with
    anything already pointing at ``<stem>__masks``).
    """
    api_key = api_key_for_uri(server_uri)
    client = get_tiled_client(server_uri, api_key)

    parts = [p for p in source.strip("/").split("/") if p]
    stem = parts[-1]
    parent: Any = client
    for part in parts[:-1]:
        parent = parent[part]

    container_key = f"{stem}__masks{container_suffix}"
    try:
        container: Any = parent[container_key]
    except KeyError:
        container = None

    existing = _read_existing_masks(container) if container is not None else None
    # H/W mismatch → can't merge; replace instead.
    if existing is not None and existing["semantic"].shape[1:] != volumes["semantic"].shape[1:]:
        logger.warning("mask merge: shape changed for %s — replacing existing masks", source)
        existing = None
    # slice_indices (container metadata) and the registered semantic array's own
    # slice count can disagree — e.g. a prior interrupted/partial write, or stale
    # metadata left over from before a fix landed — and merge_mask_volumes indexes
    # the array positionally by `enumerate(slice_indices)`, so a longer metadata
    # list than the array actually holds raises "index N is out of bounds for
    # axis 0" deep inside the merge. Same "can't trust it, replace" contract as
    # the H/W-mismatch guard above, rather than crashing the whole write.
    if existing is not None and existing["semantic"].shape[0] != len(existing["slice_indices"]):
        logger.warning(
            "mask merge: slice_indices (%d) != stored semantic slices (%d) for %s — replacing existing masks",
            len(existing["slice_indices"]), existing["semantic"].shape[0], source,
        )
        existing = None

    merged = merge_mask_volumes(existing, volumes)

    now_iso = datetime.now(timezone.utc).isoformat()
    slice_ts: dict[str, str] = dict((existing or {}).get("slice_updated_at", {})) if existing else {}
    for idx in merged["updated_indices"]:
        slice_ts[str(idx)] = now_iso

    container_meta = {
        "studio_type": "segmentation_masks",
        "source": source,
        "updated_at": now_iso,
        "n_slices": len(merged["slice_indices"]),
        "slice_indices": merged["slice_indices"],
        "slice_updated_at": slice_ts,
        "last_updated_slices": merged["updated_indices"],
        "classes": merged["legend"],
        "legend": merged["legend"],
    }

    if container is not None:
        # Clear prior arrays (external_only=False → also internally-managed data).
        container.delete_contents(recursive=True, external_only=False)
        container.update_metadata(container_meta)
    else:
        container = parent.create_container(key=container_key, metadata=container_meta)

    dims = ["slice", "y", "x"]
    # Registered as a real OME-NGFF multiscale node (mask_pyramid.py), not a
    # bare write_array — that's what lets the volume viewer's loadMask()
    # actually open this as a Zarr store instead of rejecting it for
    # "missing multiscales". Only `semantic` needs this: it's the one array
    # the viewer's single combined class-id mask texture reads.
    mask_pyramid.register_mask_pyramid(
        merged["semantic"], key="semantic", container=container, cache_key=container_key,
    )
    for name, vol in merged["class_vols"].items():
        container.write_array(
            vol, key=_safe_name(name), dims=dims,
            metadata={"studio_type": "segmentation_class", "class_name": name},
        )

    path = "/".join(parts[:-1] + [container_key])
    logger.info(
        "tiled_mask_sync: merged masks into %s (%d total, %d updated)",
        path, len(merged["slice_indices"]), len(merged["updated_indices"]),
    )
    return {
        "path": path,
        "n_slices": len(merged["slice_indices"]),
        "updated": len(merged["updated_indices"]),
        "n_classes": len(merged["class_vols"]),
    }


def run_mask_sync_job(jid: str, source_items: list[Any], payload: Any) -> None:
    """Background worker: rasterize each Tiled source's masks and write them back."""
    try:
        export_jobs.update(jid, state="running", phase="reading")
        tiled_items = [it for it in source_items if it.kind == "tiled"]
        total = 0
        for it in tiled_items:
            total += len({k for k, s in (it.slices or {}).items() if s} | {str(k) for k in (it.negative_slices or [])})
        export_jobs.set_total(jid, total)

        if not tiled_items:
            export_jobs.update(
                jid, state="done", phase="done",
                result={"written": [], "note": "skipped — no Tiled sources (masks only write back to Tiled)"},
            )
            export_jobs.log(jid, "No Tiled sources — nothing to write.")
            return

        written: list[dict[str, Any]] = []
        for item in tiled_items:
            export_jobs.log(jid, f"Rasterizing {item.source} …")
            node = arrays_mod.resolve_array(item.source, item.kind, item.server_uri)
            meta = arrays_mod.array_shape_meta(node)

            def _cb(message: str, _jid: str = jid) -> None:
                export_jobs.bump(_jid, 1)
                export_jobs.log(_jid, message)

            volumes = build_mask_volumes(item, payload.classes, meta, progress_cb=_cb)
            if volumes is None:
                export_jobs.log(jid, f"{item.source}: no annotated slices — skipped.")
                continue

            export_jobs.update(jid, phase="writing")
            info = write_masks_to_tiled(item.source, item.server_uri, volumes, payload.classes)
            written.append({
                "source": item.source,
                "container": info["path"],
                "n_slices": info["n_slices"],
                "updated": info["updated"],
                "n_classes": info["n_classes"],
            })
            export_jobs.log(
                jid,
                f"Wrote {info['path']} ({info['n_slices']} slices total, {info['updated']} updated).",
            )

        export_jobs.update(jid, state="done", phase="done", result={"written": written})
        export_jobs.log(jid, "Mask sync complete.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Mask sync job failed: %s", exc)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))
