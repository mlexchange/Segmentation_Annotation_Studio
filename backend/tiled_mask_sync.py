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
from coco_export import _safe_name, shape_to_mask
from tiled_clients import get_tiled_client

logger = logging.getLogger(__name__)


class ExistingMasksUnreadable(RuntimeError):
    """Raised when an existing mask set cannot be read safely for a merge."""


def _validate_safe_class_names(names: list[str]) -> None:
    """Reject distinct class names that collapse to the same Tiled node key."""
    by_key: dict[str, str] = {}
    for name in names:
        safe = _safe_name(name)
        previous = by_key.get(safe)
        if previous is not None and previous != name:
            raise ValueError(
                f"class names {previous!r} and {name!r} map to the same Tiled key {safe!r}"
            )
        by_key[safe] = name


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
    for hard negatives), in sorted numeric order.
    """
    h, w = int(meta["height"]), int(meta["width"])
    cat_id_map, cat_id_to_name, legend = _category_maps(classes)

    slices: dict[str, list[dict[str, Any]]] = item.slices or {}
    neg = {str(k) for k in (item.negative_slices or [])}
    keys = sorted(
        {k for k, shapes in slices.items() if shapes} | neg,
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
        for shape in slices.get(key, []):
            shape_dict = shape if isinstance(shape, dict) else shape.model_dump()
            mask = shape_to_mask(shape_dict, h, w)
            if float(mask.sum()) < 1:
                continue
            cat_id = cat_id_map.get(int(shape_dict.get("classId", 1)), 1)
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


def _resolve_masks_parent(client: Any, source: str) -> tuple[Any, str]:
    """Walk to ``source``'s parent container and the path its ``<stem>__masks``
    sibling would have. Shared by ``write_masks_to_tiled`` (which also needs
    the parent itself, to create the container on a first write) and
    ``_masks_container`` (which doesn't) — so every caller derives the exact
    same location from a source string, never a slightly different one.
    """
    parts = [p for p in source.strip("/").split("/") if p]
    stem = parts[-1]
    parent: Any = client
    for part in parts[:-1]:
        parent = parent[part]
    return parent, "/".join(parts[:-1] + [f"{stem}__masks"])


def _masks_container(client: Any, source: str) -> tuple[Any | None, str]:
    """Resolve the ``<stem>__masks`` sibling container next to ``source``.

    Returns ``(container, path)`` if it exists, or ``(None, path)`` if it
    doesn't — ``path`` is always returned so callers can report it either way.
    """
    parent, path = _resolve_masks_parent(client, source)
    container_key = path.rsplit("/", 1)[-1]
    try:
        return parent[container_key], path
    except KeyError:
        return None, path


def _read_existing_masks(container: Any) -> dict[str, Any] | None:
    """Read prior masks, failing closed if any expected data is unreadable."""
    try:
        meta = dict(container.metadata)
        legend = meta.get("legend") or meta.get("classes") or []
        _validate_safe_class_names([str(entry["name"]) for entry in legend])
        indices = [int(i) for i in (meta.get("slice_indices") or [])]
        semantic = np.asarray(container["semantic"][...])
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
            "metadata": meta,
        }
    except ExistingMasksUnreadable:
        raise
    except Exception as exc:  # noqa: BLE001 - preserving existing data takes priority
        raise ExistingMasksUnreadable(f"could not read existing masks: {exc}") from exc


def _write_mask_arrays(container: Any, volumes: dict[str, Any]) -> None:
    """Write one complete semantic/per-class mask set to an empty container."""
    dims = ["slice", "y", "x"]
    container.write_array(
        volumes["semantic"],
        key="semantic",
        dims=dims,
        metadata={"studio_type": "segmentation_semantic"},
    )
    for name, volume in volumes["class_vols"].items():
        container.write_array(
            volume,
            key=_safe_name(name),
            dims=dims,
            metadata={"studio_type": "segmentation_class", "class_name": name},
        )


def _restore_existing_masks(container: Any, existing: dict[str, Any]) -> None:
    """Best-effort rollback of a prior mask snapshot after a failed replacement."""
    container.delete_contents(recursive=True, external_only=False)
    container.update_metadata(metadata=existing["metadata"])
    prior = {
        "semantic": existing["semantic"],
        "class_vols": existing["class_arrays"],
    }
    _write_mask_arrays(container, prior)


def write_masks_to_tiled(
    source: str,
    server_uri: str | None,
    volumes: dict[str, Any],
    classes: list[Any],
) -> dict[str, Any]:
    """Merge stacked mask volumes into a ``<source_stem>__masks`` sibling container.

    Slices in this push overwrite the same index; previously-pushed slices are
    kept (merge). Metadata records ``updated_at`` and a per-slice
    ``slice_updated_at`` map plus ``last_updated_slices`` so the latest version of
    each slice is explicit. Returns ``{path, n_slices, updated, n_classes}``.
    """
    client = get_tiled_client(server_uri)
    parent, path = _resolve_masks_parent(client, source)
    container_key = path.rsplit("/", 1)[-1]
    try:
        container: Any = parent[container_key]
    except KeyError:
        container = None

    existing = _read_existing_masks(container) if container is not None else None
    # A dimension change requires an explicit migration/new target. Replacing here
    # would discard every previously annotated slice without a recoverable merge.
    if existing is not None and existing["semantic"].shape[1:] != volumes["semantic"].shape[1:]:
        raise ValueError("mask dimensions changed; refusing to replace existing masks")

    merged = merge_mask_volumes(existing, volumes)
    _validate_safe_class_names(list(merged["class_vols"]))

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

    created = False
    try:
        if container is not None:
            # Keep a complete in-memory snapshot above, then clear and publish. If
            # any write fails, the except block restores that prior snapshot.
            container.delete_contents(recursive=True, external_only=False)
            container.update_metadata(metadata=container_meta)
        else:
            container = parent.create_container(key=container_key, metadata=container_meta)
            created = True
        _write_mask_arrays(container, merged)
    except Exception as write_exc:  # noqa: BLE001 - rollback must cover any client error
        if existing is not None and container is not None:
            try:
                _restore_existing_masks(container, existing)
            except Exception as rollback_exc:  # noqa: BLE001
                logger.exception("mask rollback failed for %s", source)
                raise RuntimeError(
                    f"mask write failed and rollback also failed for {source}"
                ) from rollback_exc
        elif created:
            try:
                parent.delete_contents(container_key, recursive=True, external_only=False)
            except Exception:  # noqa: BLE001 - preserve the original write error
                logger.exception("could not remove partial new mask container %s", container_key)
        raise write_exc

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


def read_masks_summary(source: str, server_uri: str | None) -> dict[str, Any]:
    """Metadata-only probe for whether ``<stem>__masks`` exists for ``source``.

    Powers the "Load saved masks" panel's disabled/enabled state without
    launching a job destined to fail on a sample with nothing saved. Never
    reads the ``semantic`` array itself — container metadata only.

    Returns ``{"available": False}`` (optionally with an ``"error"`` string if
    the check itself failed unexpectedly — a probe must never raise) or
    ``{"available": True, "path", "n_slices", "slice_indices", "updated_at",
    "classes"}``.
    """
    try:
        client = get_tiled_client(server_uri)
        container, path = _masks_container(client, source)
        if container is None:
            return {"available": False}
        meta = dict(container.metadata)
        legend = meta.get("legend") or meta.get("classes") or []
        slice_indices = [int(i) for i in (meta.get("slice_indices") or [])]
        return {
            "available": True,
            "path": path,
            "n_slices": int(meta.get("n_slices") or len(slice_indices)),
            "slice_indices": slice_indices,
            "updated_at": meta.get("updated_at"),
            "classes": legend,
        }
    except Exception as exc:  # noqa: BLE001 - a probe must never crash the caller
        logger.warning("read_masks_summary failed for %s: %s", source, exc)
        return {"available": False, "error": str(exc)}


def _legend_lut(legend: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build the ``run_classes`` list ``infer_jobs._vectorize_label_map``
    expects, plus a 256-entry lookup table remapping a stored semantic pixel
    value to that list's positional convention.

    ``_vectorize_label_map`` matches classes POSITIONALLY — ``label_map == i +
    1`` for the i-th entry of ``run_classes`` — but the semantic volume stores
    each pixel's LEGEND id, which is only contiguous-from-1 by the convention
    ``_category_maps``/``merge_mask_volumes`` currently happen to follow, not
    by any enforced invariant. Routing every stored frame through this LUT
    before vectorizing makes read-back correct even if that ever changes
    (e.g. a class removed from the middle of the legend across several
    merges), for the cost of one array-index lookup per frame.
    """
    ordered = sorted(legend, key=lambda e: int(e["id"]))
    run_classes = [
        {"classId": int(e["id"]), "label": e["name"], "color": e.get("color")}
        for e in ordered
    ]
    lut = np.zeros(256, dtype=np.uint8)
    for position, entry in enumerate(ordered):
        stored_id = int(entry["id"])
        if 0 <= stored_id < 256:
            lut[stored_id] = position + 1
    return run_classes, lut


def run_masks_readback_job(jid: str, request: Any) -> None:
    """Background worker: vectorize a saved ``<stem>__masks`` semantic volume
    back into shape dicts, in the SAME result shape ``infer_jobs.run_infer_job``
    produces (``classes``/``slices``/``n_shapes``), so the frontend's existing
    "Import as annotations" machinery works completely unchanged regardless of
    whether the shapes came from a live model run or a previously-saved mask.

    No ``train_common.ML_LOCK`` — this is pure I/O + skimage, not a model
    forward pass, and must not contend with (or wait behind) an actual
    training/inference job for the device.

    Reads the semantic volume PER FRAME (``sem_node[pos]``), never the whole
    volume at once: a 38x3232x3232 uint8 semantic volume is ~397MB whole vs
    ~10MB for a single frame — the same principle ``volumes.py``'s strided
    reads follow for the raw 3-D volume endpoint.
    """
    # Lazy, mirroring infer_jobs.run_write_tiled_job's own lazy import of this
    # module — avoids a module-level import cycle between the two.
    from infer_jobs import _vectorize_label_map

    try:
        export_jobs.update(jid, state="running", phase="reading")
        client = get_tiled_client(request.server_uri)
        container, path = _masks_container(client, request.source)
        if container is None:
            export_jobs.update(
                jid, state="error", phase="error",
                error="No saved masks found for this sample.",
            )
            return

        meta = dict(container.metadata)
        legend = meta.get("legend") or meta.get("classes") or []
        if not legend:
            export_jobs.update(
                jid, state="error", phase="error",
                error="Saved masks have no class legend — cannot map classes.",
            )
            return

        slice_indices = [int(i) for i in (meta.get("slice_indices") or [])]
        if not slice_indices:
            export_jobs.update(
                jid, state="error", phase="error",
                error="Saved masks have no recorded slices.",
            )
            return

        try:
            sem_node = container["semantic"]
        except KeyError:
            export_jobs.update(
                jid, state="error", phase="error",
                error="Saved masks are missing the semantic volume.",
            )
            return

        sem_shape = tuple(int(d) for d in sem_node.shape)
        if sem_shape[0] != len(slice_indices):
            export_jobs.update(
                jid, state="error", phase="error",
                error="Saved masks are inconsistent — re-write them from the Train tab.",
            )
            return

        run_classes, lut = _legend_lut(legend)
        # Only feeds the emitted shapes' id prefix (infer_jobs.py's own
        # convention); the frontend re-uuids everything on import regardless.
        run_id = f"stored{abs(hash(path)) % 10**8:08d}"

        export_jobs.set_total(jid, len(slice_indices))
        export_jobs.update(jid, phase="vectorizing")

        slices_result: dict[str, list[dict[str, Any]]] = {}
        errors: list[dict[str, Any]] = []
        n_shapes = 0
        cancelled = False

        for pos, real_idx in enumerate(slice_indices):
            if export_jobs.cancel_requested(jid):
                cancelled = True
                break
            try:
                frame = lut[np.asarray(sem_node[pos])]
                shapes = _vectorize_label_map(
                    frame, run_classes, request.min_area, request.simplify_tol, run_id, real_idx,
                )
                slices_result[str(real_idx)] = shapes
                n_shapes += len(shapes)
                export_jobs.log(jid, f"slice {real_idx}: {len(shapes)} region(s)")
            except Exception as exc:  # noqa: BLE001 - one bad frame must not abort the rest
                logger.warning("masks readback: skipping unreadable slice %d: %s", real_idx, exc)
                errors.append({"slice": real_idx, "error": str(exc)})
            export_jobs.bump(jid, 1)

        if not slices_result and errors:
            export_jobs.update(
                jid, state="error", phase="error",
                error="Could not read any saved mask slice.",
            )
            return

        result = {
            "classes": run_classes,
            "slices": slices_result,
            "n_shapes": n_shapes,
            "cancelled": cancelled,
            "errors": errors,
            "path": path,
        }
        export_jobs.update(jid, state="done", phase="done", result=result)
        export_jobs.log(jid, "Masks readback cancelled; partial results kept." if cancelled else "Masks loaded.")
    except Exception as exc:  # noqa: BLE001 - reported as a job error, never a crash
        logger.error("Masks readback job %s failed: %s", jid, exc)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))


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
