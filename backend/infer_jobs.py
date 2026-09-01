"""Inference job orchestration for the Train tab.

Runs a saved fine-tuned dlsia TUNet run over requested slices, producing
editable polygon annotations, a colourised overlay preview, and an optional
push of the raw label maps into Tiled (reusing the existing mask-sync
writer). Progress/cancellation share the same :mod:`export_jobs` registry as
training and export jobs.

DINOv3 LoRA inference is out of scope here (see Phase 5.5) — only
``dlsia_tunet`` runs are supported; ``dlsia_denoiser`` runs are explicitly
refused (a denoiser is 1->1 regression, not classification).

Prediction label-map convention matches ``tiled_mask_sync``'s semantic masks
exactly (0 = no confident prediction/background, 1..n = predicted class index
+ 1) — NOT the training-time ``IGNORE_INDEX=255`` convention, which is a
different concept (unannotated ground truth, not a model's confidence gate).
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Any

import numpy as np
from fastapi import HTTPException
from PIL import Image as PILImage

import export_jobs
import train_common
from coco_export import _mask_to_polygons
from schemas import InferRequest

logger = logging.getLogger(__name__)

# job_id -> {"run_id", "classes", "source", "kind", "server_uri", "height",
#            "width", "label_pngs": {slice_idx: png_bytes}}
# Bounded to the newest few jobs — previews are only needed for the session
# that just ran inference, not forever.
_MAX_CACHED_JOBS = 4
_cache: dict[str, dict[str, Any]] = {}
_cache_order: list[str] = []
_cache_lock = threading.Lock()


def _cache_put(job_id: str, entry: dict[str, Any]) -> None:
    with _cache_lock:
        _cache[job_id] = entry
        _cache_order.append(job_id)
        while len(_cache_order) > _MAX_CACHED_JOBS:
            oldest = _cache_order.pop(0)
            _cache.pop(oldest, None)


def _cache_get(job_id: str) -> dict[str, Any] | None:
    with _cache_lock:
        return _cache.get(job_id)


def _hex_to_rgb(color: str | None) -> tuple[int, int, int]:
    if not color or not color.startswith("#"):
        return (255, 0, 0)
    hex_part = color[1:]
    if len(hex_part) in (3, 4):
        hex_part = "".join(c * 2 for c in hex_part[:3])
    try:
        return tuple(int(hex_part[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]  # noqa: E203
    except ValueError:
        return (255, 0, 0)


def _mask_to_polygons_padded(component: np.ndarray) -> list[list[float]]:
    """Like ``coco_export._mask_to_polygons``, but safe for a region that
    touches the array border.

    ``skimage.measure.find_contours`` only traces a transition *within* the
    array — a mask that's ``True`` all the way to an edge (e.g. a dominant
    class spanning the whole slice) has no such transition there, so it finds
    zero contours (confirmed: a full-frame True mask yields 0 contours,
    verified against skimage directly). Padding with a 1px False border
    guarantees a transition exists everywhere the shape meets the canvas edge;
    the offset is subtracted back out of the resulting coordinates.
    """
    padded = np.pad(component, pad_width=1, mode="constant", constant_values=False)
    polygons = _mask_to_polygons(padded)
    return [[v - 1 for v in flat] for flat in polygons]


def _enclosed_area(ring: np.ndarray) -> float:
    """Shoelace area enclosed by a closed (n, 2) ring, ignoring winding."""
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _vectorize_label_map(
    label_map: np.ndarray,
    run_classes: list[dict[str, Any]],
    min_area: int,
    simplify_tol: float,
    run_id: str,
    slice_idx: int,
) -> list[dict[str, Any]]:
    """Connected-component-vectorize a semantic label map (0=bg, 1..n=class) into
    polygon shape dicts, one per component per class.

    Holes matter here, not just cosmetically. A label map assigns each pixel to
    exactly one class, so a component that surrounds another class (a background
    region between grains, a ring around a bore) traces an outer contour *and*
    one contour per enclosed region. Emitting those inner contours as separate
    solid same-class polygons — as this used to — double-claims those pixels and
    makes the outer polygon paint straight over whichever class actually sits
    there: shapes are drawn in list order, so a later class's filled outer
    contour hides the earlier ones entirely (the label-map preview PNG never
    showed this, since it colours one class per pixel by construction).
    Encoding them as real ``holes`` keeps every pixel claimed exactly once, so
    the imported annotations match the preview regardless of draw order.
    """
    from skimage import measure

    shapes: list[dict[str, Any]] = []
    counter = 0
    for c, cls in enumerate(run_classes):
        binary = label_map == (c + 1)
        if not binary.any():
            continue
        labeled = measure.label(binary, connectivity=2)
        for region in measure.regionprops(labeled):
            if region.area < min_area:
                continue
            component = labeled == region.label
            rings: list[np.ndarray] = []
            for flat_points in _mask_to_polygons_padded(component):
                coords = np.asarray(flat_points, dtype=np.float64).reshape(-1, 2)
                if simplify_tol > 0:
                    coords = measure.approximate_polygon(coords, tolerance=simplify_tol)
                if len(coords) < 3:
                    continue
                rings.append(coords)
            if not rings:
                continue
            # A connected region has a single outer boundary, so the widest ring
            # is it and everything else it traced is enclosed by it.
            rings.sort(key=_enclosed_area, reverse=True)
            counter += 1
            shape: dict[str, Any] = {
                "id": f"pred_{run_id[:8]}_{slice_idx}_{counter}",
                "classId": cls["classId"],
                "kind": "polygon",
                "points": [round(float(v), 2) for v in rings[0].ravel().tolist()],
            }
            if len(rings) > 1:
                shape["holes"] = [
                    [round(float(v), 2) for v in ring.ravel().tolist()] for ring in rings[1:]
                ]
            shapes.append(shape)
    return shapes


def run_infer_job(jid: str, request: InferRequest) -> None:
    """Background worker: predict + vectorize each requested slice."""
    if not train_common.ML_LOCK.acquire(blocking=False):
        export_jobs.update(
            jid,
            state="error",
            phase="error",
            error="Another training or inference job is already running",
        )
        return
    try:
        export_jobs.update(jid, state="running", phase="loading")
        device = train_common.pick_device()
        if device is None:
            raise RuntimeError("torch is not installed on this server")

        config = train_common.load_run_config(request.run_id)
        adapter_state = train_common.load_adapter_state(request.run_id)
        run_classes: list[dict[str, Any]] = config["classes"]
        n_classes = len(run_classes)
        image_size = int(config["image_size"])
        render = request.render.model_dump() if request.render is not None else config["render"]
        # Predict with the geometry this run was TRAINED with — reading the flag off
        # the run (not the request) keeps runs saved before tiling existed on the
        # original whole-slice-rescale path, where their weights are valid.
        tiled = bool((config.get("hyperparams") or {}).get("tiling", False))
        # Input denoising is read off the RUN, never off the request — same rule
        # as `tiled` above. The model was trained on these exact pixels, so
        # letting a caller choose differently at predict time would be a silent
        # distribution shift: no error, just quietly worse predictions. Runs
        # saved before this field existed have no "denoise" key and get plain
        # render_slice, which is exactly what they were trained with.
        render_slice_fn = train_common.denoising_render_slice_fn(config.get("denoise"))
        if config.get("denoise"):
            export_jobs.log(
                jid,
                f"Applying the run's {config['denoise'].get('method')} input denoising "
                "(recorded at training time).",
            )
        if tiled:
            import tiling

            if not tiling.qlty_available():
                raise RuntimeError(
                    "This run was trained with tiling and needs the 'qlty' package to predict, "
                    "which is not installed on this server"
                )

        if config["model_family"] == "dlsia_tunet":
            import dlsia_runtime as fam

            if not train_common.dlsia_available():
                raise RuntimeError("dlsia is not installed on this server")
            model = fam.load_model(adapter_state, device)
            model.eval()
            forward_fn = fam.make_forward_fn(model)
            to_tensor_fn = fam.make_to_tensor_fn()
        elif config["model_family"] == "dlsia_denoiser":
            # A denoiser is a 1->1 regression model with no class channels, so
            # the label-map path below (softmax/argmax over n_classes, then
            # vectorising into shapes) is meaningless for it. Refuse clearly
            # instead of loading it through dlsia_runtime, which is what an
            # unconditional catch-all `else` would do — that would produce a
            # shape mismatch deep in the forward pass rather than an explanation.
            raise RuntimeError(
                "This is a denoiser run, not a segmentation model — it produces a denoised "
                "image, not labelled regions. Apply it from the Annotate tab's Denoise panel."
            )
        else:
            raise RuntimeError(f"Unsupported model family: {config['model_family']!r}")

        import torch
        import torch.nn.functional as F

        import arrays as arrays_mod
        import images as images_mod

        node = arrays_mod.resolve_array(request.source, request.kind, request.server_uri)
        meta = arrays_mod.array_shape_meta(node)
        h, w = meta["height"], meta["width"]
        global_range = images_mod._sample_global_stats(node, meta) if render.get("norm") == "global" else None

        export_jobs.set_total(jid, len(request.slice_indices))
        export_jobs.update(jid, phase="predicting")

        label_pngs: dict[int, bytes] = {}
        slices_result: dict[str, list[dict[str, Any]]] = {}
        n_shapes = 0
        cancelled = False
        logged_tiling = False

        with torch.no_grad():
            for slice_idx in request.slice_indices:
                if export_jobs.cancel_requested(jid):
                    cancelled = True
                    break

                arr = arrays_mod.read_slice(node, meta, slice_idx)
                rgb = render_slice_fn(arr, render, global_range)

                if tiled:
                    import tiling

                    label_map = tiling.predict_label_map_tiled(
                        rgb,
                        forward_fn=forward_fn,
                        to_tensor_fn=to_tensor_fn,
                        window=image_size,
                        min_confidence=request.min_confidence,
                        device=device,
                        cancel_cb=lambda: export_jobs.cancel_requested(jid),
                        # Logged once: every slice of a volume has the same shape, so
                        # the tiling geometry never changes between them. Repeating it
                        # per slice would just double an already per-slice log.
                        progress_cb=(lambda msg: export_jobs.log(jid, f"tiling: {msg}")) if not logged_tiling else None,
                    )
                    logged_tiling = True
                    if label_map is None:  # cancelled mid-slice
                        cancelled = True
                        break
                else:
                    img_l, _ = train_common.letterbox(rgb, np.zeros((h, w), dtype=np.uint8), image_size)
                    batch = to_tensor_fn(img_l).unsqueeze(0).to(device)

                    logits = forward_fn(batch)[0]
                    probs = F.softmax(logits, dim=0)
                    confidence, pred_class = probs.max(dim=0)
                    pred_np = pred_class.cpu().numpy()
                    conf_np = confidence.cpu().numpy()
                    label_letterboxed = np.where(conf_np >= request.min_confidence, pred_np + 1, 0).astype(np.uint8)
                    label_map = train_common.unletterbox(label_letterboxed, h, w, image_size)

                shapes = _vectorize_label_map(
                    label_map,
                    run_classes,
                    request.min_area,
                    request.simplify_tol,
                    request.run_id,
                    slice_idx,
                )
                n_shapes += len(shapes)
                slices_result[str(slice_idx)] = shapes

                buf = io.BytesIO()
                PILImage.fromarray(label_map, mode="L").save(buf, format="PNG")
                label_pngs[slice_idx] = buf.getvalue()

                export_jobs.bump(jid, 1)
                export_jobs.log(jid, f"slice {slice_idx}: {len(shapes)} region(s)")

        _cache_put(
            jid,
            {
                "run_id": request.run_id,
                "classes": run_classes,
                "source": request.source,
                "kind": request.kind,
                "server_uri": request.server_uri,
                "height": h,
                "width": w,
                "label_pngs": label_pngs,
            },
        )

        result = {
            "run_id": request.run_id,
            "classes": run_classes,
            "slices": slices_result,
            "n_shapes": n_shapes,
            "preview_slices": sorted(label_pngs.keys()),
            "cancelled": cancelled,
        }
        export_jobs.update(jid, state="done", phase="done", result=result)
        export_jobs.log(jid, "Inference cancelled; partial results kept." if cancelled else "Inference complete.")
    except Exception as exc:  # noqa: BLE001 — reported as a job error, never a crash
        logger.error("Inference job %s failed: %s", jid, exc)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))
    finally:
        train_common.ML_LOCK.release()


def preview_png(job_id: str, slice_index: int) -> bytes:
    """Colourise a cached predicted label map into an RGBA overlay PNG.

    Raises:
        HTTPException: 404 if the job or slice isn't cached.
    """
    entry = _cache_get(job_id)
    if entry is None:
        raise HTTPException(404, "No cached inference results for this job (they expire after a few jobs)")
    png_bytes = entry["label_pngs"].get(slice_index)
    if png_bytes is None:
        raise HTTPException(404, f"Slice {slice_index} was not part of this inference job")

    label = np.asarray(PILImage.open(io.BytesIO(png_bytes)))
    rgba = np.zeros((*label.shape, 4), dtype=np.uint8)
    for c, cls in enumerate(entry["classes"]):
        mask = label == (c + 1)
        if not mask.any():
            continue
        r, g, b = _hex_to_rgb(cls.get("color"))
        rgba[mask] = (r, g, b, 180)

    buf = io.BytesIO()
    PILImage.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def run_write_tiled_job(jid: str, infer_job_id: str) -> None:
    """Background worker: push a completed inference job's label maps into
    Tiled as a ``<source_stem>__masks`` sibling container (same writer the
    manual "sync masks to Tiled" flow uses)."""
    try:
        export_jobs.update(jid, state="running", phase="writing")
        entry = _cache_get(infer_job_id)
        if entry is None:
            raise ValueError("No cached inference results for this job (they expire after a few jobs)")
        if entry["kind"] != "tiled":
            raise ValueError("Inference source was not a Tiled array — nothing to write back")

        import tiled_mask_sync

        classes = entry["classes"]
        slice_indices = sorted(entry["label_pngs"].keys())
        semantic = np.stack(
            [np.asarray(PILImage.open(io.BytesIO(entry["label_pngs"][i]))) for i in slice_indices],
            axis=0,
        )
        class_vols = {cls["label"]: ((semantic == (c + 1)) * 255).astype(np.uint8) for c, cls in enumerate(classes)}
        legend = [{"id": c + 1, "name": cls["label"], "color": cls.get("color")} for c, cls in enumerate(classes)]
        volumes = {
            "semantic": semantic,
            "class_vols": class_vols,
            "slice_indices": slice_indices,
            "legend": legend,
        }

        export_jobs.set_total(jid, 1)
        # "_deep" keeps this in its own container, separate from whatever the
        # manual "sync masks to Tiled" action (iPred's fast results) has
        # written for the same source — letting both be loaded as independent
        # mask layers in the 3-D viewer instead of merging into one.
        info = tiled_mask_sync.write_masks_to_tiled(
            entry["source"], entry["server_uri"], volumes, classes, container_suffix="_deep",
        )
        export_jobs.bump(jid, 1)
        export_jobs.update(jid, state="done", phase="done", result=info)
        export_jobs.log(jid, f"Wrote predicted masks to {info['path']}.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Write-to-Tiled job %s failed: %s", jid, exc)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))
