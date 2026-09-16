"""Apply a denoise filter to a whole volume and save the result as a NEW Tiled dataset.

The display preview in Annotate (``GET /api/image/slice``'s ``denoise_*``
params) is deliberately non-destructive — it changes what you see, not the data,
and exports still use the original pixels. This job is the other half: it writes
a denoised copy as a first-class dataset you can open, annotate, train on and
export.

Written as a sibling dataset under the ingest root (``browse/<name>_denoised``),
NOT as a ``<stem>__masks``-style sidecar: anything matching
``sidecars.SIDECAR_SUFFIXES`` is deliberately hidden from Browse and from slice
enumeration, which is exactly wrong for an output the user needs to open.

Reuses ``ingest.py``'s write path (``validate_container_path`` /
``_ensure_container`` / the same metadata shape) so Browse describes, facets and
orders the result identically to an ingested dataset.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

import arrays as arrays_mod
import denoise as denoise_mod
import export_jobs
import ingest as ingest_mod
from tiled_clients import get_tiled_client

logger = logging.getLogger(__name__)

# Minimum zero-pad width for `image_number`, matching ingest's convention: the
# Tiled node KEY is the raw stem, so keys sort lexically — `image_number` is what
# makes numeric slice order recoverable, and it only works if it's padded.
_PAD_WIDTH = 4


def default_target_path(source: str, suffix: str = "denoised") -> str:
    """``browse/dataset`` -> ``browse/dataset_denoised``.

    Keeps the new dataset a sibling of its source so it appears right next to it
    in Browse.
    """
    # .strip("/") alone leaves a whitespace-only path intact, which would happily
    # produce "   _denoised" — strip whitespace first, and drop blank segments.
    parts = [p for p in (seg.strip() for seg in source.strip().strip("/").split("/")) if p]
    if not parts:
        raise ValueError("source path is empty")
    parts[-1] = f"{parts[-1]}_{suffix}"
    return "/".join(parts)


def _slice_key(index: int) -> str:
    return f"slice_{index:0{_PAD_WIDTH}d}"


def run_denoise_bake_job(jid: str, request: Any) -> None:
    """Background worker: denoise every slice and write a new Tiled dataset.

    ``request`` carries ``source``, ``server_uri``, ``method``, ``strength``,
    ``target_path`` and ``description`` (see ``schemas.DenoiseBakeRequest``).

    Deliberately does NOT take ``train_common.ML_LOCK``: classical denoising is
    CPU work with no GPU contention, so baking must not be blocked by — or
    block — a running training job.

    Per-slice failures are tolerated (logged, recorded in ``result.errors``,
    written as an unfiltered copy of the source slice so the output volume keeps
    a 1:1 slice correspondence with its source rather than silently shifting
    every later index).
    """
    model_denoiser = None  # bound before the try so `finally` can always see it
    try:
        export_jobs.update(jid, state="running", phase="preparing")

        method = request.method
        if method == "none":
            export_jobs.update(
                jid, state="error", phase="error",
                error="Pick a denoise method before saving a denoised copy.",
            )
            return
        # "model" applies a trained Noise2Noise/Noise2Void run rather than a
        # classical filter, so it is validated against the run registry instead
        # of the filter menu (and needs the GPU, see _ModelDenoiser).
        if method == "model":
            if not request.run_id:
                export_jobs.update(
                    jid, state="error", phase="error",
                    error="Applying a trained denoiser needs a run_id.",
                )
                return
        elif method not in denoise_mod.available_methods():
            export_jobs.update(
                jid, state="error", phase="error",
                error=f"Denoise method {method!r} is unavailable on this server.",
            )
            return

        target_path = request.target_path or default_target_path(request.source)
        try:
            target_parts = ingest_mod.validate_container_path(target_path)
        except ValueError as exc:
            export_jobs.update(jid, state="error", phase="error", error=str(exc))
            return

        client = get_tiled_client(request.server_uri)
        # Refuse rather than merge into an existing dataset: a half-overwritten
        # volume mixing two filters is far worse than a clear error.
        try:
            client[target_path]
        except KeyError:
            pass
        else:
            export_jobs.update(
                jid, state="error", phase="error",
                error=f"{target_path!r} already exists — delete it or choose another name.",
            )
            return

        node = arrays_mod.resolve_array(request.source, "tiled", request.server_uri)
        meta = arrays_mod.array_shape_meta(node)
        n_slices = int(meta["n_slices"])
        radius = 0 if method == "model" else denoise_mod.z_radius_for(method)

        # Held for the whole volume when applying a trained model: re-acquiring
        # per slice would let a training job interleave and thrash the GPU.
        model_denoiser = None
        if method == "model":
            try:
                model_denoiser = _ModelDenoiser(request.run_id, node, meta)
            except Exception as exc:  # noqa: BLE001 — surfaced as a job error
                export_jobs.update(jid, state="error", phase="error", error=str(exc))
                return

        export_jobs.set_total(jid, n_slices)
        export_jobs.update(jid, phase="denoising")

        container = ingest_mod._ensure_container(client, target_parts)
        now_iso = datetime.now(timezone.utc).isoformat()
        sample_name = target_parts[-1]
        keywords = ingest_mod.parse_keywords(request.description or "")

        errors: list[dict[str, Any]] = []
        written = 0
        cancelled = False

        for index in range(n_slices):
            if export_jobs.cancel_requested(jid):
                cancelled = True
                break
            try:
                if model_denoiser is not None:
                    out = model_denoiser.denoise(index)
                else:
                    out = _denoise_one(node, meta, index, method, request.strength, radius, n_slices)
            except Exception as exc:  # noqa: BLE001 — one bad slice must not abort the volume
                logger.warning("denoise bake: slice %d failed (%s); copying source", index, exc)
                errors.append({"slice": index, "error": str(exc)})
                try:
                    out = np.asarray(arrays_mod.read_slice(node, meta, index))
                except Exception as read_exc:  # noqa: BLE001 — truly unreadable
                    logger.warning("denoise bake: slice %d unreadable (%s); skipping", index, read_exc)
                    errors[-1]["error"] = f"{exc}; source also unreadable: {read_exc}"
                    export_jobs.bump(jid, 1)
                    continue

            key = _slice_key(index)
            container.write_array(
                out,
                key=key,
                dims=["y", "x"] if out.ndim == 2 else None,
                metadata={
                    "image_number": str(index).zfill(_PAD_WIDTH),
                    "size": ingest_mod._size_str(out),
                    "sample_name": sample_name,
                    # Provenance: enough to reproduce this dataset exactly.
                    "denoise_source": request.source,
                    "denoise_method": method,
                    "denoise_strength": float(request.strength),
                    **({"description": request.description} if request.description else {}),
                    **({"keywords": keywords} if keywords else {}),
                },
            )
            written += 1
            export_jobs.bump(jid, 1)
            export_jobs.log(jid, f"slice {index}: denoised")

        if model_denoiser is not None:
            model_denoiser.close()
            model_denoiser = None

        export_jobs.update(jid, phase="finalizing")
        container.update_metadata(metadata={
            "sample_name": sample_name,
            # Browse derives n_slices from actual children; n_images is the
            # display/facet value and must match what was really written.
            "n_images": written,
            "denoise_source": request.source,
            "denoise_method": method,
            "denoise_strength": float(request.strength),
            "denoise_created_at": now_iso,
            **({"description": request.description} if request.description else {}),
            **({"keywords": keywords} if keywords else {}),
        })

        if written == 0:
            export_jobs.update(
                jid, state="error", phase="error",
                error="No slices could be denoised — nothing was written.",
            )
            return

        result = {
            "path": target_path,
            "n_slices": written,
            "method": method,
            "strength": float(request.strength),
            "cancelled": cancelled,
            "errors": errors,
        }
        export_jobs.update(jid, state="done", phase="done", result=result)
        export_jobs.log(
            jid,
            f"Denoise bake {'cancelled after' if cancelled else 'complete —'} "
            f"{written} slice(s) written to {target_path}.",
        )
    except Exception as exc:  # noqa: BLE001 — reported as a job error, never a crash
        logger.error("Denoise bake job %s failed: %s", jid, exc)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))
    finally:
        # A model bake holds ML_LOCK for the whole volume. Releasing only on the
        # happy path would leave it held forever after any mid-bake failure,
        # deadlocking every later training job and preview.
        if model_denoiser is not None:
            model_denoiser.close()


class _ModelDenoiser:
    """Applies a trained Noise2Noise/Noise2Void run across a whole volume.

    Loads the model and takes ``ML_LOCK`` ONCE for the entire bake rather than
    per slice: re-acquiring per slice would let a training job interleave, and
    reloading weights per slice would dominate the runtime.

    Preprocessing is delegated to ``denoise_train``'s own helpers, so the
    network sees exactly the display-mapped uint8 grayscale it was trained on
    (volume-global bounds, the run's saved render options). Reimplementing that
    here is how the two halves of the contract would silently drift apart.

    Output dtype differs from the classical filters on purpose: a denoiser
    returns continuous values in ``[0, 1]``, so this writes **uint8** (the scale
    it actually operated in) rather than pretending to the source's uint16
    precision it never had access to.
    """

    def __init__(self, run_id: str, node: Any, meta: dict[str, Any]) -> None:
        import denoise_train
        import tiling
        import train_common

        config = train_common.load_run_config(run_id)
        if config.get("model_family") != "dlsia_denoiser" or config.get("task") != "denoising":
            raise ValueError(f"Run {run_id!r} is not a denoiser run.")
        # Which network this run is; defaults to TUNet for runs saved before the
        # field existed. Only the TUNet architecture needs dlsia.
        fam = train_common.denoiser_runtime_for(config)
        if train_common.denoiser_needs_dlsia(config) and not train_common.dlsia_available():
            raise ValueError("dlsia is not installed on this server.")
        if not tiling.qlty_available():
            raise ValueError("Applying a denoiser needs the 'qlty' package, which is missing.")
        device = train_common.pick_device()
        if device is None:
            raise ValueError("torch is not installed on this server.")

        self._tiling = tiling
        self._train_common = train_common
        self._denoise_train = denoise_train
        self._node = node
        self._meta = meta
        self._device = device
        self._window = int(config["image_size"])
        self._opts = denoise_train._render_opts(config.get("render") or {})

        import images as images_mod

        self._global_range = images_mod._sample_global_stats(node, meta)

        if not train_common.ML_LOCK.acquire(blocking=False):
            raise ValueError("The device is busy with another job — try again when it finishes.")
        self._locked = True
        try:
            state = train_common.load_adapter_state(run_id)
            model = fam.load_model(state, device)
            model.eval()
            self._forward_fn = fam.make_forward_fn(model)
            self._to_tensor_fn = fam.make_to_tensor_fn()
        except Exception:
            self.close()  # never hold the lock past a failed load
            raise

    def denoise(self, index: int) -> np.ndarray:
        import torch

        gray = self._denoise_train._slice_to_gray_uint8(
            self._node, self._meta, index, self._opts, self._global_range
        )
        with torch.no_grad():
            out = self._tiling.denoise_image_tiled(
                gray,
                forward_fn=self._forward_fn,
                to_tensor_fn=self._to_tensor_fn,
                window=self._window,
                device=self._device,
            )
        if out is None:
            raise RuntimeError("denoising returned no result")
        unit = np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)
        return (unit * 255.0).round().astype(np.uint8)

    def close(self) -> None:
        """Release ``ML_LOCK``. Idempotent — callers release on both the happy
        path and in a ``finally``."""
        if getattr(self, "_locked", False):
            self._train_common.ML_LOCK.release()
            self._locked = False


def _denoise_one(
    node: Any,
    meta: dict[str, Any],
    index: int,
    method: str,
    strength: float,
    radius: int,
    n_slices: int,
) -> np.ndarray:
    """Denoise slice *index*, reading a z-window when the method needs one.

    The window clamps at the volume ends, so the target slice is not always the
    centre of what was read — its position is tracked explicitly.
    """
    if radius == 0:
        return denoise_mod.denoise_slice(
            np.asarray(arrays_mod.read_slice(node, meta, index)), method, strength
        )

    lo = max(0, index - radius)
    hi = min(n_slices - 1, index + radius)
    frames = [np.asarray(arrays_mod.read_slice(node, meta, i)) for i in range(lo, hi + 1)]
    if len(frames) < 2:
        fallback = "gaussian" if method == "gaussian3d" else "median"
        return denoise_mod.denoise_slice(frames[0], fallback, strength)
    return denoise_mod.denoise_stack(np.stack(frames, axis=0), method, strength)[index - lo]
