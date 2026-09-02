"""Shared plumbing for the Train tab.

Two model families share this module's data prep, generic training loop, and
run persistence:

* ``"dlsia_tunet"`` — dlsia's tunable U-Net trained from scratch, no
  pretrained checkpoint (see :mod:`dlsia_runtime`). Needs ``torch`` + the
  optional ``dlsia`` package.
* ``"dlsia_denoiser"`` — a self-supervised single-channel denoiser, either a
  dlsia TUNet (:mod:`denoise_runtime`) or a plain convolutional autoencoder
  (:mod:`autoencoder_runtime`).

Both are optional dependencies (``backend/pyproject.toml``'s ``ml`` extra) —
this module is import-safe without torch installed; only the functions that
actually need it import it lazily and are guarded by :func:`torch_available`.

DINOv3 LoRA fine-tuning is deliberately NOT supported here — see Phase 5.5 in
the integration plan. It needs vendored, Meta-licensed (research/non-commercial)
model code and checkpoint provisioning this repo doesn't have a story for yet.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
from fastapi import HTTPException
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# One fine-tune (or inference) job at a time — they contend for the same GPU
# memory, and neither this hand-rolled loop nor dlsia's TUNet is written to be
# safely reentrant across concurrent callers.
import threading  # noqa: E402

ML_LOCK = threading.Lock()

# Separate from ML_LOCK: ML_LOCK gates whole JOBS against each other (a
# training run, an inference run, a denoise bake, a batch probe — all four
# hold it for their entire duration so no two heavy ML jobs contend for the
# GPU at once). GPU_FORWARD_LOCK is finer-grained — it serializes only the
# actual model forward call, so a per-job worker pool (see infer_jobs.py's
# per-slice pool) can run I/O, preprocessing, and vectorization for multiple
# slices concurrently while still guaranteeing only one thread ever calls
# into the model at a given instant. Do not use this in place of ML_LOCK —
# it does not protect against two different JOBS running at once, only
# against two threads calling the model at the same time within one job.
GPU_FORWARD_LOCK = threading.Lock()

IGNORE_INDEX = 255  # unannotated pixels — see coco_export.build_export_plan(lightly=True)


def torch_available() -> bool:
    """True if ``torch`` is importable, without paying the import cost."""
    return importlib.util.find_spec("torch") is not None


def dlsia_available() -> bool:
    """True if ``dlsia`` is importable, without paying the import cost."""
    return importlib.util.find_spec("dlsia") is not None


def pick_device() -> str | None:
    """Return ``"mps"|"cuda"|"cpu"``, or ``None`` if torch is unavailable.

    ``TRAIN_DEVICE`` env var overrides auto-detection (e.g. to force ``cpu``
    on a machine where MPS is flaky for a particular op).
    """
    if not torch_available():
        return None
    override = (os.getenv("TRAIN_DEVICE") or "").strip().lower()
    if override in {"mps", "cuda", "cpu"}:
        return override
    import torch  # noqa: PLC0415 — optional dependency, imported lazily

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def runs_dir() -> Path:
    """Server-owned directory holding saved fine-tune runs (both families)."""
    configured = (os.getenv("DINO_RUNS_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    local_root = Path(os.getenv("LOCAL_DATA_ROOT", "~/data")).expanduser().resolve()
    return (local_root / "models" / "runs").resolve()


def _validate_run_id(run_id: str) -> str:
    """Reject anything that isn't a single safe path component."""
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id or "\x00" in run_id:
        raise HTTPException(400, "Invalid run_id")
    return run_id


def run_dir(run_id: str) -> Path:
    """Resolved, containment-checked directory for one run."""
    base = runs_dir()
    candidate = (base / _validate_run_id(run_id)).resolve()
    if not candidate.is_relative_to(base):
        raise HTTPException(400, "Invalid run_id")
    return candidate


# ---------------------------------------------------------------------------
# Run persistence (shared shape across both families)
# ---------------------------------------------------------------------------


def save_run(
    run_id: str,
    *,
    model_family: str,
    model_config: dict[str, Any],
    classes: list[dict[str, Any]],
    render: dict[str, Any],
    image_size: int,
    hyperparams: dict[str, Any],
    source_keys: list[str],
    adapter_state: dict[str, Any],
    metrics: dict[str, Any],
    resumed_from: str | None = None,
    task: str = "segmentation",
    denoise: dict[str, Any] | None = None,
) -> None:
    """Persist one run's config + adapter/weights + metrics to ``runs_dir()``.

    ``resumed_from`` records the run this one continued fine-tuning from, so a
    chain of successive refinements stays traceable (a resume always writes a
    NEW run — the parent is never modified).

    ``task`` distinguishes a segmentation run from a self-supervised denoiser
    run (see ``schemas.TrainRequest.task``); defaults to ``"segmentation"``
    for callers that predate the denoiser family.

    ``denoise`` records any denoising applied to the model's INPUT pixels, so
    inference can reapply exactly the same preprocessing off the run instead of
    trusting the caller to remember it (see :func:`denoising_render_slice_fn`).
    """
    import torch  # noqa: PLC0415

    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": run_id,
        "model_family": model_family,
        "task": task,
        "model_config": model_config,
        "classes": classes,
        "render": render,
        "denoise": denoise,
        "image_size": image_size,
        "hyperparams": hyperparams,
        "source_keys": source_keys,
        "resumed_from": resumed_from,
        "created_at": _now_iso(),
    }
    (d / "config.json").write_text(json.dumps(config, indent=2))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(adapter_state, d / "adapter.pt")


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def list_runs() -> list[dict[str, Any]]:
    """List saved runs, newest first. Malformed run directories are skipped."""
    base = runs_dir()
    if not base.exists():
        return []
    results: list[dict[str, Any]] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        try:
            config = json.loads((d / "config.json").read_text())
            # Runs saved before `task` existed have no such key on disk — treat
            # that as "segmentation", the only thing this app trained before.
            config.setdefault("task", "segmentation")
            metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
            results.append({**config, "metrics": metrics})
        except Exception as exc:  # noqa: BLE001 — one bad run dir must not break the list
            logger.warning("Skipping malformed run directory %s: %s", d, exc)
            continue
    results.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return results


def delete_run(run_id: str) -> None:
    """Permanently remove a saved run's directory (config, metrics, weights)."""
    d = run_dir(run_id)
    if not d.is_dir():
        raise HTTPException(404, f"Unknown run: {run_id!r}")
    shutil.rmtree(d)


def load_run_config(run_id: str) -> dict[str, Any]:
    """Return the parsed ``config.json`` for *run_id*, or raise 404.

    ``task`` is a field added after this app already had saved runs on disk —
    an absent key means the run predates the field, and every run trained
    before it existed was a segmentation run, so that is the default filled
    in here rather than leaving callers (e.g. ``check_resume_compatible``) to
    each re-derive the same backward-compat fallback.
    """
    d = run_dir(run_id)
    config_path = d / "config.json"
    if not config_path.exists():
        raise HTTPException(404, f"Unknown run: {run_id!r}")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(500, "Run config is corrupt") from exc
    config.setdefault("task", "segmentation")
    return config


def load_adapter_state(run_id: str) -> dict[str, Any]:
    """Load the saved ``adapter.pt`` state dict for *run_id* onto the CPU."""
    import torch  # noqa: PLC0415

    d = run_dir(run_id)
    adapter_path = d / "adapter.pt"
    if not adapter_path.exists():
        raise HTTPException(404, f"No saved weights for run: {run_id!r}")
    return torch.load(adapter_path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Training-data preparation (shared: reuses the Lightly export path)
# ---------------------------------------------------------------------------


def denoising_render_slice_fn(denoise: Any) -> Callable[..., np.ndarray]:
    """A ``render_slice``-compatible callable that denoises the RAW slice first.

    The single place model-input denoising is applied, used by BOTH training
    (:func:`prepare_datasets`) and inference (``infer_jobs``). Keeping one
    implementation is the point: a model must see the same pixel distribution at
    predict time that it saw during training, and two copies of this would be
    free to drift apart silently — the failure mode is not an error, just
    quietly worse predictions.

    Denoising runs on raw intensity units, before ``render_slice`` normalises,
    matching where the Annotate preview applies it.

    Args:
        denoise: Anything with ``.method``/``.strength`` (a
            :class:`schemas.DenoiseTrainOpts`) or the equivalent dict as
            reloaded from a run's ``config.json``. Falsy, or a ``"none"``
            method, yields plain ``images.render_slice``.
    """
    import images as images_mod  # noqa: PLC0415

    method, strength = _denoise_params(denoise)
    if method is None:
        return images_mod.render_slice

    import denoise as denoise_mod  # noqa: PLC0415

    def _render(arr: np.ndarray, opts: dict[str, Any], global_range: Any = None) -> np.ndarray:
        # Colour sources are left alone: the classical filters are 2-D
        # grayscale, and render_slice early-returns for RGB anyway.
        if arr.ndim == 2:
            arr = denoise_mod.denoise_slice(arr, method, strength)
        return images_mod.render_slice(arr, opts, global_range)

    return _render


def _denoise_params(denoise: Any) -> tuple[str | None, float]:
    """Normalise a DenoiseTrainOpts / dict / None into ``(method, strength)``.

    Returns ``(None, ...)`` when no denoising should be applied — including for
    ``"model"``, which would need its own run and a GPU pass per slice and is
    deliberately not supported as a preprocessor for another model.
    """
    if not denoise:
        return None, 0.5
    if hasattr(denoise, "method"):
        method, strength = denoise.method, float(denoise.strength)
    else:
        method, strength = denoise.get("method"), float(denoise.get("strength", 0.5))
    if not method or method in ("none", "model"):
        return None, strength
    return method, strength


def prepare_datasets(
    sources: list[Any],
    classes: list[Any],
    render: Any,
    auto_split: dict[str, Any],
    progress_cb: Callable[[str], None] | None = None,
    denoise: Any = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Render + rasterize every annotated source into in-memory train/val pairs.

    Reuses ``coco_export.build_export_plan(lightly=True)`` per source — the
    same in-memory rendering/rasterization the Lightly export already does —
    so no disk export is needed just to assemble training tensors. Valid +
    test folds into val (same convention as the Lightly export).

    Returns:
        ``{"train": [(rgb_uint8_hwc, label_uint8_hw), ...], "val": [...]}``.
    """
    import arrays as arrays_mod
    import images as images_mod
    from coco_export import build_export_plan
    from schemas import ExportRequest

    train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    val_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for item in sources:
        node = arrays_mod.resolve_array(item.source, item.kind, item.server_uri)
        shim = ExportRequest(
            kind=item.kind,
            source=item.source,
            server_uri=item.server_uri,
            slices=item.slices,
            split_by_slice=item.split_by_slice,
            negative_slices=item.negative_slices,
            classes=classes,
            render=render,
            auto_split=auto_split,
        )
        plan = build_export_plan(
            node,
            shim,
            render_slice_fn=denoising_render_slice_fn(denoise),
            array_shape_meta_fn=arrays_mod.array_shape_meta,
            read_slice_fn=arrays_mod.read_slice,
            sample_global_stats_fn=images_mod._sample_global_stats,
            progress_cb=progress_cb,
            lightly=True,
        )
        for split_name, split_data in plan["splits"].items():
            bucket = train_pairs if split_name == "train" else val_pairs
            for img in split_data["images"]:
                rgb = np.asarray(PILImage.open(io.BytesIO(img["png_bytes"])).convert("RGB"))
                label = np.asarray(PILImage.open(io.BytesIO(img["label_png_bytes"])))
                bucket.append((rgb, label))

    return {"train": train_pairs, "val": val_pairs}


def letterbox(image: np.ndarray, label: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize-keep-aspect + pad *image* (uint8 HWC) and *label* (uint8 HW) to
    a ``size`` x ``size`` square. Label padding uses :data:`IGNORE_INDEX` so
    padded pixels never contribute to the loss.

    Also returns enough to invert the transform (see :func:`unletterbox`),
    packed into the returned label's dtype-preserving companion — callers that
    need to invert should use :func:`letterbox_params` directly instead of
    re-deriving it, since floating-point resize choices must match exactly.
    """
    h, w = image.shape[:2]
    if (h, w) == (size, size) and label.shape[:2] == (size, size):
        # Already exactly the target square — every tiled-training patch hits
        # this. The full computation below is a no-op here anyway (scale=1,
        # zero padding), just paid for with 4 array copies per patch per epoch;
        # skip straight to returning the inputs. Safe to hand back unchanged
        # (not a defensive copy): callers only ever read these, or derive a NEW
        # array via reversal/dtype-cast, never mutate in place.
        return image, label
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))

    img_pil = PILImage.fromarray(image).resize((nw, nh), PILImage.Resampling.BILINEAR)
    lbl_pil = PILImage.fromarray(label).resize((nw, nh), PILImage.Resampling.NEAREST)

    out_img = np.zeros((size, size, image.shape[2]), dtype=np.uint8)
    out_lbl = np.full((size, size), IGNORE_INDEX, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out_img[top : top + nh, left : left + nw] = np.asarray(img_pil)  # noqa: E203
    out_lbl[top : top + nh, left : left + nw] = np.asarray(lbl_pil)  # noqa: E203
    return out_img, out_lbl


def letterbox_params(h: int, w: int, size: int) -> dict[str, int]:
    """Return the placement used by :func:`letterbox` for an ``h``x``w`` image,
    so a prediction on the ``size``x``size`` canvas can be cropped/resized back
    to the original resolution (see :func:`unletterbox`)."""
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - nh) // 2, (size - nw) // 2
    return {"top": top, "left": left, "nh": nh, "nw": nw}


def unletterbox(label: np.ndarray, orig_h: int, orig_w: int, size: int) -> np.ndarray:
    """Invert :func:`letterbox` on a predicted label map: crop the padding,
    then nearest-resize back to ``(orig_h, orig_w)``."""
    p = letterbox_params(orig_h, orig_w, size)
    cropped = label[p["top"] : p["top"] + p["nh"], p["left"] : p["left"] + p["nw"]]  # noqa: E203
    if (p["nh"], p["nw"]) == (orig_h, orig_w):
        return cropped
    return np.asarray(PILImage.fromarray(cropped).resize((orig_w, orig_h), PILImage.Resampling.NEAREST))


# ---------------------------------------------------------------------------
# Model construction (shared: the one place a model gets built from a
# schemas.ModelConfig, used by both train_jobs.py and batch_probe.py)
# ---------------------------------------------------------------------------


class BuiltFamily(NamedTuple):
    """Everything :func:`run_training_loop` (or the batch-size probe) needs
    from a constructed model, plus enough to save it afterward.

    ``adapter_state_fn`` defers building the saved-weights dict (the dlsia
    network dict) until it's actually called: only :mod:`train_jobs` calls
    it, after training completes; :mod:`batch_probe` never saves anything and
    simply ignores the field.
    """

    forward_fn: Callable[["Any"], "Any"]
    to_tensor_fn: Callable[["Any"], "Any"]
    trainable_params: list["Any"]
    set_train_mode: Callable[[bool], None] | None
    model_config_snapshot: dict[str, Any]
    adapter_state_fn: Callable[[], dict[str, Any]]


def denoiser_runtime_for(config: dict[str, Any]) -> "Any":
    """The runtime module that can load *config*'s saved denoiser.

    Both inference paths (the single-slice canvas preview in
    ``annotation_server`` and the whole-volume bake in ``denoise_bake``) need
    this, and both need the same backward-compatible default — hence one
    implementation rather than two that could drift.

    A run's ``model_config["architecture"]`` names the network. Runs saved
    before that field existed have no such key and are dlsia TUNets by
    definition, so an absent value MUST default to ``"tunet"`` or every existing
    run on disk becomes unloadable.

    Raises:
        ValueError: the run names an architecture this build doesn't know (e.g.
            a run produced by a newer version).
    """
    architecture = (config.get("model_config") or {}).get("architecture", "tunet")
    if architecture == "cnn_ae":
        import autoencoder_runtime  # noqa: PLC0415

        return autoencoder_runtime
    if architecture == "tunet":
        import denoise_runtime  # noqa: PLC0415

        return denoise_runtime
    raise ValueError(f"Unknown denoiser architecture on this run: {architecture!r}")


def denoiser_needs_dlsia(config: dict[str, Any]) -> bool:
    """Whether *config*'s denoiser architecture requires the dlsia dependency.

    Only the TUNet architecture does; the convolutional autoencoder is plain
    torch. Lets the inference gates refuse for the right reason instead of
    demanding dlsia for a network that never touches it.
    """
    return (config.get("model_config") or {}).get("architecture", "tunet") == "tunet"


def build_family(
    model_cfg: "Any",
    n_classes: int,
    device: str,
    log_cb: Callable[[str], None],
    init_state: dict[str, Any] | None = None,
) -> BuiltFamily:
    """Build the model, forward pass, and tensor conversion for *model_cfg*'s
    family.

    This is the single place a model gets constructed from hyperparameters —
    :mod:`train_jobs` and :mod:`batch_probe` both call it, so the probe's
    "builds exactly like training does" claim is enforced by sharing code
    rather than by keeping two copies in sync by hand.

    ``init_state`` is a saved run's ``adapter.pt`` to warm-start from (continue
    fine-tuning) instead of starting from scratch. Optimizer/scheduler state is
    not saved and so is not restored — a resume gets a fresh AdamW and a cosine
    schedule starting again at ``lr``, which is the normal fine-tune-again
    behaviour.

    Every recognized ``schemas.ModelConfig`` member has its own explicit
    branch below; anything else raises :class:`ValueError` rather than being
    silently treated as a dlsia TUNet — that used to be this function's
    implicit ``else``, which would happily (and wrongly) build a segmentation
    TUNet for a config type nobody had actually written a branch for yet.
    """
    from schemas import DlsiaDenoiserConfig, DlsiaTunetConfig  # noqa: PLC0415 — avoid a hard import-time cycle

    hp = model_cfg.hyperparams

    if isinstance(model_cfg, DlsiaTunetConfig):
        import dlsia_runtime as fam  # noqa: PLC0415

        if not dlsia_available():
            raise RuntimeError("dlsia is not installed on this server")
        if init_state is not None:
            # load_model rebuilds the net from the saved topo_dict, so the resumed
            # topology is the run's own — depth/base_channels/growth_rate/image_size
            # from the request are deliberately NOT used here (they can't be: a
            # different topology cannot load these weights). run_train_job forces
            # them to match the saved run before getting here.
            log_cb(f"Resuming dlsia TUNet from saved weights on {device}…")
            model = fam.load_model(init_state, device)
            topo = init_state.get("topo_dict", {})
            snapshot = {
                "depth": topo.get("depth", hp.depth),
                "base_channels": topo.get("base_channels", hp.base_channels),
                "growth_rate": topo.get("growth_rate", hp.growth_rate),
            }
        else:
            log_cb(f"Building dlsia TUNet (depth={hp.depth}, base_channels={hp.base_channels}) on {device}…")
            model = fam.build_model(n_classes, hp.image_size, hp.depth, hp.base_channels, hp.growth_rate, device)
            snapshot = {"depth": hp.depth, "base_channels": hp.base_channels, "growth_rate": hp.growth_rate}
        return BuiltFamily(
            forward_fn=fam.make_forward_fn(model),
            to_tensor_fn=fam.make_to_tensor_fn(),
            trainable_params=list(model.parameters()),
            set_train_mode=fam.make_set_train_mode_fn(model),
            model_config_snapshot=snapshot,
            adapter_state_fn=lambda: fam.network_dict(model),
        )

    if isinstance(model_cfg, DlsiaDenoiserConfig):
        # Two architectures share this family; both expose the identical
        # six-function runtime template, so everything below the module choice
        # is common. See DlsiaDenoiserConfig for why they aren't separate
        # model_family values.
        if model_cfg.architecture == "cnn_ae":
            import autoencoder_runtime as fam  # noqa: PLC0415

            # No dlsia_available() gate here, unlike the TUNet branch below:
            # this network is plain torch and needs no optional dependency.
            if init_state is not None:
                log_cb(f"Resuming autoencoder denoiser from saved weights on {device}…")
                model = fam.load_model(init_state, device)
                topo = init_state.get("topo_dict", {})
                snapshot = {
                    "architecture": "cnn_ae",
                    "depth": topo.get("depth", hp.depth),
                    "base_channels": topo.get("base_channels", hp.base_channels),
                    "ae_compression": topo.get("compression", model_cfg.ae_compression),
                    "latent_channels": topo.get("latent_channels"),
                }
            else:
                latent = fam.latent_channels_for(hp.depth, model_cfg.ae_compression)
                log_cb(
                    f"Building autoencoder denoiser (depth={hp.depth}, "
                    f"base_channels={hp.base_channels}, {model_cfg.ae_compression}x compression "
                    f"-> {latent} latent channels) on {device}…"
                )
                model = fam.build_model(
                    hp.image_size, hp.depth, hp.base_channels, model_cfg.ae_compression, device
                )
                snapshot = {
                    "architecture": "cnn_ae",
                    "depth": hp.depth,
                    "base_channels": hp.base_channels,
                    "ae_compression": model_cfg.ae_compression,
                    "latent_channels": latent,
                }
        else:
            import denoise_runtime as fam  # noqa: PLC0415

            if not dlsia_available():
                raise RuntimeError("dlsia is not installed on this server")
            if init_state is not None:
                # Same warm-start contract as the segmentation TUNet branch above:
                # the saved topo_dict wins, the request's topology knobs don't.
                log_cb(f"Resuming dlsia denoiser TUNet from saved weights on {device}…")
                model = fam.load_model(init_state, device)
                topo = init_state.get("topo_dict", {})
                snapshot = {
                    "architecture": "tunet",
                    "depth": topo.get("depth", hp.depth),
                    "base_channels": topo.get("base_channels", hp.base_channels),
                    "growth_rate": topo.get("growth_rate", hp.growth_rate),
                }
            else:
                log_cb(
                    f"Building dlsia denoiser TUNet (depth={hp.depth}, base_channels={hp.base_channels}) "
                    f"on {device}…"
                )
                # No n_classes: denoise_runtime.build_model fixes in/out channels to
                # 1 (single-channel regression) — there is nothing to parameterize.
                model = fam.build_model(hp.image_size, hp.depth, hp.base_channels, hp.growth_rate, device)
                snapshot = {
                    "architecture": "tunet",
                    "depth": hp.depth,
                    "base_channels": hp.base_channels,
                    "growth_rate": hp.growth_rate,
                }
        return BuiltFamily(
            forward_fn=fam.make_forward_fn(model),
            to_tensor_fn=fam.make_to_tensor_fn(),
            trainable_params=list(model.parameters()),
            set_train_mode=fam.make_set_train_mode_fn(model),
            model_config_snapshot=snapshot,
            adapter_state_fn=lambda: fam.network_dict(model),
        )

    raise ValueError(
        f"Unknown model family/config type for build_family: {type(model_cfg).__name__!r} "
        f"(model_family={getattr(model_cfg, 'model_family', None)!r})"
    )


# ---------------------------------------------------------------------------
# Generic training loop (family-agnostic: takes a forward callable + params)
# ---------------------------------------------------------------------------


def compute_miou(pred: "Any", target: "Any", n_classes: int, ignore_index: int = IGNORE_INDEX) -> float:
    """Mean IoU over ``n_classes`` (0-indexed), ignoring ``ignore_index`` pixels.

    ``pred``/``target`` are torch tensors of matching shape (after argmax);
    a class absent from both prediction and target on this batch is skipped
    (rather than counted as a perfect or zero score) so small batches don't
    bias the running average toward classes that simply didn't appear.
    """
    valid = target != ignore_index
    ious = []
    for c in range(n_classes):
        p = (pred == c) & valid
        t = (target == c) & valid
        union = (p | t).sum().item()
        if union == 0:
            continue
        intersection = (p & t).sum().item()
        ious.append(intersection / union)
    return float(sum(ious) / len(ious)) if ious else 0.0


def run_training_loop(
    *,
    train_pairs: list[tuple[np.ndarray, np.ndarray]],
    val_pairs: list[tuple[np.ndarray, np.ndarray]],
    image_size: int,
    n_classes: int,
    epochs: int,
    batch_size: int,
    seed: int,
    flip_augment: bool,
    to_tensor_fn: Callable[["Any"], "Any"],
    forward_fn: Callable[["Any"], "Any"],
    trainable_params: list["Any"],
    lr: float,
    device: str,
    make_optimizer_fn: Callable[[list["Any"], float], "Any"] | None = None,
    on_batch: Callable[[], bool] | None = None,
    on_epoch: Callable[[int, float, float | None, float | None], bool] | None = None,
    set_train_mode: Callable[[bool], None] | None = None,
) -> dict[str, Any]:
    """Shared epoch/batch loop used by both model families.

    ``to_tensor_fn(rgb_uint8_hwc) -> float tensor (C,H,W)`` applies whatever
    normalisation the family wants (plain /255 for dlsia). ``forward_fn(batch_images)
    -> logits (B,n_classes,H,W)`` runs the family's model. Both
    ``on_batch``/``on_epoch`` return ``True`` to request a cooperative stop
    (checkpoints are still saved by the caller either way).

    ``set_train_mode(is_training)`` toggles ``nn.Module.train()``/``eval()``
    around validation — matters for the dlsia family (TUNet defaults to
    BatchNorm2d, whose running stats should stay fixed during validation
    rather than being computed from that batch). Omit for a model with no such
    distinction.

    Returns ``{"epochs_completed", "final_train_loss", "final_val_loss",
    "val_miou", "cancelled"}``.
    """
    import torch
    import torch.nn as nn

    if not train_pairs:
        raise ValueError("No training data: at least one annotated slice is required")

    rng = np.random.default_rng(seed)
    optimizer = (make_optimizer_fn or (lambda params, lr_: torch.optim.AdamW(params, lr=lr_, weight_decay=0.01)))(
        trainable_params, lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def _prep(rgb: np.ndarray, label: np.ndarray, flip: bool) -> tuple["Any", "Any"]:
        img_l, lbl_l = letterbox(rgb, label, image_size)
        if flip:
            img_l = np.ascontiguousarray(img_l[:, ::-1])
            lbl_l = np.ascontiguousarray(lbl_l[:, ::-1])
        return to_tensor_fn(img_l), torch.from_numpy(lbl_l.astype(np.int64))

    cancelled = False
    final_train_loss = 0.0
    final_val_loss: float | None = None
    val_miou: float | None = None
    epochs_completed = 0

    for epoch in range(epochs):
        order = rng.permutation(len(train_pairs))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]  # noqa: E203
            imgs, lbls = [], []
            for i in batch_idx:
                rgb, label = train_pairs[int(i)]
                flip = bool(flip_augment and rng.random() < 0.5)
                img_t, lbl_t = _prep(rgb, label, flip)
                imgs.append(img_t)
                lbls.append(lbl_t)
            batch_img = torch.stack(imgs).to(device)
            batch_lbl = torch.stack(lbls).to(device)

            logits = forward_fn(batch_img)
            loss = criterion(logits, batch_lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().item())
            n_batches += 1
            if on_batch is not None and on_batch():
                cancelled = True
                break
        scheduler.step()
        final_train_loss = epoch_loss / max(1, n_batches)
        epochs_completed = epoch + 1

        if val_pairs and not cancelled:
            if set_train_mode is not None:
                set_train_mode(False)
            final_val_loss, val_miou = _evaluate(
                val_pairs, image_size, n_classes, to_tensor_fn, forward_fn, device, criterion, batch_size
            )
            if set_train_mode is not None:
                set_train_mode(True)

        if on_epoch is not None and on_epoch(epochs_completed, final_train_loss, final_val_loss, val_miou):
            cancelled = True
        if cancelled:
            break

    return {
        "epochs_completed": epochs_completed,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "val_miou": val_miou,
        "cancelled": cancelled,
    }


def _evaluate(
    val_pairs: list[tuple[np.ndarray, np.ndarray]],
    image_size: int,
    n_classes: int,
    to_tensor_fn: Callable[["Any"], "Any"],
    forward_fn: Callable[["Any"], "Any"],
    device: str,
    criterion: "Any",
    batch_size: int = 1,
) -> tuple[float, float]:
    """Average validation loss + per-sample mIoU over ``val_pairs``.

    Batches the forward pass by ``batch_size`` (mirrors the training loop)
    instead of one sample at a time — tiling's holdout can leave dozens of
    validation patches, and running under ``torch.no_grad()`` means the
    activation memory a bigger batch needs stays bounded regardless.

    mIoU is still scored per sample, not per batch: :func:`compute_miou`
    pools every matching pixel in whatever tensor it's given, so handing it a
    whole batch at once would silently compute one pooled-pixel score across
    samples instead of the average of each sample's own score. That per-sample
    split is cheap CPU-side tensor slicing after the one batched forward, not
    a second forward pass, so it doesn't undo the batching's benefit.
    """
    import torch

    total_loss = 0.0
    total_miou = 0.0
    n = len(val_pairs)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = val_pairs[start : start + batch_size]  # noqa: E203
            imgs, lbls = [], []
            for rgb, label in chunk:
                img_l, lbl_l = letterbox(rgb, label, image_size)
                imgs.append(to_tensor_fn(img_l))
                lbls.append(torch.from_numpy(lbl_l.astype(np.int64)))
            batch_img = torch.stack(imgs).to(device)
            batch_lbl = torch.stack(lbls).to(device)

            logits = forward_fn(batch_img)
            # criterion's mean reduction is over the whole batch, so weight by
            # sample count to keep this a per-sample average overall — same
            # convention the training loop already uses for its own loss.
            total_loss += float(criterion(logits, batch_lbl).item()) * len(chunk)
            preds = logits.argmax(dim=1)
            for i in range(len(chunk)):
                total_miou += compute_miou(preds[i], batch_lbl[i], n_classes)
    return total_loss / n, total_miou / n


# ---------------------------------------------------------------------------
# Capability probe (never raises — feeds GET /api/train/capability)
# ---------------------------------------------------------------------------


def capability() -> dict[str, Any]:
    """Best-effort snapshot of Train-tab readiness. Never raises.

    No DINOv3 fields — that model family is deferred to Phase 5.5 and isn't
    scaffolded here at all.
    """
    result: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "device": None,
        "dlsia": {"available": False},
        "tiling": {"available": False},
        # Classical denoising is pure CPU (scipy/skimage) and never touches
        # ML_LOCK, so it stays usable while a training job runs.
        "denoise": {"available": False, "methods": []},
        "runs_dir": str(runs_dir()),
        "busy": ML_LOCK.locked(),
    }
    try:
        result["torch_available"] = torch_available()
        if result["torch_available"]:
            import torch  # noqa: PLC0415

            result["torch_version"] = torch.__version__
            result["device"] = pick_device()

        result["dlsia"] = {"available": result["torch_available"] and dlsia_available()}

        import tiling  # noqa: PLC0415 — avoid a hard import-time cycle (tiling imports train_common)

        result["tiling"] = {"available": tiling.qlty_available()}

        import denoise  # noqa: PLC0415 — cheap, but keep the probe self-contained

        # `available: True` unconditionally: scipy/skimage are hard dependencies
        # here (unlike torch/dlsia), so the classical filters always work. The
        # per-method `available` flags carry the real gating — wavelet needs
        # PyWavelets, which skimage imports lazily and which is not installed.
        result["denoise"] = {"available": True, "methods": denoise.describe_methods()}
    except Exception as exc:  # noqa: BLE001 — a capability probe must never 500
        logger.warning("Train capability probe failed: %s", exc)
        result["error"] = str(exc)
    return result
