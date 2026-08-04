"""Background job orchestration for the Train tab.

Dispatches a :class:`schemas.TrainRequest` to the requested model family
(DINOv3 LoRA or dlsia TUNet), sharing data preparation, the generic training
loop, and run persistence via :mod:`train_common`. Progress/cancellation are
reported through the existing :mod:`export_jobs` registry — the frontend polls
the same ``GET /api/export/status/{job_id}`` route already used by exports and
mask-sync jobs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import export_jobs
import train_common
from schemas import TrainRequest

logger = logging.getLogger(__name__)


def new_run_id(model_family: str) -> str:
    """``<UTC timestamp>_<model_family>_<short random>`` — sorts newest-last
    lexically within a day, globally unique enough for a single-user tool."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{model_family}_{uuid.uuid4().hex[:4]}"


def check_resume_compatible(parent_config: dict, request: TrainRequest) -> None:
    """Raise if *request* cannot continue fine-tuning from *parent_config*'s run.

    The saved head has one output channel per class, in the parent's class
    ORDER (``infer_jobs`` maps channel ``c`` to ``classes[c]["classId"]``), so
    the class list has to line up positionally for a resume to mean anything.
    A count mismatch would fail loudly inside ``load_state_dict`` anyway, but a
    same-count-different-labels resume would train happily against the wrong
    semantics and never say so — which is the case this exists to catch.
    """
    if parent_config.get("model_family") != request.model.model_family:
        raise ValueError(
            f"Cannot continue fine-tuning a {parent_config.get('model_family')} run "
            f"as {request.model.model_family} — pick the matching model family, or train a new run."
        )

    parent_labels = [str(c.get("label", "")).strip().lower() for c in parent_config.get("classes", [])]
    current_labels = [c.label.strip().lower() for c in request.classes]
    if parent_labels != current_labels:
        raise ValueError(
            "The classes changed since that run was trained, so its saved weights no longer apply "
            f"(run has {len(parent_labels)}: {', '.join(parent_labels) or '—'}; "
            f"now {len(current_labels)}: {', '.join(current_labels) or '—'}). "
            "Restore the original classes to continue fine-tuning, or train a new run instead."
        )


def _apply_parent_architecture(parent_config: dict, request: TrainRequest) -> None:
    """Force the architecture-defining settings to the parent run's values.

    Anything that changes tensor shapes cannot differ across a resume — the
    saved weights simply would not load. Rather than trusting the client to
    echo these back correctly (or 400-ing on every mismatch), the server just
    overrides them, so a resume is always weight-compatible by construction.
    Genuinely re-tunable knobs — epochs, lr, batch_size, seed, flip_augment —
    are left as the caller sent them; that is the point of resuming.
    """
    model_cfg = request.model
    hp = model_cfg.hyperparams
    parent_model = parent_config.get("model_config", {})
    parent_hp = parent_config.get("hyperparams", {})

    # image_size + tiling define the geometry the weights were fit to.
    for field in ("image_size", "tiling"):
        if field in parent_hp:
            setattr(hp, field, parent_hp[field])

    if model_cfg.model_family == "dinov3_lora":
        # arch/checkpoint fix embed_dim; rank/alpha fix the LoRA tensor shapes.
        if parent_model.get("arch"):
            model_cfg.arch = parent_model["arch"]
        if parent_model.get("checkpoint"):
            model_cfg.checkpoint = parent_model["checkpoint"]
        for field in ("lora_rank", "lora_alpha"):
            if field in parent_hp:
                setattr(hp, field, parent_hp[field])
    else:
        # TUNet's topology comes back from the saved topo_dict regardless; mirror
        # it onto the request so the saved config records what actually ran.
        for field in ("depth", "base_channels", "growth_rate"):
            if field in parent_hp:
                setattr(hp, field, parent_hp[field])


def run_train_job(jid: str, request: TrainRequest, run_id: str) -> None:
    """Background worker: prepare data, build the requested model family, run
    the shared training loop, and persist the resulting run.

    Holds :data:`train_common.ML_LOCK` for the whole job — training and
    inference contend for the same device memory, so only one ML job (of
    either kind) runs at a time.
    """
    if not train_common.ML_LOCK.acquire(blocking=False):
        export_jobs.update(
            jid,
            state="error",
            phase="error",
            error="Another training or inference job is already running",
        )
        return
    try:
        export_jobs.update(jid, state="running", phase="preparing")
        device = train_common.pick_device()
        if device is None:
            raise RuntimeError("torch is not installed on this server")

        def _progress(msg: str) -> None:
            export_jobs.log(jid, msg)

        model_cfg = request.model
        n_classes = len(request.classes)

        # Resolve a resume BEFORE prepare_datasets: an incompatible one must fail
        # in seconds, not after minutes of rendering. Loading the weights here
        # too means a corrupt/missing adapter.pt is caught just as early.
        init_state = None
        resume_id = request.resume_from_run_id
        if resume_id:
            parent_config = train_common.load_run_config(resume_id)
            check_resume_compatible(parent_config, request)
            _apply_parent_architecture(parent_config, request)
            init_state = train_common.load_adapter_state(resume_id)
            export_jobs.log(jid, f"Continuing fine-tuning from run {resume_id}.")

        hp = model_cfg.hyperparams

        # Check qlty BEFORE the render/rasterize pass below (prepare_datasets can
        # take minutes on a large source list) so a torch-only install fails fast
        # instead of after paying for it.
        if hp.tiling:
            import tiling

            if not tiling.qlty_available():
                raise RuntimeError("Tiling requires the 'qlty' package, which is not installed on this server")

        datasets = train_common.prepare_datasets(
            request.sources,
            request.classes,
            request.render,
            request.auto_split,
            progress_cb=_progress,
        )

        # Tiling: keep native resolution by cutting `image_size` windows out of each
        # slice, instead of letting the training loop rescale whole slices down to
        # `image_size`. Patches come out exactly window-sized, which letterbox()
        # passes through, so the loop below is unchanged either way.
        if model_cfg.hyperparams.tiling:
            import tiling  # already confirmed available above; re-import is a cheap sys.modules hit

            export_jobs.update(jid, phase="tiling")
            export_jobs.log(jid, f"Tiling slices into {model_cfg.hyperparams.image_size}px windows…")

            datasets = tiling.tile_datasets(
                datasets,
                model_cfg.hyperparams.image_size,
                progress_cb=_progress,
                cancel_cb=lambda: export_jobs.cancel_requested(jid),
            )
            if datasets is None:
                # Cancelled before any model existed to save — a bare "cancelled"
                # result, not the usual partial-run shape from a training-loop
                # cancel (train_common.run_training_loop's own cancel path, below).
                export_jobs.update(jid, state="done", phase="done", result={"cancelled": True})
                export_jobs.log(jid, "Training cancelled while tiling; nothing was trained yet.")
                return
            # Splits are per-slice, so a one- or two-slice dataset leaves validation
            # empty and the run reports no metrics. Tiling yields enough patches to
            # hold a few back instead.
            datasets, n_held = tiling.holdout_val_patches(
                datasets, seed=model_cfg.hyperparams.seed
            )
            if n_held:
                export_jobs.log(
                    jid,
                    f"No validation slices — held back {n_held} training patch(es) for validation. "
                    "They come from the same image(s) as the training patches, so mIoU reads "
                    "optimistically compared with an unseen slice.",
                )

        n_train = len(datasets["train"])
        if n_train == 0:
            raise ValueError("No annotated slices to train on")

        built = train_common.build_family(model_cfg, n_classes, device, _progress, init_state=init_state)
        forward_fn = built.forward_fn
        to_tensor_fn = built.to_tensor_fn
        trainable = built.trainable_params
        set_train_mode = built.set_train_mode
        model_config_snapshot = built.model_config_snapshot

        batches_per_epoch = max(1, -(-n_train // hp.batch_size))
        export_jobs.set_total(jid, hp.epochs * batches_per_epoch)
        export_jobs.update(jid, phase="training")

        def _on_batch() -> bool:
            export_jobs.bump(jid, 1)
            return export_jobs.cancel_requested(jid)

        def _on_epoch(epoch: int, train_loss: float, val_loss: float | None, val_miou: float | None) -> bool:
            msg = f"epoch {epoch}/{hp.epochs} — train loss {train_loss:.4f}"
            if val_loss is not None:
                msg += f", val loss {val_loss:.4f}, mIoU {val_miou:.3f}"
            export_jobs.log(jid, msg)
            return export_jobs.cancel_requested(jid)

        metrics = train_common.run_training_loop(
            train_pairs=datasets["train"],
            val_pairs=datasets["val"],
            image_size=hp.image_size,
            n_classes=n_classes,
            epochs=hp.epochs,
            batch_size=hp.batch_size,
            seed=hp.seed,
            flip_augment=hp.flip_augment,
            to_tensor_fn=to_tensor_fn,
            forward_fn=forward_fn,
            trainable_params=trainable,
            lr=hp.lr,
            device=device,
            on_batch=_on_batch,
            on_epoch=_on_epoch,
            set_train_mode=set_train_mode,
        )

        export_jobs.update(jid, phase="saving")
        adapter_state = built.adapter_state_fn()

        source_keys = [
            (f"tiled:{item.server_uri or ''}:{item.source}" if item.kind == "tiled" else f"local:{item.source}")
            for item in request.sources
        ]
        train_common.save_run(
            run_id,
            model_family=model_cfg.model_family,
            model_config=model_config_snapshot,
            classes=[c.model_dump() for c in request.classes],
            render=request.render.model_dump(),
            image_size=hp.image_size,
            hyperparams=hp.model_dump(),
            source_keys=source_keys,
            adapter_state=adapter_state,
            metrics=metrics,
            resumed_from=resume_id,
        )

        result = {"run_id": run_id, **metrics}
        export_jobs.update(jid, state="done", phase="done", result=result)
        export_jobs.log(
            jid,
            "Training cancelled; partial run saved." if metrics["cancelled"] else "Training complete.",
        )
    except Exception as exc:  # noqa: BLE001 — reported as a job error, never a crash
        logger.error("Training job %s failed: %s", jid, exc)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))
    finally:
        train_common.ML_LOCK.release()
