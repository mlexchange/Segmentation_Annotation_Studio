"""Background job orchestration for the Train tab.

Dispatches a :class:`schemas.TrainRequest` to the requested model family
(DINOv3 LoRA or dlsia TUNet), sharing data preparation, the generic training
loop, and run persistence via :mod:`train_common`. Progress/cancellation are
reported through the existing :mod:`export_jobs` registry — the frontend polls
the same ``GET /api/export/status/{job_id}`` route already used by exports and
mask-sync jobs.

Two *tasks* are dispatched from here, distinguished by
``TrainRequest.task``:

* ``"segmentation"`` — annotation-driven, via
  :func:`train_common.prepare_datasets` and
  :func:`train_common.run_training_loop`.
* ``"denoising"`` — self-supervised (Noise2Noise / Noise2Void), via
  :mod:`denoise_train`'s raw-slice samplers and its own regression loop.

Everything either task has in common — the ML lock, device selection, resume
resolution, the qlty guard, ``export_jobs`` progress/cancel reporting, and
persistence through :func:`train_common.save_run` — is shared in
:func:`run_train_job`; only data preparation, the loss, and the validation
metric differ, and those live behind the split at the end of its preamble.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import export_jobs
import train_common
from schemas import (
    DenoiseTrainOpts,
    DinoV3LoraConfig,
    DlsiaDenoiserConfig,
    DlsiaTunetConfig,
    TrainRequest,
)

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

    A self-supervised denoiser has no class taxonomy at all, so the class-list
    comparison is skipped when BOTH the parent run and *request* are
    ``task == "denoising"``. That is not the same as skipping the check
    whenever either side is denoising: resuming a denoiser as a segmentation
    run (or vice versa) is a genuine incompatibility — the saved weights are
    single-channel regression output, not per-class logits, or the reverse —
    and must still be reported as such rather than silently allowed through.
    ``parent_config`` may predate the ``task`` field entirely (see
    ``train_common.load_run_config``), so an absent key defaults to
    ``"segmentation"`` here too, not just at load time.
    """
    if parent_config.get("model_family") != request.model.model_family:
        raise ValueError(
            f"Cannot continue fine-tuning a {parent_config.get('model_family')} run "
            f"as {request.model.model_family} — pick the matching model family, or train a new run."
        )

    parent_task = parent_config.get("task", "segmentation")
    request_task = request.task
    if parent_task != request_task:
        raise ValueError(
            f"Cannot continue fine-tuning a {parent_task!r}-task run as a {request_task!r}-task "
            "request — a segmentation model and a denoiser are not interchangeable. Pick the "
            "matching task, or train a new run."
        )
    if parent_task == "denoising":
        # Both sides are confirmed "denoising" above — there is no class
        # taxonomy to compare for a self-supervised denoiser. But the two
        # architectures in this family share a model_family, so the check above
        # passes for a TUNet-vs-autoencoder mismatch; without this, the resume
        # would reach load_state_dict and fail on an opaque key/shape mismatch
        # instead of saying what is actually wrong. Runs saved before
        # `architecture` existed are TUNets by definition.
        parent_arch = (parent_config.get("model_config") or {}).get("architecture", "tunet")
        request_arch = getattr(request.model, "architecture", "tunet")
        if parent_arch != request_arch:
            raise ValueError(
                f"Cannot continue fine-tuning a {parent_arch!r} denoiser as {request_arch!r} — "
                "the two architectures have different weights entirely. Pick the matching "
                "architecture, or train a new run."
            )
        return

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

    Every recognized ``schemas.ModelConfig`` member has its own explicit
    branch below; anything else raises :class:`ValueError` — mirrors
    ``train_common.build_family``'s dispatch, which used to have the same
    implicit "anything not DINOv3 must be a TUNet" assumption here.
    """
    model_cfg = request.model
    hp = model_cfg.hyperparams
    parent_model = parent_config.get("model_config", {})
    parent_hp = parent_config.get("hyperparams", {})

    # image_size + tiling define the geometry the weights were fit to.
    for field in ("image_size", "tiling"):
        if field in parent_hp:
            setattr(hp, field, parent_hp[field])

    # Input denoising is inherited for the same reason as the geometry above:
    # the parent's weights were fit to that pixel distribution, so continuing to
    # train them on differently-preprocessed pixels degrades them silently. It
    # is NOT a re-tunable knob like epochs or lr — leaving it to the caller
    # meant a fine-tune of a denoise-trained run quietly reverted to raw pixels
    # whenever the client forgot to echo it back.
    parent_denoise = parent_config.get("denoise")
    request.denoise = (
        DenoiseTrainOpts(**parent_denoise) if parent_denoise else None
    )

    if isinstance(model_cfg, DinoV3LoraConfig):
        # arch/checkpoint fix embed_dim; rank/alpha fix the LoRA tensor shapes.
        if parent_model.get("arch"):
            model_cfg.arch = parent_model["arch"]
        if parent_model.get("checkpoint"):
            model_cfg.checkpoint = parent_model["checkpoint"]
        for field in ("lora_rank", "lora_alpha"):
            if field in parent_hp:
                setattr(hp, field, parent_hp[field])
    elif isinstance(model_cfg, DlsiaTunetConfig):
        # TUNet's topology comes back from the saved topo_dict regardless; mirror
        # it onto the request so the saved config records what actually ran.
        for field in ("depth", "base_channels", "growth_rate"):
            if field in parent_hp:
                setattr(hp, field, parent_hp[field])
    elif isinstance(model_cfg, DlsiaDenoiserConfig):
        # Same TUNet topology knobs as the segmentation family above — dlsia's
        # TUNet is the shared architecture underneath both; only the fixed
        # in/out channel counts differ, and those aren't user-configurable.
        for field in ("depth", "base_channels", "growth_rate"):
            if field in parent_hp:
                setattr(hp, field, parent_hp[field])
        # The architecture itself, and the bottleneck width that follows from it,
        # are inherited for the same reason as the geometry above: they define
        # the tensor shapes, so a resume that changed them could not load the
        # saved weights. A run saved before `architecture` existed has no such
        # key and is a TUNet by definition.
        parent_arch = parent_model.get("architecture", "tunet")
        model_cfg.architecture = parent_arch
        if parent_arch == "cnn_ae":
            # `ae` is the only scheme the schema permits with cnn_ae, so a
            # resume must land on it or the request would be self-inconsistent.
            model_cfg.training_scheme = "ae"
            if parent_model.get("ae_compression") is not None:
                model_cfg.ae_compression = int(parent_model["ae_compression"])
    else:
        raise ValueError(
            f"Unknown model config type for resume: {type(model_cfg).__name__!r} "
            f"(model_family={getattr(model_cfg, 'model_family', None)!r})"
        )


def check_task_matches_model(request: TrainRequest) -> None:
    """Raise unless ``request.task`` and the model family agree.

    ``task`` and ``model.model_family`` are independent fields on the schema,
    and ``task`` defaults to ``"segmentation"`` — so a client that sends a
    ``dlsia_denoiser`` config and forgets ``task`` produces a request that
    validates cleanly and then means something incoherent. Left unchecked it
    would route into the annotation-driven path and train a single-channel
    regression network against ``CrossEntropyLoss`` over zero classes. The
    mirror case (``task="denoising"`` with a segmentation family) would ask
    :mod:`denoise_train` to feed grayscale into a 3-channel model. Both are
    caught here, before any data is read.
    """
    is_denoiser_family = isinstance(request.model, DlsiaDenoiserConfig)
    if request.task == "denoising" and not is_denoiser_family:
        raise ValueError(
            "task='denoising' needs a denoiser model family, but got "
            f"{request.model.model_family!r}. Use model_family='dlsia_denoiser', or set task='segmentation'."
        )
    if request.task != "denoising" and is_denoiser_family:
        raise ValueError(
            "model_family='dlsia_denoiser' is a self-supervised denoiser and cannot be trained as a "
            f"{request.task!r} task — send task='denoising' (and classes: []) instead."
        )


def _source_keys(request: TrainRequest) -> list[str]:
    """Stable per-source identifiers recorded on the saved run."""
    return [
        (f"tiled:{item.server_uri or ''}:{item.source}" if item.kind == "tiled" else f"local:{item.source}")
        for item in request.sources
    ]


def _run_denoise_training(
    jid: str,
    request: TrainRequest,
    run_id: str,
    *,
    device: str,
    model_cfg: DlsiaDenoiserConfig,
    init_state: dict | None,
    resume_id: str | None,
    progress_cb,
) -> None:
    """Self-supervised denoiser branch of :func:`run_train_job`.

    Called with :data:`train_common.ML_LOCK` already held and the resume
    already resolved, and reports through ``export_jobs`` with the same phase
    names (``preparing`` → ``tiling`` → ``training`` → ``saving`` → ``done``)
    and the same cancellation contract as the segmentation path, so the
    frontend's existing job polling needs no denoiser-specific handling.

    Weights and run metadata go through the ordinary
    :func:`train_common.save_run`, with ``task="denoising"`` and an empty class
    list, so the run appears in ``list_runs`` and the Learned Denoiser panel's
    ``model_family == "dlsia_denoiser"`` filter finds it.
    """
    import denoise_train
    import tiling

    hp = model_cfg.hyperparams
    scheme = model_cfg.training_scheme

    # phase is already "preparing" — set by run_train_job before the split.
    scheme_label = {
        "n2n": "Noise2Noise",
        "n2v": "Noise2Void",
        "ae": "autoencoder",
    }.get(scheme, scheme)
    export_jobs.log(
        jid,
        f"Preparing self-supervised {scheme_label} data from raw slices (no annotations needed)…",
    )
    if scheme == "n2n":
        datasets = denoise_train.prepare_noise2noise_datasets(
            request.sources, request.render, progress_cb=progress_cb
        )
    else:
        # n2v and dae are both single-slice schemes: each item is a slice paired
        # with itself, and the objective differs only in how the INPUT is
        # perturbed (blind-spot masking vs. added synthetic noise). So they share
        # this sampler rather than needing a third one.
        datasets = denoise_train.prepare_noise2void_datasets(
            request.sources, request.render, progress_cb=progress_cb
        )

    if hp.tiling:
        export_jobs.update(jid, phase="tiling")
        export_jobs.log(jid, f"Cutting {hp.image_size}px windows…")
        datasets = denoise_train.tile_denoise_datasets(
            datasets,
            hp.image_size,
            progress_cb=progress_cb,
            cancel_cb=lambda: export_jobs.cancel_requested(jid),
        )
        if datasets is None:
            # Same "cancelled before a model existed" shape the segmentation
            # path uses — deliberately not the partial-run result, which implies
            # save_run produced something.
            export_jobs.update(jid, state="done", phase="done", result={"cancelled": True})
            export_jobs.log(jid, "Training cancelled while tiling; nothing was trained yet.")
            return

    # The denoiser's samplers put everything in "train" (there are no
    # annotation-driven splits to inherit), so the seeded patch holdout is what
    # produces a validation set at all. Applied whether or not tiling ran: with
    # tiling off the items are whole slices, and the function no-ops below its
    # minimum count rather than starving a small run of training data.
    datasets, n_held = tiling.holdout_val_patches(datasets, seed=hp.seed)
    if n_held:
        export_jobs.log(
            jid,
            f"Held back {n_held} training patch(es) for validation. They come from the same "
            "slice(s) as the training patches, and both schemes' targets are themselves noisy, "
            "so the reported correlation is a convergence signal — not an image-quality score.",
        )

    n_train = len(datasets["train"])
    if n_train == 0:
        raise ValueError("No slices to train the denoiser on")

    # n_classes is ignored by build_family's denoiser branch (in/out channels
    # are fixed at 1); 0 is passed to make that explicit rather than incidental.
    built = train_common.build_family(model_cfg, 0, device, progress_cb, init_state=init_state)

    batches_per_epoch = max(1, -(-n_train // hp.batch_size))
    export_jobs.set_total(jid, hp.epochs * batches_per_epoch)
    export_jobs.update(jid, phase="training")

    def _on_batch() -> bool:
        export_jobs.bump(jid, 1)
        return export_jobs.cancel_requested(jid)

    def _on_epoch(epoch: int, train_loss: float, val_loss: float | None, val_metric: float | None) -> bool:
        msg = f"epoch {epoch}/{hp.epochs} — train loss {train_loss:.5f}"
        if val_loss is not None:
            # Labelled as correlation with the NOISY target, never as quality.
            msg += f", val loss {val_loss:.5f}, noisy-target r {val_metric:.3f}"
        export_jobs.log(jid, msg)
        return export_jobs.cancel_requested(jid)

    metrics = denoise_train.run_denoise_training_loop(
        train_pairs=datasets["train"],
        val_pairs=datasets["val"],
        image_size=hp.image_size,
        training_scheme=scheme,
        epochs=hp.epochs,
        batch_size=hp.batch_size,
        seed=hp.seed,
        flip_augment=hp.flip_augment,
        to_tensor_fn=built.to_tensor_fn,
        forward_fn=built.forward_fn,
        trainable_params=built.trainable_params,
        lr=hp.lr,
        device=device,
        on_batch=_on_batch,
        on_epoch=_on_epoch,
        set_train_mode=built.set_train_mode,
    )

    export_jobs.update(jid, phase="saving")
    train_common.save_run(
        run_id,
        model_family=model_cfg.model_family,
        # training_scheme belongs on the run: n2n and n2v produce different
        # models from the same topology, and the runs list surfaces which.
        model_config={
            **built.model_config_snapshot,
            "training_scheme": scheme,
            # Recorded only where it means something, so a run's config doesn't
            # imply a knob that had no effect on how it was trained.
            # `architecture` is always recorded: both inference sites dispatch
            # on it, defaulting to "tunet" for runs saved before it existed.
            "architecture": model_cfg.architecture,
        },
        classes=[],
        render=request.render.model_dump(),
        image_size=hp.image_size,
        hyperparams=hp.model_dump(),
        source_keys=_source_keys(request),
        adapter_state=built.adapter_state_fn(),
        metrics=metrics,
        resumed_from=resume_id,
        task="denoising",
    )

    result = {"run_id": run_id, **metrics}
    export_jobs.update(jid, state="done", phase="done", result=result)
    export_jobs.log(
        jid,
        "Training cancelled; partial run saved." if metrics["cancelled"] else "Denoiser training complete.",
    )


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
        check_task_matches_model(request)
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

        # Everything above is task-agnostic (lock, device, resume, qlty guard).
        # From here the two tasks diverge: a denoiser reads raw slices instead
        # of rendering annotations, and optimises a regression loss.
        if request.task == "denoising":
            _run_denoise_training(
                jid,
                request,
                run_id,
                device=device,
                model_cfg=model_cfg,
                init_state=init_state,
                resume_id=resume_id,
                progress_cb=_progress,
            )
            return

        datasets = train_common.prepare_datasets(
            request.sources,
            request.classes,
            request.render,
            request.auto_split,
            progress_cb=_progress,
            denoise=request.denoise,
        )
        if request.denoise is not None:
            _progress(
                f"Training on {request.denoise.method}-denoised input "
                f"({request.denoise.strength:.0%}); inference will reapply it automatically."
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

        train_common.save_run(
            run_id,
            model_family=model_cfg.model_family,
            model_config=model_config_snapshot,
            classes=[c.model_dump() for c in request.classes],
            render=request.render.model_dump(),
            image_size=hp.image_size,
            hyperparams=hp.model_dump(),
            source_keys=_source_keys(request),
            adapter_state=adapter_state,
            metrics=metrics,
            resumed_from=resume_id,
            task=request.task,
            # Recorded so inference reapplies the same input preprocessing
            # without the caller having to remember it.
            denoise=request.denoise.model_dump() if request.denoise else None,
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
