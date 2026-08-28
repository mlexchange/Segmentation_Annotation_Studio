"""Background jobs for sample-scale iPred operations: multi-slice train and
whole-volume apply.

Both dispatch onto the shared ``export_jobs`` registry — mirrors
``denoise_bake.py``'s worker pattern exactly (thread target, cooperative
cancel, bump/log per unit of work). Poll ``GET /api/export/status/{job_id}``;
the routes that spawn these live in ``ipred_routes.py``.

The apply job deliberately stops at "run inference per slice" and does NOT
vectorize the conformal commit/status maps into polygon shapes server-side —
that stays client-side (``AnnotatePage.tsx`` + ``pixelClf.ts``), reusing the
already-proven, already-tested polygon tracer instead of porting it to Python.
"""

from __future__ import annotations

import logging
from typing import Any

import export_jobs
import ipred_client as ipred_client_mod

logger = logging.getLogger(__name__)


def run_ipred_multi_train_job(
    jid: str,
    *,
    session_id: str,
    per_slice_shapes: dict[int, list[dict[str, Any]]],
    composition_id: str | None,
    feature_setup_id: str | None,
    trainer_id: str,
    config: dict[str, Any] | None,
) -> None:
    """Preprocess every given slice (feature banks are cache-aware, so an
    already-computed slice is cheap), then train ONE model pooling all their
    labeled pixels via ``POST /train/multi``.

    A slice's preprocess failing aborts the whole job rather than being
    skipped — unlike the apply job below, a pooled model that silently
    dropped one of the slices the user asked for isn't the model they asked
    to train, so this fails loudly instead.
    """
    try:
        export_jobs.update(jid, state="running", phase="preprocessing")
        slice_indices = sorted(per_slice_shapes)
        # +1 unit for the final pooled-train phase, so the bar doesn't read
        # 100% while the (potentially slow) pooled fit is still running.
        export_jobs.set_total(jid, len(slice_indices) + 1)

        feature_ids: dict[str, str] = {}
        for slice_index in slice_indices:
            if export_jobs.cancel_requested(jid):
                export_jobs.update(jid, state="error", phase="cancelled", error="Cancelled.")
                return
            bank = ipred_client_mod.preprocess(
                session_id=session_id,
                feature_setup_id=feature_setup_id,
                composition_id=composition_id,
                slice_index=slice_index,
            )
            feature_ids[str(slice_index)] = bank["feature_id"]
            export_jobs.bump(jid, 1)
            cache_note = "cached" if bank.get("cache_hit") else "computed"
            export_jobs.log(jid, f"slice {slice_index}: features ready ({cache_note})")

        if export_jobs.cancel_requested(jid):
            export_jobs.update(jid, state="error", phase="cancelled", error="Cancelled.")
            return

        export_jobs.update(jid, phase="training")
        result = ipred_client_mod.train_multi(
            session_id=session_id,
            slices={str(k): v for k, v in per_slice_shapes.items()},
            feature_ids=feature_ids,
            trainer_id=trainer_id,
            config=config,
        )
        export_jobs.bump(jid, 1)
        export_jobs.update(jid, state="done", phase="done", result=result)
        export_jobs.log(
            jid,
            f"Trained on {len(slice_indices)} slice(s), "
            f"{result.get('n_samples', 0):,} pooled samples.",
        )
    except Exception as exc:  # noqa: BLE001 — reported as a job error, never a crash
        logger.exception("ipred multi-slice train job %s failed", jid)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))


def run_ipred_volume_apply_job(
    jid: str,
    *,
    session_id: str,
    model_id: str,
    slice_indices: list[int],
    composition_id: str | None,
    feature_setup_id: str | None,
    alpha: float,
) -> None:
    """Run inference on every given slice, ensuring each has a feature bank
    first. Stores ``result.runs = {slice_index: run_id}`` for the frontend to
    fetch commit/status PNGs from and vectorize per slice.

    Unlike the train job, one slice failing does not abort the rest — a
    volume-wide apply that stops at the first unreadable slice would be far
    more disruptive than a job that finishes with a handful of gaps reported
    in ``result.errors``.
    """
    try:
        export_jobs.update(jid, state="running", phase="predicting")
        export_jobs.set_total(jid, len(slice_indices))

        runs: dict[str, str] = {}
        errors: list[dict[str, Any]] = []
        cancelled = False
        for slice_index in slice_indices:
            if export_jobs.cancel_requested(jid):
                cancelled = True
                break
            try:
                bank = ipred_client_mod.preprocess(
                    session_id=session_id,
                    feature_setup_id=feature_setup_id,
                    composition_id=composition_id,
                    slice_index=slice_index,
                )
                run = ipred_client_mod.infer(
                    session_id=session_id,
                    model_id=model_id,
                    feature_id=bank["feature_id"],
                    alpha=alpha,
                )
                runs[str(slice_index)] = run["run_id"]
                export_jobs.log(jid, f"slice {slice_index}: predicted (run {run['run_id'][:8]})")
            except Exception as exc:  # noqa: BLE001 — one bad slice must not abort the volume
                logger.warning("ipred volume apply: slice %d failed (%s)", slice_index, exc)
                errors.append({"slice": slice_index, "error": str(exc)})
            export_jobs.bump(jid, 1)

        result = {"runs": runs, "errors": errors, "cancelled": cancelled}
        if not runs:
            export_jobs.update(
                jid,
                state="error",
                phase="error",
                error="No slices could be predicted.",
                result=result,
            )
            return
        export_jobs.update(jid, state="done", phase="done", result=result)
        export_jobs.log(
            jid,
            f"{'Cancelled after' if cancelled else 'Applied to'} "
            f"{len(runs)}/{len(slice_indices)} slice(s).",
        )
    except Exception as exc:  # noqa: BLE001 — reported as a job error, never a crash
        logger.exception("ipred volume apply job %s failed", jid)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))
