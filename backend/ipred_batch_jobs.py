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

import concurrent.futures
import logging
import os
from typing import Any

import httpx

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


#: Slices processed concurrently in a volume-wide apply job. Each in-flight
#: slice holds ~1 GB (its feature bank, released immediately after use — see
#: `Catalog.delete_feature_bank`) resident on the ipred service process, so
#: this directly multiplies peak memory there. Overridable per machine
#: without a code change via IPRED_APPLY_CONCURRENCY; kept conservative by
#: default. `preprocess`/`infer` are sync FastAPI routes, which Starlette
#: already runs in its own thread pool even under a single uvicorn process —
#: nothing on the ipred service side needs to change to make this safe (see
#: the encoder's own `_session_lock`/`_model_lock`, which already serialize
#: concurrent ONNX access correctly).
_DEFAULT_APPLY_CONCURRENCY = 4


def _apply_one_slice(
    slice_index: int,
    *,
    client: httpx.Client,
    session_id: str,
    model_id: str,
    composition_id: str | None,
    feature_setup_id: str | None,
    alpha: float,
) -> tuple[int, str | None, str | None]:
    """Preprocess + infer one slice on a worker thread. Returns
    ``(slice_index, run_id, error)`` — exactly one of the last two is set —
    never raises, so a pool of these can be driven with plain
    ``future.result()`` and no per-future try/except at the call site."""
    bank: dict[str, Any] | None = None
    try:
        bank = ipred_client_mod.preprocess(
            session_id=session_id,
            feature_setup_id=feature_setup_id,
            composition_id=composition_id,
            slice_index=slice_index,
            client=client,
        )
        run = ipred_client_mod.infer(
            session_id=session_id,
            model_id=model_id,
            feature_id=bank["feature_id"],
            alpha=alpha,
            # Volume-apply never reads a run's proba.npy back (only
            # commit.png, client-side) — see run_infer's own doc for why
            # this is safe to skip here but not for interactive infer.
            store_probabilities=False,
            client=client,
        )
        return slice_index, run["run_id"], None
    except Exception as exc:  # noqa: BLE001 — one bad slice must not abort the volume
        return slice_index, None, str(exc)
    finally:
        # Release the feature bank whether infer succeeded or failed —
        # a failed infer still leaves an orphaned ~1 GB bank behind
        # otherwise. Best-effort: a cleanup failure must not fail the
        # slice that already predicted successfully.
        if bank is not None:
            try:
                ipred_client_mod.delete_feature_bank(bank["feature_id"], client=client)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ipred volume apply: could not release feature bank for slice %d (%s)",
                    slice_index, exc,
                )


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

    Slices are processed by a small bounded thread pool (see
    ``_DEFAULT_APPLY_CONCURRENCY``) sharing one persistent HTTP connection
    (``ipred_client_mod.new_shared_client``) instead of one-at-a-time with a
    fresh connection per call — see the plan doc for why each piece of
    shared state here (export_jobs' lock, Catalog's per-call connections,
    CatBoost's read-only predict, the encoder's existing session locks) is
    already safe under this concurrency.
    """
    try:
        export_jobs.update(jid, state="running", phase="predicting")
        export_jobs.set_total(jid, len(slice_indices))

        runs: dict[str, str] = {}
        errors: list[dict[str, Any]] = []
        cancelled = False
        pool_size = max(1, int(os.getenv("IPRED_APPLY_CONCURRENCY", _DEFAULT_APPLY_CONCURRENCY)))

        def _publish_result() -> None:
            """Update the job's `result` after every completed slice, not
            just once at the end — otherwise a UI polling this job while it
            runs sees `result: null` the entire time, no matter how many
            slices have actually already predicted (same class of bug fixed
            in infer_jobs.py's dlsia inference job for the same reason)."""
            export_jobs.update(jid, result={"runs": dict(runs), "errors": list(errors), "cancelled": cancelled})

        with ipred_client_mod.new_shared_client() as client, \
                concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool:
            pending = iter(slice_indices)
            in_flight: dict[concurrent.futures.Future, int] = {}

            def submit_next() -> bool:
                si = next(pending, None)
                if si is None:
                    return False
                fut = pool.submit(
                    _apply_one_slice,
                    si,
                    client=client,
                    session_id=session_id,
                    model_id=model_id,
                    composition_id=composition_id,
                    feature_setup_id=feature_setup_id,
                    alpha=alpha,
                )
                in_flight[fut] = si
                return True

            for _ in range(pool_size):
                if not submit_next():
                    break

            while in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    del in_flight[fut]
                    slice_index, run_id, error = fut.result()
                    if error is not None:
                        logger.warning("ipred volume apply: slice %d failed (%s)", slice_index, error)
                        errors.append({"slice": slice_index, "error": error})
                    else:
                        runs[str(slice_index)] = run_id  # type: ignore[assignment]
                        export_jobs.log(jid, f"slice {slice_index}: predicted (run {run_id[:8]})")
                    export_jobs.bump(jid, 1)
                    _publish_result()

                if export_jobs.cancel_requested(jid):
                    cancelled = True
                    # Already-submitted work can't be un-submitted (Python
                    # threads have no preemptive cancel) — just stop
                    # refilling the pool so it drains rather than growing.
                    continue
                # Refill one slot per completed future, keeping ~pool_size
                # in flight rather than waiting for a full batch to finish.
                for _ in done:
                    submit_next()

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
