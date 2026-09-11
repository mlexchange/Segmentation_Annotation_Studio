"""Measure the largest training batch size that fits in device memory.

An analytic estimate would have to model activation storage, attention kernels
and allocator fragmentation, and be recomputed for every arch — so instead this
runs the real thing: build the model via :func:`train_common.build_family`, the
same function :mod:`train_jobs` uses, then do a genuine forward + backward +
optimizer step at 1, 2, 4, 8… until it runs out of memory. The largest size
that actually completed is what gets reported, because it was actually
executed rather than predicted.

Reports through the :mod:`export_jobs` registry, so the frontend polls the same
``GET /api/export/status/{job_id}`` route it already uses for exports, training
and inference. Holds :data:`train_common.ML_LOCK` for the same reason training
does — the probe deliberately fills device memory, so nothing else may be using
it at the same time.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

import export_jobs
import train_common
from schemas import BatchProbeRequest

logger = logging.getLogger(__name__)

# Doubling from 1 finds the ceiling in log2(max) steps. Capped by the schema's own
# batch_size bound, so a suggestion is always a value the user could submit.
_MAX_PROBE = 64
# Keep this fraction of the largest working size, so normal variation between
# slices (and whatever else the machine picks up later) doesn't OOM a long run.
_SAFETY_FACTOR = 0.8
# How often to check for a cancel request while an attempt is still running.
# A single real training step for a large backbone at a large batch size can
# take far longer than the whole rest of the probe combined — without this,
# "cancel" only ever took effect between attempts, so a stuck/slow attempt at,
# say, batch 16 on a 7B-parameter model made cancel look broken: the request
# was recorded immediately (see `cancel_requested`) but nothing visibly
# happened until that one attempt finally finished, however long that took.
_ATTEMPT_POLL_SECONDS = 1.0


def schema_batch_cap(hyperparams: Any, default: int = _MAX_PROBE) -> int:
    """Upper bound the schema allows for ``batch_size``.

    Suggesting a value the user cannot actually submit would be useless, so the
    probe never reports above this. Pydantic keeps the constraints as an unordered
    metadata list (``[Ge(...), Le(...)]``), hence the scan rather than an index.
    """
    try:
        for constraint in type(hyperparams).model_fields["batch_size"].metadata:
            le = getattr(constraint, "le", None)
            if le is not None:
                return min(default, int(le))
    except Exception:  # noqa: BLE001 — fall back to the probe's own ceiling
        pass
    return default


def _attempt_count(cap: int) -> int:
    """Number of doubling attempts (1, 2, 4, …) up to and including *cap* —
    the progress total, since that is the loop's actual step count."""
    return max(1, cap.bit_length())


def _suggest(largest_ok: int, cap: int) -> int:
    """Batch size to report given the largest one that actually completed.

    Backs off to :data:`_SAFETY_FACTOR` of it: the probe runs synthetic zeros
    on an otherwise-idle device, so real training has slightly more to hold,
    and anything else running on the machine later competes for the same
    memory. Never below 1 or above what was actually measured.
    """
    return max(1, min(cap, int(largest_ok * _SAFETY_FACTOR)))


def _device_memory_gib(device: str) -> tuple[float, float] | None:
    """``(allocated, budget)`` in GiB for *device*, or None if not reportable."""
    import torch  # noqa: PLC0415

    try:
        if device == "mps":
            return (
                torch.mps.current_allocated_memory() / 2**30,
                torch.mps.recommended_max_memory() / 2**30,
            )
        if device == "cuda":
            free, total = torch.cuda.mem_get_info()
            return ((total - free) / 2**30, total / 2**30)
    except Exception:  # noqa: BLE001 — reporting only, never fatal
        return None
    return None


def _release(device: str) -> None:
    """Drop cached blocks so the next attempt starts from a clean allocator."""
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    gc.collect()
    try:
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _is_oom(exc: BaseException) -> bool:
    """True for an out-of-memory failure rather than a genuine bug.

    MPS and CUDA report OOM as differently-worded RuntimeErrors, so match on the
    message; anything unrecognised propagates instead of being misreported as a
    memory ceiling.
    """
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("out of memory", "insufficient memory", "can't allocate", "cannot allocate")
    )


def _try_batch(
    batch_size: int,
    *,
    image_size: int,
    forward_fn: Any,
    trainable: list[Any],
    device: str,
) -> None:
    """Run one real training step at *batch_size*, raising on OOM.

    Uses synthetic tensors of the exact shape and dtype the training loop feeds,
    and includes ``backward()`` plus an optimizer step — the backward pass is what
    actually holds activations, so a forward-only probe would overestimate badly.
    All-zero labels work regardless of class count: it's already baked into
    `forward_fn`'s output channels (fixed when the model was built), so
    there's nothing here for an explicit `n_classes` to do.
    """
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    images = torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.float32, device=device)
    labels = torch.zeros((batch_size, image_size, image_size), dtype=torch.int64, device=device)
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=train_common.IGNORE_INDEX)

    logits = forward_fn(images)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()  # errors surface lazily otherwise


def _run_attempt_cancellable(
    executor: concurrent.futures.ThreadPoolExecutor,
    jid: str,
    batch_size: int,
    *,
    image_size: int,
    forward_fn: Any,
    trainable: list[Any],
    device: str,
) -> str:
    """Run one :func:`_try_batch` call on *executor*, polling for cancellation
    every :data:`_ATTEMPT_POLL_SECONDS` while it's in flight.

    Returns ``"ok"``, ``"oom"``, or ``"cancelled"``. A real torch op can't be
    safely interrupted mid-flight (there's no signal that unwinds a native
    MPS/CUDA kernel), so ``"cancelled"`` does NOT mean the attempt has
    actually stopped — it may still be running on *executor*'s thread. The
    caller must not submit another attempt, or let the caller's caller
    release the device lock, until that thread is confirmed done (see
    ``executor.shutdown(wait=True)`` in :func:`run_probe_job`) — two attempts
    racing on the same device, or a second job starting while this one is
    still quietly using it, would corrupt or OOM both.
    """
    future = executor.submit(
        _try_batch, batch_size, image_size=image_size, forward_fn=forward_fn, trainable=trainable, device=device
    )
    while True:
        try:
            future.result(timeout=_ATTEMPT_POLL_SECONDS)
            return "ok"
        except concurrent.futures.TimeoutError:
            if export_jobs.cancel_requested(jid):
                return "cancelled"
            continue
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                return "oom"
            raise


def run_probe_job(jid: str, request: BatchProbeRequest) -> None:
    """Background worker: find the largest batch size that completes a real step."""
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

        hp = request.model.hyperparams
        image_size = hp.image_size
        cap = schema_batch_cap(hp)
        attempts = _attempt_count(cap)

        def _log(msg: str) -> None:
            export_jobs.log(jid, msg)

        mem = _device_memory_gib(device)
        if mem:
            _log(f"Device {device}: {mem[1]:.0f} GiB budget, {mem[0]:.1f} GiB already in use")

        built = train_common.build_family(request.model, request.n_classes, device, _log)
        forward_fn, trainable = built.forward_fn, built.trainable_params

        export_jobs.update(jid, phase="probing")
        export_jobs.set_total(jid, attempts)

        largest_ok = 0
        first_failure: int | None = None
        cancelled = False
        in_flight_at_cancel: int | None = None
        size = 1
        # A dedicated single-worker executor, not a bare thread, because of what
        # happens on a "cancelled" outcome below: the in-flight attempt is NOT
        # actually stopped (no signal safely unwinds a native MPS/CUDA kernel
        # mid-flight), just no longer waited on eagerly. The `with` block's
        # implicit `shutdown(wait=True)` on exit — whether via `return`, `raise`,
        # or falling off the end below — is what guarantees that attempt has
        # really finished before this function can return and let the caller
        # release ML_LOCK. Every report to `export_jobs` happens BEFORE that,
        # i.e. still inside this block: the whole point is that the user sees
        # "cancelled" within `_ATTEMPT_POLL_SECONDS`, not whenever that last
        # attempt happens to actually finish. `capability.busy` staying true for
        # a bit after the job itself reports done is that drain, made visible —
        # real, not a bug.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while size <= cap:
                if export_jobs.cancel_requested(jid):
                    cancelled = True
                    break
                outcome = _run_attempt_cancellable(
                    executor, jid, size, image_size=image_size, forward_fn=forward_fn, trainable=trainable,
                    device=device,
                )
                if outcome == "cancelled":
                    cancelled = True
                    in_flight_at_cancel = size
                    break
                if outcome == "oom":
                    first_failure = size
                    _log(f"batch {size}: out of memory")
                    export_jobs.bump(jid, 1)
                    _release(device)
                    break
                used = _device_memory_gib(device)
                _log(f"batch {size}: ok" + (f" ({used[0]:.1f} GiB in use)" if used else ""))
                largest_ok = size
                export_jobs.bump(jid, 1)
                _release(device)
                size *= 2

            still_finishing = (
                f" (batch {in_flight_at_cancel} may still finish in the background momentarily)"
                if in_flight_at_cancel is not None
                else ""
            )

            # Cancelling before any size completed is a deliberate stop, not the
            # device genuinely refusing batch 1 — those must not read the same. A
            # real "batch 1 doesn't fit" still raises, since there is no batch
            # size to suggest.
            if largest_ok == 0 and cancelled:
                note = f"Cancelled before any batch size could be measured.{still_finishing}"
                _log(note)
                export_jobs.update(
                    jid,
                    state="done",
                    phase="done",
                    result={
                        "suggested_batch_size": None,
                        "largest_ok": 0,
                        "first_failure": first_failure,
                        "image_size": image_size,
                        "device": device,
                        "probe_cap": cap,
                        "cancelled": True,
                        "note": note,
                    },
                )
                return
            if largest_ok == 0:
                raise RuntimeError(
                    f"Even batch size 1 ran out of memory at {image_size}px. "
                    "Reduce the patch size, or pick a smaller model."
                )

            suggested = _suggest(largest_ok, cap)
            if cancelled:
                note = (
                    f"Cancelled after batch {largest_ok} — this measurement is partial. "
                    f"Suggesting {suggested} to leave headroom.{still_finishing}"
                )
            else:
                note = (
                    f"Largest that ran: {largest_ok}"
                    + (f"; {first_failure} ran out of memory" if first_failure else f"; stopped at the {cap} cap")
                    + f". Suggesting {suggested} to leave headroom."
                )
            _log(note)

            export_jobs.update(
                jid,
                state="done",
                phase="done",
                result={
                    "suggested_batch_size": suggested,
                    "largest_ok": largest_ok,
                    "first_failure": first_failure,
                    "image_size": image_size,
                    "device": device,
                    "probe_cap": cap,
                    "cancelled": cancelled,
                    "note": note,
                },
            )
    except Exception as exc:  # noqa: BLE001 — reported as a job error, never a crash
        logger.error("Batch-size probe %s failed: %s", jid, exc)
        export_jobs.update(jid, state="error", phase="error", error=str(exc))
    finally:
        try:
            _release(train_common.pick_device() or "cpu")
        finally:
            train_common.ML_LOCK.release()
