"""In-memory job registry for COCO/mask exports.

Mirrors the ingest job pattern (ingest.py): export runs on a background thread
and reports phase/progress/log lines that the UI polls via
``/api/export/status/{job_id}``. Kept deliberately simple — a process-local dict
guarded by a lock; jobs are ephemeral and fine to lose on restart.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_MAX_LOG = 200

# Per-stage timing is opt-in (off by default in production) — see log()'s own
# doc for why. Read once at import: this flag is meant for a deliberate
# profiling run (locally or in CI), not something toggled mid-process.
_PROFILE = os.getenv("PROFILE_JOBS") == "1"


def new_job(dataset_path: str) -> str:
    """Register a new export job and return its id."""
    jid = uuid.uuid4().hex
    with _lock:
        _jobs[jid] = {
            "state": "pending",   # pending | running | done | error
            "phase": "queued",
            "done": 0,
            "total": 0,
            "log": [],
            "result": None,
            "error": None,
            "dataset_path": dataset_path,
            "zip_path": None,     # internal; surfaced as result.zip_available
            "cancel_requested": False,
            "_last_log_at": time.monotonic(),  # internal; PROFILE_JOBS timing only
        }
    return jid


def request_cancel(jid: str) -> bool:
    """Ask a running job to stop; returns False if the job is unknown.

    Cooperative: this only sets a flag. Long-running jobs poll
    :func:`cancel_requested` between units of work and stop at a clean boundary,
    which is what lets a half-finished dataset be discarded rather than left
    looking complete.
    """
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return False
        job["cancel_requested"] = True
        return True


def cancel_requested(jid: str) -> bool:
    """True if cancellation was requested for *jid*.

    Reads the flag directly under the lock rather than through :func:`get_job`,
    which copies the whole job dict (log lines included) just to read one bool —
    and this is polled on every slice of a running job.
    """
    with _lock:
        job = _jobs.get(jid)
        return bool(job and job.get("cancel_requested"))


def get_job(jid: str) -> dict | None:
    """Return a snapshot copy of the job (without the internal zip_path)."""
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return None
        snap = dict(job)
        snap.pop("zip_path", None)
        snap.pop("_last_log_at", None)
        return snap


def zip_path(jid: str) -> str | None:
    with _lock:
        job = _jobs.get(jid)
        return job.get("zip_path") if job else None


def update(jid: str, **kw: Any) -> None:
    with _lock:
        if jid in _jobs:
            _jobs[jid].update(kw)


def set_total(jid: str, total: int) -> None:
    with _lock:
        if jid in _jobs:
            _jobs[jid]["total"] = total


def bump(jid: str, n: int = 1) -> None:
    with _lock:
        if jid in _jobs:
            _jobs[jid]["done"] += n


def log(jid: str, message: str) -> None:
    """Append a log line, called at every natural stage boundary by every job
    type (iPred batch jobs, dlsia train/infer).

    When ``PROFILE_JOBS=1`` is set, each line gets an elapsed-since-previous-
    log suffix (e.g. ``"Wrote foo (+1.23s)"``) — a cheap per-stage timing
    signal with zero new call sites, since every job already calls this at
    the boundaries that matter. Off by default: real users/deployments should
    see zero timing overhead, not "probably negligible" — this is meant to
    run in the test suite (real job round-trips with profiling on) and local
    benchmarking, not every production request.
    """
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return
        if _PROFILE:
            now = time.monotonic()
            elapsed = now - job.get("_last_log_at", now)
            job["_last_log_at"] = now
            message = f"{message} (+{elapsed:.2f}s)"
        job["log"].append(message)
        if len(job["log"]) > _MAX_LOG:
            del job["log"][: len(job["log"]) - _MAX_LOG]
