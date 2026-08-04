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

# Terminal job records older than this are pruned on the next new_job() call,
# so a long-running server doesn't accumulate an unbounded in-memory registry.
_JOB_TTL_SECONDS = float(os.getenv("EXPORT_JOB_TTL_SECONDS", str(6 * 3600)))
_MAX_JOBS = int(os.getenv("EXPORT_MAX_JOBS", "500"))
_TERMINAL_STATES = frozenset({"done", "error"})


def _prune_jobs_locked() -> None:
    """Evict terminal jobs by age, then by count, while holding ``_lock``.

    Running/pending jobs are never evicted, and a completed job's ``zip_path``
    stays valid for as long as its record does — pruning here is also what
    eventually stops referencing (and lets the OS reclaim) old export zips.
    """
    now = time.monotonic()
    for jid, job in list(_jobs.items()):
        if job.get("state") in _TERMINAL_STATES and now - job.get("_created", now) > _JOB_TTL_SECONDS:
            del _jobs[jid]
    if len(_jobs) <= _MAX_JOBS:
        return
    terminal_by_age = sorted(
        (jid for jid, job in _jobs.items() if job.get("state") in _TERMINAL_STATES),
        key=lambda jid: _jobs[jid].get("_created", 0),
    )
    for jid in terminal_by_age[: len(_jobs) - _MAX_JOBS]:
        del _jobs[jid]


def new_job(dataset_path: str) -> str:
    """Register a new export job and return its id."""
    jid = uuid.uuid4().hex
    with _lock:
        _prune_jobs_locked()
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
            "_created": time.monotonic(),
        }
    return jid


def get_job(jid: str) -> dict | None:
    """Return a snapshot copy of the job (without internal bookkeeping fields)."""
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return None
        snap = dict(job)
        snap.pop("zip_path", None)
        snap.pop("_created", None)
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


def cancel_requested(jid: str) -> bool:
    """True if cancellation was requested for *jid* (see the ``*/cancel`` routes).

    Reads the flag directly under the lock rather than through :func:`get_job`,
    which copies the whole job dict (log lines included) just to read one bool —
    and this is called on every batch/slice/probe-step of a running job.
    """
    with _lock:
        job = _jobs.get(jid)
        return bool(job and job.get("cancel_requested"))


def log(jid: str, message: str) -> None:
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return
        job["log"].append(message)
        if len(job["log"]) > _MAX_LOG:
            del job["log"][: len(job["log"]) - _MAX_LOG]
