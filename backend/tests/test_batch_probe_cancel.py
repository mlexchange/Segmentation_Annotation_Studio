"""run_probe_job's cancellation semantics — orchestration only, no real model.

build_family and _try_batch are mocked so these run torch-free and exercise
ONLY run_probe_job's own loop/result-shape logic: distinguishing "cancelled"
from a genuine out-of-memory ceiling, and never reporting a suggestion that
wasn't actually measured.
"""

from __future__ import annotations

import threading
import time

import batch_probe
import export_jobs
import train_common
from schemas import BatchProbeRequest, DlsiaTunetConfig


def _request() -> BatchProbeRequest:
    return BatchProbeRequest(model=DlsiaTunetConfig(hyperparams={"image_size": 64}))


def _fake_built() -> train_common.BuiltFamily:
    return train_common.BuiltFamily(
        forward_fn=None,
        to_tensor_fn=None,
        trainable_params=[],
        set_train_mode=None,
        model_config_snapshot={},
        adapter_state_fn=lambda: {},
    )


def _install_fakes(monkeypatch, try_batch) -> None:
    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")
    monkeypatch.setattr(train_common, "build_family", lambda model_cfg, n_classes, device, log: _fake_built())
    monkeypatch.setattr(batch_probe, "_try_batch", try_batch)
    monkeypatch.setattr(batch_probe, "_release", lambda device: None)
    monkeypatch.setattr(batch_probe, "_device_memory_gib", lambda device: None)


def test_cancel_before_the_first_attempt_is_not_reported_as_an_oom(monkeypatch) -> None:
    """Regression: cancelling before batch 1 completed used to raise "Even
    batch size 1 ran out of memory" — a real device-limitation message for
    what was actually a deliberate stop, and the frontend has no way to tell
    the two apart without a `cancelled` flag."""

    def _must_not_run(size, **kwargs):
        raise AssertionError("must not attempt any batch size when already cancelled")

    _install_fakes(monkeypatch, _must_not_run)
    jid = export_jobs.new_job("test")
    export_jobs.update(jid, cancel_requested=True)  # cancelled before the loop's first check

    batch_probe.run_probe_job(jid, _request())
    job = export_jobs.get_job(jid)

    assert job["state"] == "done"  # not "error" — this was not a real OOM
    assert not job.get("error")
    assert job["result"]["cancelled"] is True
    assert job["result"]["suggested_batch_size"] is None
    assert job["result"]["largest_ok"] == 0


def test_cancel_after_some_successes_keeps_the_measurement_but_flags_it(monkeypatch) -> None:
    """Regression: cancelling after e.g. batch 4 succeeded used to report
    state=done with NO `cancelled` flag and a note claiming it "stopped at the
    cap" — both false — so the frontend silently adopted a partial measurement
    as if it were the real ceiling."""
    def _try_batch(size, **kwargs):
        if size == 4:  # cancellation is only checked at the TOP of the next
            # iteration, so requesting it during batch 4's own attempt still
            # lets that attempt finish and count as a success.
            export_jobs.update(jid, cancel_requested=True)

    _install_fakes(monkeypatch, _try_batch)
    jid = export_jobs.new_job("test")

    batch_probe.run_probe_job(jid, _request())
    job = export_jobs.get_job(jid)

    assert job["state"] == "done"
    assert job["result"]["cancelled"] is True
    assert job["result"]["largest_ok"] == 4
    assert job["result"]["first_failure"] is None  # it was cancelled, not OOM'd
    assert isinstance(job["result"]["suggested_batch_size"], int)
    assert job["result"]["suggested_batch_size"] <= 4


def test_a_genuine_oom_at_batch_one_is_still_reported_as_an_error(monkeypatch) -> None:
    """The fix must not soften a REAL "doesn't even fit at 1" failure into a
    success — only actual cancellation gets the gentler treatment."""

    def _try_batch(size, **kwargs):
        raise RuntimeError("MPS backend out of memory (MPS allocated: 100.00 GB)")

    _install_fakes(monkeypatch, _try_batch)
    jid = export_jobs.new_job("test")

    batch_probe.run_probe_job(jid, _request())
    job = export_jobs.get_job(jid)

    assert job["state"] == "error"
    assert "out of memory" in job["error"].lower()


def test_reaching_the_cap_without_cancelling_is_not_reported_as_cancelled(monkeypatch) -> None:
    """The ordinary success path must still say `cancelled: False`, and the note
    must still say it hit the cap — the regression tests above must not have
    made every completed probe look cancelled."""

    def _try_batch(size, **kwargs):
        pass  # every size "succeeds"

    _install_fakes(monkeypatch, _try_batch)
    jid = export_jobs.new_job("test")

    batch_probe.run_probe_job(jid, _request())
    job = export_jobs.get_job(jid)

    assert job["state"] == "done"
    assert job["result"]["cancelled"] is False
    assert "cap" in job["result"]["note"]


# ---------------------------------------------------------------------------
# Cancelling an attempt that's still in flight (not just between attempts)
#
# Real regression: a single real training step for a large backbone at a large
# batch size can run far longer than the rest of the probe combined. Before
# _run_attempt_cancellable existed, `cancel_requested` was only ever checked
# BETWEEN attempts, so requesting cancellation while one was still running had
# no visible effect until that attempt finally finished — however long that
# took (in practice: user reports "cancel does nothing", repeated clicks, a
# probe stuck for many minutes on a 7B/Huge+ backbone).
# ---------------------------------------------------------------------------


def _slow_try_batch(release_event: threading.Event, hold_seconds: float = 5.0):
    """A `_try_batch` stand-in that blocks until `release_event` is set (or
    `hold_seconds` elapses, as a backstop so a failing test doesn't hang)."""

    def _fn(size, **kwargs):
        release_event.wait(timeout=hold_seconds)

    return _fn


def test_cancelling_a_genuinely_in_flight_attempt_is_noticed_within_the_poll_interval(monkeypatch) -> None:
    """The actual fix: a concurrent observer (the frontend's poller) must see
    "done" quickly — it must NOT have to wait for the slow attempt itself to
    return, which is exactly the old behaviour this replaces."""
    monkeypatch.setattr(batch_probe, "_ATTEMPT_POLL_SECONDS", 0.02)
    release = threading.Event()
    _install_fakes(monkeypatch, _slow_try_batch(release, hold_seconds=5.0))
    jid = export_jobs.new_job("test")

    job_thread = threading.Thread(target=batch_probe.run_probe_job, args=(jid, _request()))
    started = time.monotonic()
    job_thread.start()
    time.sleep(0.06)  # a few poll intervals in — batch 1 is "still running"
    export_jobs.update(jid, cancel_requested=True)

    deadline = time.monotonic() + 2.0
    while export_jobs.get_job(jid)["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    noticed_elapsed = time.monotonic() - started

    try:
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["cancelled"] is True
        assert job["result"]["largest_ok"] == 0
        # Well under the 5s the held attempt would otherwise take to return —
        # that gap is exactly what the old "check only between attempts"
        # behaviour cost the user.
        assert noticed_elapsed < 2.0
    finally:
        release.set()
        job_thread.join(timeout=5.0)


def test_the_note_says_the_in_flight_attempt_may_still_finish_in_the_background(monkeypatch) -> None:
    """capability.busy can stay true for a moment after this job reports
    done — that's the drained-before-lock-release guarantee, not a bug — so
    the note has to say so rather than implying the device is free right away."""
    monkeypatch.setattr(batch_probe, "_ATTEMPT_POLL_SECONDS", 0.02)
    release = threading.Event()
    _install_fakes(monkeypatch, _slow_try_batch(release, hold_seconds=5.0))
    jid = export_jobs.new_job("test")

    job_thread = threading.Thread(target=batch_probe.run_probe_job, args=(jid, _request()))
    job_thread.start()
    time.sleep(0.06)
    export_jobs.update(jid, cancel_requested=True)

    deadline = time.monotonic() + 2.0
    while export_jobs.get_job(jid)["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        note = export_jobs.get_job(jid)["result"]["note"]
        assert "batch 1" in note
        assert "background" in note
    finally:
        release.set()
        job_thread.join(timeout=5.0)


def test_ml_lock_is_not_released_until_the_in_flight_attempt_actually_finishes(monkeypatch) -> None:
    """The safety half of the fix: releasing the lock while the cancelled
    attempt still quietly holds the device would let a second job start and
    corrupt or OOM both. run_probe_job must not return (and so must not let
    its caller release ML_LOCK) until that attempt is confirmed done."""
    monkeypatch.setattr(batch_probe, "_ATTEMPT_POLL_SECONDS", 0.02)
    release = threading.Event()
    _install_fakes(monkeypatch, _slow_try_batch(release, hold_seconds=5.0))
    jid = export_jobs.new_job("test")
    lock_state_when_job_returned = {}

    def _run():
        batch_probe.run_probe_job(jid, _request())
        # By the time run_probe_job returns, the held attempt must have
        # actually finished — the fake only finishes when `release` is set,
        # so if the lock is already free here, it was released too early.
        lock_state_when_job_returned["release_was_set_first"] = release.is_set()

    runner = threading.Thread(target=_run)
    runner.start()
    time.sleep(0.06)
    export_jobs.update(jid, cancel_requested=True)

    deadline = time.monotonic() + 2.0
    while export_jobs.get_job(jid)["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    # The job already reports done/cancelled here, but the attempt is still
    # deliberately held — ML_LOCK must reflect that, not the report.
    still_locked_while_attempt_lingers = train_common.ML_LOCK.locked()

    release.set()
    runner.join(timeout=5.0)

    assert export_jobs.get_job(jid)["state"] == "done"
    assert still_locked_while_attempt_lingers is True
    assert lock_state_when_job_returned["release_was_set_first"] is True
