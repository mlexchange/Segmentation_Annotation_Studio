"""run_probe_job's cancellation semantics — orchestration only, no real model.

build_family and _try_batch are mocked so these run torch-free and exercise
ONLY run_probe_job's own loop/result-shape logic: distinguishing "cancelled"
from a genuine out-of-memory ceiling, and never reporting a suggestion that
wasn't actually measured.
"""

from __future__ import annotations

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
