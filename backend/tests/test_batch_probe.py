"""Tests for batch_probe.py — pure helpers directly, plus real (no mocks)
CPU forward+backward+optimizer-step round trips via train_common.build_family
(same tiny-model pattern as test_train_e2e_real_ml.py), and the job-level
guard/OOM/cancellation paths via a monkeypatched _try_batch."""
from __future__ import annotations

import concurrent.futures

import pytest

torch = pytest.importorskip("torch")

import batch_probe  # noqa: E402
import export_jobs  # noqa: E402
import train_common  # noqa: E402
from schemas import BatchProbeRequest, DlsiaTunetConfig  # noqa: E402


def _request(image_size=64, batch_size=2, depth=2, base_channels=4):
    return BatchProbeRequest(
        model=DlsiaTunetConfig(
            hyperparams={
                "depth": depth, "base_channels": base_channels,
                "image_size": image_size, "batch_size": batch_size,
            },
        ),
        n_classes=2,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestSchemaBatchCap:
    def test_returns_the_schema_upper_bound_when_lower_than_default(self):
        hp = _request().model.hyperparams
        cap = batch_probe.schema_batch_cap(hp, default=1000)
        assert cap <= 1000

    def test_default_used_when_no_le_constraint_found(self):
        class NoConstraints:
            model_fields = {}

        assert batch_probe.schema_batch_cap(NoConstraints(), default=42) == 42

    def test_broken_metadata_falls_back_to_default(self):
        class Broken:
            pass

        assert batch_probe.schema_batch_cap(Broken(), default=7) == 7


class TestAttemptCount:
    def test_cap_one_is_one_attempt(self):
        assert batch_probe._attempt_count(1) == 1

    def test_higher_caps_scale_with_bit_length(self):
        assert batch_probe._attempt_count(64) == 64 .bit_length()

    def test_never_below_one(self):
        assert batch_probe._attempt_count(0) >= 1


class TestSuggest:
    def test_applies_safety_factor(self):
        assert batch_probe._suggest(10, cap=64) == int(10 * batch_probe._SAFETY_FACTOR)

    def test_never_below_one(self):
        assert batch_probe._suggest(1, cap=64) >= 1

    def test_never_exceeds_cap(self):
        assert batch_probe._suggest(1000, cap=8) <= 8


class TestIsOom:
    @pytest.mark.parametrize("message", [
        "CUDA out of memory. Tried to allocate...",
        "MPS backend out of memory",
        "Insufficient Memory!",
        "can't allocate memory",
        "Cannot allocate 4GB",
    ])
    def test_recognizes_oom_messages_case_insensitively(self, message):
        assert batch_probe._is_oom(RuntimeError(message)) is True

    def test_unrelated_error_is_not_oom(self):
        assert batch_probe._is_oom(RuntimeError("shape mismatch")) is False


class TestDeviceMemoryGib:
    def test_cpu_reports_nothing(self):
        assert batch_probe._device_memory_gib("cpu") is None

    def test_unknown_device_reports_nothing(self):
        assert batch_probe._device_memory_gib("weird") is None


class TestRelease:
    def test_cpu_does_not_raise(self):
        batch_probe._release("cpu")  # gc.collect() only; no torch cache to clear


# ---------------------------------------------------------------------------
# _try_batch — real forward + backward + optimizer step on CPU
# ---------------------------------------------------------------------------

class TestTryBatchReal:
    def test_completes_a_real_step_without_raising(self):
        req = _request()
        built = train_common.build_family(req.model, req.n_classes, "cpu", lambda msg: None)
        batch_probe._try_batch(
            2, image_size=16, forward_fn=built.forward_fn, trainable=built.trainable_params, device="cpu",
        )

    def test_updates_model_parameters(self):
        req = _request()
        built = train_common.build_family(req.model, req.n_classes, "cpu", lambda msg: None)
        before = [p.clone() for p in built.trainable_params]
        batch_probe._try_batch(
            2, image_size=16, forward_fn=built.forward_fn, trainable=built.trainable_params, device="cpu",
        )
        after = built.trainable_params
        assert any(not torch.equal(b, a) for b, a in zip(before, after))


# ---------------------------------------------------------------------------
# _run_attempt_cancellable
# ---------------------------------------------------------------------------

class TestRunAttemptCancellable:
    def test_returns_ok_on_success(self):
        # A real trainable module, not an identity lambda: `_try_batch` calls
        # `loss.backward()`, which needs a real grad_fn in the graph.
        conv = torch.nn.Conv2d(3, 3, kernel_size=1)
        jid = export_jobs.new_job("x")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            outcome = batch_probe._run_attempt_cancellable(
                executor, jid, 1, image_size=8, forward_fn=conv, trainable=list(conv.parameters()), device="cpu",
            )
        assert outcome == "ok"

    def test_returns_oom_when_try_batch_raises_oom_message(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(batch_probe, "_try_batch", boom)
        jid = export_jobs.new_job("x")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            outcome = batch_probe._run_attempt_cancellable(
                executor, jid, 1, image_size=8, forward_fn=lambda x: x, trainable=[], device="cpu",
            )
        assert outcome == "oom"

    def test_reraises_non_oom_exceptions(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("something else broke")

        monkeypatch.setattr(batch_probe, "_try_batch", boom)
        jid = export_jobs.new_job("x")
        with pytest.raises(RuntimeError, match="something else broke"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                batch_probe._run_attempt_cancellable(
                    executor, jid, 1, image_size=8, forward_fn=lambda x: x, trainable=[], device="cpu",
                )

    def test_polls_for_cancellation_while_attempt_runs(self, monkeypatch):
        import time

        def slow(*a, **k):
            time.sleep(1.5)

        monkeypatch.setattr(batch_probe, "_try_batch", slow)
        monkeypatch.setattr(batch_probe, "_ATTEMPT_POLL_SECONDS", 0.1)
        jid = export_jobs.new_job("x")
        export_jobs.request_cancel(jid)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            outcome = batch_probe._run_attempt_cancellable(
                executor, jid, 1, image_size=8, forward_fn=lambda x: x, trainable=[], device="cpu",
            )
        assert outcome == "cancelled"


# ---------------------------------------------------------------------------
# run_probe_job — guard paths and outcome-shaping logic
# ---------------------------------------------------------------------------

class TestRunProbeJobGuards:
    def test_busy_ml_lock_reports_error_not_a_crash(self):
        train_common.ML_LOCK.acquire()
        try:
            jid = export_jobs.new_job("x")
            batch_probe.run_probe_job(jid, _request())
            job = export_jobs.get_job(jid)
            assert job["state"] == "error"
            assert "already running" in job["error"]
        finally:
            train_common.ML_LOCK.release()

    def test_no_device_reports_error(self, monkeypatch):
        monkeypatch.setattr(train_common, "pick_device", lambda: None)
        jid = export_jobs.new_job("x")
        batch_probe.run_probe_job(jid, _request())
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "torch is not installed" in job["error"]
        assert train_common.ML_LOCK.locked() is False  # released even on early failure


class TestRunProbeJobOutcomes:
    def test_real_probe_succeeds_and_releases_lock(self):
        jid = export_jobs.new_job("x")
        batch_probe.run_probe_job(jid, _request())
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["suggested_batch_size"] >= 1
        assert job["result"]["largest_ok"] >= 1
        assert job["result"]["cancelled"] is False
        assert train_common.ML_LOCK.locked() is False

    def test_oom_on_first_attempt_reports_no_suggestion(self, monkeypatch):
        def always_oom(*a, **k):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(batch_probe, "_try_batch", always_oom)
        jid = export_jobs.new_job("x")
        batch_probe.run_probe_job(jid, _request())
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "Even batch size 1" in job["error"]

    def test_oom_after_some_success_suggests_a_smaller_batch(self, monkeypatch):
        calls = []

        def sometimes_oom(batch_size, **k):
            calls.append(batch_size)
            if batch_size > 2:
                raise RuntimeError("out of memory")

        monkeypatch.setattr(batch_probe, "_try_batch", sometimes_oom)
        jid = export_jobs.new_job("x")
        batch_probe.run_probe_job(jid, _request(batch_size=32))
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["largest_ok"] == 2
        assert job["result"]["first_failure"] == 4
        assert job["result"]["suggested_batch_size"] <= 2

    def test_cancelled_before_any_success_reports_done_not_error(self, monkeypatch):
        monkeypatch.setattr(export_jobs, "cancel_requested", lambda jid: True)
        jid = export_jobs.new_job("x")
        batch_probe.run_probe_job(jid, _request())
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["cancelled"] is True
        assert job["result"]["suggested_batch_size"] is None

    def test_cancelled_after_a_success_suggests_from_partial_progress(self, monkeypatch):
        calls = {"n": 0}

        def cancel_after_first(jid):
            calls["n"] += 1
            return calls["n"] > 1

        monkeypatch.setattr(export_jobs, "cancel_requested", cancel_after_first)
        jid = export_jobs.new_job("x")
        batch_probe.run_probe_job(jid, _request(batch_size=32))
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["cancelled"] is True
        assert job["result"]["largest_ok"] >= 1
