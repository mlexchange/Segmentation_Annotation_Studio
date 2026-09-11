"""Tests for the parallelized iPred volume-apply job (ipred_batch_jobs.py).

Uses real export_jobs (in-memory registry, no I/O) and monkeypatches
ipred_client_mod so no real HTTP/ipred service is needed.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest

import export_jobs
import ipred_batch_jobs
import ipred_client as ipred_client_mod


@contextmanager
def _fake_shared_client():
    yield object()  # never actually used for HTTP in these tests


@pytest.fixture(autouse=True)
def fake_shared_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ipred_client_mod, "new_shared_client", _fake_shared_client)


def test_one_failing_slice_does_not_abort_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_preprocess(*, session_id, feature_setup_id, composition_id, slice_index, client):
        if slice_index == 2:
            raise RuntimeError("boom")
        return {"feature_id": f"fid-{slice_index}"}

    def fake_infer(*, session_id, model_id, feature_id, alpha, store_probabilities, client):
        assert store_probabilities is False  # volume-apply must never persist proba.npy
        return {"run_id": f"run-{feature_id}"}

    deleted: list[str] = []

    def fake_delete(feature_id, *, client):
        deleted.append(feature_id)
        return {"deleted": True}

    monkeypatch.setattr(ipred_client_mod, "preprocess", fake_preprocess)
    monkeypatch.setattr(ipred_client_mod, "infer", fake_infer)
    monkeypatch.setattr(ipred_client_mod, "delete_feature_bank", fake_delete)

    jid = export_jobs.new_job("")
    ipred_batch_jobs.run_ipred_volume_apply_job(
        jid,
        session_id="s1",
        model_id="m1",
        slice_indices=[0, 1, 2, 3, 4],
        composition_id=None,
        feature_setup_id=None,
        alpha=0.05,
    )

    job = export_jobs.get_job(jid)
    assert job["state"] == "done"
    result = job["result"]
    assert sorted(int(k) for k in result["runs"]) == [0, 1, 3, 4]
    assert result["errors"] == [{"slice": 2, "error": "boom"}]
    assert result["cancelled"] is False
    # No feature bank leaked, including for the failed slice (preprocess for
    # slice 2 never succeeded, so there's nothing of its to delete).
    assert sorted(deleted) == ["fid-0", "fid-1", "fid-3", "fid-4"]


def test_feature_bank_is_released_even_when_infer_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_preprocess(*, session_id, feature_setup_id, composition_id, slice_index, client):
        return {"feature_id": f"fid-{slice_index}"}

    def fake_infer(*, session_id, model_id, feature_id, alpha, store_probabilities, client):
        raise RuntimeError("infer exploded")

    deleted: list[str] = []
    monkeypatch.setattr(ipred_client_mod, "preprocess", fake_preprocess)
    monkeypatch.setattr(ipred_client_mod, "infer", fake_infer)
    monkeypatch.setattr(
        ipred_client_mod, "delete_feature_bank",
        lambda feature_id, *, client: deleted.append(feature_id),
    )

    jid = export_jobs.new_job("")
    ipred_batch_jobs.run_ipred_volume_apply_job(
        jid, session_id="s1", model_id="m1", slice_indices=[0, 1],
        composition_id=None, feature_setup_id=None, alpha=0.05,
    )

    job = export_jobs.get_job(jid)
    # Both slices failed (infer always raises) -> job reports an error state,
    # but the orphaned-bank cleanup must still have run for each.
    assert job["state"] == "error"
    assert sorted(deleted) == ["fid-0", "fid-1"]


def test_cancellation_stops_submitting_new_slices(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[int] = []
    lock = threading.Lock()

    def fake_preprocess(*, session_id, feature_setup_id, composition_id, slice_index, client):
        with lock:
            started.append(slice_index)
        # First slice requests cancellation while still "in flight", so the
        # pool must not keep refilling after it's observed.
        if slice_index == 0:
            export_jobs.request_cancel(jid_holder["jid"])
            time.sleep(0.05)
        return {"feature_id": f"fid-{slice_index}"}

    def fake_infer(*, session_id, model_id, feature_id, alpha, store_probabilities, client):
        return {"run_id": f"run-{feature_id}"}

    monkeypatch.setattr(ipred_client_mod, "preprocess", fake_preprocess)
    monkeypatch.setattr(ipred_client_mod, "infer", fake_infer)
    monkeypatch.setattr(ipred_client_mod, "delete_feature_bank", lambda feature_id, *, client: None)
    monkeypatch.setenv("IPRED_APPLY_CONCURRENCY", "1")  # deterministic: one slice at a time

    jid_holder: dict[str, str] = {}
    jid = export_jobs.new_job("")
    jid_holder["jid"] = jid

    ipred_batch_jobs.run_ipred_volume_apply_job(
        jid, session_id="s1", model_id="m1", slice_indices=[0, 1, 2, 3, 4],
        composition_id=None, feature_setup_id=None, alpha=0.05,
    )

    job = export_jobs.get_job(jid)
    assert job["result"]["cancelled"] is True
    # With concurrency=1, cancellation is observed right after slice 0
    # completes — no later slice should ever have started.
    assert started == [0]


def test_result_is_published_incrementally_while_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: result.runs must be visible WHILE the job is still
    `running`, not only once the whole volume finishes — otherwise a live
    per-slice preview in the UI (Annotate's PixelClassifierPanel) has nothing
    to show for slices that already predicted, no matter how far along the
    job is (this is exactly what a user reported live: slice 2 of a 690-slice
    apply job stayed blank even though 11 slices had already predicted)."""
    def fake_preprocess(*, session_id, feature_setup_id, composition_id, slice_index, client):
        time.sleep(0.03)
        return {"feature_id": f"fid-{slice_index}"}

    def fake_infer(*, session_id, model_id, feature_id, alpha, store_probabilities, client):
        return {"run_id": f"run-{feature_id}"}

    monkeypatch.setattr(ipred_client_mod, "preprocess", fake_preprocess)
    monkeypatch.setattr(ipred_client_mod, "infer", fake_infer)
    monkeypatch.setattr(ipred_client_mod, "delete_feature_bank", lambda feature_id, *, client: None)
    monkeypatch.setenv("IPRED_APPLY_CONCURRENCY", "2")

    n_slices = 8
    jid = export_jobs.new_job("")
    thread = threading.Thread(
        target=ipred_batch_jobs.run_ipred_volume_apply_job,
        kwargs=dict(
            jid=jid, session_id="s1", model_id="m1", slice_indices=list(range(n_slices)),
            composition_id=None, feature_setup_id=None, alpha=0.05,
        ),
        daemon=True,
    )
    thread.start()

    saw_partial_progress = False
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        job = export_jobs.get_job(jid)
        if job["state"] == "running" and job.get("result"):
            n = len(job["result"].get("runs", {}))
            if 0 < n < n_slices:
                saw_partial_progress = True
                break
        if job["state"] in ("done", "error"):
            break
        time.sleep(0.005)
    thread.join(timeout=10.0)

    assert saw_partial_progress, "expected to observe a partial (running) result with some but not all slices done"
    job = export_jobs.get_job(jid)
    assert job["state"] == "done"
    assert len(job["result"]["runs"]) == n_slices
