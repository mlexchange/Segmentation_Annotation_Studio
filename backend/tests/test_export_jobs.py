"""Tests for export_jobs' shared job registry, focused on PROFILE_JOBS timing."""
import importlib
import os

import pytest

import export_jobs


@pytest.fixture(autouse=True)
def _restore_default_module_state():
    """PROFILE_JOBS is read once at import — reloading the module to flip it
    (see _reload_with_profile) mutates process-global state that outlives
    monkeypatch's own env var cleanup, regardless of fixture teardown order.
    Explicitly clear the env var and reload back to the un-profiled default
    after every test so other test files never see a leftover reload here."""
    yield
    os.environ.pop("PROFILE_JOBS", None)
    importlib.reload(export_jobs)


def _reload_with_profile(monkeypatch, enabled: bool):
    """PROFILE_JOBS is read once at import time — reload the module under a
    patched env var rather than mutating a private flag directly, so the test
    exercises the real startup path."""
    monkeypatch.setenv("PROFILE_JOBS", "1" if enabled else "0")
    module = importlib.reload(export_jobs)
    return module


def test_log_has_no_timing_suffix_by_default(monkeypatch):
    module = _reload_with_profile(monkeypatch, enabled=False)
    jid = module.new_job("some/path")
    module.log(jid, "starting")
    job = module.get_job(jid)
    assert job["log"] == ["starting"]


def test_log_adds_elapsed_suffix_when_profiling_enabled(monkeypatch):
    module = _reload_with_profile(monkeypatch, enabled=True)
    jid = module.new_job("some/path")
    module.log(jid, "starting")
    module.log(jid, "done")
    job = module.get_job(jid)
    assert len(job["log"]) == 2
    for line in job["log"]:
        assert "(+" in line and line.endswith("s)")


def test_internal_last_log_at_field_never_leaks_into_snapshot(monkeypatch):
    module = _reload_with_profile(monkeypatch, enabled=True)
    jid = module.new_job("some/path")
    module.log(jid, "starting")
    job = module.get_job(jid)
    assert "_last_log_at" not in job
    assert "zip_path" not in job
