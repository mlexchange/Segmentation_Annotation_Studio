"""Tests for export_jobs' cancel_requested helper.

Consolidates what used to be five separate inline spellings of
`bool((get_job(jid) or {}).get("cancel_requested"))` across train_jobs.py,
infer_jobs.py and batch_probe.py.
"""

from __future__ import annotations

import export_jobs


def test_cancel_requested_is_false_by_default() -> None:
    jid = export_jobs.new_job("test")
    assert export_jobs.cancel_requested(jid) is False


def test_cancel_requested_true_after_being_set() -> None:
    jid = export_jobs.new_job("test")
    export_jobs.update(jid, cancel_requested=True)
    assert export_jobs.cancel_requested(jid) is True


def test_cancel_requested_is_false_for_an_unknown_job() -> None:
    assert export_jobs.cancel_requested("does-not-exist") is False
