"""Tests for run_train_job's qlty-tiling wiring, torch-free.

Mocks train_common.prepare_datasets/run_training_loop/save_run and the family
builders so these exercise ONLY the orchestration in train_jobs.py: guard
ordering, phase transitions, and that a tiling/holdout failure is reported as a
job error rather than raised out of the background thread.
"""

from __future__ import annotations

import export_jobs
import train_common
import train_jobs
from schemas import AnnotationClass, DlsiaTunetConfig, ExportSourceItem, TrainRequest


def _classes() -> list[AnnotationClass]:
    return [AnnotationClass(classId=1, label="pore", color="#ff0000")]


def _source() -> ExportSourceItem:
    return ExportSourceItem(
        kind="local",
        source="sample.tif",
        slices={"0": [{"id": "s1", "classId": 1, "kind": "rectangle", "x": 0, "y": 0, "w": 5, "h": 5}]},
    )


def _request(*, tiling: bool) -> TrainRequest:
    return TrainRequest(
        sources=[_source()],
        classes=_classes(),
        model=DlsiaTunetConfig(hyperparams={"image_size": 64, "tiling": tiling}),
    )


def _run(monkeypatch, request: TrainRequest) -> dict:
    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")
    jid = export_jobs.new_job("test")
    train_jobs.run_train_job(jid, request, "run-1")
    return export_jobs.get_job(jid)


def test_qlty_unavailable_fails_before_preparing_datasets(monkeypatch) -> None:
    """Regression: the qlty check used to run AFTER prepare_datasets, so a
    torch-only install paid for rendering/rasterising every annotated slice
    before failing. It must fail before that work starts."""
    import tiling

    monkeypatch.setattr(tiling, "qlty_available", lambda: False)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("prepare_datasets must not run when qlty is unavailable")

    monkeypatch.setattr(train_common, "prepare_datasets", _must_not_run)

    job = _run(monkeypatch, _request(tiling=True))

    assert job["state"] == "error"
    assert "qlty" in job["error"]
    assert train_common.ML_LOCK.locked() is False  # released even on this early failure


def test_qlty_unavailable_does_not_block_a_non_tiled_run(monkeypatch) -> None:
    """The guard is scoped to tiling=True — a non-tiled run must not care
    whether qlty is installed at all."""
    import tiling

    monkeypatch.setattr(tiling, "qlty_available", lambda: False)
    monkeypatch.setattr(
        train_common,
        "prepare_datasets",
        lambda *a, **k: {"train": [], "val": []},
    )

    job = _run(monkeypatch, _request(tiling=False))

    # Gets past the qlty guard fine; fails later for the ordinary "no data" reason.
    assert job["state"] == "error"
    assert "No annotated slices" in job["error"]


def test_tiling_runs_after_qlty_check_passes(monkeypatch) -> None:
    """When qlty IS available, tiling.tile_datasets is actually invoked (and the
    phase transitions to "tiling") — the guard must not short-circuit the
    tiling step itself, only gate it."""
    import tiling

    monkeypatch.setattr(tiling, "qlty_available", lambda: True)
    monkeypatch.setattr(
        train_common,
        "prepare_datasets",
        lambda *a, **k: {"train": [(1, 2)], "val": []},
    )

    seen_phases: list[str] = []
    real_update = export_jobs.update

    def _tracking_update(jid, **kw):
        if "phase" in kw:
            seen_phases.append(kw["phase"])
        real_update(jid, **kw)

    monkeypatch.setattr(export_jobs, "update", _tracking_update)
    monkeypatch.setattr(
        tiling, "tile_datasets", lambda datasets, *a, **k: {"train": [], "val": []}
    )

    job = _run(monkeypatch, _request(tiling=True))

    assert "tiling" in seen_phases
    # tile_datasets above returns an EMPTY train set, so this must still hit the
    # ordinary "no annotated slices" check downstream — tiling doesn't bypass it.
    assert job["state"] == "error"
    assert "No annotated slices" in job["error"]


def test_cancel_during_tiling_reports_cancelled_without_a_run_id(monkeypatch) -> None:
    """Regression: tiling had no cancellation path at all. Cancelling here means
    no model was ever built, so this must NOT go through the ordinary
    partial-run-saved shape (that implies save_run ran and produced a run_id)."""
    import tiling

    monkeypatch.setattr(tiling, "qlty_available", lambda: True)
    monkeypatch.setattr(
        train_common, "prepare_datasets", lambda *a, **k: {"train": [(1, 2)], "val": []}
    )
    monkeypatch.setattr(tiling, "tile_datasets", lambda datasets, *a, **k: None)

    def _save_run_must_not_be_called(*a, **k):
        raise AssertionError("save_run must not run — no model was built before the cancel")

    monkeypatch.setattr(train_common, "save_run", _save_run_must_not_be_called)

    job = _run(monkeypatch, _request(tiling=True))

    assert job["state"] == "done"
    assert job["result"] == {"cancelled": True}


def test_holdout_log_message_appears_when_it_holds_patches_back(monkeypatch) -> None:
    import tiling

    monkeypatch.setattr(tiling, "qlty_available", lambda: True)
    monkeypatch.setattr(
        train_common, "prepare_datasets", lambda *a, **k: {"train": [(1, 2)], "val": []}
    )
    monkeypatch.setattr(
        tiling, "tile_datasets", lambda datasets, *a, **k: {"train": list(range(30)), "val": []}
    )
    monkeypatch.setattr(
        tiling, "holdout_val_patches", lambda datasets, **k: ({"train": list(range(27)), "val": list(range(3))}, 3)
    )

    job = _run(monkeypatch, _request(tiling=True))

    assert any("held back 3 training patch" in line for line in job["log"])
