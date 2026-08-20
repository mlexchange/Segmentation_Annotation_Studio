"""Tests for train_jobs' task dispatch: routing a task="denoising" request to
denoise_train instead of the annotation-driven segmentation path.

Most of these mock denoise_train/train_common so they exercise ONLY the
orchestration in train_jobs.py — guard ordering, phase transitions, what
reaches save_run — in the style of test_train_jobs.py. The last one is a real
end-to-end run (torch/dlsia/qlty required) proving the saved run shows up in
list_runs the way the Learned Denoiser panel expects.

See test_denoiser_train_jobs.py for the resume-compatibility half of the
denoiser's train_jobs surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import denoise_train
import export_jobs
import train_common
import train_jobs
from schemas import AnnotationClass, DlsiaDenoiserConfig, DlsiaTunetConfig, ExportSourceItem, TrainRequest


def _source(indices=(0, 1, 2, 3)) -> ExportSourceItem:
    return ExportSourceItem(
        kind="local",
        source="volume.tif",
        slices={str(i): [] for i in indices},
    )


def _denoise_request(*, scheme: str = "n2n", tiling: bool = True, **hp) -> TrainRequest:
    return TrainRequest(
        task="denoising",
        sources=[_source()],
        classes=[],
        model=DlsiaDenoiserConfig(
            training_scheme=scheme,
            hyperparams={"image_size": 64, "tiling": tiling, "epochs": 2, "batch_size": 2, **hp},
        ),
    )


def _run(monkeypatch, request: TrainRequest) -> dict:
    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")
    jid = export_jobs.new_job("test")
    train_jobs.run_train_job(jid, request, "run-denoise-1")
    return export_jobs.get_job(jid)


def _fake_built(monkeypatch, captured: dict | None = None):
    """A BuiltFamily whose pieces are inert — nothing here is exercised once
    run_denoise_training_loop is itself mocked."""

    def _build_family(model_cfg, n_classes, device, log_cb, init_state=None):
        if captured is not None:
            captured["n_classes"] = n_classes
            captured["model_cfg"] = model_cfg
            captured["init_state"] = init_state
        return train_common.BuiltFamily(
            forward_fn=lambda x: x,
            to_tensor_fn=lambda x: x,
            trainable_params=[],
            set_train_mode=None,
            model_config_snapshot={"depth": 2, "base_channels": 4, "growth_rate": 1.5},
            adapter_state_fn=lambda: {"topo_dict": {}, "state_dict": {}},
        )

    monkeypatch.setattr(train_common, "build_family", _build_family)


_METRICS = {
    "epochs_completed": 2,
    "final_train_loss": 0.01,
    "final_val_loss": 0.02,
    denoise_train.VAL_METRIC_KEY: 0.9,
    "cancelled": False,
}


def _mock_happy_path(monkeypatch, captured: dict):
    """Stub out every expensive step, recording what the dispatch chose."""
    import tiling

    monkeypatch.setattr(tiling, "qlty_available", lambda: True)

    def _must_not_run(*a, **k):
        raise AssertionError("the annotation-driven segmentation path must not run for a denoiser")

    monkeypatch.setattr(train_common, "prepare_datasets", _must_not_run)
    monkeypatch.setattr(train_common, "run_training_loop", _must_not_run)

    def _sampler(name):
        def _prepare(sources, render, **kwargs):
            captured["sampler"] = name
            captured["render"] = render
            return {"train": [(np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8))] * 4, "val": []}

        return _prepare

    monkeypatch.setattr(denoise_train, "prepare_noise2noise_datasets", _sampler("n2n"))
    monkeypatch.setattr(denoise_train, "prepare_noise2void_datasets", _sampler("n2v"))

    def _tile(datasets, window, **kwargs):
        captured["tiled_window"] = window
        patch = (np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8))
        return {"train": [patch] * 20, "val": []}

    monkeypatch.setattr(denoise_train, "tile_denoise_datasets", _tile)

    def _loop(**kwargs):
        captured["loop_kwargs"] = kwargs
        captured["lock_held_during_training"] = train_common.ML_LOCK.locked()
        return dict(_METRICS)

    monkeypatch.setattr(denoise_train, "run_denoise_training_loop", _loop)
    _fake_built(monkeypatch, captured)

    def _save_run(run_id, **kwargs):
        captured["save_run"] = {"run_id": run_id, **kwargs}

    monkeypatch.setattr(train_common, "save_run", _save_run)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_denoising_task_routes_to_the_denoiser_loop(monkeypatch) -> None:
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)

    job = _run(monkeypatch, _denoise_request())

    assert job["state"] == "done"
    assert captured["sampler"] == "n2n"
    assert captured["loop_kwargs"]["training_scheme"] == "n2n"
    assert job["result"]["run_id"] == "run-denoise-1"
    assert job["result"][denoise_train.VAL_METRIC_KEY] == 0.9


def test_the_training_scheme_selects_the_sampler(monkeypatch) -> None:
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)

    _run(monkeypatch, _denoise_request(scheme="n2v"))

    assert captured["sampler"] == "n2v"
    assert captured["loop_kwargs"]["training_scheme"] == "n2v"


def test_segmentation_task_is_untouched_by_the_new_branch(monkeypatch) -> None:
    """The dispatch must not divert an ordinary segmentation request."""
    import tiling

    monkeypatch.setattr(tiling, "qlty_available", lambda: True)
    seen: list[str] = []
    monkeypatch.setattr(
        train_common,
        "prepare_datasets",
        lambda *a, **k: seen.append("prepare_datasets") or {"train": [], "val": []},
    )

    request = TrainRequest(
        sources=[_source()],
        classes=[AnnotationClass(classId=1, label="pore", color="#ff0000")],
        model=DlsiaTunetConfig(hyperparams={"image_size": 64, "tiling": False}),
    )
    job = _run(monkeypatch, request)

    assert seen == ["prepare_datasets"]
    assert "No annotated slices" in job["error"]  # the ordinary downstream check


def test_hyperparameters_reach_the_denoiser_loop(monkeypatch) -> None:
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)

    _run(monkeypatch, _denoise_request(epochs=7, batch_size=3, seed=99, image_size=128))

    kwargs = captured["loop_kwargs"]
    assert kwargs["epochs"] == 7
    assert kwargs["batch_size"] == 3
    assert kwargs["seed"] == 99
    assert kwargs["image_size"] == 128
    assert captured["tiled_window"] == 128


def test_build_family_is_called_with_no_classes(monkeypatch) -> None:
    """A denoiser has no taxonomy; denoise_runtime fixes in/out channels at 1."""
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)

    _run(monkeypatch, _denoise_request())

    assert captured["n_classes"] == 0
    assert isinstance(captured["model_cfg"], DlsiaDenoiserConfig)
    assert captured["init_state"] is None  # a fresh run starts from scratch


def test_resuming_a_denoiser_warm_starts_the_model_and_records_the_parent(monkeypatch) -> None:
    """The resume preamble is shared with segmentation, so the denoiser branch
    only has to carry `init_state` through to build_family and `resumed_from`
    through to save_run — but nothing else checks that it does."""
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)
    parent_state = {"topo_dict": {"depth": 2}, "state_dict": {}}
    monkeypatch.setattr(
        train_common,
        "load_run_config",
        lambda run_id: {"model_family": "dlsia_denoiser", "task": "denoising", "hyperparams": {"depth": 2}},
    )
    monkeypatch.setattr(train_common, "load_adapter_state", lambda run_id: parent_state)

    request = _denoise_request()
    request.resume_from_run_id = "parent-run"
    _run(monkeypatch, request)

    assert captured["init_state"] is parent_state
    assert captured["save_run"]["resumed_from"] == "parent-run"


# ---------------------------------------------------------------------------
# Coherence guards between `task` and `model_family`
# ---------------------------------------------------------------------------


def test_denoising_task_with_a_segmentation_family_is_rejected(monkeypatch) -> None:
    request = TrainRequest(
        task="denoising",
        sources=[_source()],
        classes=[],
        model=DlsiaTunetConfig(hyperparams={"image_size": 64, "tiling": False}),
    )

    job = _run(monkeypatch, request)

    assert job["state"] == "error"
    assert "dlsia_denoiser" in job["error"]


def _denoiser_config_on_segmentation_task() -> TrainRequest:
    """`task` defaults to "segmentation", so a client that sends a denoiser
    config and forgets the field produces this. The schema can't catch it —
    with a class list present the request validates cleanly — and unguarded it
    would train a 1-channel regression net against CrossEntropyLoss."""
    return TrainRequest(
        sources=[_source()],
        classes=[AnnotationClass(classId=1, label="pore", color="#ff0000")],
        model=DlsiaDenoiserConfig(training_scheme="n2v", hyperparams={"image_size": 64, "tiling": False}),
    )


def test_denoiser_family_left_on_the_default_segmentation_task_is_rejected(monkeypatch) -> None:
    job = _run(monkeypatch, _denoiser_config_on_segmentation_task())

    assert job["state"] == "error"
    assert "task='denoising'" in job["error"]


def test_the_coherence_guard_runs_before_any_data_is_read(monkeypatch) -> None:
    def _must_not_run(*a, **k):
        raise AssertionError("no data should be read for an incoherent request")

    monkeypatch.setattr(denoise_train, "prepare_noise2noise_datasets", _must_not_run)
    monkeypatch.setattr(denoise_train, "prepare_noise2void_datasets", _must_not_run)
    monkeypatch.setattr(train_common, "prepare_datasets", _must_not_run)

    job = _run(monkeypatch, _denoiser_config_on_segmentation_task())

    assert job["state"] == "error"


# ---------------------------------------------------------------------------
# Persistence, progress and cancellation
# ---------------------------------------------------------------------------


def test_save_run_records_the_denoising_task_and_an_empty_class_list(monkeypatch) -> None:
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)

    _run(monkeypatch, _denoise_request(scheme="n2v"))

    saved = captured["save_run"]
    assert saved["task"] == "denoising"
    assert saved["classes"] == []
    # The panel filters the runs list on this exact value.
    assert saved["model_family"] == "dlsia_denoiser"
    # The scheme is part of what distinguishes two runs of the same topology.
    assert saved["model_config"]["training_scheme"] == "n2v"
    assert saved["source_keys"] == ["local:volume.tif"]
    assert saved["metrics"][denoise_train.VAL_METRIC_KEY] == 0.9


def test_denoiser_job_reports_the_standard_phase_sequence(monkeypatch) -> None:
    """The frontend polls the same export-job route for every job kind, so the
    denoiser must not invent phase names of its own."""
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)
    phases: list[str] = []
    real_update = export_jobs.update

    def _tracking_update(jid, **kw):
        if "phase" in kw:
            phases.append(kw["phase"])
        real_update(jid, **kw)

    monkeypatch.setattr(export_jobs, "update", _tracking_update)

    _run(monkeypatch, _denoise_request())

    assert phases == ["preparing", "tiling", "training", "saving", "done"]


def test_denoiser_job_holds_the_ml_lock_and_releases_it(monkeypatch) -> None:
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)

    _run(monkeypatch, _denoise_request())

    assert captured["lock_held_during_training"] is True
    assert train_common.ML_LOCK.locked() is False


def test_denoiser_job_refuses_to_start_while_another_ml_job_runs(monkeypatch) -> None:
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)

    train_common.ML_LOCK.acquire()
    try:
        job = _run(monkeypatch, _denoise_request())
    finally:
        train_common.ML_LOCK.release()

    assert job["state"] == "error"
    assert "already running" in job["error"]
    assert "save_run" not in captured


def test_cancel_during_denoiser_tiling_reports_cancelled_without_saving(monkeypatch) -> None:
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)
    monkeypatch.setattr(denoise_train, "tile_denoise_datasets", lambda *a, **k: None)

    job = _run(monkeypatch, _denoise_request())

    assert job["state"] == "done"
    assert job["result"] == {"cancelled": True}
    assert "save_run" not in captured  # no model existed yet


def test_a_cancelled_denoiser_run_is_still_saved(monkeypatch) -> None:
    """Cancelling during TRAINING is different from cancelling during tiling —
    there are weights worth keeping."""
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)
    monkeypatch.setattr(
        denoise_train, "run_denoise_training_loop", lambda **k: {**_METRICS, "cancelled": True}
    )

    job = _run(monkeypatch, _denoise_request())

    assert job["state"] == "done"
    assert captured["save_run"]["task"] == "denoising"
    assert any("partial run saved" in line for line in job["log"])


def test_a_sampler_failure_is_reported_as_a_job_error(monkeypatch) -> None:
    """A background thread must never raise out; e.g. a one-slice n2n scope."""
    captured: dict = {}
    _mock_happy_path(monkeypatch, captured)

    def _no_pairs(*a, **k):
        raise ValueError("Noise2Noise needs at least two slices")

    monkeypatch.setattr(denoise_train, "prepare_noise2noise_datasets", _no_pairs)

    job = _run(monkeypatch, _denoise_request())

    assert job["state"] == "error"
    assert "at least two slices" in job["error"]
    assert train_common.ML_LOCK.locked() is False


def test_qlty_is_still_required_before_any_slices_are_read(monkeypatch) -> None:
    import tiling

    monkeypatch.setattr(tiling, "qlty_available", lambda: False)

    def _must_not_run(*a, **k):
        raise AssertionError("slices must not be read when qlty is unavailable")

    monkeypatch.setattr(denoise_train, "prepare_noise2noise_datasets", _must_not_run)

    job = _run(monkeypatch, _denoise_request(tiling=True))

    assert job["state"] == "error"
    assert "qlty" in job["error"]


# ---------------------------------------------------------------------------
# Real end-to-end run
# ---------------------------------------------------------------------------


def test_end_to_end_denoiser_run_is_saved_and_listed(monkeypatch, tmp_path: Path) -> None:
    """The whole pipeline for real — raw slices in, trained weights and a run
    record out — with only the data source faked. Guards the join between the
    pieces the mocked tests above each check in isolation, and the contract the
    Learned Denoiser panel reads: model_family == "dlsia_denoiser" in
    list_runs."""
    pytest.importorskip("torch")
    pytest.importorskip("dlsia")
    pytest.importorskip("qlty")

    import arrays
    import images

    images._stats_cache.clear()
    rng = np.random.default_rng(0)
    clean = np.tile(np.linspace(0, 255, 80).astype(np.int16), (80, 1))
    volume = np.stack(
        [np.clip(clean + rng.integers(-30, 31, size=(80, 80)), 0, 255).astype(np.uint8) for _ in range(4)]
    )
    monkeypatch.setattr(arrays, "resolve_array", lambda *a, **k: volume)
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")

    request = _denoise_request(scheme="n2n", depth=2, base_channels=4)
    jid = export_jobs.new_job("test")
    train_jobs.run_train_job(jid, request, "e2e-denoise")
    job = export_jobs.get_job(jid)

    assert job["state"] == "done", job["error"]
    assert job["result"]["epochs_completed"] == 2
    assert job["result"]["cancelled"] is False

    runs = train_common.list_runs()
    assert [r["run_id"] for r in runs] == ["e2e-denoise"]
    run = runs[0]
    assert run["model_family"] == "dlsia_denoiser"
    assert run["task"] == "denoising"
    assert run["classes"] == []
    assert run["model_config"]["training_scheme"] == "n2n"
    assert denoise_train.VAL_METRIC_KEY in run["metrics"]

    # Weights are real and reloadable as a denoiser.
    import denoise_runtime

    state = train_common.load_adapter_state("e2e-denoise")
    model = denoise_runtime.load_model(state, "cpu")
    assert json.loads((tmp_path / "e2e-denoise" / "config.json").read_text())["task"] == "denoising"
    assert model is not None


def test_end_to_end_noise2void_run_trains_from_a_single_slice(monkeypatch, tmp_path: Path) -> None:
    """The case Noise2Void exists for, and the one Noise2Noise cannot do: a
    scope of exactly one slice."""
    pytest.importorskip("torch")
    pytest.importorskip("dlsia")
    pytest.importorskip("qlty")

    import arrays
    import images

    images._stats_cache.clear()
    rng = np.random.default_rng(1)
    volume = rng.integers(0, 256, size=(3, 80, 80), dtype=np.uint8)
    monkeypatch.setattr(arrays, "resolve_array", lambda *a, **k: volume)
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")

    request = TrainRequest(
        task="denoising",
        sources=[_source(indices=(1,))],
        classes=[],
        model=DlsiaDenoiserConfig(
            training_scheme="n2v",
            hyperparams={"image_size": 64, "tiling": True, "epochs": 1, "batch_size": 2,
                         "depth": 2, "base_channels": 4},
        ),
    )
    jid = export_jobs.new_job("test")
    train_jobs.run_train_job(jid, request, "e2e-n2v")

    job = export_jobs.get_job(jid)
    assert job["state"] == "done", job["error"]
    assert train_common.list_runs()[0]["model_config"]["training_scheme"] == "n2v"
