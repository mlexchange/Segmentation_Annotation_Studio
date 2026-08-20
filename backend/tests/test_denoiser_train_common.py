"""Tests for the denoiser's slice through train_common: build_family's
explicit dispatch (including its new, no-longer-silent error branch) and the
`task` backward-compat default on the run-registry load path.

See test_train_common_build_family.py for the pre-existing segmentation
(DINOv3 / dlsia TUNet) coverage this parallels and must not regress.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import denoise_runtime  # noqa: E402
import train_common  # noqa: E402
from schemas import DlsiaDenoiserConfig  # noqa: E402


def _log(_msg: str) -> None:
    pass


# ---------------------------------------------------------------------------
# load_run_config: `task` backward compatibility
# ---------------------------------------------------------------------------


def _write_bare_config(base: Path, run_id: str, **extra) -> None:
    """A run config with no `task` key at all — exactly what every run saved
    before this field existed looks like on disk."""
    d = base / run_id
    d.mkdir(parents=True)
    config = {
        "run_id": run_id,
        "model_family": "dlsia_tunet",
        "model_config": {},
        "classes": [{"classId": 1, "label": "pore", "color": "#ff0000"}],
        "render": {},
        "image_size": 512,
        "hyperparams": {},
        "source_keys": [],
        "created_at": "2025-01-01T00:00:00+00:00",
        **extra,
    }
    (d / "config.json").write_text(json.dumps(config))
    (d / "metrics.json").write_text(json.dumps({}))


def test_load_run_config_defaults_absent_task_to_segmentation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    _write_bare_config(tmp_path, "old_run")

    loaded = train_common.load_run_config("old_run")

    assert loaded["task"] == "segmentation"


def test_list_runs_also_defaults_absent_task_to_segmentation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    _write_bare_config(tmp_path, "old_run")

    runs = train_common.list_runs()

    assert runs[0]["task"] == "segmentation"


def test_load_run_config_preserves_an_explicit_task(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    _write_bare_config(tmp_path, "denoiser_run", task="denoising", model_family="dlsia_denoiser", classes=[])

    loaded = train_common.load_run_config("denoiser_run")

    assert loaded["task"] == "denoising"


def test_save_run_records_the_task(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))

    train_common.save_run(
        "denoiser_run",
        model_family="dlsia_denoiser",
        model_config={},
        classes=[],
        render={},
        image_size=512,
        hyperparams={},
        source_keys=[],
        adapter_state={},
        metrics={},
        task="denoising",
    )

    config = json.loads((tmp_path / "denoiser_run" / "config.json").read_text())
    assert config["task"] == "denoising"


def test_save_run_defaults_task_to_segmentation_for_old_style_callers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))

    train_common.save_run(
        "seg_run",
        model_family="dlsia_tunet",
        model_config={},
        classes=[{"classId": 1, "label": "pore", "color": "#ff0000"}],
        render={},
        image_size=512,
        hyperparams={},
        source_keys=[],
        adapter_state={},
        metrics={},
    )

    config = json.loads((tmp_path / "seg_run" / "config.json").read_text())
    assert config["task"] == "segmentation"


# ---------------------------------------------------------------------------
# build_family: explicit dispatch, no more silent TUNet fallthrough
# ---------------------------------------------------------------------------


def test_build_family_raises_value_error_for_an_unrecognized_config_type() -> None:
    """The old code's implicit `else` silently treated ANY non-DINOv3 config
    as a segmentation TUNet. Prove that branch is now a real error instead of
    a permissive fallthrough, using a config type build_family has never
    heard of."""

    class _BogusModelConfig:
        model_family = "totally_unknown_family"
        hyperparams = object()

    with pytest.raises(ValueError, match="Unknown model"):
        train_common.build_family(_BogusModelConfig(), n_classes=2, device="cpu", log_cb=_log)


def test_denoiser_config_gets_its_own_family_not_the_tunet_fallthrough(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1)  # real nn.Module: build_family calls model.parameters()
    monkeypatch.setattr(train_common, "dlsia_available", lambda: True)
    monkeypatch.setattr(
        denoise_runtime,
        "build_model",
        lambda image_size, depth, base_channels, growth_rate, device: model,
    )
    monkeypatch.setattr(denoise_runtime, "make_forward_fn", lambda m: (lambda x: x))
    monkeypatch.setattr(denoise_runtime, "make_to_tensor_fn", lambda: (lambda gray: gray))
    monkeypatch.setattr(denoise_runtime, "make_set_train_mode_fn", lambda m: (lambda is_training: None))
    monkeypatch.setattr(denoise_runtime, "network_dict", lambda m: {"weights": True})

    model_cfg = DlsiaDenoiserConfig(
        training_scheme="n2n", hyperparams={"depth": 3, "base_channels": 16, "growth_rate": 2.0}
    )
    built = train_common.build_family(model_cfg, n_classes=0, device="cpu", log_cb=_log)

    assert built.model_config_snapshot == {
        "architecture": "tunet", "depth": 3, "base_channels": 16, "growth_rate": 2.0,
    }
    assert callable(built.set_train_mode)
    assert built.trainable_params == list(model.parameters())
    assert built.adapter_state_fn() == {"weights": True}


def test_denoiser_config_without_dlsia_reports_missing_dlsia(monkeypatch) -> None:
    monkeypatch.setattr(train_common, "dlsia_available", lambda: False)

    model_cfg = DlsiaDenoiserConfig(training_scheme="n2v")
    with pytest.raises(RuntimeError, match="dlsia"):
        train_common.build_family(model_cfg, n_classes=0, device="cpu", log_cb=_log)


def test_denoiser_warm_start_rebuilds_from_the_saved_topology(monkeypatch) -> None:
    """Mirrors the segmentation TUNet's warm-start contract: build_model must
    NOT run, and the snapshot must reflect the saved topology, not the
    request's (possibly different) hyperparameters."""
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(train_common, "dlsia_available", lambda: True)

    def _must_not_build(*a, **k):
        raise AssertionError("build_model must not run for a warm start — use load_model")

    monkeypatch.setattr(denoise_runtime, "build_model", _must_not_build)
    monkeypatch.setattr(denoise_runtime, "load_model", lambda state, device: model)
    monkeypatch.setattr(denoise_runtime, "make_forward_fn", lambda m: (lambda x: x))
    monkeypatch.setattr(denoise_runtime, "make_to_tensor_fn", lambda: (lambda gray: gray))
    monkeypatch.setattr(denoise_runtime, "make_set_train_mode_fn", lambda m: (lambda t: None))

    init_state = {"topo_dict": {"depth": 5, "base_channels": 16, "growth_rate": 2.0}, "state_dict": {}}
    built = train_common.build_family(
        DlsiaDenoiserConfig(training_scheme="n2n", hyperparams={"depth": 2, "base_channels": 4}),
        n_classes=0,
        device="cpu",
        log_cb=_log,
        init_state=init_state,
    )

    assert built.model_config_snapshot == {
        "architecture": "tunet", "depth": 5, "base_channels": 16, "growth_rate": 2.0,
    }
