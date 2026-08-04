"""run_infer_job's tiled-vs-legacy path selection, and the qlty guard.

The backward-compatibility contract stated in infer_jobs.py's comment — a run
saved before tiling existed predicts on its original whole-slice-rescale
geometry, never the tiled path — had no test before this. Real torch is used
(installed in this project's venv) since the legacy path runs genuine
softmax/argmax tensor ops; only qlty's own availability is mocked.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import arrays as arrays_mod  # noqa: E402
import dlsia_runtime  # noqa: E402
import export_jobs  # noqa: E402
import images as images_mod  # noqa: E402
import infer_jobs  # noqa: E402
import tiling  # noqa: E402
import train_common  # noqa: E402
from schemas import InferRequest  # noqa: E402


def _fake_config(*, tiling_flag: bool | None) -> dict:
    """A saved-run config. `tiling_flag=None` omits the key entirely — exactly
    what a run saved before tiling existed looks like on disk."""
    hyperparams: dict = {"image_size": 8}
    if tiling_flag is not None:
        hyperparams["tiling"] = tiling_flag
    return {
        "model_family": "dlsia_tunet",
        "model_config": {},
        "hyperparams": hyperparams,
        "classes": [{"classId": 1, "label": "pore", "color": "#ff0000"}],
        "image_size": 8,
        # "slice" (not "global") so this never touches images._sample_global_stats.
        "render": {"norm": "slice", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0, "cmap": "gray"},
    }


class _FakeModel:
    """Stands in for the loaded TUNet — infer_jobs calls .eval() on it directly."""

    def eval(self) -> None:
        pass


def _install_common_fakes(monkeypatch, config: dict) -> None:
    """Everything both the legacy and tiled paths need, with real (tiny) torch
    ops standing in for the model — cheap, but genuine tensor code."""
    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")
    monkeypatch.setattr(train_common, "load_run_config", lambda run_id: config)
    monkeypatch.setattr(train_common, "load_adapter_state", lambda run_id: {})
    monkeypatch.setattr(dlsia_runtime, "load_model", lambda state, device: _FakeModel())
    monkeypatch.setattr(
        dlsia_runtime,
        "make_forward_fn",
        lambda model: (lambda batch: torch.rand(batch.shape[0], 2, batch.shape[2], batch.shape[3])),
    )
    monkeypatch.setattr(
        dlsia_runtime,
        "make_to_tensor_fn",
        lambda: (lambda rgb: torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0),
    )
    monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri: object())
    monkeypatch.setattr(arrays_mod, "array_shape_meta", lambda node: {"height": 8, "width": 8})
    monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: np.zeros((8, 8), dtype=np.uint8))
    monkeypatch.setattr(
        images_mod, "render_slice", lambda arr, render, global_range: np.zeros((8, 8, 3), dtype=np.uint8)
    )


def _run(config: dict) -> dict:
    request = InferRequest(run_id="run-1", kind="local", source="s.tif", slice_indices=[0])
    jid = export_jobs.new_job("test")
    infer_jobs.run_infer_job(jid, request)
    return export_jobs.get_job(jid)


def test_run_with_no_tiling_key_takes_the_legacy_path(monkeypatch) -> None:
    """A run saved BEFORE tiling existed has no "tiling" key at all —
    `config.get("hyperparams", {}).get("tiling", False)` must default to False,
    not KeyError or (worse) silently True."""
    config = _fake_config(tiling_flag=None)
    _install_common_fakes(monkeypatch, config)

    def _must_not_run(*a, **k):
        raise AssertionError("tiled inference must not run for a legacy (no 'tiling' key) run")

    monkeypatch.setattr(tiling, "predict_label_map_tiled", _must_not_run)

    assert _run(config)["state"] == "done"


def test_run_with_tiling_false_also_takes_the_legacy_path(monkeypatch) -> None:
    config = _fake_config(tiling_flag=False)
    _install_common_fakes(monkeypatch, config)

    def _must_not_run(*a, **k):
        raise AssertionError("tiled inference must not run when tiling=False")

    monkeypatch.setattr(tiling, "predict_label_map_tiled", _must_not_run)

    assert _run(config)["state"] == "done"


def test_run_with_tiling_true_takes_the_tiled_path(monkeypatch) -> None:
    config = _fake_config(tiling_flag=True)
    _install_common_fakes(monkeypatch, config)
    monkeypatch.setattr(tiling, "qlty_available", lambda: True)
    called: list[bool] = []

    def _fake_tiled(rgb, **kwargs):
        called.append(True)
        return np.ones((8, 8), dtype=np.uint8)

    monkeypatch.setattr(tiling, "predict_label_map_tiled", _fake_tiled)

    job = _run(config)

    assert called == [True]
    assert job["state"] == "done"


def test_tiled_run_fails_fast_when_qlty_is_unavailable(monkeypatch) -> None:
    """Regression: the tiled path had no qlty guard at all — a torch-only
    install died with a raw ModuleNotFoundError deep inside tiling.py, after
    already loading the model. It must fail before that, with a clear message."""
    config = _fake_config(tiling_flag=True)
    _install_common_fakes(monkeypatch, config)
    monkeypatch.setattr(tiling, "qlty_available", lambda: False)

    def _must_not_load(*a, **k):
        raise AssertionError("the model must not be loaded when qlty is unavailable")

    monkeypatch.setattr(dlsia_runtime, "load_model", _must_not_load)

    job = _run(config)

    assert job["state"] == "error"
    assert "qlty" in job["error"]
    assert train_common.ML_LOCK.locked() is False
