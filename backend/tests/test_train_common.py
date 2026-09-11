"""train_common.py + the denoise_bake.py model-denoiser fix it enables.

Ported alongside train_common/tiling/denoise_train/denoise_runtime/
autoencoder_runtime (Phase 5, Stage A). The `ml` extra (torch/dlsia/qlty) may
or may not be installed in whatever environment runs this suite — these tests
don't assume either way. The property that matters is: every module here is
import-safe regardless, and denoise_bake.py's `_ModelDenoiser` (which already
called into these modules before they existed) fails with clear, specific
errors instead of a raw `ModuleNotFoundError` when a dependency truly is
missing — verified here by monkeypatching availability rather than by relying
on the dev machine's actual install state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import denoise_bake  # noqa: E402
import train_common  # noqa: E402


def test_modules_import_cleanly():
    """All Stage A modules must be import-safe whether or not torch/dlsia are installed."""
    import autoencoder_runtime  # noqa: F401
    import denoise_runtime  # noqa: F401
    import denoise_train  # noqa: F401
    import dlsia_runtime  # noqa: F401
    import tiling  # noqa: F401


def test_capability_never_raises_and_has_no_dinov3_fields():
    """DINOv3 is deferred to Phase 5.5 — capability() must not report it at all,
    regardless of what's actually installed."""
    result = train_common.capability()
    assert "dinov3" not in result
    assert "error" not in result
    assert set(result) == {
        "torch_available", "torch_version", "device", "dlsia", "tiling",
        "denoise", "runs_dir", "busy",
    }


def test_capability_reports_unavailable_without_torch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(train_common, "torch_available", lambda: False)
    monkeypatch.setattr(train_common, "dlsia_available", lambda: False)
    result = train_common.capability()
    assert result["torch_available"] is False
    assert result["torch_version"] is None
    assert result["device"] is None
    assert result["dlsia"] == {"available": False}


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _write_run_config(runs_dir: Path, run_id: str, config: dict) -> None:
    d = runs_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))


def test_model_denoiser_unknown_run_is_a_clean_404(runs_dir: Path) -> None:
    """Previously: ModuleNotFoundError before train_common.py existed."""
    with pytest.raises(Exception) as exc_info:
        denoise_bake._ModelDenoiser("nonexistent-run", None, {})
    assert "Unknown run" in str(exc_info.value)


def test_model_denoiser_refuses_non_denoiser_run(runs_dir: Path) -> None:
    _write_run_config(runs_dir, "seg-run", {"model_family": "dlsia_tunet", "task": "segmentation"})
    with pytest.raises(ValueError, match="not a denoiser run"):
        denoise_bake._ModelDenoiser("seg-run", None, {})


def test_model_denoiser_reports_missing_dlsia_by_name(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tunet-architecture denoiser run needs dlsia — confirm the error names it
    specifically, not just a generic import failure. Forces the "not installed"
    branch via monkeypatch so this doesn't depend on the test machine's actual
    dlsia install state."""
    monkeypatch.setattr(train_common, "dlsia_available", lambda: False)
    _write_run_config(
        runs_dir,
        "denoiser-run",
        {
            "model_family": "dlsia_denoiser",
            "task": "denoising",
            "model_config": {"architecture": "tunet"},
        },
    )
    with pytest.raises(ValueError, match="dlsia is not installed"):
        denoise_bake._ModelDenoiser("denoiser-run", None, {})


def test_model_denoiser_reports_missing_torch_by_name(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cnn_ae denoiser needs no dlsia, but still needs torch for a device."""
    import tiling

    monkeypatch.setattr(tiling, "qlty_available", lambda: True)
    monkeypatch.setattr(train_common, "pick_device", lambda: None)
    _write_run_config(
        runs_dir,
        "ae-run",
        {
            "model_family": "dlsia_denoiser",
            "task": "denoising",
            "model_config": {"architecture": "cnn_ae"},
            "image_size": 256,
        },
    )
    with pytest.raises(ValueError, match="torch is not installed"):
        denoise_bake._ModelDenoiser("ae-run", None, {})


# ---------------------------------------------------------------------------
# pick_device
# ---------------------------------------------------------------------------

class TestPickDevice:
    def test_no_torch_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(train_common, "torch_available", lambda: False)
        assert train_common.pick_device() is None

    @pytest.mark.parametrize("override", ["mps", "cuda", "cpu"])
    def test_train_device_env_override_short_circuits_detection(self, monkeypatch: pytest.MonkeyPatch, override):
        # A valid override returns before ever importing torch's backends —
        # exercised for real (torch IS available in this env) without needing
        # the overridden device to actually exist on this machine.
        monkeypatch.setenv("TRAIN_DEVICE", override)
        assert train_common.pick_device() == override

    def test_invalid_override_falls_through_to_real_detection(self, monkeypatch: pytest.MonkeyPatch):
        pytest.importorskip("torch")
        monkeypatch.setenv("TRAIN_DEVICE", "not-a-real-device")
        assert train_common.pick_device() in ("mps", "cuda", "cpu")


# ---------------------------------------------------------------------------
# run_dir / _validate_run_id
# ---------------------------------------------------------------------------

class TestRunDirValidation:
    @pytest.mark.parametrize("bad_id", ["", ".", "..", "a/b", "a\\b", "a\x00b"])
    def test_rejects_unsafe_run_ids(self, runs_dir: Path, bad_id):
        with pytest.raises(HTTPException) as exc:
            train_common.run_dir(bad_id)
        assert exc.value.status_code == 400

    def test_accepts_a_safe_run_id(self, runs_dir: Path):
        d = train_common.run_dir("my-run-1")
        assert d.name == "my-run-1"
        assert d.parent == train_common.runs_dir()


# ---------------------------------------------------------------------------
# Run persistence: list_runs / delete_run / load_run_config
# (config.json written directly via the existing _write_run_config helper —
# these three never touch torch, unlike save_run/load_adapter_state below)
# ---------------------------------------------------------------------------

class TestListRuns:
    def test_empty_when_runs_dir_does_not_exist(self, runs_dir: Path):
        assert train_common.list_runs() == []

    def test_lists_saved_runs_sorted_newest_first(self, runs_dir: Path):
        _write_run_config(runs_dir, "old", {"model_family": "dlsia_tunet", "created_at": "2024-01-01T00:00:00"})
        _write_run_config(runs_dir, "new", {"model_family": "dlsia_tunet", "created_at": "2024-06-01T00:00:00"})
        results = train_common.list_runs()
        assert [r["created_at"] for r in results] == ["2024-06-01T00:00:00", "2024-01-01T00:00:00"]

    def test_backward_compat_missing_task_defaults_to_segmentation(self, runs_dir: Path):
        _write_run_config(runs_dir, "old-run", {"model_family": "dlsia_tunet", "created_at": "x"})
        results = train_common.list_runs()
        assert results[0]["task"] == "segmentation"

    def test_non_directory_entries_are_skipped(self, runs_dir: Path):
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "stray_file.txt").write_text("not a run")
        assert train_common.list_runs() == []

    def test_malformed_run_directory_is_skipped_not_raised(self, runs_dir: Path):
        d = runs_dir / "broken"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{ not valid json")
        assert train_common.list_runs() == []

    def test_metrics_defaults_to_empty_dict_when_missing(self, runs_dir: Path):
        _write_run_config(runs_dir, "no-metrics", {"model_family": "dlsia_tunet", "created_at": "x"})
        results = train_common.list_runs()
        assert results[0]["metrics"] == {}


class TestDeleteRun:
    def test_unknown_run_is_404(self, runs_dir: Path):
        with pytest.raises(HTTPException) as exc:
            train_common.delete_run("nonexistent")
        assert exc.value.status_code == 404

    def test_removes_the_run_directory(self, runs_dir: Path):
        _write_run_config(runs_dir, "to-delete", {"model_family": "dlsia_tunet"})
        d = train_common.run_dir("to-delete")
        assert d.is_dir()
        train_common.delete_run("to-delete")
        assert not d.exists()


class TestLoadRunConfig:
    def test_unknown_run_is_404(self, runs_dir: Path):
        with pytest.raises(HTTPException) as exc:
            train_common.load_run_config("nonexistent")
        assert exc.value.status_code == 404

    def test_corrupt_config_is_500(self, runs_dir: Path):
        d = runs_dir / "corrupt"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{ broken")
        with pytest.raises(HTTPException) as exc:
            train_common.load_run_config("corrupt")
        assert exc.value.status_code == 500

    def test_missing_task_defaults_to_segmentation(self, runs_dir: Path):
        _write_run_config(runs_dir, "old", {"model_family": "dlsia_tunet"})
        config = train_common.load_run_config("old")
        assert config["task"] == "segmentation"


class TestSaveRunAndLoadAdapterState:
    def test_round_trips_config_metrics_and_weights(self, runs_dir: Path):
        torch = pytest.importorskip("torch")
        train_common.save_run(
            "run1",
            model_family="dlsia_tunet",
            model_config={"depth": 2},
            classes=[{"classId": 1, "label": "a"}],
            render={},
            image_size=64,
            hyperparams={"depth": 2},
            source_keys=["local:x.tif"],
            adapter_state={"weight": torch.zeros(2, 2)},
            metrics={"epochs_completed": 1},
        )
        config = train_common.load_run_config("run1")
        assert config["model_family"] == "dlsia_tunet"
        assert config["classes"] == [{"classId": 1, "label": "a"}]

        adapter = train_common.load_adapter_state("run1")
        assert torch.equal(adapter["weight"], torch.zeros(2, 2))

    def test_load_adapter_state_missing_weights_is_404(self, runs_dir: Path):
        pytest.importorskip("torch")
        _write_run_config(runs_dir, "no-weights", {"model_family": "dlsia_tunet"})
        with pytest.raises(HTTPException) as exc:
            train_common.load_adapter_state("no-weights")
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# letterbox / letterbox_params / unletterbox — pure numpy/PIL, no torch
# ---------------------------------------------------------------------------

class TestLetterbox:
    def test_already_square_is_returned_unchanged(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        lbl = np.ones((32, 32), dtype=np.uint8)
        out_img, out_lbl = train_common.letterbox(img, lbl, 32)
        assert out_img is img
        assert out_lbl is lbl

    def test_pads_a_non_square_image_to_a_square_canvas(self):
        img = np.full((20, 40, 3), 255, dtype=np.uint8)
        lbl = np.ones((20, 40), dtype=np.uint8)
        out_img, out_lbl = train_common.letterbox(img, lbl, 40)
        assert out_img.shape == (40, 40, 3)
        assert out_lbl.shape == (40, 40)

    def test_padded_label_regions_use_ignore_index(self):
        img = np.full((10, 40, 3), 255, dtype=np.uint8)
        lbl = np.ones((10, 40), dtype=np.uint8)
        _out_img, out_lbl = train_common.letterbox(img, lbl, 40)
        # Top/bottom padding bands must be IGNORE_INDEX, not a real class.
        assert np.all(out_lbl[0, :] == train_common.IGNORE_INDEX)
        assert np.all(out_lbl[-1, :] == train_common.IGNORE_INDEX)


class TestLetterboxParamsAndUnletterbox:
    def test_letterbox_params_centers_the_scaled_image(self):
        params = train_common.letterbox_params(20, 40, 40)
        assert params["nh"] == 20
        assert params["nw"] == 40
        assert params["top"] == 10
        assert params["left"] == 0

    def test_unletterbox_inverts_letterbox_for_an_exact_fit(self):
        img = np.zeros((20, 40, 3), dtype=np.uint8)
        lbl = np.full((20, 40), 3, dtype=np.uint8)
        boxed_img, boxed_lbl = train_common.letterbox(img, lbl, 40)
        recovered = train_common.unletterbox(boxed_lbl, 20, 40, 40)
        assert recovered.shape == (20, 40)
        assert np.all(recovered == 3)

    def test_unletterbox_resizes_back_when_original_was_downscaled(self):
        # image_size smaller than the original -> nh/nw != orig, exercising the
        # resize-back branch rather than the exact-crop shortcut.
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        lbl = np.full((100, 200), 5, dtype=np.uint8)
        boxed_img, boxed_lbl = train_common.letterbox(img, lbl, 40)
        recovered = train_common.unletterbox(boxed_lbl, 100, 200, 40)
        assert recovered.shape == (100, 200)
        # Nearest-neighbor resize of a constant label stays constant.
        assert np.all(recovered == 5)


# ---------------------------------------------------------------------------
# denoising_render_slice_fn / _denoise_params
# ---------------------------------------------------------------------------

class TestDenoiseParams:
    def test_none_denoise_returns_none_method(self):
        assert train_common._denoise_params(None) == (None, 0.5)

    def test_falsy_dict_returns_none_method(self):
        assert train_common._denoise_params({}) == (None, 0.5)

    def test_method_none_string_is_treated_as_no_denoise(self):
        method, strength = train_common._denoise_params({"method": "none", "strength": 0.7})
        assert method is None
        assert strength == 0.7

    def test_model_method_is_not_supported_as_a_preprocessor(self):
        method, _ = train_common._denoise_params({"method": "model", "strength": 0.5})
        assert method is None

    def test_real_classical_method_passes_through(self):
        assert train_common._denoise_params({"method": "median", "strength": 0.3}) == ("median", 0.3)

    def test_object_with_method_attribute(self):
        class Opts:
            method = "gaussian"
            strength = 0.4

        assert train_common._denoise_params(Opts()) == ("gaussian", 0.4)


class TestDenoisingRenderSliceFn:
    def test_no_denoise_returns_plain_render_slice(self):
        import images as images_mod

        fn = train_common.denoising_render_slice_fn(None)
        assert fn is images_mod.render_slice

    def test_denoise_method_filters_grayscale_input_before_rendering(self, monkeypatch):
        calls = []
        import denoise as denoise_mod

        def fake_denoise_slice(arr, method, strength):
            calls.append((method, strength))
            return arr

        monkeypatch.setattr(denoise_mod, "denoise_slice", fake_denoise_slice)
        fn = train_common.denoising_render_slice_fn({"method": "median", "strength": 0.2})
        arr = np.zeros((8, 8), dtype=np.float32)
        fn(arr, {"norm": "slice", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0, "cmap": "gray"}, None)
        assert calls == [("median", 0.2)]

    def test_rgb_input_skips_denoising(self, monkeypatch):
        calls = []
        import denoise as denoise_mod

        monkeypatch.setattr(denoise_mod, "denoise_slice", lambda *a: calls.append(1))
        fn = train_common.denoising_render_slice_fn({"method": "median", "strength": 0.2})
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        fn(arr, {"norm": "slice", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0, "cmap": "gray"}, None)
        assert calls == []


# ---------------------------------------------------------------------------
# capability()'s exception-swallowing path
# ---------------------------------------------------------------------------

class TestCapabilityErrorHandling:
    def test_internal_failure_is_reported_not_raised(self, monkeypatch: pytest.MonkeyPatch):
        def boom():
            raise RuntimeError("tiling import exploded")

        monkeypatch.setattr(train_common, "torch_available", boom)
        result = train_common.capability()
        assert "error" in result
        assert "tiling import exploded" in result["error"]
