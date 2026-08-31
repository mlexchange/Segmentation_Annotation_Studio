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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_common  # noqa: E402
import denoise_bake  # noqa: E402


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
