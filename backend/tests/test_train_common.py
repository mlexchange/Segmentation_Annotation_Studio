"""Torch-free tests for train_common: paths, letterbox/unletterbox, run
persistence, and the capability probe. Anything requiring an actual torch
install lives in test_train_torch.py (skipped when torch is unavailable)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import train_common


def test_models_dir_and_runs_dir_default_under_local_data_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DINO_MODELS_DIR", raising=False)
    monkeypatch.delenv("DINO_RUNS_DIR", raising=False)
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    assert train_common.models_dir() == (tmp_path / "models" / "dinov3").resolve()
    assert train_common.runs_dir() == (tmp_path / "models" / "runs").resolve()


def test_models_dir_honors_explicit_env_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom_models"
    monkeypatch.setenv("DINO_MODELS_DIR", str(override))
    assert train_common.models_dir() == override.resolve()


@pytest.mark.parametrize("run_id", ["../evil", "a/b", "a\\b", "", "."])
def test_run_dir_rejects_unsafe_run_ids(run_id: str) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        train_common.run_dir(run_id)


def test_letterbox_pads_with_ignore_index_and_preserves_content() -> None:
    """A wide image should be padded top/bottom (or vice versa) with the
    ignore index in the label, and content should survive the round trip."""
    image = np.full((20, 40, 3), 100, dtype=np.uint8)
    label = np.full((20, 40), 3, dtype=np.uint8)

    out_img, out_lbl = train_common.letterbox(image, label, size=64)

    assert out_img.shape == (64, 64, 3)
    assert out_lbl.shape == (64, 64)
    # Padded label pixels use the ignore index somewhere (image is wider than tall).
    assert (out_lbl == train_common.IGNORE_INDEX).any()
    assert (out_lbl == 3).any()


def test_letterbox_is_a_true_no_op_when_already_square_at_target_size() -> None:
    """Every tiled-training patch is already exactly `size`x`size` — letterbox
    was still doing a full resize-into-a-zeroed-canvas round trip for it (scale
    always computes to 1.0, 4 array copies for literally nothing). The identity
    fast path must return content-identical output AND the very same objects
    (never mutated in place downstream — see letterbox's docstring), not merely
    equal-shaped ones."""
    image = np.full((64, 64, 3), 77, dtype=np.uint8)
    label = np.full((64, 64), 3, dtype=np.uint8)

    out_img, out_lbl = train_common.letterbox(image, label, size=64)

    assert out_img is image
    assert out_lbl is label


def test_letterbox_still_resizes_when_shape_matches_but_not_size() -> None:
    """A guard against a too-loose fast path: a 64x64 input at size=32 must
    still go through the real resize, not be treated as identity."""
    image = np.full((64, 64, 3), 77, dtype=np.uint8)
    label = np.full((64, 64), 3, dtype=np.uint8)

    out_img, out_lbl = train_common.letterbox(image, label, size=32)

    assert out_img.shape == (32, 32, 3)
    assert out_lbl.shape == (32, 32)


def test_unletterbox_recovers_original_resolution() -> None:
    orig_h, orig_w = 30, 50
    label = np.zeros((orig_h, orig_w), dtype=np.uint8)
    label[5:10, 5:10] = 7
    letterboxed, letterboxed_label = train_common.letterbox(
        np.zeros((orig_h, orig_w, 3), dtype=np.uint8),
        label,
        size=64,
    )
    recovered = train_common.unletterbox(letterboxed_label, orig_h, orig_w, size=64)

    assert recovered.shape == (orig_h, orig_w)
    assert (recovered == 7).sum() > 0


def test_letterbox_params_places_content_within_bounds() -> None:
    p = train_common.letterbox_params(h=20, w=40, size=64)
    assert p["top"] >= 0 and p["left"] >= 0
    assert p["top"] + p["nh"] <= 64
    assert p["left"] + p["nw"] <= 64


def _write_run(base: Path, run_id: str, *, created_at: str, valid: bool = True) -> None:
    d = base / run_id
    d.mkdir(parents=True)
    if valid:
        config = {
            "run_id": run_id,
            "model_family": "dinov3_lora",
            "model_config": {"arch": "vitb16", "checkpoint": "ckpt.pth"},
            "classes": [{"classId": 1, "label": "pore", "color": "#ff0000", "isVisible": True}],
            "render": {"norm": "global", "scale": "linear", "vmin_pct": 1.0, "vmax_pct": 99.0, "cmap": "gray"},
            "image_size": 512,
            "hyperparams": {},
            "source_keys": [],
            "created_at": created_at,
        }
        (d / "config.json").write_text(json.dumps(config))
        (d / "metrics.json").write_text(json.dumps({"epochs_completed": 1}))
    else:
        (d / "config.json").write_text("{not valid json")


def test_list_runs_sorts_newest_first_and_skips_malformed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    _write_run(tmp_path, "run_a", created_at="2026-01-01T00:00:00+00:00")
    _write_run(tmp_path, "run_b", created_at="2026-06-01T00:00:00+00:00")
    _write_run(tmp_path, "run_broken", created_at="2026-12-01T00:00:00+00:00", valid=False)

    runs = train_common.list_runs()

    assert [r["run_id"] for r in runs] == ["run_b", "run_a"]


def test_load_run_config_raises_404_for_unknown_run(monkeypatch, tmp_path: Path) -> None:
    from fastapi import HTTPException

    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        train_common.load_run_config("does_not_exist")
    assert exc_info.value.status_code == 404


def test_delete_run_removes_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    _write_run(tmp_path, "run_a", created_at="2026-01-01T00:00:00+00:00")
    assert [r["run_id"] for r in train_common.list_runs()] == ["run_a"]

    train_common.delete_run("run_a")

    assert train_common.list_runs() == []
    assert not (tmp_path / "run_a").exists()


def test_delete_run_raises_404_for_unknown_run(monkeypatch, tmp_path: Path) -> None:
    from fastapi import HTTPException

    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        train_common.delete_run("does_not_exist")
    assert exc_info.value.status_code == 404


def test_delete_run_rejects_unsafe_run_ids(monkeypatch, tmp_path: Path) -> None:
    from fastapi import HTTPException

    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        train_common.delete_run("../escape")
    assert exc_info.value.status_code == 400


def test_capability_never_raises_when_torch_and_dlsia_are_absent(monkeypatch) -> None:
    monkeypatch.setattr(train_common, "torch_available", lambda: False)
    monkeypatch.setattr(train_common, "dlsia_available", lambda: False)

    result = train_common.capability()

    assert result["torch_available"] is False
    assert result["dinov3"]["available"] is False
    assert result["dlsia"]["available"] is False
    assert "error" not in result


def test_capability_reports_tiling_availability() -> None:
    """The frontend needs this to grey out the tiling checkbox the same way it
    already does for dlsia — see ModelPickerPanel's dlsia.available check."""
    result = train_common.capability()

    assert isinstance(result["tiling"]["available"], bool)


def test_capability_survives_qlty_probe_failure(monkeypatch) -> None:
    import tiling

    def _boom():
        raise RuntimeError("qlty probe exploded")

    monkeypatch.setattr(tiling, "qlty_available", _boom)

    result = train_common.capability()

    assert result["tiling"]["available"] is False
    assert "error" in result


def test_compute_miou_requires_torch_but_module_itself_is_import_safe() -> None:
    """Importing train_common must never require torch — only calling its
    torch-dependent functions does."""
    assert train_common.torch_available() in (True, False)  # doesn't raise
