"""Torch-free tests for dino_runtime: checkpoint discovery and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

import dino_runtime


def _touch(path: Path, size: int = 100) -> None:
    path.write_bytes(b"0" * size)


def test_list_checkpoints_parses_recognised_filenames(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_MODELS_DIR", str(tmp_path))
    _touch(tmp_path / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    _touch(tmp_path / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", size=200)
    _touch(tmp_path / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")

    checkpoints = dino_runtime.list_checkpoints()

    by_arch = {c["arch"]: c for c in checkpoints}
    assert set(by_arch) == {"vitb16", "vitl16", "vits16plus"}
    assert by_arch["vitb16"]["embed_dim"] == 768
    assert by_arch["vitl16"]["size_bytes"] == 200
    assert by_arch["vits16plus"]["weights"] == "lvd1689m"


def test_list_checkpoints_distinguishes_plus_variant_from_base(monkeypatch, tmp_path: Path) -> None:
    """vits16plus must not be misparsed as vits16 (regression: alternation ordering)."""
    monkeypatch.setenv("DINO_MODELS_DIR", str(tmp_path))
    _touch(tmp_path / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    _touch(tmp_path / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")

    checkpoints = dino_runtime.list_checkpoints()

    archs = sorted(c["arch"] for c in checkpoints)
    assert archs == ["vits16", "vits16plus"]


def test_list_checkpoints_skips_unrecognised_and_convnext_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_MODELS_DIR", str(tmp_path))
    _touch(tmp_path / "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth")  # not a ViT
    _touch(tmp_path / "not_a_checkpoint.txt")
    _touch(tmp_path / "dinov3_vitl16_pretrain.pth")  # missing the -<hash> suffix

    assert dino_runtime.list_checkpoints() == []


@pytest.mark.parametrize("weights", ["lvd1689m-a955f4ea", "sat493m-a6675841"])
def test_list_checkpoints_recognises_vit7b16(monkeypatch, tmp_path: Path, weights: str) -> None:
    """ViT-7B was originally excluded as too large; it is supported now that both
    published weight variants load (see vendor/dinov3/hub_backbones.py)."""
    monkeypatch.setenv("DINO_MODELS_DIR", str(tmp_path))
    _touch(tmp_path / f"dinov3_vit7b16_pretrain_{weights}.pth")

    checkpoints = dino_runtime.list_checkpoints()

    assert len(checkpoints) == 1
    assert checkpoints[0]["arch"] == "vit7b16"
    assert checkpoints[0]["embed_dim"] == 4096


def test_list_checkpoints_returns_empty_list_when_dir_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_MODELS_DIR", str(tmp_path / "does_not_exist"))
    assert dino_runtime.list_checkpoints() == []


def test_resolve_checkpoint_accepts_a_matching_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_MODELS_DIR", str(tmp_path))
    fname = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    _touch(tmp_path / fname)

    resolved = dino_runtime.resolve_checkpoint("vitb16", fname)

    assert resolved == (tmp_path / fname).resolve()


@pytest.mark.parametrize("file_name", ["../etc/passwd", "sub/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", "a\\b.pth"])
def test_resolve_checkpoint_rejects_path_separators(monkeypatch, tmp_path: Path, file_name: str) -> None:
    monkeypatch.setenv("DINO_MODELS_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        dino_runtime.resolve_checkpoint("vitb16", file_name)
    assert exc_info.value.status_code == 400


def test_resolve_checkpoint_rejects_arch_mismatch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_MODELS_DIR", str(tmp_path))
    fname = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    _touch(tmp_path / fname)

    with pytest.raises(HTTPException) as exc_info:
        dino_runtime.resolve_checkpoint("vitl16", fname)
    assert exc_info.value.status_code == 400


def test_resolve_checkpoint_rejects_missing_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DINO_MODELS_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        dino_runtime.resolve_checkpoint("vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    assert exc_info.value.status_code == 404
