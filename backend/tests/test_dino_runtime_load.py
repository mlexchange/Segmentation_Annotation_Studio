"""dino_runtime.load_backbone's checkpoint-loading memory behaviour.

Loading directly onto the device briefly holds BOTH the checkpoint's state
dict and the already-allocated model on-device at once — for the largest
supported arch (vit7b16, ~27GB of weights) that is ~54GB of peak device
memory just to load it, which is what actually trips a device OOM (MPS's
allocation ceiling is a Metal-tracked accounting limit, not a literal "is
there free RAM" check, so it triggers even on unified-memory Apple Silicon).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

import dino_runtime  # noqa: E402


def test_checkpoint_is_loaded_onto_cpu_not_the_target_device(monkeypatch, tmp_path: Path) -> None:
    backbone = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        dino_runtime, "_ensure_vendor_on_path", lambda: None
    )
    fake_build = MagicMock(return_value=backbone)
    monkeypatch.setitem(
        __import__("sys").modules,
        "dinov3.hub_backbones",
        MagicMock(build_untrained_backbone=fake_build),
    )
    captured: dict = {}

    def _fake_load(path, *, map_location=None, weights_only=None):
        captured["map_location"] = map_location
        return backbone.state_dict()

    monkeypatch.setattr(torch, "load", _fake_load)

    ckpt = tmp_path / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    ckpt.write_bytes(b"unused - torch.load is mocked")

    dino_runtime.load_backbone("vitl16", ckpt, "cpu")

    assert captured["map_location"] == "cpu"
    assert captured["map_location"] != "mps"
    assert captured["map_location"] != "cuda"


class _FakeLoRA:
    """Stands in for an injected LoRALinear — only the two tensors matter here."""

    def __init__(self, rank: int = 2, dim: int = 4) -> None:
        self.lora_A = torch.nn.Parameter(torch.zeros(rank, dim))
        self.lora_B = torch.nn.Parameter(torch.zeros(dim, rank))


def test_load_lora_state_dict_reports_how_many_modules_it_loaded() -> None:
    """Callers need the count: names missing from the saved dict are skipped
    rather than raising, so a total mismatch (wrong arch/rank) would otherwise
    be indistinguishable from a clean load — see train_common.build_family's
    warm-start guard."""
    modules = {"a": _FakeLoRA(), "b": _FakeLoRA()}
    saved = {
        name: {"lora_A": torch.ones(2, 4), "lora_B": torch.ones(4, 2)}
        for name in ("a", "b")
    }

    assert dino_runtime.load_lora_state_dict(modules, saved) == 2
    assert torch.allclose(modules["a"].lora_A, torch.ones(2, 4))


def test_load_lora_state_dict_counts_only_the_names_present() -> None:
    modules = {"a": _FakeLoRA(), "b": _FakeLoRA()}
    saved = {"a": {"lora_A": torch.ones(2, 4), "lora_B": torch.ones(4, 2)}}

    assert dino_runtime.load_lora_state_dict(modules, saved) == 1
    # "b" was skipped, so it must still hold its freshly-initialised value.
    assert torch.allclose(modules["b"].lora_A, torch.zeros(2, 4))


def test_load_lora_state_dict_returns_zero_when_nothing_matched() -> None:
    """The signal build_family turns into a loud "architecture does not match"."""
    modules = {"a": _FakeLoRA()}
    assert dino_runtime.load_lora_state_dict(modules, {"totally": "different"}) == 0


def test_loaded_lora_tensors_stay_trainable() -> None:
    """A resume has to be able to keep training these — copy_ under no_grad
    must not clear requires_grad."""
    modules = {"a": _FakeLoRA()}
    dino_runtime.load_lora_state_dict(
        modules, {"a": {"lora_A": torch.ones(2, 4), "lora_B": torch.ones(4, 2)}}
    )
    assert modules["a"].lora_A.requires_grad
    assert modules["a"].lora_B.requires_grad
