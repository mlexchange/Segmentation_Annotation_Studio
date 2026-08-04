"""Tests for train_common.build_family — the single place a model gets built
from a schemas.ModelConfig, shared by train_jobs.py and batch_probe.py.

Real (tiny) torch ops stand in for the model, same pattern as
test_infer_jobs_tiling.py: only the family-runtime constructors
(dino_runtime/dlsia_runtime) are mocked, so this exercises build_family's own
branch selection and BuiltFamily shape rather than a real backbone/checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import dino_runtime  # noqa: E402
import dlsia_runtime  # noqa: E402
import train_common  # noqa: E402
from schemas import DinoV3LoraConfig, DlsiaTunetConfig  # noqa: E402


def _log(_msg: str) -> None:
    pass


def test_dino_branch_has_no_set_train_mode_and_snapshot_has_arch_and_checkpoint(monkeypatch) -> None:
    """DINOv3's frozen backbone has no batch-dependent layers, so there is
    nothing for set_train_mode to toggle — unlike TUNet's BatchNorm."""
    head = torch.nn.Linear(1, 1)  # real nn.Module: adapter_state_fn calls head.state_dict()
    monkeypatch.setattr(dino_runtime, "resolve_checkpoint", lambda arch, checkpoint: "fake.pth")
    monkeypatch.setattr(
        dino_runtime, "build_model", lambda arch, ckpt, n_classes, rank, alpha, device: ("backbone", head, {})
    )
    monkeypatch.setattr(dino_runtime, "build_trainable_params", lambda head, lora_modules: ["param"])
    monkeypatch.setattr(dino_runtime, "make_forward_fn", lambda backbone, head: (lambda x: x))
    monkeypatch.setattr(dino_runtime, "make_to_tensor_fn", lambda: (lambda rgb: rgb))
    monkeypatch.setattr(dino_runtime, "lora_state_dict", lambda lora_modules: {"lora": True})

    model_cfg = DinoV3LoraConfig(arch="vits16", checkpoint="ckpt.pth")
    built = train_common.build_family(model_cfg, n_classes=3, device="cpu", log_cb=_log)

    assert built.set_train_mode is None
    assert built.model_config_snapshot == {"arch": "vits16", "checkpoint": "ckpt.pth"}
    assert built.trainable_params == ["param"]
    assert built.forward_fn("x") == "x"
    adapter_state = built.adapter_state_fn()
    assert adapter_state["lora"] == {"lora": True}
    assert adapter_state["head"].keys() == head.state_dict().keys()


def test_tunet_branch_has_set_train_mode_and_snapshot_has_depth_and_channels(monkeypatch) -> None:
    """TUNet's BatchNorm needs train()/eval() toggled around validation, so
    set_train_mode must be a real callable here, unlike the DINOv3 branch."""
    model = torch.nn.Linear(1, 1)  # real nn.Module: build_family calls model.parameters()
    monkeypatch.setattr(train_common, "dlsia_available", lambda: True)
    monkeypatch.setattr(
        dlsia_runtime,
        "build_model",
        lambda n_classes, image_size, depth, base_channels, growth_rate, device: model,
    )
    monkeypatch.setattr(dlsia_runtime, "make_forward_fn", lambda model: (lambda x: x))
    monkeypatch.setattr(dlsia_runtime, "make_to_tensor_fn", lambda: (lambda rgb: rgb))
    monkeypatch.setattr(dlsia_runtime, "make_set_train_mode_fn", lambda model: (lambda is_training: None))
    monkeypatch.setattr(dlsia_runtime, "network_dict", lambda model: {"weights": True})

    model_cfg = DlsiaTunetConfig(hyperparams={"depth": 3, "base_channels": 16, "growth_rate": 2.0})
    built = train_common.build_family(model_cfg, n_classes=3, device="cpu", log_cb=_log)

    assert callable(built.set_train_mode)
    assert built.model_config_snapshot == {"depth": 3, "base_channels": 16, "growth_rate": 2.0}
    assert built.trainable_params == list(model.parameters())
    assert built.adapter_state_fn() == {"weights": True}


def test_tunet_branch_without_a_real_model_object_still_reports_missing_dlsia(monkeypatch) -> None:
    """A torch-only install (no dlsia) must fail with a clear error, not an
    AttributeError from trying to build a TUNet that doesn't exist."""
    monkeypatch.setattr(train_common, "dlsia_available", lambda: False)

    model_cfg = DlsiaTunetConfig()
    with pytest.raises(RuntimeError, match="dlsia"):
        train_common.build_family(model_cfg, n_classes=3, device="cpu", log_cb=_log)


# ---------------------------------------------------------------------------
# Warm start (continue fine-tuning from a saved run's weights)
# ---------------------------------------------------------------------------


def test_dino_warm_start_loads_the_saved_lora_and_head(monkeypatch) -> None:
    head = torch.nn.Linear(1, 1)
    loaded: dict = {}
    monkeypatch.setattr(dino_runtime, "resolve_checkpoint", lambda arch, checkpoint: "fake.pth")
    monkeypatch.setattr(
        dino_runtime, "build_model", lambda arch, ckpt, n_classes, rank, alpha, device: ("backbone", head, {"m": 1})
    )
    monkeypatch.setattr(dino_runtime, "build_trainable_params", lambda h, lm: ["param"])
    monkeypatch.setattr(dino_runtime, "make_forward_fn", lambda backbone, h: (lambda x: x))
    monkeypatch.setattr(dino_runtime, "make_to_tensor_fn", lambda: (lambda rgb: rgb))

    def _fake_load_lora(lora_modules, state):
        loaded["lora_state"] = state
        return len(lora_modules)

    monkeypatch.setattr(dino_runtime, "load_lora_state_dict", _fake_load_lora)
    monkeypatch.setattr(head, "load_state_dict", lambda state: loaded.update({"head_state": state}))

    init_state = {"lora": {"m": {"lora_A": 1, "lora_B": 2}}, "head": {"weight": 3}}
    train_common.build_family(
        DinoV3LoraConfig(arch="vits16", checkpoint="ckpt.pth"),
        n_classes=3, device="cpu", log_cb=_log, init_state=init_state,
    )

    assert loaded["lora_state"] == init_state["lora"]
    assert loaded["head_state"] == init_state["head"]


def test_dino_warm_start_fails_loudly_when_no_lora_module_matched(monkeypatch) -> None:
    """load_lora_state_dict skips unknown names instead of raising, so a total
    mismatch (wrong arch/rank) would otherwise train from the pretrained base
    while the log claimed it was resuming. That must be an error."""
    head = torch.nn.Linear(1, 1)
    monkeypatch.setattr(dino_runtime, "resolve_checkpoint", lambda arch, checkpoint: "fake.pth")
    monkeypatch.setattr(
        dino_runtime, "build_model", lambda arch, ckpt, n_classes, rank, alpha, device: ("backbone", head, {})
    )
    monkeypatch.setattr(dino_runtime, "load_lora_state_dict", lambda lora_modules, state: 0)

    with pytest.raises(RuntimeError, match="does not match"):
        train_common.build_family(
            DinoV3LoraConfig(arch="vits16", checkpoint="ckpt.pth"),
            n_classes=3, device="cpu", log_cb=_log,
            init_state={"lora": {}, "head": {}},
        )


def test_tunet_warm_start_rebuilds_from_the_saved_topology(monkeypatch) -> None:
    """load_model reconstructs the net from the saved topo_dict, so build_model
    must NOT be called — the request's depth/channels can't apply to weights
    trained at a different topology."""
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(train_common, "dlsia_available", lambda: True)

    def _must_not_build(*a, **k):
        raise AssertionError("build_model must not run for a warm start — use load_model")

    monkeypatch.setattr(dlsia_runtime, "build_model", _must_not_build)
    monkeypatch.setattr(dlsia_runtime, "load_model", lambda state, device: model)
    monkeypatch.setattr(dlsia_runtime, "make_forward_fn", lambda m: (lambda x: x))
    monkeypatch.setattr(dlsia_runtime, "make_to_tensor_fn", lambda: (lambda rgb: rgb))
    monkeypatch.setattr(dlsia_runtime, "make_set_train_mode_fn", lambda m: (lambda t: None))

    init_state = {"topo_dict": {"depth": 5, "base_channels": 16, "growth_rate": 2.0}, "state_dict": {}}
    built = train_common.build_family(
        DlsiaTunetConfig(hyperparams={"depth": 2, "base_channels": 4}),
        n_classes=3, device="cpu", log_cb=_log, init_state=init_state,
    )

    # The snapshot must describe what actually ran (the saved topology), not the
    # request's ignored values — otherwise the child run's config lies.
    assert built.model_config_snapshot == {"depth": 5, "base_channels": 16, "growth_rate": 2.0}
    assert built.trainable_params == list(model.parameters())


def test_tunet_warm_start_model_is_left_trainable(monkeypatch) -> None:
    """A resumed model that arrived frozen would train nothing and report
    success — every parameter must still require grad."""
    model = torch.nn.Linear(4, 2)
    monkeypatch.setattr(train_common, "dlsia_available", lambda: True)
    monkeypatch.setattr(dlsia_runtime, "load_model", lambda state, device: model)
    monkeypatch.setattr(dlsia_runtime, "make_forward_fn", lambda m: (lambda x: x))
    monkeypatch.setattr(dlsia_runtime, "make_to_tensor_fn", lambda: (lambda rgb: rgb))
    monkeypatch.setattr(dlsia_runtime, "make_set_train_mode_fn", lambda m: (lambda t: None))

    built = train_common.build_family(
        DlsiaTunetConfig(), n_classes=2, device="cpu", log_cb=_log,
        init_state={"topo_dict": {}, "state_dict": {}},
    )

    assert built.trainable_params
    assert all(p.requires_grad for p in built.trainable_params)
