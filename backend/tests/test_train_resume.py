"""Tests for continuing fine-tuning from a saved run (``resume_from_run_id``).

The compatibility guard is the point of this feature's risk: the saved head has
one output channel per class in the parent run's class ORDER (infer_jobs maps
channel c to classes[c]["classId"]), so a class list that changed shape — or
merely reordered/relabelled — makes the saved weights mean something different.
A count mismatch fails loudly inside load_state_dict on its own; a
same-count-different-labels resume would not, which is what these pin down.
"""

from __future__ import annotations

import pytest

import train_common
import train_jobs
from schemas import AnnotationClass, DinoV3LoraConfig, DlsiaTunetConfig, ExportSourceItem, TrainRequest


def _classes(*labels: str) -> list[AnnotationClass]:
    return [
        AnnotationClass(classId=i + 1, label=label, color="#ff0000")
        for i, label in enumerate(labels)
    ]


def _source() -> ExportSourceItem:
    return ExportSourceItem(
        kind="local",
        source="sample.tif",
        slices={"0": [{"id": "s1", "classId": 1, "kind": "rectangle", "x": 0, "y": 0, "w": 5, "h": 5}]},
    )


def _request(*, labels: tuple[str, ...] = ("pore",), family: str = "dlsia_tunet", **hp) -> TrainRequest:
    model = (
        DinoV3LoraConfig(arch="vits16", checkpoint="ckpt.pth", hyperparams=hp or {})
        if family == "dinov3_lora"
        else DlsiaTunetConfig(hyperparams=hp or {})
    )
    return TrainRequest(sources=[_source()], classes=_classes(*labels), model=model)


def _parent(*, labels: tuple[str, ...] = ("pore",), family: str = "dlsia_tunet", **overrides) -> dict:
    config = {
        "model_family": family,
        "classes": [{"classId": i + 1, "label": label, "color": "#ff0000"} for i, label in enumerate(labels)],
        "model_config": {},
        "hyperparams": {},
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Compatibility guard
# ---------------------------------------------------------------------------


def test_identical_classes_are_compatible() -> None:
    train_jobs.check_resume_compatible(_parent(labels=("pore", "grain")), _request(labels=("pore", "grain")))


def test_label_case_and_whitespace_do_not_block_a_resume() -> None:
    """Matching is case/whitespace-insensitive — a user retyping "Pore" for
    "pore" has not actually changed what the model was trained to find."""
    parent = _parent(labels=("Pore", " Grain "))
    train_jobs.check_resume_compatible(parent, _request(labels=("pore", "grain")))


def test_a_different_class_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="classes changed"):
        train_jobs.check_resume_compatible(_parent(labels=("pore",)), _request(labels=("pore", "grain")))


def test_reordered_classes_are_rejected() -> None:
    """Same labels, different order: the head's channel c still means the
    parent's class c, so this would silently train against swapped semantics."""
    with pytest.raises(ValueError, match="classes changed"):
        train_jobs.check_resume_compatible(
            _parent(labels=("pore", "grain")), _request(labels=("grain", "pore"))
        )


def test_renamed_class_at_the_same_count_is_rejected() -> None:
    """The dangerous case — nothing downstream would have complained."""
    with pytest.raises(ValueError, match="classes changed"):
        train_jobs.check_resume_compatible(_parent(labels=("pore",)), _request(labels=("crack",)))


def test_a_different_model_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="model family"):
        train_jobs.check_resume_compatible(
            _parent(family="dinov3_lora"), _request(family="dlsia_tunet")
        )


def test_the_error_message_names_both_class_lists() -> None:
    """The user has to be able to tell what to put back — so the message must
    say what the run expects, not just that something is wrong."""
    with pytest.raises(ValueError) as excinfo:
        train_jobs.check_resume_compatible(_parent(labels=("pore",)), _request(labels=("crack", "void")))
    message = str(excinfo.value)
    assert "pore" in message
    assert "crack" in message and "void" in message


# ---------------------------------------------------------------------------
# Architecture override
# ---------------------------------------------------------------------------


def test_geometry_settings_are_forced_to_the_parents_values() -> None:
    """image_size/tiling define the geometry the weights were fit to, so a
    resume must use the parent's regardless of what the client asked for."""
    request = _request(image_size=256, tiling=False)
    parent = _parent(hyperparams={"image_size": 512, "tiling": True})

    train_jobs._apply_parent_architecture(parent, request)

    assert request.model.hyperparams.image_size == 512
    assert request.model.hyperparams.tiling is True


def test_retunable_hyperparams_are_left_alone() -> None:
    """Epochs/lr/batch_size are the whole point of resuming — overriding them
    from the parent would make "continue for 10 more epochs" impossible."""
    request = _request(epochs=10, lr=5e-4, batch_size=3, seed=99, flip_augment=False)
    parent = _parent(hyperparams={"epochs": 60, "lr": 1e-3, "batch_size": 4, "seed": 1234, "flip_augment": True})

    train_jobs._apply_parent_architecture(parent, request)

    hp = request.model.hyperparams
    assert (hp.epochs, hp.lr, hp.batch_size, hp.seed, hp.flip_augment) == (10, 5e-4, 3, 99, False)


def test_tunet_topology_is_forced_to_the_parents_values() -> None:
    request = _request(depth=2, base_channels=4, growth_rate=1.1)
    parent = _parent(hyperparams={"depth": 5, "base_channels": 16, "growth_rate": 2.0})

    train_jobs._apply_parent_architecture(parent, request)

    hp = request.model.hyperparams
    assert (hp.depth, hp.base_channels, hp.growth_rate) == (5, 16, 2.0)


def test_dino_arch_checkpoint_and_lora_shapes_are_forced_to_the_parents_values() -> None:
    """arch/checkpoint fix embed_dim and rank/alpha fix the LoRA tensor shapes —
    all four would make the saved weights unloadable if they drifted."""
    request = _request(family="dinov3_lora", lora_rank=4, lora_alpha=8.0)
    parent = _parent(
        family="dinov3_lora",
        model_config={"arch": "vitl16", "checkpoint": "parent.pth"},
        hyperparams={"lora_rank": 16, "lora_alpha": 32.0},
    )

    train_jobs._apply_parent_architecture(parent, request)

    assert request.model.arch == "vitl16"
    assert request.model.checkpoint == "parent.pth"
    assert request.model.hyperparams.lora_rank == 16
    assert request.model.hyperparams.lora_alpha == 32.0


def test_a_parent_missing_optional_hyperparams_leaves_the_request_untouched() -> None:
    """Runs saved before a field existed have no value to copy — the request's
    own value has to survive rather than becoming None."""
    request = _request(image_size=256)
    train_jobs._apply_parent_architecture(_parent(hyperparams={}), request)
    assert request.model.hyperparams.image_size == 256


# ---------------------------------------------------------------------------
# Job wiring
# ---------------------------------------------------------------------------


def test_run_train_job_rejects_an_incompatible_resume_before_preparing_datasets(monkeypatch) -> None:
    """The guard must run before the (minutes-long) render pass, so a mistake
    costs seconds — same fail-fast contract as the qlty check."""
    import export_jobs

    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")
    monkeypatch.setattr(train_common, "load_run_config", lambda run_id: _parent(labels=("pore",)))

    def _must_not_run(*args, **kwargs):
        raise AssertionError("prepare_datasets must not run for an incompatible resume")

    monkeypatch.setattr(train_common, "prepare_datasets", _must_not_run)

    request = _request(labels=("crack", "void"), tiling=False)
    request.resume_from_run_id = "parent-run"
    jid = export_jobs.new_job("test")
    train_jobs.run_train_job(jid, request, "child-run")
    job = export_jobs.get_job(jid)

    assert job["state"] == "error"
    assert "classes changed" in job["error"]
    assert train_common.ML_LOCK.locked() is False


def test_run_train_job_passes_the_parents_weights_to_build_family(monkeypatch) -> None:
    """The actual warm start: build_family has to receive the saved adapter
    state, otherwise a "resume" silently trains from scratch."""
    import export_jobs

    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")
    monkeypatch.setattr(train_common, "load_run_config", lambda run_id: _parent(labels=("pore",)))
    monkeypatch.setattr(train_common, "load_adapter_state", lambda run_id: {"sentinel": True})
    monkeypatch.setattr(train_common, "prepare_datasets", lambda *a, **k: {"train": [(1, 2)], "val": []})

    seen: dict = {}

    def _fake_build_family(model_cfg, n_classes, device, log_cb, init_state=None):
        seen["init_state"] = init_state
        raise RuntimeError("stop here — build_family reached with the state we care about")

    monkeypatch.setattr(train_common, "build_family", _fake_build_family)

    request = _request(labels=("pore",), tiling=False)
    request.resume_from_run_id = "parent-run"
    jid = export_jobs.new_job("test")
    train_jobs.run_train_job(jid, request, "child-run")

    assert seen["init_state"] == {"sentinel": True}


def test_a_non_resume_run_passes_no_init_state(monkeypatch) -> None:
    """Training from scratch must stay exactly as it was — no accidental
    warm start when resume_from_run_id is absent."""
    import export_jobs

    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")
    monkeypatch.setattr(train_common, "prepare_datasets", lambda *a, **k: {"train": [(1, 2)], "val": []})

    def _must_not_load(*a, **k):
        raise AssertionError("load_adapter_state must not run without resume_from_run_id")

    monkeypatch.setattr(train_common, "load_adapter_state", _must_not_load)

    seen: dict = {}

    def _fake_build_family(model_cfg, n_classes, device, log_cb, init_state=None):
        seen["init_state"] = init_state
        raise RuntimeError("stop here")

    monkeypatch.setattr(train_common, "build_family", _fake_build_family)

    jid = export_jobs.new_job("test")
    train_jobs.run_train_job(jid, _request(tiling=False), "run-1")

    assert seen["init_state"] is None


def test_save_run_records_the_parent_run_id(monkeypatch, tmp_path) -> None:
    """Lineage: a chain of refinements has to stay traceable on disk."""
    import json

    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    pytest.importorskip("torch")

    train_common.save_run(
        "child",
        model_family="dlsia_tunet",
        model_config={},
        classes=[{"classId": 1, "label": "pore", "color": "#ff0000"}],
        render={},
        image_size=512,
        hyperparams={},
        source_keys=[],
        adapter_state={},
        metrics={},
        resumed_from="parent",
    )

    config = json.loads((tmp_path / "child" / "config.json").read_text())
    assert config["resumed_from"] == "parent"


def test_save_run_records_none_for_a_fresh_run(monkeypatch, tmp_path) -> None:
    import json

    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path))
    pytest.importorskip("torch")

    train_common.save_run(
        "fresh",
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

    config = json.loads((tmp_path / "fresh" / "config.json").read_text())
    assert config["resumed_from"] is None
