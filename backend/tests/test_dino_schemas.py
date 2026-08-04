"""Tests for the Train-tab (DINOv3 LoRA / dlsia TUNet fine-tune + inference) schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import (
    AnnotationClass,
    DinoHyperParams,
    DinoV3LoraConfig,
    DlsiaTunetConfig,
    ExportSourceItem,
    InferRequest,
    TrainRequest,
    TunetHyperParams,
)


def _classes() -> list[AnnotationClass]:
    return [AnnotationClass(classId=1, label="pore", color="#ff0000")]


def _source(class_id: int = 1) -> ExportSourceItem:
    return ExportSourceItem(
        kind="local",
        source="sample.tif",
        slices={"0": [{"id": "s1", "classId": class_id, "kind": "rectangle", "x": 0, "y": 0, "w": 5, "h": 5}]},
    )


def _dino_model(checkpoint: str = "dinov3_vitb16_pretrain_lvd1689m-abc123.pth") -> DinoV3LoraConfig:
    return DinoV3LoraConfig(arch="vitb16", checkpoint=checkpoint)


def test_train_request_accepts_a_valid_dinov3_payload() -> None:
    """A single source/class/checkpoint should validate cleanly."""
    req = TrainRequest(sources=[_source()], classes=_classes(), model=_dino_model())
    assert req.model.model_family == "dinov3_lora"
    assert req.model.arch == "vitb16"
    assert req.model.hyperparams.epochs == 30
    assert req.model.hyperparams.image_size % 16 == 0


def test_train_request_accepts_a_valid_dlsia_tunet_payload() -> None:
    """A dlsia TUNet request needs no checkpoint/arch."""
    req = TrainRequest(sources=[_source()], classes=_classes(), model=DlsiaTunetConfig())
    assert req.model.model_family == "dlsia_tunet"
    assert req.model.hyperparams.depth == 4


def test_train_request_dispatches_model_family_from_a_plain_dict() -> None:
    """The API boundary receives JSON dicts, not model instances — confirm the
    discriminator resolves correctly from a raw dict payload."""
    req = TrainRequest(
        sources=[_source()],
        classes=_classes(),
        model={"model_family": "dlsia_tunet", "hyperparams": {"depth": 3}},
    )
    assert isinstance(req.model, DlsiaTunetConfig)
    assert req.model.hyperparams.depth == 3

    req2 = TrainRequest(
        sources=[_source()],
        classes=_classes(),
        model={"model_family": "dinov3_lora", "arch": "vitl16", "checkpoint": "ckpt.pth"},
    )
    assert isinstance(req2.model, DinoV3LoraConfig)
    assert req2.model.arch == "vitl16"


def test_train_request_rejects_empty_sources_or_classes() -> None:
    with pytest.raises(ValidationError):
        TrainRequest(sources=[], classes=_classes(), model=_dino_model())
    with pytest.raises(ValidationError):
        TrainRequest(sources=[_source()], classes=[], model=_dino_model())


def test_train_request_rejects_shapes_referencing_undefined_classes() -> None:
    """A source shape referencing a classId absent from `classes` must fail closed."""
    with pytest.raises(ValidationError):
        TrainRequest(sources=[_source(class_id=99)], classes=_classes(), model=_dino_model())


@pytest.mark.parametrize("checkpoint", ["../secret.pth", "a/b.pth", "a\\b.pth", ".", ".."])
def test_dinov3_config_rejects_unsafe_checkpoint_names(checkpoint: str) -> None:
    """Checkpoint must be a single filename component, never a path."""
    with pytest.raises(ValidationError):
        DinoV3LoraConfig(arch="vitb16", checkpoint=checkpoint)


@pytest.mark.parametrize("run_name", ["../evil", "a/b", "a\\b"])
def test_train_request_rejects_unsafe_run_names(run_name: str) -> None:
    with pytest.raises(ValidationError):
        TrainRequest(sources=[_source()], classes=_classes(), model=_dino_model(), run_name=run_name)


def test_dinov3_config_rejects_unknown_arch() -> None:
    with pytest.raises(ValidationError):
        DinoV3LoraConfig(arch="vitg16", checkpoint="ckpt.pth")


def test_dino_arch_literal_matches_the_archs_the_runtime_actually_builds() -> None:
    """dino_runtime.list_checkpoints() advertises every key of VIT_ARCHS via
    /api/train/capability, and the frontend submits whatever arch it's given
    verbatim — so an arch missing from DinoArch is a checkpoint the UI shows
    but every train/estimate request for it 422s. Regression for vit7b16."""
    from typing import get_args

    import dino_runtime
    from schemas import DinoArch

    assert set(get_args(DinoArch)) == set(dino_runtime.VIT_ARCHS)


def test_train_request_rejects_more_than_255_classes() -> None:
    classes = [AnnotationClass(classId=i, label=f"c{i}", color="#ff0000") for i in range(256)]
    with pytest.raises(ValidationError):
        TrainRequest(sources=[_source(class_id=0)], classes=classes, model=_dino_model())


def test_train_request_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        TrainRequest(
            sources=[_source()],
            classes=_classes(),
            model=_dino_model(),
            unexpected_field="nope",
        )


def test_model_configs_reject_cross_family_fields() -> None:
    """A dlsia payload must not accept dinov3-only fields (extra='forbid')."""
    with pytest.raises(ValidationError):
        DlsiaTunetConfig(arch="vitb16", checkpoint="ckpt.pth")
    with pytest.raises(ValidationError):
        DinoV3LoraConfig(arch="vitb16", checkpoint="ckpt.pth", depth=4)


@pytest.mark.parametrize("image_size", [500, 100, 1025])
def test_dino_hyperparams_reject_non_multiple_of_16_or_out_of_range_image_size(image_size: int) -> None:
    with pytest.raises(ValidationError):
        DinoHyperParams(image_size=image_size)


def test_dino_hyperparams_accept_valid_image_size() -> None:
    assert DinoHyperParams(image_size=384).image_size == 384


@pytest.mark.parametrize("image_size", [500, 100, 2049])
def test_tunet_hyperparams_reject_non_multiple_of_64_or_out_of_range_image_size(image_size: int) -> None:
    with pytest.raises(ValidationError):
        TunetHyperParams(image_size=image_size)


def test_tunet_hyperparams_accept_valid_image_size() -> None:
    assert TunetHyperParams(image_size=256).image_size == 256


def test_infer_request_accepts_a_valid_minimal_payload() -> None:
    req = InferRequest(run_id="20260101_000000_vitb16_ab12", kind="local", source="sample.tif", slice_indices=[0, 1, 2])
    assert req.min_confidence == 0.5
    assert req.render is None


@pytest.mark.parametrize("run_id", ["../evil", "a/b", "a\\b", ""])
def test_infer_request_rejects_unsafe_run_id(run_id: str) -> None:
    with pytest.raises(ValidationError):
        InferRequest(run_id=run_id, kind="local", source="sample.tif", slice_indices=[0])


def test_infer_request_rejects_empty_or_duplicate_slice_indices() -> None:
    with pytest.raises(ValidationError):
        InferRequest(run_id="run1", kind="local", source="sample.tif", slice_indices=[])
    with pytest.raises(ValidationError):
        InferRequest(run_id="run1", kind="local", source="sample.tif", slice_indices=[0, 0])


def test_infer_request_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        InferRequest(run_id="run1", kind="local", source="sample.tif", slice_indices=[0], min_confidence=1.5)
