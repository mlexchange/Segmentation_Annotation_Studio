"""Tests for the denoiser (Noise2Noise/Noise2Void) additions to the Train-tab
schemas: TrainRequest.task, the now-conditional `classes` requirement, and
DlsiaDenoiserConfig's place in the ModelConfig discriminated union.

See test_dino_schemas.py for the pre-existing segmentation-only coverage this
must not regress.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from schemas import (
    AnnotationClass,
    DlsiaDenoiserConfig,
    DlsiaTunetConfig,
    ExportSourceItem,
    ModelConfig,
    TrainRequest,
)


def _source() -> ExportSourceItem:
    """A source with no annotated shapes at all — a denoiser needs raw images,
    not a taxonomy, so its sources typically carry no slices."""
    return ExportSourceItem(kind="local", source="sample.tif")


def _seg_classes() -> list[AnnotationClass]:
    return [AnnotationClass(classId=1, label="pore", color="#ff0000")]


# ---------------------------------------------------------------------------
# task field / conditional classes requirement
# ---------------------------------------------------------------------------


def test_denoising_task_accepts_empty_classes() -> None:
    req = TrainRequest(
        sources=[_source()],
        classes=[],
        model=DlsiaDenoiserConfig(training_scheme="n2n"),
        task="denoising",
    )
    assert req.task == "denoising"
    assert req.classes == []


def test_segmentation_task_still_rejects_empty_classes() -> None:
    """Regression: the original hard `Field(min_length=1)` behaviour must
    survive the move to a task-aware model_validator."""
    with pytest.raises(ValidationError):
        TrainRequest(
            sources=[_source()],
            classes=[],
            model=DlsiaTunetConfig(),
        )


def test_task_defaults_to_segmentation_for_old_style_requests() -> None:
    """A request body that never mentions `task` at all (every pre-existing
    caller) must behave exactly as it did before this field existed."""
    req = TrainRequest(sources=[_source()], classes=_seg_classes(), model=DlsiaTunetConfig())
    assert req.task == "segmentation"


def test_denoising_task_with_classes_present_still_validates_the_taxonomy() -> None:
    """Relaxing the *minimum* doesn't relax the other taxonomy rules — a
    denoising request that does carry classes is still checked for
    uniqueness/references like any other."""
    dup_classes = [
        AnnotationClass(classId=1, label="pore", color="#ff0000"),
        AnnotationClass(classId=1, label="grain", color="#00ff00"),
    ]
    with pytest.raises(ValidationError):
        TrainRequest(
            sources=[_source()],
            classes=dup_classes,
            model=DlsiaDenoiserConfig(training_scheme="n2n"),
            task="denoising",
        )


def test_unknown_task_literal_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainRequest(
            sources=[_source()],
            classes=[],
            model=DlsiaDenoiserConfig(training_scheme="n2n"),
            task="bogus",
        )


# ---------------------------------------------------------------------------
# DlsiaDenoiserConfig / ModelConfig union
# ---------------------------------------------------------------------------


def test_dlsia_denoiser_config_round_trips_through_the_discriminated_union() -> None:
    adapter: TypeAdapter = TypeAdapter(ModelConfig)
    cfg = adapter.validate_python({"model_family": "dlsia_denoiser", "training_scheme": "n2v"})

    assert isinstance(cfg, DlsiaDenoiserConfig)
    assert cfg.training_scheme == "n2v"
    assert cfg.hyperparams.depth == 4  # TunetHyperParams default, reused verbatim

    round_tripped = adapter.validate_python(cfg.model_dump())
    assert round_tripped == cfg


def test_train_request_dispatches_denoiser_config_from_a_plain_dict() -> None:
    """The API boundary receives JSON dicts, not model instances — confirm the
    discriminator resolves DlsiaDenoiserConfig from a raw dict payload."""
    req = TrainRequest(
        sources=[_source()],
        classes=[],
        model={"model_family": "dlsia_denoiser", "training_scheme": "n2n"},
        task="denoising",
    )
    assert isinstance(req.model, DlsiaDenoiserConfig)
    assert req.model.training_scheme == "n2n"


def test_dlsia_denoiser_config_requires_training_scheme() -> None:
    with pytest.raises(ValidationError):
        DlsiaDenoiserConfig()


def test_dlsia_denoiser_config_rejects_unknown_training_scheme() -> None:
    with pytest.raises(ValidationError):
        DlsiaDenoiserConfig(training_scheme="bogus")


def test_dlsia_denoiser_config_rejects_cross_family_fields() -> None:
    """extra='forbid' — a denoiser payload must not accept DINOv3-only fields."""
    with pytest.raises(ValidationError):
        DlsiaDenoiserConfig(training_scheme="n2n", arch="vitb16", checkpoint="ckpt.pth")
