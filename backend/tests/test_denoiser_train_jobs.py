"""Tests for the denoiser's slice through train_jobs: check_resume_compatible
and _apply_parent_architecture with task="denoising".

See test_train_resume.py for the pre-existing segmentation-only coverage —
these guard specifically against the denoiser's "skip the class comparison"
bypass (1) swallowing a genuine segmentation class-label mismatch, or
(2) treating a segmentation/denoising task mismatch as compatible.
"""

from __future__ import annotations

import pytest

import train_jobs
from schemas import AnnotationClass, DlsiaDenoiserConfig, DlsiaTunetConfig, ExportSourceItem, TrainRequest


def _source() -> ExportSourceItem:
    return ExportSourceItem(kind="local", source="sample.tif")


def _seg_classes(*labels: str) -> list[AnnotationClass]:
    return [AnnotationClass(classId=i + 1, label=label, color="#ff0000") for i, label in enumerate(labels)]


def _seg_request(*, labels: tuple[str, ...] = ("pore",)) -> TrainRequest:
    return TrainRequest(sources=[_source()], classes=_seg_classes(*labels), model=DlsiaTunetConfig())


def _denoise_request(*, task: str = "denoising") -> TrainRequest:
    return TrainRequest(
        sources=[_source()],
        classes=[],
        model=DlsiaDenoiserConfig(training_scheme="n2n"),
        task=task,
    )


def _parent(*, task: str = "segmentation", family: str = "dlsia_tunet", labels: tuple[str, ...] = ("pore",)) -> dict:
    config = {
        "model_family": family,
        "task": task,
        "classes": [{"classId": i + 1, "label": label, "color": "#ff0000"} for i, label in enumerate(labels)],
        "model_config": {},
        "hyperparams": {},
    }
    return config


# ---------------------------------------------------------------------------
# check_resume_compatible: the denoising bypass
# ---------------------------------------------------------------------------


def test_two_denoising_configs_with_empty_classes_are_compatible() -> None:
    parent = _parent(task="denoising", family="dlsia_denoiser", labels=())
    train_jobs.check_resume_compatible(parent, _denoise_request())


def test_segmentation_class_mismatch_is_still_rejected() -> None:
    """Regression guard: the denoising bypass must not swallow a genuine
    segmentation class-label mismatch — the whole point of this function."""
    parent = _parent(task="segmentation", family="dlsia_tunet", labels=("pore",))
    with pytest.raises(ValueError, match="classes changed"):
        train_jobs.check_resume_compatible(parent, _seg_request(labels=("pore", "grain")))


def test_segmentation_class_rename_at_same_count_is_still_rejected() -> None:
    """The historically dangerous case (same count, different labels) must
    still be caught — nothing about the denoiser bypass should weaken it."""
    parent = _parent(task="segmentation", family="dlsia_tunet", labels=("pore",))
    with pytest.raises(ValueError, match="classes changed"):
        train_jobs.check_resume_compatible(parent, _seg_request(labels=("crack",)))


def test_task_mismatch_is_rejected_even_when_model_family_matches() -> None:
    """Isolates the task check from the (separately covered) model_family
    check: same family both sides, only `task` differs."""
    parent = _parent(task="segmentation", family="dlsia_tunet", labels=("pore",))
    mismatched_request = TrainRequest(
        sources=[_source()], classes=[], model=DlsiaTunetConfig(), task="denoising"
    )
    with pytest.raises(ValueError, match="task"):
        train_jobs.check_resume_compatible(parent, mismatched_request)


def test_denoising_parent_rejects_a_segmentation_resume_request() -> None:
    parent = _parent(task="denoising", family="dlsia_denoiser", labels=())
    with pytest.raises(ValueError):
        train_jobs.check_resume_compatible(parent, _seg_request())


def test_segmentation_parent_rejects_a_denoising_resume_request() -> None:
    parent = _parent(task="segmentation", family="dlsia_tunet", labels=("pore",))
    with pytest.raises(ValueError):
        train_jobs.check_resume_compatible(parent, _denoise_request())


def test_parent_missing_task_key_defaults_to_segmentation_for_compatibility_check() -> None:
    """A run saved before `task` existed has no key at all — must be treated
    as segmentation for this check too, not crash or silently mismatch."""
    parent = _parent(labels=("pore",))
    del parent["task"]
    train_jobs.check_resume_compatible(parent, _seg_request(labels=("pore",)))


def test_model_family_mismatch_is_still_checked_before_task() -> None:
    """The pre-existing model_family guard must still fire first/independently
    of the new task check."""
    parent = _parent(task="segmentation", family="dinov3_lora", labels=("pore",))
    with pytest.raises(ValueError, match="model family"):
        train_jobs.check_resume_compatible(parent, _seg_request(labels=("pore",)))


# ---------------------------------------------------------------------------
# _apply_parent_architecture: explicit per-type dispatch
# ---------------------------------------------------------------------------


def test_denoiser_topology_is_forced_to_the_parents_values() -> None:
    request = _denoise_request()
    request.model.hyperparams.depth = 2
    request.model.hyperparams.base_channels = 4
    request.model.hyperparams.growth_rate = 1.1
    parent = _parent(task="denoising", family="dlsia_denoiser", labels=())
    parent["hyperparams"] = {"depth": 5, "base_channels": 16, "growth_rate": 2.0}

    train_jobs._apply_parent_architecture(parent, request)

    hp = request.model.hyperparams
    assert (hp.depth, hp.base_channels, hp.growth_rate) == (5, 16, 2.0)


def test_apply_parent_architecture_raises_for_an_unrecognized_config_type() -> None:
    """Same guard as build_family: an unrecognized model config type must be a
    real error, not silently treated as a TUNet."""

    class _BogusModelConfig:
        hyperparams = type("HP", (), {})()

    class _BogusRequest:
        model = _BogusModelConfig()

    with pytest.raises(ValueError, match="Unknown model config type"):
        train_jobs._apply_parent_architecture({"hyperparams": {}}, _BogusRequest())
