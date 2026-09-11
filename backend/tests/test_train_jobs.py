"""Tests for train_jobs.py — the pure resume/validation helpers directly, and
run_train_job's early guard/error paths (lock contention, task/model
mismatch). The core train/save/load contract itself is proven for real (no
mocks) in test_train_e2e_real_ml.py; a full run_train_job segmentation round
trip would need real annotated sources assembled the way prepare_datasets
expects, which is out of scope for this pass.
"""
from __future__ import annotations

import pytest

import export_jobs
import train_common
import train_jobs
from schemas import (
    AnnotationClass,
    DlsiaDenoiserConfig,
    DlsiaTunetConfig,
    ExportSourceItem,
    TrainRequest,
)


def _classes():
    return [AnnotationClass(classId=1, label="A", color="#f00"), AnnotationClass(classId=2, label="B", color="#0f0")]


def _sources():
    return [ExportSourceItem(kind="local", source="fake.tif", slices={"0": []})]


def _tunet_request(**overrides):
    defaults = dict(sources=_sources(), classes=_classes(), model=DlsiaTunetConfig())
    defaults.update(overrides)
    return TrainRequest(**defaults)


def _denoiser_request(**overrides):
    defaults = dict(
        task="denoising", sources=_sources(), classes=[],
        model=DlsiaDenoiserConfig(architecture="cnn_ae", training_scheme="ae", ae_compression=4),
    )
    defaults.update(overrides)
    return TrainRequest(**defaults)


class TestNewRunId:
    def test_includes_model_family_and_is_unique(self):
        a = train_jobs.new_run_id("dlsia_tunet")
        b = train_jobs.new_run_id("dlsia_tunet")
        assert "dlsia_tunet" in a
        assert a != b


class TestCheckTaskMatchesModel:
    def test_segmentation_task_with_denoiser_family_raises(self):
        # classes=_classes() needed to get past the schema's own "segmentation
        # needs >=1 class" validator so this reaches check_task_matches_model's
        # family-specific check instead of failing at construction for an
        # unrelated reason.
        request = _denoiser_request(task="segmentation", classes=_classes())
        with pytest.raises(ValueError, match="cannot be trained as a"):
            train_jobs.check_task_matches_model(request)

    def test_denoising_task_with_segmentation_family_raises(self):
        with pytest.raises(ValueError, match="needs a denoiser model family"):
            train_jobs.check_task_matches_model(_tunet_request(task="denoising"))

    def test_matching_pairs_do_not_raise(self):
        train_jobs.check_task_matches_model(_tunet_request())
        train_jobs.check_task_matches_model(_denoiser_request())


class TestCheckResumeCompatible:
    def test_model_family_mismatch_raises(self):
        parent = {"model_family": "dlsia_denoiser", "task": "segmentation", "classes": []}
        with pytest.raises(ValueError, match="Cannot continue fine-tuning a dlsia_denoiser run"):
            train_jobs.check_resume_compatible(parent, _tunet_request())

    def test_task_mismatch_raises(self):
        parent = {"model_family": "dlsia_tunet", "task": "denoising", "classes": []}
        with pytest.raises(ValueError, match="Cannot continue fine-tuning a 'denoising'-task run"):
            train_jobs.check_resume_compatible(parent, _tunet_request())

    def test_denoiser_architecture_mismatch_raises(self):
        parent = {
            "model_family": "dlsia_denoiser", "task": "denoising",
            "model_config": {"architecture": "tunet"},
        }
        with pytest.raises(ValueError, match="Cannot continue fine-tuning a 'tunet' denoiser"):
            train_jobs.check_resume_compatible(parent, _denoiser_request())

    def test_denoiser_matching_architecture_skips_class_check(self):
        parent = {
            "model_family": "dlsia_denoiser", "task": "denoising",
            "model_config": {"architecture": "cnn_ae"},
        }
        train_jobs.check_resume_compatible(parent, _denoiser_request())  # no raise

    def test_denoiser_defaults_missing_architecture_to_tunet(self):
        parent = {"model_family": "dlsia_denoiser", "task": "denoising"}
        with pytest.raises(ValueError, match="Cannot continue fine-tuning a 'tunet' denoiser"):
            train_jobs.check_resume_compatible(parent, _denoiser_request())

    def test_class_list_changed_raises(self):
        parent = {
            "model_family": "dlsia_tunet", "task": "segmentation",
            "classes": [{"label": "A"}, {"label": "C"}],
        }
        with pytest.raises(ValueError, match="classes changed since that run was trained"):
            train_jobs.check_resume_compatible(parent, _tunet_request())

    def test_class_list_case_and_whitespace_insensitive_match(self):
        parent = {
            "model_family": "dlsia_tunet", "task": "segmentation",
            "classes": [{"label": " a "}, {"label": "B"}],
        }
        train_jobs.check_resume_compatible(parent, _tunet_request())  # no raise

    def test_missing_task_defaults_to_segmentation(self):
        parent = {"model_family": "dlsia_tunet", "classes": [{"label": "a"}, {"label": "b"}]}
        train_jobs.check_resume_compatible(parent, _tunet_request())  # no raise


class TestApplyParentArchitecture:
    def test_tunet_inherits_geometry_and_topology(self):
        request = _tunet_request()
        parent = {
            "hyperparams": {"image_size": 128, "tiling": True, "depth": 5, "base_channels": 8, "growth_rate": 1.5},
        }
        train_jobs._apply_parent_architecture(parent, request)
        hp = request.model.hyperparams
        assert hp.image_size == 128
        assert hp.tiling is True
        assert hp.depth == 5
        assert hp.base_channels == 8
        assert hp.growth_rate == 1.5

    def test_tunet_inherits_denoise_settings(self):
        request = _tunet_request()
        parent = {"hyperparams": {}, "denoise": {"method": "gaussian", "strength": 0.5}}
        train_jobs._apply_parent_architecture(parent, request)
        assert request.denoise is not None
        assert request.denoise.method == "gaussian"

    def test_tunet_clears_denoise_when_parent_had_none(self):
        from schemas import DenoiseTrainOpts
        request = _tunet_request(denoise=DenoiseTrainOpts(method="gaussian", strength=0.5))
        train_jobs._apply_parent_architecture({"hyperparams": {}}, request)
        assert request.denoise is None

    def test_denoiser_cnn_ae_forces_scheme_and_compression(self):
        request = _denoiser_request()
        request.model.architecture = "tunet"  # caller sent something else
        parent = {
            "hyperparams": {},
            "model_config": {"architecture": "cnn_ae", "ae_compression": 8},
        }
        train_jobs._apply_parent_architecture(parent, request)
        assert request.model.architecture == "cnn_ae"
        assert request.model.training_scheme == "ae"
        assert request.model.ae_compression == 8

    def test_denoiser_defaults_missing_parent_architecture_to_tunet(self):
        request = _denoiser_request()
        train_jobs._apply_parent_architecture({"hyperparams": {}, "model_config": {}}, request)
        assert request.model.architecture == "tunet"


class TestSourceKeys:
    def test_tiled_and_local_formats(self):
        request = _tunet_request(
            sources=[
                ExportSourceItem(kind="tiled", source="a/b", server_uri="http://x:1", slices={}),
                ExportSourceItem(kind="local", source="c/d", slices={}),
            ],
        )
        assert train_jobs._source_keys(request) == ["tiled:http://x:1:a/b", "local:c/d"]


class TestRunTrainJobGuards:
    def test_reports_error_when_ml_lock_already_held(self):
        jid = export_jobs.new_job("x")
        assert train_common.ML_LOCK.acquire(blocking=False)
        try:
            train_jobs.run_train_job(jid, _tunet_request(), "run-1")
        finally:
            train_common.ML_LOCK.release()
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "already running" in job["error"]

    def test_task_model_mismatch_reports_error_not_a_crash(self):
        jid = export_jobs.new_job("x")
        # task defaults to "segmentation" but the model is a denoiser -> caught
        # by check_task_matches_model inside run_train_job's try block.
        request = _tunet_request()
        request.model = DlsiaDenoiserConfig(architecture="cnn_ae", training_scheme="ae")
        train_jobs.run_train_job(jid, request, "run-2")
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert not train_common.ML_LOCK.locked()
