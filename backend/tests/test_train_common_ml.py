"""Real (no mocks) torch round trips for train_common.py's model-construction
and training-loop functions — build_family, compute_miou, run_training_loop,
_evaluate. Kept in a separate file from test_train_common.py so that file can
stay import-safe/runnable without torch (its own stated design), while this
one skips outright when the `ml` extra isn't installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_common  # noqa: E402
from schemas import DlsiaDenoiserConfig, DlsiaTunetConfig  # noqa: E402

IMAGE_SIZE = 64


def _tunet_config(**overrides):
    hp = {"depth": 2, "base_channels": 4, "growth_rate": 1.2, "image_size": IMAGE_SIZE, "batch_size": 2}
    hp.update(overrides)
    return DlsiaTunetConfig(hyperparams=hp)


def _denoiser_config(architecture="tunet", **overrides):
    hp = {"depth": 2, "base_channels": 4, "growth_rate": 1.2, "image_size": IMAGE_SIZE, "batch_size": 2}
    hp.update(overrides)
    scheme = "ae" if architecture == "cnn_ae" else "n2n"
    return DlsiaDenoiserConfig(architecture=architecture, hyperparams=hp, training_scheme=scheme)


class TestBuildFamilyTunet:
    def test_builds_from_scratch(self):
        built = train_common.build_family(_tunet_config(), n_classes=3, device="cpu", log_cb=lambda m: None)
        assert built.model_config_snapshot == {"depth": 2, "base_channels": 4, "growth_rate": 1.2}
        assert len(built.trainable_params) > 0

    def test_dlsia_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(train_common, "dlsia_available", lambda: False)
        with pytest.raises(RuntimeError, match="dlsia is not installed"):
            train_common.build_family(_tunet_config(), n_classes=3, device="cpu", log_cb=lambda m: None)

    def test_resume_uses_saved_topology_not_the_request(self):
        built = train_common.build_family(_tunet_config(), n_classes=3, device="cpu", log_cb=lambda m: None)
        state = built.adapter_state_fn()
        resumed = train_common.build_family(
            _tunet_config(depth=6), n_classes=3, device="cpu", log_cb=lambda m: None, init_state=state,
        )
        # depth=6 in the new request must be ignored; the saved topology wins.
        assert resumed.model_config_snapshot["depth"] == 2

    def test_forward_produces_correct_class_channels(self):
        built = train_common.build_family(_tunet_config(), n_classes=3, device="cpu", log_cb=lambda m: None)
        batch = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        out = built.forward_fn(batch)
        assert out.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


class TestBuildFamilyDenoiserTunet:
    def test_builds_from_scratch(self):
        built = train_common.build_family(_denoiser_config("tunet"), n_classes=2, device="cpu", log_cb=lambda m: None)
        assert built.model_config_snapshot["architecture"] == "tunet"

    def test_dlsia_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(train_common, "dlsia_available", lambda: False)
        with pytest.raises(RuntimeError, match="dlsia is not installed"):
            train_common.build_family(_denoiser_config("tunet"), n_classes=2, device="cpu", log_cb=lambda m: None)

    def test_resume_uses_saved_topology(self):
        built = train_common.build_family(_denoiser_config("tunet"), n_classes=2, device="cpu", log_cb=lambda m: None)
        state = built.adapter_state_fn()
        resumed = train_common.build_family(
            _denoiser_config("tunet", depth=6), n_classes=2, device="cpu", log_cb=lambda m: None, init_state=state,
        )
        assert resumed.model_config_snapshot["depth"] == 2


class TestBuildFamilyDenoiserCnnAe:
    def test_builds_from_scratch_no_dlsia_gate(self, monkeypatch):
        # cnn_ae is plain torch — must succeed even when dlsia is "unavailable".
        monkeypatch.setattr(train_common, "dlsia_available", lambda: False)
        built = train_common.build_family(_denoiser_config("cnn_ae"), n_classes=2, device="cpu", log_cb=lambda m: None)
        assert built.model_config_snapshot["architecture"] == "cnn_ae"
        assert built.model_config_snapshot["latent_channels"] > 0

    def test_resume_uses_saved_topology(self):
        built = train_common.build_family(_denoiser_config("cnn_ae"), n_classes=2, device="cpu", log_cb=lambda m: None)
        state = built.adapter_state_fn()
        resumed = train_common.build_family(
            _denoiser_config("cnn_ae", depth=6), n_classes=2, device="cpu", log_cb=lambda m: None, init_state=state,
        )
        assert resumed.model_config_snapshot["depth"] == 2


class TestBuildFamilyUnknownConfig:
    def test_unrecognized_config_type_raises_value_error(self):
        class NotAModelConfig:
            hyperparams = None
            model_family = "bogus"

        with pytest.raises(ValueError, match="Unknown model family"):
            train_common.build_family(NotAModelConfig(), n_classes=2, device="cpu", log_cb=lambda m: None)


class TestComputeMiou:
    def test_perfect_match_gives_miou_one(self):
        pred = torch.tensor([0, 0, 1, 1])
        target = torch.tensor([0, 0, 1, 1])
        assert train_common.compute_miou(pred, target, n_classes=2) == pytest.approx(1.0)

    def test_no_overlap_gives_miou_zero(self):
        pred = torch.tensor([0, 0, 0, 0])
        target = torch.tensor([1, 1, 1, 1])
        assert train_common.compute_miou(pred, target, n_classes=2) == pytest.approx(0.0)

    def test_ignore_index_pixels_are_excluded(self):
        pred = torch.tensor([0, 0, 1, 1])
        target = torch.tensor([0, 0, 255, 255])
        # Only the first two pixels count; both match -> class 0 IoU 1.0, class 1 has no valid pixels (skipped).
        assert train_common.compute_miou(pred, target, n_classes=2, ignore_index=255) == pytest.approx(1.0)

    def test_all_ignored_returns_zero(self):
        pred = torch.tensor([0, 1])
        target = torch.tensor([255, 255])
        assert train_common.compute_miou(pred, target, n_classes=2, ignore_index=255) == 0.0


def _synthetic_pairs(n, n_classes=2, size=IMAGE_SIZE, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (
            rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8),
            rng.integers(0, n_classes, size=(size, size), dtype=np.uint8),
        )
        for _ in range(n)
    ]


class TestRunTrainingLoop:
    def _built(self, n_classes=2):
        return train_common.build_family(_tunet_config(), n_classes=n_classes, device="cpu", log_cb=lambda m: None)

    def test_no_training_data_raises(self):
        built = self._built()
        with pytest.raises(ValueError, match="No training data"):
            train_common.run_training_loop(
                train_pairs=[], val_pairs=[], image_size=IMAGE_SIZE, n_classes=2, epochs=1,
                batch_size=2, seed=0, flip_augment=False, to_tensor_fn=built.to_tensor_fn,
                forward_fn=built.forward_fn, trainable_params=built.trainable_params, lr=1e-3, device="cpu",
                set_train_mode=built.set_train_mode,
            )

    def test_completes_epochs_and_reports_loss(self):
        built = self._built()
        result = train_common.run_training_loop(
            train_pairs=_synthetic_pairs(4), val_pairs=[], image_size=IMAGE_SIZE, n_classes=2,
            epochs=2, batch_size=2, seed=0, flip_augment=True, to_tensor_fn=built.to_tensor_fn,
            forward_fn=built.forward_fn, trainable_params=built.trainable_params, lr=1e-3, device="cpu",
            set_train_mode=built.set_train_mode,
        )
        assert result["epochs_completed"] == 2
        assert result["cancelled"] is False
        assert result["final_val_loss"] is None
        assert result["val_miou"] is None

    def test_validation_pairs_produce_val_loss_and_miou(self):
        built = self._built()
        result = train_common.run_training_loop(
            train_pairs=_synthetic_pairs(4), val_pairs=_synthetic_pairs(2, seed=1),
            image_size=IMAGE_SIZE, n_classes=2, epochs=1, batch_size=2, seed=0, flip_augment=False,
            to_tensor_fn=built.to_tensor_fn, forward_fn=built.forward_fn,
            trainable_params=built.trainable_params, lr=1e-3, device="cpu",
            set_train_mode=built.set_train_mode,
        )
        assert result["final_val_loss"] is not None
        assert result["val_miou"] is not None

    def test_on_batch_cancel_stops_immediately(self):
        built = self._built()
        result = train_common.run_training_loop(
            train_pairs=_synthetic_pairs(8), val_pairs=[], image_size=IMAGE_SIZE, n_classes=2,
            epochs=3, batch_size=2, seed=0, flip_augment=False, to_tensor_fn=built.to_tensor_fn,
            forward_fn=built.forward_fn, trainable_params=built.trainable_params, lr=1e-3, device="cpu",
            on_batch=lambda: True, set_train_mode=built.set_train_mode,
        )
        assert result["cancelled"] is True
        assert result["epochs_completed"] == 1

    def test_on_epoch_cancel_stops_after_that_epoch(self):
        built = self._built()
        epochs_seen = []
        result = train_common.run_training_loop(
            train_pairs=_synthetic_pairs(4), val_pairs=[], image_size=IMAGE_SIZE, n_classes=2,
            epochs=5, batch_size=2, seed=0, flip_augment=False, to_tensor_fn=built.to_tensor_fn,
            forward_fn=built.forward_fn, trainable_params=built.trainable_params, lr=1e-3, device="cpu",
            on_epoch=lambda epoch, tl, vl, miou: epochs_seen.append(epoch) or epoch >= 2,
            set_train_mode=built.set_train_mode,
        )
        assert result["cancelled"] is True
        assert result["epochs_completed"] == 2
        assert epochs_seen == [1, 2]

    def test_custom_optimizer_factory_is_used(self):
        built = self._built()
        seen = []

        def make_optimizer(params, lr):
            seen.append(lr)
            return torch.optim.SGD(params, lr=lr)

        train_common.run_training_loop(
            train_pairs=_synthetic_pairs(2), val_pairs=[], image_size=IMAGE_SIZE, n_classes=2,
            epochs=1, batch_size=2, seed=0, flip_augment=False, to_tensor_fn=built.to_tensor_fn,
            forward_fn=built.forward_fn, trainable_params=built.trainable_params, lr=5e-4, device="cpu",
            make_optimizer_fn=make_optimizer, set_train_mode=built.set_train_mode,
        )
        assert seen == [5e-4]
