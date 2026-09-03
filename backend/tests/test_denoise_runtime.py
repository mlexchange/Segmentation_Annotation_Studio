"""Real (no mocks) build/save/load/forward round trip for denoise_runtime.py
— mirrors test_train_e2e_real_ml.py's pattern for the sibling dlsia_runtime.py
segmentation family.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("dlsia")

import denoise_runtime  # noqa: E402

IMAGE_SIZE = 64


def _model():
    return denoise_runtime.build_model(
        image_size=IMAGE_SIZE, depth=2, base_channels=4, growth_rate=1.2, device="cpu",
    )


class TestBuildModel:
    def test_forward_pass_produces_single_channel_output(self):
        model = _model()
        model.eval()
        forward_fn = denoise_runtime.make_forward_fn(model)
        to_tensor_fn = denoise_runtime.make_to_tensor_fn()
        gray = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        batch = to_tensor_fn(gray).unsqueeze(0)
        with torch.no_grad():
            out = forward_fn(batch)
        assert out.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)

    def test_final_activation_is_pinned_to_none(self):
        # Checked directly on the model rather than by hoping an untrained
        # (effectively random) network happens to produce an out-of-[0,1]
        # value — that was flaky (not guaranteed within a handful of seeds).
        # final_activation=None must hold: a squashed output would mean a
        # future dlsia default silently added an activation the denoiser
        # regression head must never have.
        model = _model()
        assert model.final_activation is None


class TestSaveLoadRoundTrip:
    def test_reloaded_model_reproduces_output(self):
        model = _model()
        model.eval()
        forward_fn = denoise_runtime.make_forward_fn(model)
        to_tensor_fn = denoise_runtime.make_to_tensor_fn()
        gray = np.random.default_rng(0).integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        batch = to_tensor_fn(gray).unsqueeze(0)

        with torch.no_grad():
            original_out = forward_fn(batch)

        state = denoise_runtime.network_dict(model)
        assert "topo_dict" in state and "state_dict" in state

        reloaded = denoise_runtime.load_model(state, "cpu")
        reloaded.eval()
        reloaded_forward = denoise_runtime.make_forward_fn(reloaded)
        with torch.no_grad():
            reloaded_out = reloaded_forward(batch)

        torch.testing.assert_close(original_out, reloaded_out)


class TestSetTrainMode:
    def test_toggles_module_training_flag(self):
        model = _model()
        set_train_mode = denoise_runtime.make_set_train_mode_fn(model)
        set_train_mode(False)
        assert model.training is False
        set_train_mode(True)
        assert model.training is True


class TestToTensorFn:
    def test_scales_uint8_to_unit_range(self):
        to_tensor_fn = denoise_runtime.make_to_tensor_fn()
        gray = np.full((8, 8), 255, dtype=np.uint8)
        t = to_tensor_fn(gray)
        assert t.shape == (1, 8, 8)
        assert torch.allclose(t, torch.ones_like(t))

    def test_accepts_single_channel_hwc(self):
        to_tensor_fn = denoise_runtime.make_to_tensor_fn()
        gray = np.full((8, 8, 1), 255, dtype=np.uint8)
        t = to_tensor_fn(gray)
        assert t.shape == (1, 8, 8)
