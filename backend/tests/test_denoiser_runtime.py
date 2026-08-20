"""Tests for denoise_runtime — the third model-family runtime (self-supervised
Noise2Noise/Noise2Void denoiser), mirroring dlsia_runtime.py's template but
wired for single-channel regression instead of multi-class segmentation.

Uses the real dlsia TUNet (tiny topology) rather than mocking it, since the
whole point is to prove the actual channel counts and activation wiring —
see test_train_common_build_family.py for the mocked-runtime dispatch tests.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("dlsia")

import denoise_runtime  # noqa: E402


def _tiny_model():
    return denoise_runtime.build_model(image_size=64, depth=2, base_channels=4, growth_rate=1.5, device="cpu")


def test_build_model_first_conv_layer_has_in_channels_1() -> None:
    model = _tiny_model()
    first_conv = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    assert first_conv.in_channels == 1


def test_build_model_out_channels_is_1() -> None:
    model = _tiny_model()
    assert model.out_channels == 1


def test_build_model_has_no_final_activation() -> None:
    """A regression head must not have a softmax/sigmoid squashing its output
    range — confirm the network was built with final_activation disabled."""
    model = _tiny_model()
    assert model.final_activation is None


def test_forward_pass_on_a_single_channel_tensor_returns_matching_shape() -> None:
    model = _tiny_model()
    forward = denoise_runtime.make_forward_fn(model)

    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        y = forward(x)

    assert y.shape == (1, 1, 64, 64)


def test_forward_output_is_not_squashed_into_the_unit_interval() -> None:
    """Regression, not classification: the raw output should not come back
    clamped into [0, 1] the way a lingering softmax/sigmoid would clamp it.
    Seeded for a deterministic (untrained, random-init) output range."""
    torch.manual_seed(0)
    model = _tiny_model()
    forward = denoise_runtime.make_forward_fn(model)

    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        y = forward(x)

    assert not torch.all((y >= 0) & (y <= 1))


def test_network_dict_round_trips_through_load_model() -> None:
    model = _tiny_model()
    state = denoise_runtime.network_dict(model)
    reloaded = denoise_runtime.load_model(state, device="cpu")

    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        original_out = model(x)
        reloaded_out = reloaded(x)

    assert torch.allclose(original_out, reloaded_out)
    assert reloaded.in_channels == 1
    assert reloaded.out_channels == 1


def test_make_set_train_mode_fn_toggles_training_mode() -> None:
    model = _tiny_model()
    set_train_mode = denoise_runtime.make_set_train_mode_fn(model)

    set_train_mode(False)
    assert model.training is False
    set_train_mode(True)
    assert model.training is True


def test_make_to_tensor_fn_scales_uint8_to_unit_range_with_a_channel_dim() -> None:
    import numpy as np

    to_tensor = denoise_runtime.make_to_tensor_fn()
    gray = np.full((64, 64), 255, dtype=np.uint8)

    tensor = to_tensor(gray)

    assert tensor.shape == (1, 64, 64)
    assert torch.allclose(tensor, torch.ones(1, 64, 64))


def test_make_to_tensor_fn_accepts_a_trailing_singleton_channel_dim() -> None:
    import numpy as np

    to_tensor = denoise_runtime.make_to_tensor_fn()
    gray_hw1 = np.full((64, 64, 1), 0, dtype=np.uint8)

    tensor = to_tensor(gray_hw1)

    assert tensor.shape == (1, 64, 64)
    assert torch.allclose(tensor, torch.zeros(1, 64, 64))
