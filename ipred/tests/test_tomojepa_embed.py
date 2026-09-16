"""Mark25 / TomoJEPA dense embedding encode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipred import tomojepa_embed

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "tomojepa25.pth"
WEIGHTS11 = Path(__file__).resolve().parents[1] / "models" / "tomojepa11.pth"
needs_weights = pytest.mark.skipif(
    not WEIGHTS.is_file() or not tomojepa_embed.torch_available(),
    reason="tomojepa25.pth or torch/timm not available",
)
needs_mark11 = pytest.mark.skipif(
    not WEIGHTS11.is_file() or not tomojepa_embed.torch_available(),
    reason="tomojepa11.pth or torch/timm not available",
)


def test_resolve_weights_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOMOJEPA_WEIGHTS", raising=False)
    resolved = tomojepa_embed.resolve_weights_path(None)
    if WEIGHTS.is_file():
        assert resolved is not None
        assert resolved.name == "tomojepa25.pth"
    else:
        assert resolved is None


def test_resolve_weights_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "custom.pth"
    fake.write_bytes(b"x")
    monkeypatch.setenv("TOMOJEPA_WEIGHTS", str(fake))
    assert tomojepa_embed.resolve_weights_path(None) == fake.resolve()


def test_to_minus_one_one_linear() -> None:
    x = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    y = tomojepa_embed.to_minus_one_one(x)
    np.testing.assert_allclose(y, [[-1.0, 0.0, 1.0]], atol=1e-6)


@needs_weights
def test_encode_feeds_model_in_minus_one_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """All Mark25 paths must pass intensities in [-1, 1] into the network."""
    captured: list[float] = []

    real_cached = tomojepa_embed._cached_model

    def _wrap(path_str: str):
        net, device = real_cached(path_str)
        orig_forward = net.forward

        def forward(x):  # noqa: ANN001
            t = x.detach().cpu().numpy()
            captured.append(float(t.min()))
            captured.append(float(t.max()))
            return orig_forward(x)

        net.forward = forward  # type: ignore[method-assign]
        return net, device

    monkeypatch.setattr(tomojepa_embed, "_cached_model", _wrap)
    arr = np.linspace(10, 200, 64 * 48, dtype=np.float32).reshape(64, 48)
    tomojepa_embed.encode_dense_embeddings(
        arr, weights_path=str(WEIGHTS), input_size=128
    )
    assert captured, "model forward was not called"
    assert min(captured) >= -1.0 - 1e-5
    assert max(captured) <= 1.0 + 1e-5
    # After min-max + linear map, extremes should reach near ±1.
    assert min(captured) < -0.9
    assert max(captured) > 0.9


@needs_weights
def test_load_state_dict_strict() -> None:
    net = tomojepa_embed.build_encoder()
    state = tomojepa_embed.load_checkpoint_state(WEIGHTS)
    net.load_state_dict(state, strict=True)


@needs_weights
def test_encode_dense_shape_512() -> None:
    arr = np.linspace(0, 1, 400 * 300, dtype=np.float32).reshape(400, 300)
    emb, orig_hw, reshaped_hw = tomojepa_embed.encode_dense_embeddings(
        arr, weights_path=str(WEIGHTS), input_size=512
    )
    assert orig_hw == (400, 300)
    assert reshaped_hw == (512, 512)
    assert emb.shape == (32, 32, 64)
    assert emb.dtype == np.float32


@needs_weights
def test_encode_dense_honors_input_size() -> None:
    arr = np.random.default_rng(0).random((128, 96), dtype=np.float32)
    emb, _, reshaped = tomojepa_embed.encode_dense_embeddings(
        arr, weights_path=str(WEIGHTS), input_size=256
    )
    assert reshaped == (256, 256)
    assert emb.shape == (16, 16, 64)


@needs_weights
def test_encode_native_no_resize() -> None:
    arr = np.random.default_rng(1).random((100, 80), dtype=np.float32)
    emb, orig_hw, reshaped = tomojepa_embed.encode_dense_embeddings(
        arr, weights_path=str(WEIGHTS), resize=False
    )
    assert orig_hw == (100, 80)
    # padded to multiple of 16
    assert reshaped == (112, 80)
    assert emb.shape == (7, 5, 64)


@needs_mark11
def test_encode_mark11_dense_is_256d() -> None:
    """Mark11 dense projector is 256-D (Mark25 is 64-D)."""
    tomojepa_embed._cached_model.cache_clear()
    arr = np.linspace(0, 1, 64 * 48, dtype=np.float32).reshape(64, 48)
    emb, _, _ = tomojepa_embed.encode_dense_embeddings(
        arr, weights_path=str(WEIGHTS11), input_size=128
    )
    assert emb.shape == (8, 8, 256)
