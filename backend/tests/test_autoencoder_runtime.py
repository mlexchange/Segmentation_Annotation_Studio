"""Contracts the convolutional autoencoder must satisfy to be usable at all.

Three of these are load-bearing rather than cosmetic:

* **Shape preservation** — ``tiling._blend_tiled_forward`` accumulates each
  window with ``canvas[:, y:y+w, x:x+w] += out[i] * weight``, so an output whose
  spatial size differs from the input window raises inside tiled inference.
* **The bottleneck actually compresses** — if it didn't, the network could pass
  its input straight through and "denoising" by reconstruction would be a no-op.
  This is the structural half of the guarantee; ``test_denoiser_ae.py`` covers
  the behavioural half.
* **save -> load reproduces identical output** — unlike dlsia's TUNet there is
  no vendor ``save_network_parameters``/``topology_dict`` helper here, so both
  sides of that contract are hand-written and are the likeliest thing to break
  silently. A run whose weights reload into a differently-shaped network is
  worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import autoencoder_runtime as ar  # noqa: E402


# ---------------------------------------------------------------------------
# compression -> latent channels
# ---------------------------------------------------------------------------

def test_latent_channels_follow_the_documented_ratio() -> None:
    """``L = 4**depth / compression`` — independent of patch size, which is what
    lets one "compression" number mean the same thing at 128px and 512px."""
    assert ar.latent_channels_for(4, 16) == 16
    assert ar.latent_channels_for(4, 4) == 64
    assert ar.latent_channels_for(3, 8) == 8


def test_more_compression_means_fewer_latent_channels() -> None:
    channels = [ar.latent_channels_for(4, c) for c in (4, 8, 16, 32, 64)]
    assert channels == sorted(channels, reverse=True)
    assert len(set(channels)) == len(channels), "each step should actually change the bottleneck"


def test_latent_channels_never_collapses_to_zero() -> None:
    """A zero-channel bottleneck would sever the network entirely."""
    for depth in (1, 2, 3, 4, 5, 6):
        for compression in (4, 32, 64):
            assert ar.latent_channels_for(depth, compression) >= 1


def test_degenerate_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="depth"):
        ar.latent_channels_for(0, 16)
    with pytest.raises(ValueError, match="compression"):
        ar.latent_channels_for(4, 0)


# ---------------------------------------------------------------------------
# The network's shape contract
# ---------------------------------------------------------------------------

def _model(image_size: int = 128, depth: int = 4, base: int = 8, compression: int = 16):
    return ar.build_model(
        image_size=image_size, depth=depth, base_channels=base,
        compression=compression, device="cpu",
    )


@pytest.mark.parametrize("image_size,depth", [(64, 3), (128, 4), (256, 4)])
def test_output_matches_input_size_exactly(image_size: int, depth: int) -> None:
    """The tiled-blend contract. A mismatch here breaks inference, not training."""
    model = _model(image_size, depth)
    model.eval()

    with torch.no_grad():
        out = ar.make_forward_fn(model)(torch.rand(1, 1, image_size, image_size))

    assert tuple(out.shape) == (1, 1, image_size, image_size)


def test_a_batch_of_windows_is_handled() -> None:
    """``_blend_tiled_forward`` forwards INFER_TILE_BATCH windows at a time, so
    the model must accept an arbitrary leading batch dim."""
    model = _model()
    model.eval()

    with torch.no_grad():
        out = ar.make_forward_fn(model)(torch.rand(5, 1, 128, 128))

    assert tuple(out.shape) == (5, 1, 128, 128)


def test_output_is_single_channel() -> None:
    model = _model()
    model.eval()
    with torch.no_grad():
        out = ar.make_forward_fn(model)(torch.rand(2, 1, 128, 128))
    assert out.shape[1] == 1


def test_an_image_size_the_decoder_cannot_reconstruct_is_rejected() -> None:
    """Caught at construction rather than surfacing as a tensor-size error deep
    inside the tiled blend."""
    with pytest.raises(ValueError, match="divisible"):
        _model(image_size=100, depth=4)


# ---------------------------------------------------------------------------
# The bottleneck is real
# ---------------------------------------------------------------------------

def test_the_bottleneck_holds_fewer_values_than_the_input() -> None:
    """The structural guarantee behind reconstruction-as-denoising: the input
    cannot survive the round trip unchanged, so the network has to choose what
    to keep — and noise is the part that doesn't fit."""
    model = _model(image_size=128, depth=4, compression=16)
    model.eval()

    with torch.no_grad():
        latent = model.encoder(torch.rand(1, 1, 128, 128))

    assert latent.numel() < 128 * 128
    assert (128 * 128) / latent.numel() == pytest.approx(16, rel=0.01)


def test_requesting_more_compression_narrows_the_bottleneck() -> None:
    sizes = []
    for compression in (4, 16, 64):
        model = _model(compression=compression)
        model.eval()
        with torch.no_grad():
            sizes.append(model.encoder(torch.rand(1, 1, 128, 128)).numel())
    assert sizes == sorted(sizes, reverse=True)


def test_no_skip_connections_exist() -> None:
    """The whole design rests on this. A skip path would make ``f(x) = x``
    learnable and turn pure reconstruction into a silent no-op — the exact
    failure the previous add-synthetic-noise scheme existed to dodge.

    Asserted structurally: the module graph is exactly encoder -> decoder, with
    the forward pass composing the two and nothing else.
    """
    model = _model()

    assert set(dict(model.named_children())) == {"encoder", "decoder"}


# ---------------------------------------------------------------------------
# Checkpoint round trip
# ---------------------------------------------------------------------------

def test_network_dict_uses_the_key_names_the_rest_of_the_codebase_expects() -> None:
    """``train_common.build_family``'s warm-start reads
    ``init_state.get("topo_dict", {})``, so these names are a contract, not an
    implementation detail."""
    state = ar.network_dict(_model())

    assert set(state) == {"topo_dict", "state_dict"}
    assert {"image_size", "depth", "base_channels", "compression", "latent_channels"} <= set(
        state["topo_dict"]
    )


def test_save_then_load_reproduces_identical_output() -> None:
    """The single most important test in this file. Weights that reload into a
    differently-shaped network make the run worthless, and nothing else would
    notice — inference would just return something plausible and wrong."""
    model = _model(image_size=64, depth=3, base=8, compression=8)
    model.eval()
    sample = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        before = ar.make_forward_fn(model)(sample)

    reloaded = ar.load_model(ar.network_dict(model), "cpu")
    reloaded.eval()
    with torch.no_grad():
        after = ar.make_forward_fn(reloaded)(sample)

    assert torch.equal(before, after)


def test_a_reloaded_model_keeps_its_topology() -> None:
    model = _model(image_size=64, depth=3, base=16, compression=32)

    reloaded = ar.load_model(ar.network_dict(model), "cpu")

    assert reloaded.topo == model.topo


# ---------------------------------------------------------------------------
# Tensor conversion + train-mode plumbing (same template as denoise_runtime)
# ---------------------------------------------------------------------------

def test_to_tensor_matches_the_tunet_runtime_convention() -> None:
    """Both architectures consume the same uint8 grayscale from
    ``denoise_train._slice_to_gray_uint8``, so a run trained under one can be
    compared against the other on equal footing."""
    import denoise_runtime

    gray = (np.arange(16 * 16, dtype=np.uint8).reshape(16, 16))

    mine = ar.make_to_tensor_fn()(gray)
    theirs = denoise_runtime.make_to_tensor_fn()(gray)

    assert tuple(mine.shape) == (1, 16, 16)
    assert torch.equal(mine, theirs)


def test_to_tensor_accepts_a_trailing_singleton_channel() -> None:
    gray = np.zeros((8, 8, 1), dtype=np.uint8)
    assert tuple(ar.make_to_tensor_fn()(gray).shape) == (1, 8, 8)


def test_set_train_mode_toggles_batchnorm() -> None:
    """BatchNorm running stats must not update while validating."""
    model = _model()
    toggle = ar.make_set_train_mode_fn(model)

    toggle(False)
    assert not model.training
    toggle(True)
    assert model.training
