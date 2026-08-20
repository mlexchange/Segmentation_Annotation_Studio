"""The ``"ae"`` scheme: pure self-reconstruction through a bottleneck.

This is the inverse of what the removed ``dae`` scheme required. There, input and
target had to DIFFER (synthetic noise added to the input), because TUNet's skip
connections make ``f(x) = x`` trivially learnable and ``target == input`` would
have converged to copying the input while still reporting a falling loss. Here
input and target are deliberately IDENTICAL — which is only safe because the
``cnn_ae`` architecture has no skip connections and cannot pass its input
through unchanged.

So the two things worth testing are the pairing (input == target, no mask) and,
more importantly, that a real short training run genuinely denoises rather than
learning the identity. The latter has to be actual training — a mock cannot tell
you whether the objective is sound.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import autoencoder_runtime as ar  # noqa: E402
import denoise_train as dt  # noqa: E402


def _to_tensor(arr: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0) / 255.0


def _patch(size: int = 32, value: int = 120) -> np.ndarray:
    arr = np.full((size, size), value, dtype=np.uint8)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 200
    return arr


def _ae_batch(patch: np.ndarray, seed: int = 0):
    return dt._prep_batch(
        [(patch, patch)],
        [0],
        image_size=patch.shape[0],
        scheme="ae",
        to_tensor_fn=_to_tensor,
        rng=np.random.default_rng(seed),
        flip_augment=False,
        mask_fraction=dt.DEFAULT_N2V_MASK_FRACTION,
        neighbourhood=dt.DEFAULT_N2V_NEIGHBOURHOOD,
    )


# ---------------------------------------------------------------------------
# The pairing
# ---------------------------------------------------------------------------

def test_input_equals_target_and_there_is_no_mask() -> None:
    """Pure reconstruction. The opposite of dae's requirement, and safe only
    because cnn_ae has no skip connections (the schema enforces that pairing)."""
    imgs, tgts, mask = _ae_batch(_patch())

    assert mask is None, "ae is unmasked — plain MSE over all pixels"
    assert torch.equal(imgs, tgts)


def test_a_flip_is_applied_to_input_and_target_together() -> None:
    """A self-paired item's target must be the SAME flip its input took —
    otherwise the model would be asked to un-mirror the image, which is not
    denoising. This is why 'ae' must not fall through to the n2n branch, which
    flips the target independently.
    """
    patch = _patch()
    # Deterministic flip: seed whose first random() < 0.5.
    for seed in range(20):
        imgs, tgts, _ = dt._prep_batch(
            [(patch, patch)], [0], image_size=patch.shape[0], scheme="ae",
            to_tensor_fn=_to_tensor, rng=np.random.default_rng(seed),
            flip_augment=True, mask_fraction=dt.DEFAULT_N2V_MASK_FRACTION,
            neighbourhood=dt.DEFAULT_N2V_NEIGHBOURHOOD,
        )
        assert torch.equal(imgs, tgts), f"input/target diverged under flip at seed {seed}"


def test_asymmetric_patch_survives_the_pairing() -> None:
    """Guards the flip property with content that would expose a mirror."""
    patch = np.zeros((32, 32), dtype=np.uint8)
    patch[:, :8] = 255  # only the left edge is bright

    for seed in range(10):
        imgs, tgts, _ = dt._prep_batch(
            [(patch, patch)], [0], image_size=32, scheme="ae",
            to_tensor_fn=_to_tensor, rng=np.random.default_rng(seed),
            flip_augment=True, mask_fraction=0.015, neighbourhood=5,
        )
        assert torch.equal(imgs, tgts)


# ---------------------------------------------------------------------------
# It genuinely denoises, and does not learn the identity
# ---------------------------------------------------------------------------

def _phantom(size: int = 64):
    """Piecewise-constant discs on a flat field — grains-in-epoxy-like."""
    clean = np.zeros((size, size), dtype=np.float64)
    for cy, cx, r, v in [(20, 20, 11, 0.80), (44, 42, 13, 0.55), (18, 46, 8, 0.35)]:
        y, x = np.ogrid[:size, :size]
        clean[(y - cy) ** 2 + (x - cx) ** 2 < r * r] = v
    return clean + 0.10


def test_training_reduces_error_against_clean_ground_truth() -> None:
    """The property the whole design rests on, measured the only way that
    counts: against ground truth the model never saw.

    Deliberately a real (if tiny) training run. A mocked forward pass would
    prove the plumbing works while saying nothing about whether reconstructing
    a noisy image through a bottleneck removes its noise.
    """
    size = 64
    clean = _phantom(size)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    def noisy_uint8(n: int, sd: float = 0.09):
        return [
            (np.clip(clean + rng.normal(0, sd, clean.shape), 0, 1) * 255).astype(np.uint8)
            for _ in range(n)
        ]

    train = [(p, p) for p in noisy_uint8(48)]
    model = ar.build_model(image_size=size, depth=3, base_channels=16, compression=8, device="cpu")

    result = dt.run_denoise_training_loop(
        train_pairs=train, val_pairs=train[:4], image_size=size,
        training_scheme="ae", epochs=30, batch_size=8, seed=0, flip_augment=False,
        to_tensor_fn=ar.make_to_tensor_fn(), forward_fn=ar.make_forward_fn(model),
        trainable_params=list(model.parameters()), lr=2e-3, device="cpu",
        set_train_mode=ar.make_set_train_mode_fn(model),
    )
    assert result["epochs_completed"] == 30
    assert not result["cancelled"]

    model.eval()
    held_out = noisy_uint8(8, sd=0.09)
    inputs = torch.stack([ar.make_to_tensor_fn()(p) for p in held_out])
    with torch.no_grad():
        out = ar.make_forward_fn(model)(inputs).clamp(0, 1)

    truth = torch.from_numpy(clean).float().unsqueeze(0).unsqueeze(0)
    rmse = lambda t: float(((t - truth) ** 2).mean().sqrt())  # noqa: E731

    assert rmse(out) < rmse(inputs), (
        f"autoencoder did not denoise: input RMSE {rmse(inputs):.4f} -> output {rmse(out):.4f}"
    )


def test_the_trained_model_does_not_just_copy_its_input() -> None:
    """The identity-collapse guard. A network that learned ``f(x) = x`` would
    also show a falling reconstruction loss, so loss alone cannot distinguish
    "denoised" from "did nothing" — the output has to actually differ from the
    input it was given.
    """
    size = 64
    clean = _phantom(size)
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    patches = [
        (np.clip(clean + rng.normal(0, 0.09, clean.shape), 0, 1) * 255).astype(np.uint8)
        for _ in range(32)
    ]
    model = ar.build_model(image_size=size, depth=3, base_channels=16, compression=8, device="cpu")

    dt.run_denoise_training_loop(
        train_pairs=[(p, p) for p in patches], val_pairs=[], image_size=size,
        training_scheme="ae", epochs=20, batch_size=8, seed=0, flip_augment=False,
        to_tensor_fn=ar.make_to_tensor_fn(), forward_fn=ar.make_forward_fn(model),
        trainable_params=list(model.parameters()), lr=2e-3, device="cpu",
        set_train_mode=ar.make_set_train_mode_fn(model),
    )

    model.eval()
    sample = ar.make_to_tensor_fn()(patches[0]).unsqueeze(0)
    with torch.no_grad():
        out = ar.make_forward_fn(model)(sample).clamp(0, 1)

    # Smoother than what it was handed: the noise is gone, not reproduced.
    rough = lambda t: float(t.diff(dim=-1).abs().mean())  # noqa: E731
    assert rough(out) < rough(sample), "output is as rough as the input — it copied rather than denoised"


# ---------------------------------------------------------------------------
# Scheme / architecture pairing is enforced
# ---------------------------------------------------------------------------

def test_pure_reconstruction_is_refused_on_a_skip_connected_network() -> None:
    """The footgun this whole change exists to remove. On TUNet, ``ae`` would
    learn to copy its input and silently remove nothing."""
    import pydantic

    from schemas import DlsiaDenoiserConfig

    with pytest.raises(pydantic.ValidationError, match="cnn_ae"):
        DlsiaDenoiserConfig(training_scheme="ae", architecture="tunet")


@pytest.mark.parametrize("scheme", ["n2n", "n2v"])
def test_the_autoencoder_is_refused_with_the_other_schemes(scheme: str) -> None:
    """Not unsound in principle, just untested — refused rather than shipped."""
    import pydantic

    from schemas import DlsiaDenoiserConfig

    with pytest.raises(pydantic.ValidationError):
        DlsiaDenoiserConfig(training_scheme=scheme, architecture="cnn_ae")


def test_the_removed_dae_scheme_is_rejected() -> None:
    import pydantic

    from schemas import DlsiaDenoiserConfig

    with pytest.raises(pydantic.ValidationError):
        DlsiaDenoiserConfig(training_scheme="dae")


def test_the_training_loop_accepts_ae_and_rejects_nonsense() -> None:
    patch = _patch()
    with pytest.raises(ValueError, match="scheme"):
        dt.run_denoise_training_loop(
            train_pairs=[(patch, patch)], val_pairs=[], image_size=32,
            training_scheme="dae", epochs=1, batch_size=1, seed=0, flip_augment=False,
            to_tensor_fn=_to_tensor, forward_fn=lambda b: b,
            trainable_params=[], lr=1e-3, device="cpu",
        )


# ---------------------------------------------------------------------------
# Architecture dispatch + backward compatibility
# ---------------------------------------------------------------------------

def test_a_run_with_no_architecture_key_loads_as_tunet() -> None:
    """Every run already on disk predates this field. Defaulting to TUNet is
    what keeps them loadable — the alternative is silently bricking them."""
    import denoise_runtime
    import train_common

    assert train_common.denoiser_runtime_for({"model_config": {}}) is denoise_runtime
    assert train_common.denoiser_runtime_for({}) is denoise_runtime
    assert train_common.denoiser_needs_dlsia({"model_config": {}}) is True


def test_an_autoencoder_run_dispatches_to_the_autoencoder_runtime() -> None:
    import train_common

    config = {"model_config": {"architecture": "cnn_ae"}}
    assert train_common.denoiser_runtime_for(config) is ar
    # Plain torch — demanding dlsia here would refuse for the wrong reason.
    assert train_common.denoiser_needs_dlsia(config) is False


def test_an_unknown_architecture_is_reported_not_guessed() -> None:
    import train_common

    with pytest.raises(ValueError, match="Unknown denoiser architecture"):
        train_common.denoiser_runtime_for({"model_config": {"architecture": "from-the-future"}})


def test_build_family_builds_the_autoencoder_and_records_it() -> None:
    import train_common
    from schemas import DlsiaDenoiserConfig

    cfg = DlsiaDenoiserConfig(training_scheme="ae", architecture="cnn_ae", ae_compression=32)
    cfg.hyperparams.depth = 3
    cfg.hyperparams.base_channels = 8
    cfg.hyperparams.image_size = 64

    built = train_common.build_family(cfg, 0, "cpu", lambda _m: None)

    assert built.model_config_snapshot["architecture"] == "cnn_ae"
    assert built.model_config_snapshot["ae_compression"] == 32
    assert built.model_config_snapshot["latent_channels"] == ar.latent_channels_for(3, 32)
    # Round-trips through the same {topo_dict, state_dict} contract.
    assert set(built.adapter_state_fn()) == {"topo_dict", "state_dict"}


def test_resuming_an_autoencoder_as_a_tunet_is_refused() -> None:
    """Both architectures share a model_family, so the family check passes; without
    this the resume would reach load_state_dict and fail on an opaque mismatch."""
    import train_jobs
    from schemas import DlsiaDenoiserConfig, TrainRequest

    request = TrainRequest(
        sources=[{"kind": "tiled", "source": "browse/ds", "server_uri": "u",
                  "slices": {"0": []}, "split_by_slice": {}, "negative_slices": []}],
        classes=[], task="denoising",
        model=DlsiaDenoiserConfig(training_scheme="n2v", architecture="tunet"),
    )
    parent = {
        "model_family": "dlsia_denoiser", "task": "denoising",
        "model_config": {"architecture": "cnn_ae"},
    }

    with pytest.raises(ValueError, match="architecture"):
        train_jobs.check_resume_compatible(parent, request)


def test_resuming_a_legacy_denoiser_run_still_works() -> None:
    """A parent with no `architecture` key is a TUNet, so a TUNet request matches."""
    import train_jobs
    from schemas import DlsiaDenoiserConfig, TrainRequest

    request = TrainRequest(
        sources=[{"kind": "tiled", "source": "browse/ds", "server_uri": "u",
                  "slices": {"0": []}, "split_by_slice": {}, "negative_slices": []}],
        classes=[], task="denoising",
        model=DlsiaDenoiserConfig(training_scheme="n2v"),
    )
    parent = {"model_family": "dlsia_denoiser", "task": "denoising", "model_config": {}}

    train_jobs.check_resume_compatible(parent, request)  # must not raise
