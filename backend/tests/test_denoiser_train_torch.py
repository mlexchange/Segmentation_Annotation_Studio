"""Tensor-backed tests for denoise_train: patch extraction, the masked loss,
and the training loop itself.

Skipped without torch/qlty/dlsia (all three live in the optional ``ml``
extra). See test_denoiser_train.py for the torch-free sampling/masking half.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qlty")
pytest.importorskip("dlsia")

import denoise_runtime  # noqa: E402
import denoise_train  # noqa: E402
import tiling  # noqa: E402
from train_common import IGNORE_INDEX  # noqa: E402


WINDOW = 64


def _noisy_volume(n: int = 4, size: int = 80, seed: int = 0) -> list[np.ndarray]:
    """A fixed structure (a horizontal ramp) plus independent noise per slice —
    the situation Noise2Noise is built for."""
    rng = np.random.default_rng(seed)
    clean = np.tile(np.linspace(0, 255, size).astype(np.int16), (size, 1))
    return [
        np.clip(clean + rng.integers(-30, 31, size=(size, size)), 0, 255).astype(np.uint8)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Patch extraction — the sparse-annotation weeding bypass
# ---------------------------------------------------------------------------


def test_patches_are_kept_where_the_segmentation_path_would_drop_them_all() -> None:
    """The trap this bypass exists for. ``tiling._tile_pair`` runs qlty's
    ``weed_sparse_classification_training_pairs_2D``, which drops every patch
    containing no labelled pixel — correct for sparse annotations, fatal for a
    denoiser, where the "target" is intensity and IGNORE_INDEX (255) just means
    a white pixel. Same image, same window: the segmentation path yields
    nothing, this one yields every patch."""
    size = 96
    rgb = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    unlabelled = np.full((size, size), IGNORE_INDEX, dtype=np.uint8)

    assert tiling._tile_pair(rgb, unlabelled, WINDOW) == []

    gray = rgb[:, :, 0]
    white_target = np.full((size, size), 255, dtype=np.uint8)
    patches = denoise_train.tile_denoise_pair(gray, white_target, WINDOW)

    assert len(patches) > 0
    assert all(p[0].shape == (WINDOW, WINDOW) for p in patches)


def test_patch_count_matches_the_qlty_geometry_used_at_inference() -> None:
    """Training windows must be the same grid ``tiling.denoise_image_tiled``
    blends at inference, so the model sees identically-shaped input in both."""
    size = 150
    gray = np.zeros((size, size), dtype=np.uint8)
    expected = len(tiling.tile_origins(size, WINDOW, tiling.step_for(WINDOW))) ** 2

    assert len(denoise_train.tile_denoise_pair(gray, gray, WINDOW)) == expected


def test_patch_extraction_keeps_input_and_target_spatially_aligned() -> None:
    """Both sides are unstitched through the same quilt, so patch k of one is
    the same window of the image as patch k of the other. Verified with a
    coordinate-encoded target: each pixel's value identifies where it came
    from, so any misalignment shows up as an offset."""
    size = 128
    ys, xs = np.mgrid[0:size, 0:size]
    inp = ((ys * 7) % 251).astype(np.uint8)
    tgt = ((xs * 5) % 241).astype(np.uint8)

    patches = denoise_train.tile_denoise_pair(inp, tgt, WINDOW)

    origins = tiling.tile_origins(size, WINDOW, tiling.step_for(WINDOW))
    expected = [
        (inp[y : y + WINDOW, x : x + WINDOW], tgt[y : y + WINDOW, x : x + WINDOW])  # noqa: E203
        for y in origins
        for x in origins
    ]
    assert len(patches) == len(expected)
    for (got_in, got_tgt), (want_in, want_tgt) in zip(patches, expected):
        assert np.array_equal(got_in, want_in)
        assert np.array_equal(got_tgt, want_tgt)


def test_patch_extraction_pads_an_image_smaller_than_one_window() -> None:
    small = np.full((20, 30), 128, dtype=np.uint8)

    patches = denoise_train.tile_denoise_pair(small, small, WINDOW)

    assert len(patches) == 1
    patch_in, patch_tgt = patches[0]
    assert patch_in.shape == (WINDOW, WINDOW)
    # Padding is zero on BOTH sides, so it stays a matched (trivial) region
    # rather than a bright frame the network is asked to invent.
    assert patch_in[25, 25] == 0 and patch_tgt[25, 25] == 0


def test_patch_extraction_rejects_a_mismatched_pair() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        denoise_train.tile_denoise_pair(
            np.zeros((80, 80), np.uint8), np.zeros((80, 96), np.uint8), WINDOW
        )


def test_tile_denoise_datasets_keeps_the_split_shape() -> None:
    volume = _noisy_volume(n=3)
    datasets = {"train": [(volume[0], volume[1])], "val": [(volume[1], volume[2])]}

    tiled = denoise_train.tile_denoise_datasets(datasets, WINDOW)

    assert set(tiled) == {"train", "val"}
    assert len(tiled["train"]) > 1 and len(tiled["val"]) > 1
    assert tiled["train"][0][0].shape == (WINDOW, WINDOW)


def test_tile_denoise_datasets_returns_none_when_cancelled() -> None:
    """Same cancellation contract as tiling.tile_datasets, so train_jobs checks
    it the same way."""
    volume = _noisy_volume(n=2)

    result = denoise_train.tile_denoise_datasets(
        {"train": [(volume[0], volume[1])], "val": []}, WINDOW, cancel_cb=lambda: True
    )

    assert result is None


# ---------------------------------------------------------------------------
# The masked (Noise2Void) loss
# ---------------------------------------------------------------------------


def _masked_criterion():
    from dlsia.core.custom_losses import MSELossMasked

    return MSELossMasked()


def test_n2v_loss_ignores_every_non_masked_pixel() -> None:
    """The defining property of the scheme: only the blind-spot coordinates are
    supervised. Perturbing a non-masked pixel's prediction must leave the loss
    bit-identical; perturbing a masked one must move it."""
    criterion = _masked_criterion()
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    mask[0, 0, 3, 3] = True
    mask[0, 0, 5, 1] = True

    baseline = float(denoise_train._denoise_loss(pred, target, mask, criterion))

    off_spot = pred.clone()
    off_spot[0, 0, 0, 0] = 99.0
    assert float(denoise_train._denoise_loss(off_spot, target, mask, criterion)) == baseline

    on_spot = pred.clone()
    on_spot[0, 0, 3, 3] = 1.0
    assert float(denoise_train._denoise_loss(on_spot, target, mask, criterion)) > baseline


def test_n2n_loss_by_contrast_counts_every_pixel() -> None:
    """The unmasked branch must NOT quietly inherit the masked behaviour — for
    Noise2Noise the whole patch is supervised."""
    criterion = torch.nn.MSELoss()
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)

    baseline = float(denoise_train._denoise_loss(pred, target, None, criterion))
    perturbed = pred.clone()
    perturbed[0, 0, 0, 0] = 4.0

    assert float(denoise_train._denoise_loss(perturbed, target, None, criterion)) > baseline


def test_prepped_n2v_batch_masks_the_input_but_targets_the_original() -> None:
    """End-to-end through the batch assembler: the model's input carries the
    substituted values, the target carries the originals, and the mask marks
    exactly where they differ."""
    patch = np.random.default_rng(0).integers(0, 256, size=(WINDOW, WINDOW), dtype=np.uint8)

    imgs, tgts, mask = denoise_train._prep_batch(
        [(patch, patch)],
        [0],
        image_size=WINDOW,
        scheme="n2v",
        to_tensor_fn=denoise_runtime.make_to_tensor_fn(),
        rng=np.random.default_rng(0),
        flip_augment=False,
        mask_fraction=0.02,
        neighbourhood=5,
    )

    assert mask is not None and mask.shape == imgs.shape
    assert int(mask.sum()) == round(WINDOW * WINDOW * 0.02)
    # Target is the untouched patch; the input differs only under the mask.
    assert torch.allclose(tgts[0, 0], torch.from_numpy(patch).float() / 255.0)
    assert torch.equal(imgs[~mask], tgts[~mask])


def test_prepped_n2n_batch_has_no_mask() -> None:
    volume = _noisy_volume(n=2, size=WINDOW)

    _imgs, _tgts, mask = denoise_train._prep_batch(
        [(volume[0], volume[1])],
        [0],
        image_size=WINDOW,
        scheme="n2n",
        to_tensor_fn=denoise_runtime.make_to_tensor_fn(),
        rng=np.random.default_rng(0),
        flip_augment=False,
        mask_fraction=0.02,
        neighbourhood=5,
    )

    assert mask is None


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------


def _tiny_model():
    """A 2-deep, 4-channel TUNet at 64px — small enough for a couple of epochs
    to be fast, real enough to be the model that actually ships."""
    return denoise_runtime.build_model(WINDOW, 2, 4, 1.5, "cpu")


def _tiny_dataset(scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
    volume = _noisy_volume(n=3, size=80)
    if scheme == "n2n":
        pairs = [(volume[i], volume[i + 1]) for i in range(2)]
    else:
        pairs = [(sl, sl) for sl in volume]
    return denoise_train.tile_denoise_datasets({"train": pairs}, WINDOW)["train"]


def _run(scheme: str, **overrides):
    model = _tiny_model()
    train_pairs = _tiny_dataset(scheme)
    kwargs = dict(
        train_pairs=train_pairs,
        val_pairs=train_pairs[:2],
        image_size=WINDOW,
        training_scheme=scheme,
        epochs=2,
        batch_size=2,
        seed=0,
        flip_augment=True,
        to_tensor_fn=denoise_runtime.make_to_tensor_fn(),
        forward_fn=denoise_runtime.make_forward_fn(model),
        trainable_params=list(model.parameters()),
        lr=1e-2,
        device="cpu",
        set_train_mode=denoise_runtime.make_set_train_mode_fn(model),
    )
    kwargs.update(overrides)
    return denoise_train.run_denoise_training_loop(**kwargs)


@pytest.mark.parametrize("scheme", ["n2n", "n2v"])
def test_training_loop_runs_end_to_end_and_reduces_the_training_loss(scheme: str) -> None:
    losses: list[float] = []

    metrics = _run(scheme, on_epoch=lambda e, tl, vl, vm: losses.append(tl) or False)

    assert metrics["epochs_completed"] == 2
    assert metrics["cancelled"] is False
    assert len(losses) == 2
    assert losses[-1] < losses[0], f"{scheme} loss did not fall: {losses}"


@pytest.mark.parametrize("scheme", ["n2n", "n2v"])
def test_training_loop_returns_the_shared_metrics_dict_shape(scheme: str) -> None:
    """train_jobs reads the result without special-casing the task, so the keys
    must match run_training_loop's — with the regression metric in place of
    mIoU, which would be meaningless here."""
    metrics = _run(scheme)

    assert set(metrics) == {
        "epochs_completed",
        "final_train_loss",
        "final_val_loss",
        denoise_train.VAL_METRIC_KEY,
        "cancelled",
    }
    assert "val_miou" not in metrics
    assert metrics["final_val_loss"] is not None
    assert -1.0 <= metrics[denoise_train.VAL_METRIC_KEY] <= 1.0


def test_validation_metric_is_named_for_the_noisy_target_not_for_quality() -> None:
    """Guard on the name itself. Both schemes score against a target that is
    still noisy, so anything reading like 'psnr' or 'quality' would be a lie
    wherever it surfaced in the UI."""
    key = denoise_train.VAL_METRIC_KEY
    assert "noisy" in key and ("pearson" in key or "correlation" in key)
    assert "psnr" not in key and "quality" not in key


def test_training_loop_reports_no_validation_metric_without_val_data() -> None:
    metrics = _run("n2n", val_pairs=[])

    assert metrics["final_val_loss"] is None
    assert metrics[denoise_train.VAL_METRIC_KEY] is None


@pytest.mark.parametrize("scheme", ["n2n", "n2v"])
def test_training_loop_honors_a_cooperative_cancel_from_on_batch(scheme: str) -> None:
    """Same contract as run_training_loop: returning True stops, and the
    partial result still comes back so the caller can save a checkpoint."""
    calls: list[int] = []

    def _on_batch() -> bool:
        calls.append(1)
        return True  # cancel on the very first batch

    metrics = _run(scheme, epochs=5, on_batch=_on_batch)

    assert metrics["cancelled"] is True
    assert len(calls) == 1
    assert metrics["epochs_completed"] == 1  # stopped inside the first epoch


@pytest.mark.parametrize("scheme", ["n2n", "n2v"])
def test_training_loop_honors_a_cooperative_cancel_from_on_epoch(scheme: str) -> None:
    seen: list[int] = []

    metrics = _run(scheme, epochs=5, on_epoch=lambda e, tl, vl, vm: bool(seen.append(e)) or True)

    assert metrics["cancelled"] is True
    assert metrics["epochs_completed"] == 1
    assert seen == [1]  # no further epochs ran


def test_cancelling_in_on_batch_skips_validation_for_that_epoch() -> None:
    """A half-finished epoch's weights aren't worth a validation pass, and
    run_training_loop skips it too — keep the behaviour identical."""
    metrics = _run("n2n", epochs=3, on_batch=lambda: True)

    assert metrics["final_val_loss"] is None


def test_training_loop_rejects_an_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        _run("n2x")


def test_training_loop_rejects_an_empty_dataset() -> None:
    with pytest.raises(ValueError, match="at least one slice"):
        _run("n2n", train_pairs=[])


def test_training_loop_letterboxes_a_non_tiled_pair_instead_of_failing() -> None:
    """With tiling off the items are whole slices of arbitrary size. They must
    be letterboxed as IMAGES (zero pad), never through train_common.letterbox's
    IGNORE_INDEX label padding — the model has a fixed 64px input either way."""
    volume = _noisy_volume(n=2, size=100)

    metrics = _run("n2n", train_pairs=[(volume[0], volume[1])], val_pairs=[(volume[0], volume[1])])

    assert metrics["epochs_completed"] == 2
    assert metrics["final_val_loss"] is not None
