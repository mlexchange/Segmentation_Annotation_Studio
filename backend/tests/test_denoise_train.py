"""Tests for denoise_train.py: pure sampling/geometry/masking logic directly,
plus a real (no mocks) run_denoise_training_loop round trip using a minimal
trainable model — the loop itself is generic over forward_fn/to_tensor_fn/
trainable_params, so it doesn't need a real dlsia model to exercise for real.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import arrays as arrays_mod  # noqa: E402
import denoise_train  # noqa: E402


class FakeItem:
    def __init__(self, source="fake.tif", kind="local", server_uri=None, slices=None):
        self.source = source
        self.kind = kind
        self.server_uri = server_uri
        self.slices = slices or {}


# ---------------------------------------------------------------------------
# selected_slice_indices
# ---------------------------------------------------------------------------

class TestSelectedSliceIndices:
    def test_empty_slices_returns_whole_range(self):
        assert denoise_train.selected_slice_indices(FakeItem(slices={}), 5) == [0, 1, 2, 3, 4]

    def test_explicit_keys_sorted_and_deduped(self):
        item = FakeItem(slices={"3": [], "1": [], "3": []})  # noqa: F601
        assert denoise_train.selected_slice_indices(item, 10) == [1, 3]

    def test_out_of_range_keys_dropped(self):
        item = FakeItem(slices={"1": [], "99": [], "-1": []})
        assert denoise_train.selected_slice_indices(item, 5) == [1]

    def test_non_integer_keys_ignored(self):
        item = FakeItem(slices={"abc": [], "2": []})
        assert denoise_train.selected_slice_indices(item, 5) == [2]

    def test_all_keys_out_of_range_falls_back_to_whole_range(self):
        item = FakeItem(slices={"99": []})
        assert denoise_train.selected_slice_indices(item, 3) == [0, 1, 2]


# ---------------------------------------------------------------------------
# noise2noise_pairs
# ---------------------------------------------------------------------------

class TestNoise2NoisePairs:
    def test_adjacent_pairs_both_directions(self):
        pairs = denoise_train.noise2noise_pairs([0, 1, 2], stride=1, both_directions=True)
        assert set(pairs) == {(0, 1), (1, 0), (1, 2), (2, 1)}

    def test_single_direction(self):
        pairs = denoise_train.noise2noise_pairs([0, 1, 2], stride=1, both_directions=False)
        assert set(pairs) == {(0, 1), (1, 2)}

    def test_last_slice_has_no_partner_not_clamped(self):
        pairs = denoise_train.noise2noise_pairs([0, 1], stride=1, both_directions=False)
        assert (1, 1) not in pairs

    def test_stride_skips_non_adjacent(self):
        pairs = denoise_train.noise2noise_pairs([0, 1, 2], stride=2, both_directions=False)
        assert pairs == [(0, 2)]

    def test_no_qualifying_pairs_returns_empty(self):
        assert denoise_train.noise2noise_pairs([0], stride=1) == []

    def test_stride_below_one_raises(self):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            denoise_train.noise2noise_pairs([0, 1], stride=0)


# ---------------------------------------------------------------------------
# _mirror_offset
# ---------------------------------------------------------------------------

class TestMirrorOffset:
    def test_in_range_offset_unchanged(self):
        out = denoise_train._mirror_offset(np.array([5]), np.array([2]), 10)
        assert out[0] == 7

    def test_out_of_range_flips_sign_instead_of_clamping(self):
        # centre=0, offset=-1 would clip to 0 (== centre) under naive clamping;
        # flipping gives centre - offset = 1, which differs from centre.
        out = denoise_train._mirror_offset(np.array([0]), np.array([-1]), 10)
        assert out[0] == 1

    def test_result_always_in_bounds(self):
        centre = np.array([0, 9])
        offset = np.array([-5, 5])
        out = denoise_train._mirror_offset(centre, offset, 10)
        assert np.all((out >= 0) & (out < 10))


# ---------------------------------------------------------------------------
# n2v_mask_and_replace
# ---------------------------------------------------------------------------

class TestN2vMaskAndReplace:
    def test_mask_count_matches_fraction(self):
        patch = (np.arange(400) % 256).astype(np.uint8).reshape(20, 20)
        rng = np.random.default_rng(0)
        _, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=0.05, neighbourhood=5)
        assert mask.sum() == round(400 * 0.05)

    def test_donor_never_the_masked_pixel_itself(self):
        patch = (np.arange(400) % 256).astype(np.uint8).reshape(20, 20)
        rng = np.random.default_rng(1)
        masked, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=0.1, neighbourhood=5)
        # A masked pixel that kept its original value would mean its donor was itself.
        assert not np.array_equal(masked[mask], patch[mask])

    def test_at_least_one_pixel_masked_for_tiny_fraction(self):
        patch = np.zeros((10, 10), dtype=np.uint8)
        rng = np.random.default_rng(0)
        _, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=0.0001, neighbourhood=3)
        assert mask.sum() >= 1

    def test_unmasked_pixels_untouched(self):
        patch = (np.arange(400) % 256).astype(np.uint8).reshape(20, 20)
        rng = np.random.default_rng(2)
        masked, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=0.05, neighbourhood=5)
        assert np.array_equal(masked[~mask], patch[~mask])

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_fraction_out_of_range_raises(self, fraction):
        with pytest.raises(ValueError, match="fraction must be in"):
            denoise_train.n2v_mask_and_replace(np.zeros((10, 10), dtype=np.uint8), np.random.default_rng(0), fraction=fraction)

    @pytest.mark.parametrize("neighbourhood", [2, 4, 1])
    def test_neighbourhood_must_be_odd_and_at_least_three(self, neighbourhood):
        if neighbourhood >= 3 and neighbourhood % 2 == 1:
            return
        with pytest.raises(ValueError, match="neighbourhood must be an odd size"):
            denoise_train.n2v_mask_and_replace(
                np.zeros((10, 10), dtype=np.uint8), np.random.default_rng(0), neighbourhood=neighbourhood,
            )


# ---------------------------------------------------------------------------
# letterbox_denoise_pair
# ---------------------------------------------------------------------------

class TestLetterboxDenoisePair:
    def test_already_correct_size_short_circuits(self):
        inp = np.zeros((32, 32), dtype=np.uint8)
        tgt = np.ones((32, 32), dtype=np.uint8)
        out_inp, out_tgt = denoise_train.letterbox_denoise_pair(inp, tgt, 32)
        assert out_inp is inp and out_tgt is tgt

    def test_resizes_and_pads_to_target_size(self):
        inp = np.full((16, 32), 100, dtype=np.uint8)
        tgt = np.full((16, 32), 200, dtype=np.uint8)
        out_inp, out_tgt = denoise_train.letterbox_denoise_pair(inp, tgt, 64)
        assert out_inp.shape == (64, 64)
        assert out_tgt.shape == (64, 64)
        # Corners should be zero-padded (aspect-ratio letterboxing).
        assert out_inp[0, 0] == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes differ"):
            denoise_train.letterbox_denoise_pair(
                np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 20), dtype=np.uint8), 32,
            )


# ---------------------------------------------------------------------------
# prepare_noise2noise_datasets / prepare_noise2void_datasets (real images_mod,
# fake arrays_mod — same pattern as test_infer_jobs.py's fake array source)
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_array_source(monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(1)
    volume = rng.integers(0, 256, size=(4, 16, 16), dtype=np.uint8)

    monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri: volume)
    monkeypatch.setattr(
        arrays_mod, "array_shape_meta",
        lambda node, pyramid=None: {"height": 16, "width": 16, "n_slices": 4},
    )
    monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: node[idx])
    return volume


class TestPrepareDatasets:
    def test_noise2noise_produces_pairs_from_adjacent_slices(self, fake_array_source):
        result = denoise_train.prepare_noise2noise_datasets([FakeItem()], render={})
        assert result["val"] == []
        assert len(result["train"]) > 0
        for inp, tgt in result["train"]:
            assert inp.dtype == np.uint8 and inp.shape == (16, 16)

    def test_noise2noise_raises_when_only_one_slice_in_scope(self, fake_array_source):
        item = FakeItem(slices={"0": []})
        with pytest.raises(ValueError, match="Noise2Noise needs at least two slices"):
            denoise_train.prepare_noise2noise_datasets([item], render={})

    def test_noise2void_pairs_each_slice_with_itself(self, fake_array_source):
        result = denoise_train.prepare_noise2void_datasets([FakeItem()], render={})
        assert len(result["train"]) == 4
        for inp, tgt in result["train"]:
            assert np.array_equal(inp, tgt)

    def test_noise2void_raises_with_no_slices_in_scope(self, monkeypatch, fake_array_source):
        monkeypatch.setattr(
            arrays_mod, "array_shape_meta",
            lambda node, pyramid=None: {"height": 16, "width": 16, "n_slices": 0},
        )
        with pytest.raises(ValueError, match="Noise2Void needs at least one slice"):
            denoise_train.prepare_noise2void_datasets([FakeItem()], render={})


# ---------------------------------------------------------------------------
# run_denoise_training_loop — real round trip, minimal trainable model
# ---------------------------------------------------------------------------

def _tiny_model():
    """A minimal real nn.Module — the loop is generic over forward_fn/
    to_tensor_fn/trainable_params, so it doesn't need dlsia's actual TUNet."""
    model = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward_fn(batch):
        return model(batch)

    def to_tensor_fn(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0) / 255.0

    return forward_fn, to_tensor_fn, list(model.parameters())


def _pairs(n, size=8, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (rng.integers(0, 256, size=(size, size), dtype=np.uint8), rng.integers(0, 256, size=(size, size), dtype=np.uint8))
        for _ in range(n)
    ]


class TestRunDenoiseTrainingLoop:
    @pytest.mark.parametrize("scheme", ["n2n", "n2v", "ae"])
    def test_real_round_trip_all_schemes(self, scheme):
        forward_fn, to_tensor_fn, params = _tiny_model()
        metrics = denoise_train.run_denoise_training_loop(
            train_pairs=_pairs(6), val_pairs=_pairs(2, seed=1),
            image_size=8, training_scheme=scheme, epochs=2, batch_size=2, seed=0,
            flip_augment=True, to_tensor_fn=to_tensor_fn, forward_fn=forward_fn,
            trainable_params=params, lr=1e-3, device="cpu",
        )
        assert metrics["epochs_completed"] == 2
        assert metrics["cancelled"] is False
        assert np.isfinite(metrics["final_train_loss"])
        assert np.isfinite(metrics["final_val_loss"])
        assert metrics[denoise_train.VAL_METRIC_KEY] is not None

    def test_cancellation_via_on_epoch_stops_early_with_partial_metrics(self):
        forward_fn, to_tensor_fn, params = _tiny_model()
        metrics = denoise_train.run_denoise_training_loop(
            train_pairs=_pairs(4), val_pairs=[],
            image_size=8, training_scheme="n2n", epochs=5, batch_size=2, seed=0,
            flip_augment=False, to_tensor_fn=to_tensor_fn, forward_fn=forward_fn,
            trainable_params=params, lr=1e-3, device="cpu",
            on_epoch=lambda epoch, *_: epoch == 1,
        )
        assert metrics["cancelled"] is True
        assert metrics["epochs_completed"] == 1

    def test_no_training_data_raises(self):
        forward_fn, to_tensor_fn, params = _tiny_model()
        with pytest.raises(ValueError, match="No training data"):
            denoise_train.run_denoise_training_loop(
                train_pairs=[], val_pairs=[], image_size=8, training_scheme="n2n",
                epochs=1, batch_size=2, seed=0, flip_augment=False,
                to_tensor_fn=to_tensor_fn, forward_fn=forward_fn, trainable_params=params,
                lr=1e-3, device="cpu",
            )

    def test_unknown_scheme_raises(self):
        forward_fn, to_tensor_fn, params = _tiny_model()
        with pytest.raises(ValueError, match="Unknown denoiser training scheme"):
            denoise_train.run_denoise_training_loop(
                train_pairs=_pairs(2), val_pairs=[], image_size=8, training_scheme="bogus",
                epochs=1, batch_size=2, seed=0, flip_augment=False,
                to_tensor_fn=to_tensor_fn, forward_fn=forward_fn, trainable_params=params,
                lr=1e-3, device="cpu",
            )
