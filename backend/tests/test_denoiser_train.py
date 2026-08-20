"""Torch-free tests for denoise_train's data sampling and masking.

Covers the parts of the self-supervised denoiser that are pure array work:
scope resolution, Noise2Noise pair generation, Noise2Void blind-spot masking,
and the regression-safe letterbox. See test_denoiser_train_torch.py for patch
extraction and the training loop, and test_denoiser_train_dispatch.py for the
train_jobs wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

import arrays
import denoise_train
import images
from schemas import ExportSourceItem, RenderOpts


@pytest.fixture(autouse=True)
def _clear_global_stats_cache():
    """``images._sample_global_stats`` memoizes on ``id(node)``, and a freed
    test array can hand its address to the next one — clear between tests so a
    stale entry can never be mistaken for this test's own volume."""
    images._stats_cache.clear()
    yield
    images._stats_cache.clear()


def _volume(n: int = 5, h: int = 16, w: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, h, w), dtype=np.uint8)


def _source(indices=None) -> ExportSourceItem:
    """A source item shaped the way the Learned Denoiser panel sends it:
    `slices` names the in-scope indices with empty shape lists."""
    slices = {} if indices is None else {str(i): [] for i in indices}
    return ExportSourceItem(kind="local", source="volume.tif", slices=slices)


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


def test_scope_uses_the_panels_empty_shape_list_slice_keys() -> None:
    assert denoise_train.selected_slice_indices(_source([3, 1, 2]), n_slices=10) == [1, 2, 3]


def test_scope_with_no_slices_means_the_whole_volume() -> None:
    assert denoise_train.selected_slice_indices(_source(), n_slices=4) == [0, 1, 2, 3]


def test_scope_drops_out_of_range_indices() -> None:
    assert denoise_train.selected_slice_indices(_source([0, 2, 99]), n_slices=3) == [0, 2]


def test_scope_falls_back_to_the_whole_volume_when_every_index_is_out_of_range() -> None:
    """Not an empty scope: an all-invalid mapping is indistinguishable from a
    stale client, and training on nothing is a worse answer than training on
    everything."""
    assert denoise_train.selected_slice_indices(_source([50, 60]), n_slices=3) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Noise2Noise pair generation
# ---------------------------------------------------------------------------


def test_n2n_pairs_are_structurally_adjacent_at_stride_1() -> None:
    pairs = denoise_train.noise2noise_pairs([0, 1, 2, 3], both_directions=False)
    assert pairs == [(0, 1), (1, 2), (2, 3)]
    assert all(abs(a - b) == 1 for a, b in pairs)


def test_n2n_pairs_respect_a_larger_stride() -> None:
    pairs = denoise_train.noise2noise_pairs([0, 1, 2, 3, 4, 5], stride=3, both_directions=False)
    assert pairs == [(0, 3), (1, 4), (2, 5)]
    assert all(b - a == 3 for a, b in pairs)


def test_n2n_pairs_stay_in_bounds_at_both_ends() -> None:
    """The first and last in-scope slices must not be paired with anything
    outside the scope, nor clamped onto themselves (a self-pair is a perfect
    identity target and would train the network to do nothing)."""
    indices = [0, 1, 2, 3, 4]
    pairs = denoise_train.noise2noise_pairs(indices, stride=2)

    flat = {i for pair in pairs for i in pair}
    assert flat <= set(indices)
    assert all(a != b for a, b in pairs)
    # Nothing reaches past the end: 3 and 4 have no partner at +2.
    assert (3, 5) not in pairs and (4, 6) not in pairs
    assert sorted(pairs) == [(0, 2), (1, 3), (2, 0), (2, 4), (3, 1), (4, 2)]


def test_n2n_both_directions_doubles_the_pairs_and_reverses_each() -> None:
    one_way = denoise_train.noise2noise_pairs([0, 1, 2], both_directions=False)
    two_way = denoise_train.noise2noise_pairs([0, 1, 2], both_directions=True)

    assert len(two_way) == 2 * len(one_way)
    for a, b in one_way:
        assert (a, b) in two_way and (b, a) in two_way


def test_n2n_pairs_skip_gaps_in_a_non_contiguous_scope() -> None:
    """A range selection with a hole must not pair across the hole — slices 1
    and 5 are not adjacent realizations of the same structure."""
    assert denoise_train.noise2noise_pairs([0, 1, 5, 6], both_directions=False) == [(0, 1), (5, 6)]


def test_n2n_pairs_of_a_single_slice_is_empty() -> None:
    assert denoise_train.noise2noise_pairs([7]) == []


def test_n2n_pairs_rejects_a_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride"):
        denoise_train.noise2noise_pairs([0, 1, 2], stride=0)


# ---------------------------------------------------------------------------
# Sampling raw slices into datasets
# ---------------------------------------------------------------------------


def _patch_source(monkeypatch, volume: np.ndarray) -> list[int]:
    """Point resolve_array at *volume* and record every slice index read."""
    reads: list[int] = []
    real_read_slice = arrays.read_slice

    def _counting_read_slice(node, meta, idx):
        reads.append(idx)
        return real_read_slice(node, meta, idx)

    monkeypatch.setattr(arrays, "resolve_array", lambda *a, **k: volume)
    monkeypatch.setattr(arrays, "read_slice", _counting_read_slice)
    # Pin the contrast bounds so its own sampling reads don't enter the count.
    monkeypatch.setattr(images, "_sample_global_stats", lambda node, meta: (0.0, 255.0))
    return reads


def test_prepare_noise2noise_builds_adjacent_pairs_from_raw_slices(monkeypatch) -> None:
    volume = _volume(n=4)
    _patch_source(monkeypatch, volume)

    datasets = denoise_train.prepare_noise2noise_datasets([_source()], RenderOpts(), both_directions=False)

    assert datasets["val"] == []
    assert len(datasets["train"]) == 3
    for k, (inp, tgt) in enumerate(datasets["train"]):
        assert inp.dtype == np.uint8 and inp.shape == volume.shape[1:]
        # Contrast bounds are pinned to the full uint8 range, so the mapping is
        # the identity and the pair is provably slices k and k+1.
        assert np.array_equal(inp, volume[k])
        assert np.array_equal(tgt, volume[k + 1])


def test_prepare_noise2noise_reads_each_slice_exactly_once(monkeypatch) -> None:
    """The sampler caches the scope rather than range-reading a pair per item.
    With both directions at stride 1 every interior slice belongs to four
    pairs, so a per-pair read would fetch it four times."""
    volume = _volume(n=5)
    reads = _patch_source(monkeypatch, volume)

    datasets = denoise_train.prepare_noise2noise_datasets([_source()], RenderOpts(), both_directions=True)

    assert len(datasets["train"]) == 8  # 4 adjacencies x 2 directions
    assert sorted(reads) == [0, 1, 2, 3, 4]  # each slice read once, no repeats


def test_prepare_noise2noise_honours_the_selected_scope(monkeypatch) -> None:
    volume = _volume(n=6)
    reads = _patch_source(monkeypatch, volume)

    denoise_train.prepare_noise2noise_datasets([_source([2, 3])], RenderOpts(), both_directions=False)

    assert sorted(reads) == [2, 3]  # out-of-scope slices are never touched


def test_prepare_noise2noise_rejects_a_scope_with_no_adjacent_pair(monkeypatch) -> None:
    _patch_source(monkeypatch, _volume(n=6))

    with pytest.raises(ValueError, match="Noise2Void"):
        denoise_train.prepare_noise2noise_datasets([_source([0, 4])], RenderOpts())


def test_prepare_noise2void_pairs_each_slice_with_itself(monkeypatch) -> None:
    volume = _volume(n=3)
    _patch_source(monkeypatch, volume)

    datasets = denoise_train.prepare_noise2void_datasets([_source()], RenderOpts())

    assert datasets["val"] == []
    assert len(datasets["train"]) == 3
    for k, (inp, tgt) in enumerate(datasets["train"]):
        assert np.array_equal(inp, volume[k])
        assert inp is tgt  # no needless copy; masking happens per batch


def test_prepare_noise2void_accepts_a_single_slice(monkeypatch) -> None:
    """Unlike Noise2Noise, one slice is a complete Noise2Void dataset."""
    _patch_source(monkeypatch, _volume(n=4))

    datasets = denoise_train.prepare_noise2void_datasets([_source([1])], RenderOpts())

    assert len(datasets["train"]) == 1


def test_prepare_collapses_a_colour_source_to_one_channel(monkeypatch) -> None:
    """denoise_runtime fixes in_channels=1, so an RGB source must arrive as
    (H, W), not (H, W, 3)."""
    rng = np.random.default_rng(1)
    volume = rng.integers(0, 256, size=(3, 12, 14, 3), dtype=np.uint8)
    _patch_source(monkeypatch, volume)

    datasets = denoise_train.prepare_noise2void_datasets([_source()], RenderOpts())

    assert datasets["train"][0][0].shape == (12, 14)


# ---------------------------------------------------------------------------
# Noise2Void blind-spot masking
# ---------------------------------------------------------------------------


def _unique_patch(h: int = 40, w: int = 40) -> np.ndarray:
    """Every pixel holds a distinct value, so a value identifies its origin.
    uint16 because 40x40 exceeds uint8's range — the masking code is
    dtype-agnostic, only the training path pins uint8."""
    return np.arange(h * w, dtype=np.uint16).reshape(h, w)


@pytest.mark.parametrize("fraction", [0.005, 0.01, 0.015, 0.02])
def test_n2v_masks_the_requested_pixel_fraction(fraction: float) -> None:
    rng = np.random.default_rng(0)
    patch = _unique_patch(64, 64)

    _masked, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=fraction)

    assert int(mask.sum()) == round(64 * 64 * fraction)


def test_n2v_always_masks_at_least_one_pixel() -> None:
    """dlsia's MSELossMasked divides by ``masks.sum()`` — an empty mask would
    return NaN and poison the run."""
    rng = np.random.default_rng(0)

    _masked, mask = denoise_train.n2v_mask_and_replace(_unique_patch(8, 8), rng, fraction=0.0001)

    assert int(mask.sum()) == 1


def test_n2v_leaves_every_non_masked_pixel_untouched() -> None:
    rng = np.random.default_rng(3)
    patch = _unique_patch()

    masked, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=0.02)

    assert np.array_equal(masked[~mask], patch[~mask])
    assert masked is not patch  # the caller's array is never mutated


def test_n2v_blind_spot_hides_a_masked_pixels_own_value_from_the_input() -> None:
    """The property the whole scheme rests on. With every pixel value unique, a
    masked pixel's value surviving ANYWHERE in the model input would mean the
    network could learn to copy it — the identity function — and denoise
    nothing."""
    rng = np.random.default_rng(7)
    patch = _unique_patch()

    masked, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=0.02)

    hidden_values = set(patch[mask].tolist())
    assert hidden_values.isdisjoint(set(masked.ravel().tolist()))


def test_n2v_blind_spot_holds_for_pixels_on_the_border() -> None:
    """Regression guard for donor selection at the edges: clipping an offset
    back into range collapses it onto the pixel itself (y=0, dy=-1 -> 0), which
    would hand a border pixel its own value straight back. A high fraction on a
    small patch makes border hits certain."""
    rng = np.random.default_rng(11)
    patch = _unique_patch(12, 12)

    masked, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=0.25)

    ys, xs = np.nonzero(mask)
    on_border = (ys == 0) | (xs == 0) | (ys == patch.shape[0] - 1) | (xs == patch.shape[1] - 1)
    assert on_border.any(), "test is vacuous unless some masked pixel sits on the border"
    assert set(patch[mask].tolist()).isdisjoint(set(masked.ravel().tolist()))


def test_n2v_replacement_comes_from_the_local_neighbourhood() -> None:
    """Each masked pixel takes a value from within its own NxN window — a donor
    from across the image would be a much weaker prior on the missing value."""
    rng = np.random.default_rng(5)
    patch = _unique_patch(50, 50)
    radius = denoise_train.DEFAULT_N2V_NEIGHBOURHOOD // 2

    masked, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=0.01)

    h, w = patch.shape
    for y, x in zip(*np.nonzero(mask)):
        donor_y, donor_x = divmod(int(masked[y, x]), w)
        assert abs(donor_y - y) <= radius and abs(donor_x - x) <= radius
        assert (donor_y, donor_x) != (y, x)
        assert 0 <= donor_y < h and 0 <= donor_x < w


def test_n2v_donors_are_never_themselves_masked_pixels() -> None:
    """Why the blind spot survives: if a masked pixel could donate, its value
    would reappear at the borrower's coordinates, inside the same receptive
    field."""
    rng = np.random.default_rng(9)
    patch = _unique_patch(50, 50)

    masked, mask = denoise_train.n2v_mask_and_replace(patch, rng, fraction=0.02)

    w = patch.shape[1]
    for y, x in zip(*np.nonzero(mask)):
        donor_y, donor_x = divmod(int(masked[y, x]), w)
        assert not mask[donor_y, donor_x]


def test_n2v_masks_differ_between_draws() -> None:
    """A fresh mask per patch per epoch is what eventually supervises every
    pixel; one fixed mask would only ever train ~1.5% of them."""
    patch = _unique_patch()
    _m1, mask1 = denoise_train.n2v_mask_and_replace(patch, np.random.default_rng(1), fraction=0.015)
    _m2, mask2 = denoise_train.n2v_mask_and_replace(patch, np.random.default_rng(2), fraction=0.015)

    assert not np.array_equal(mask1, mask2)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_n2v_rejects_a_fraction_outside_the_open_unit_interval(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        denoise_train.n2v_mask_and_replace(_unique_patch(8, 8), np.random.default_rng(0), fraction=fraction)


@pytest.mark.parametrize("neighbourhood", [1, 2, 4])
def test_n2v_rejects_an_even_or_degenerate_neighbourhood(neighbourhood: int) -> None:
    with pytest.raises(ValueError, match="neighbourhood"):
        denoise_train.n2v_mask_and_replace(
            _unique_patch(8, 8), np.random.default_rng(0), neighbourhood=neighbourhood
        )


def test_mirror_offset_never_maps_a_nonzero_offset_onto_the_pixel_itself() -> None:
    """The helper behind the border blind-spot guarantee, checked exhaustively
    over a small axis for every offset a 5x5 window can produce. Both obvious
    implementations fail here: clipping collapses (centre=0, offset=-1) onto 0,
    and index reflection collapses (centre=1, offset=-2) onto 1."""
    n = 9
    for centre in range(n):
        for offset in (-2, -1, 1, 2):
            landed = int(denoise_train._mirror_offset(np.array([centre]), np.array([offset]), n)[0])
            assert 0 <= landed < n
            assert landed != centre, f"centre={centre} offset={offset}"


# ---------------------------------------------------------------------------
# Regression-safe letterbox
# ---------------------------------------------------------------------------


def test_letterbox_passes_an_exact_square_straight_through() -> None:
    """Every tiled patch hits this — it must not pay for a resize."""
    inp = np.ones((32, 32), dtype=np.uint8)
    tgt = np.zeros((32, 32), dtype=np.uint8)

    out_in, out_tgt = denoise_train.letterbox_denoise_pair(inp, tgt, 32)

    assert out_in is inp and out_tgt is tgt


def test_letterbox_pads_with_zero_never_with_ignore_index() -> None:
    """train_common.letterbox pads labels with IGNORE_INDEX (255). For a
    regression target 255 is not "ignore", it is maximum brightness — a white
    frame the network would be trained to reproduce."""
    import train_common

    inp = np.full((10, 30), 7, dtype=np.uint8)
    out_in, out_tgt = denoise_train.letterbox_denoise_pair(inp, inp, 32)

    assert out_in.shape == (32, 32) and out_tgt.shape == (32, 32)
    assert (out_in == train_common.IGNORE_INDEX).sum() == 0
    assert out_in[0, 0] == 0 and out_tgt[0, 0] == 0  # padding is black


def test_letterbox_keeps_input_and_target_pixel_aligned() -> None:
    """Both sides take the same resize and the same placement, so a feature at
    a given coordinate in one lands at the same coordinate in the other. Marked
    with different intensities so the two outputs can't be confused."""
    inp = np.zeros((20, 40), dtype=np.uint8)
    tgt = np.zeros((20, 40), dtype=np.uint8)
    inp[4:12, 6:14] = 200
    tgt[4:12, 6:14] = 90

    out_in, out_tgt = denoise_train.letterbox_denoise_pair(inp, tgt, 64)

    def _bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(arr)
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    assert out_in.max() > out_tgt.max()  # the two are genuinely different images
    assert _bbox(out_in) == _bbox(out_tgt)


def test_letterbox_rejects_a_mismatched_pair() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        denoise_train.letterbox_denoise_pair(
            np.zeros((8, 8), np.uint8), np.zeros((8, 9), np.uint8), 16
        )
