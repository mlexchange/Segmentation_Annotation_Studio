"""Tests for patch-based (tiled) training/inference geometry.

The load-bearing property is that :func:`tiling.predict_label_map_tiled`'s
streamed accumulation reproduces what ``qlty``'s own ``stitch`` computes — it
exists only to bound memory, and must not change the arithmetic.

Skipped without torch/qlty (both live in the optional ``ml`` extra).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qlty")

import tiling  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from train_common import IGNORE_INDEX  # noqa: E402


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_tile_origins_match_qlty_patch_counts() -> None:
    """Our grid must yield exactly as many windows per axis as qlty expects,
    since `stitch` asserts on the total patch count."""
    from qlty.qlty2D import NCYXQuilt

    for dim in (512, 600, 906, 1038, 1500):
        window = 512
        step = tiling.step_for(window)
        quilt = NCYXQuilt(Y=dim, X=dim, window=(window, window), step=(step, step), border=tiling.border_for(window))
        assert len(tiling.tile_origins(dim, window, step)) == quilt.nY


def test_tile_origins_stay_in_bounds_and_cover_the_axis() -> None:
    window, dim = 512, 906
    origins = tiling.tile_origins(dim, window, tiling.step_for(window))

    assert origins[0] == 0
    assert all(0 <= o <= dim - window for o in origins)
    assert origins[-1] == dim - window  # last window clamped flush to the edge
    covered = np.zeros(dim, dtype=bool)
    for o in origins:
        covered[o : o + window] = True  # noqa: E203
    assert covered.all()


def test_tile_origins_rejects_an_undersized_axis() -> None:
    with pytest.raises(ValueError):
        tiling.tile_origins(100, 512, tiling.step_for(512))


def test_border_is_always_exactly_half_the_overlap() -> None:
    """`border_for(w) == (w - step_for(w)) // 2` — the invariant behind the
    BORDER_DIVISOR comment: the down-weighted/ignored ring at each window's
    edge is exactly half of the overlap between neighbours, so a down-weighted
    edge is always fully covered by a neighbour's full-weight interior. Checked
    across every size either family's schema actually allows (DinoHyperParams:
    224-1024 step 16; TunetHyperParams: 64-2048 step 64), not just the 512
    default — nothing else pins this property, and it's silent (not a crash)
    if it ever breaks."""
    dino_legal_sizes = range(224, 1024 + 1, 16)
    tunet_legal_sizes = range(64, 2048 + 1, 64)
    for window in sorted(set(dino_legal_sizes) | set(tunet_legal_sizes)):
        overlap = window - tiling.step_for(window)
        assert tiling.border_for(window) == overlap // 2, f"window={window}"


def test_pad_to_min_pads_only_when_needed_and_uses_the_fill() -> None:
    img = np.ones((300, 200, 3), dtype=np.uint8)
    padded = tiling.pad_to_min(img, 512, 512, fill=0)
    assert padded.shape == (512, 512, 3)
    assert padded[:300, :200].all() and not padded[300:, :].any()

    label = np.zeros((300, 200), dtype=np.uint8)
    padded_label = tiling.pad_to_min(label, 512, 512, fill=IGNORE_INDEX)
    assert (padded_label[300:, :] == IGNORE_INDEX).all()

    big = np.ones((600, 600, 3), dtype=np.uint8)
    assert tiling.pad_to_min(big, 512, 512, fill=0) is big  # untouched


# ---------------------------------------------------------------------------
# Training patches
# ---------------------------------------------------------------------------


def _annotated_pair(h: int, w: int, blob: tuple[slice, slice], window: int = 64):
    rgb = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    label = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
    label[blob] = 1
    return rgb, label


def test_tile_datasets_emits_window_sized_patches() -> None:
    window = 64
    rgb, label = _annotated_pair(200, 150, (slice(20, 120), slice(20, 120)))

    out = tiling.tile_datasets({"train": [(rgb, label)], "val": []}, window)

    assert out["val"] == []
    assert len(out["train"]) > 0
    for patch_rgb, patch_label in out["train"]:
        assert patch_rgb.shape == (window, window, 3)
        assert patch_label.shape == (window, window)
        assert patch_rgb.dtype == np.uint8 and patch_label.dtype == np.uint8


def test_tile_datasets_drops_fully_unannotated_patches() -> None:
    """Sparse annotation: patches with no labelled pixel contribute nothing to
    the loss, so they must not be handed to the training loop."""
    window = 64
    rgb, label = _annotated_pair(256, 256, (slice(0, 40), slice(0, 40)))

    patches = tiling.tile_datasets({"train": [(rgb, label)]}, window)["train"]

    assert len(patches) > 0
    for _, patch_label in patches:
        assert (patch_label != IGNORE_INDEX).any()
    # Far fewer than the full grid, which would be 5x5 windows here.
    assert len(patches) < 25


def test_tile_datasets_keeps_an_edge_only_annotation() -> None:
    """An annotation lying entirely in the image's outer ring is in no window's
    interior; border-masking alone would discard the slice's only labels."""
    window = 64
    rgb, label = _annotated_pair(200, 200, (slice(0, 4), slice(0, 4)))

    patches = tiling.tile_datasets({"train": [(rgb, label)]}, window)["train"]

    assert len(patches) > 0
    assert any((pl != IGNORE_INDEX).any() for _, pl in patches)


def test_tile_datasets_handles_an_image_smaller_than_the_window() -> None:
    window = 64
    rgb, label = _annotated_pair(40, 30, (slice(5, 30), slice(5, 25)))

    patches = tiling.tile_datasets({"train": [(rgb, label)]}, window)["train"]

    assert len(patches) == 1
    assert patches[0][0].shape == (window, window, 3)


def test_tile_datasets_with_no_annotations_yields_nothing() -> None:
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    label = np.full((128, 128), IGNORE_INDEX, dtype=np.uint8)

    assert tiling.tile_datasets({"train": [(rgb, label)]}, 64)["train"] == []


def test_tile_datasets_reports_progress_per_slice_not_just_per_split() -> None:
    """Regression: tiling a large annotated set used to show no progress at all
    until the whole split finished — one summary line at the very end."""
    window = 64
    pair_a = _annotated_pair(200, 200, (slice(0, 100), slice(0, 100)))
    pair_b = _annotated_pair(200, 200, (slice(0, 100), slice(0, 100)))
    messages: list[str] = []

    tiling.tile_datasets({"train": [pair_a, pair_b]}, window, progress_cb=messages.append)

    per_slice = [m for m in messages if m.startswith("train slice")]
    assert len(per_slice) == 2
    assert "1/2" in per_slice[0] and "2/2" in per_slice[1]
    # The final per-split summary line is still emitted too.
    assert any(m.startswith("train: 2 slice(s)") for m in messages)


def test_tile_datasets_stops_between_slices_when_cancelled() -> None:
    """Regression: tiling had no cancellation check at all — a long tiling pass
    on a large annotated set could not be stopped once started."""
    window = 64
    pairs = [_annotated_pair(200, 200, (slice(0, 100), slice(0, 100))) for _ in range(5)]
    calls = {"n": 0}

    def _cancel_after_first() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # let slice 0 through, then stop

    result = tiling.tile_datasets({"train": pairs}, window, cancel_cb=_cancel_after_first)

    assert result is None


def test_tile_datasets_never_cancelled_when_cancel_cb_always_false() -> None:
    window = 64
    rgb, label = _annotated_pair(200, 200, (slice(0, 100), slice(0, 100)))

    result = tiling.tile_datasets({"train": [(rgb, label)]}, window, cancel_cb=lambda: False)

    assert result is not None
    assert len(result["train"]) > 0


# ---------------------------------------------------------------------------
# Validation holdout
# ---------------------------------------------------------------------------


def _patches(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.full((4, 4, 3), i, dtype=np.uint8), np.full((4, 4), i, dtype=np.uint8)) for i in range(n)]


def test_holdout_moves_a_share_of_train_into_empty_val() -> None:
    datasets = {"train": _patches(30), "val": []}

    out, n_held = tiling.holdout_val_patches(datasets, seed=1234)

    assert n_held == 3  # 10% of 30
    assert len(out["val"]) == 3
    assert len(out["train"]) == 27
    # Every original patch still present exactly once, none duplicated across splits.
    ids = [int(p[0][0, 0, 0]) for p in out["train"] + out["val"]]
    assert sorted(ids) == list(range(30))


def test_holdout_leaves_a_populated_val_split_alone() -> None:
    """An explicit or auto-assigned validation slice must never be second-guessed."""
    datasets = {"train": _patches(30), "val": _patches(4)}

    out, n_held = tiling.holdout_val_patches(datasets, seed=1234)

    assert n_held == 0
    assert out is datasets


def test_holdout_declines_when_there_are_too_few_patches() -> None:
    """Below the threshold the lost training signal outweighs the metric."""
    datasets = {"train": _patches(5), "val": []}

    out, n_held = tiling.holdout_val_patches(datasets, seed=1234)

    assert n_held == 0
    assert len(out["train"]) == 5


def test_holdout_is_deterministic_for_a_seed() -> None:
    a, _ = tiling.holdout_val_patches({"train": _patches(30), "val": []}, seed=7)
    b, _ = tiling.holdout_val_patches({"train": _patches(30), "val": []}, seed=7)
    c, _ = tiling.holdout_val_patches({"train": _patches(30), "val": []}, seed=8)

    val_ids = lambda d: [int(p[0][0, 0, 0]) for p in d["val"]]  # noqa: E731
    assert val_ids(a) == val_ids(b)
    assert val_ids(a) != val_ids(c)


def test_holdout_never_empties_train() -> None:
    """Guard the degenerate case where the fraction would take everything."""
    datasets = {"train": _patches(10), "val": []}

    out, n_held = tiling.holdout_val_patches(datasets, seed=1, fraction=1.0, min_patches=1)

    assert n_held == 0
    assert len(out["train"]) == 10


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _identity_to_tensor(rgb: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0


def _two_class_forward(batch):
    """Class 1 where the red channel is bright, class 0 otherwise — a rule that
    can be evaluated independently on the full image to check the seams."""
    red = batch[:, 0]
    logits = torch.stack([1.0 - red, red], dim=1)
    return logits * 10.0  # confident enough to clear min_confidence


def test_streaming_accumulation_matches_qlty_stitch() -> None:
    """The whole reason the streamed path is allowed to exist: identical output
    to qlty's own stitch, at bounded memory."""
    from qlty.qlty2D import NCYXQuilt

    window, height, width = 64, 150, 130
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

    step = tiling.step_for(window)
    quilt = NCYXQuilt(
        Y=height, X=width, window=(window, window), step=(step, step),
        border=tiling.border_for(window), border_weight=tiling.BORDER_WEIGHT,
    )
    image = _identity_to_tensor(rgb).unsqueeze(0)
    patches = quilt.unstitch(image)
    reference, _ = quilt.stitch(_two_class_forward(patches))
    expected = reference[0].softmax(dim=0).max(dim=0)

    label = tiling.predict_label_map_tiled(
        rgb, forward_fn=_two_class_forward, to_tensor_fn=_identity_to_tensor,
        window=window, min_confidence=0.0, device="cpu",
    )

    expected_label = (expected.indices + 1).to(torch.uint8).numpy()
    assert label.shape == (height, width)
    np.testing.assert_array_equal(label, expected_label)


def test_tiled_prediction_agrees_with_whole_image_rule() -> None:
    """With a per-pixel forward rule, blending must reproduce that same rule
    everywhere — including across window seams."""
    window, height, width = 64, 150, 130
    rgb = np.random.default_rng(2).integers(0, 255, (height, width, 3), dtype=np.uint8)

    label = tiling.predict_label_map_tiled(
        rgb, forward_fn=_two_class_forward, to_tensor_fn=_identity_to_tensor,
        window=window, min_confidence=0.0, device="cpu",
    )

    red = _identity_to_tensor(rgb)[0]
    whole = torch.where(red > 1.0 - red, 2, 1).to(torch.uint8).numpy()
    np.testing.assert_array_equal(label, whole)


def test_tiled_prediction_crops_padding_for_a_small_image() -> None:
    rgb = np.random.default_rng(3).integers(0, 255, (30, 20, 3), dtype=np.uint8)

    label = tiling.predict_label_map_tiled(
        rgb, forward_fn=_two_class_forward, to_tensor_fn=_identity_to_tensor,
        window=64, min_confidence=0.0, device="cpu",
    )

    assert label.shape == (30, 20)


def test_min_confidence_gates_to_background() -> None:
    """Below-threshold pixels become 0 (background), matching the untiled path."""
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)

    def unconfident(batch):
        return torch.zeros((batch.shape[0], 3, batch.shape[2], batch.shape[3]))

    label = tiling.predict_label_map_tiled(
        rgb, forward_fn=unconfident, to_tensor_fn=_identity_to_tensor,
        window=64, min_confidence=0.9, device="cpu",
    )

    assert (label == 0).all()  # softmax of equal logits = 1/3 < 0.9


def test_cancellation_returns_none() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)

    label = tiling.predict_label_map_tiled(
        rgb, forward_fn=_two_class_forward, to_tensor_fn=_identity_to_tensor,
        window=64, min_confidence=0.5, device="cpu", cancel_cb=lambda: True,
    )

    assert label is None


def test_label_values_use_the_pipeline_convention() -> None:
    """0 = background, classes are 1-based — what _vectorize_label_map expects."""
    rgb = np.full((80, 80, 3), 255, dtype=np.uint8)

    label = tiling.predict_label_map_tiled(
        rgb, forward_fn=_two_class_forward, to_tensor_fn=_identity_to_tensor,
        window=64, min_confidence=0.5, device="cpu",
    )

    assert label.dtype == np.uint8
    assert set(np.unique(label)).issubset({0, 1, 2})
    assert (label == 2).all()  # bright red -> class index 1 -> label 2


# ---------------------------------------------------------------------------
# Denoising (regression) — shares `tiling._blend_tiled_forward` with the
# segmentation path above; these tests exercise it through
# `denoise_image_tiled`, which stops at the blended canvas (no softmax/argmax).
# ---------------------------------------------------------------------------


def _gradient_rgb(height: int, width: int) -> np.ndarray:
    """A smooth synthetic image (2-D gradient): no texture of its own, so any
    reconstruction error comes from the tiling/blending machinery rather than
    from image content a filter would legitimately treat unevenly."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    plane = (yy / max(height - 1, 1) + xx / max(width - 1, 1)) / 2.0
    return (np.stack([plane] * 3, axis=-1) * 255.0).astype(np.uint8)


def test_denoise_image_tiled_matches_a_simple_transform_applied_whole() -> None:
    """A pointwise transform (`batch * 0.5`) must reconstruct exactly, the same
    way segmentation's per-pixel-rule test does above — except this is
    `denoise_image_tiled` (a *regression* forward_fn), so the result is the
    blended canvas itself with no softmax/argmax on top."""
    window, height, width = 64, 150, 130
    rgb = _gradient_rgb(height, width)

    denoised = tiling.denoise_image_tiled(
        rgb, forward_fn=lambda batch: batch * 0.5, to_tensor_fn=_identity_to_tensor,
        window=window, device="cpu",
    )

    expected = (_identity_to_tensor(rgb) * 0.5).numpy()
    assert denoised.shape == (3, height, width)  # forward_fn kept all 3 channels
    np.testing.assert_allclose(denoised, expected, atol=1e-5)


def test_denoise_image_tiled_squeezes_a_single_output_channel() -> None:
    """A real denoiser emits 1 channel; the result should read like a plain
    image (H, W), not a (1, H, W) array a caller has to know to squeeze."""
    rgb = _gradient_rgb(80, 80)

    denoised = tiling.denoise_image_tiled(
        rgb, forward_fn=lambda batch: batch.mean(dim=1, keepdim=True) + 1.0,
        to_tensor_fn=_identity_to_tensor, window=64, device="cpu",
    )

    assert denoised.shape == (80, 80)


# How far into a window's own edge the synthetic "less context near the
# border" inaccuracy below reaches, and how large it gets right at the edge.
# Deliberately much bigger than the smooth gradient's own pixel-to-pixel
# variation, so a real seam would be unmistakable rather than lost in noise.
_EDGE_BIAS_RADIUS = 3
_EDGE_BIAS_MAX = 3.0


def _edge_biased_forward(batch: "torch.Tensor") -> "torch.Tensor":
    """Stand-in 'model' with a deliberately large, explicit window-edge
    inaccuracy: it adds a bias that ramps from :data:`_EDGE_BIAS_MAX` right at
    its OWN edge down to 0 a few pixels in — modelling, in an easy-to-reason-
    about way, a real conv net's reduced accuracy near its input patch's
    border (less surrounding context). Run per-window (as the tiled path
    does), that 'own edge' is the WINDOW boundary, not the true image edge —
    exactly the inaccuracy qlty's border down-weighting + overlap exist to
    hide. Collapses RGB to 1 channel, denoiser-shaped.
    """
    _, _, h, w = batch.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist_to_edge = torch.minimum(torch.minimum(yy, h - 1 - yy), torch.minimum(xx, w - 1 - xx)).float()
    bias = torch.clamp(_EDGE_BIAS_MAX - dist_to_edge, min=0.0)
    return batch.mean(dim=1, keepdim=True) + bias


def test_denoise_image_tiled_hides_window_edge_seams() -> None:
    """The real regression test for "the feathering still works": with a
    forward_fn whose accuracy genuinely depends on distance from ITS OWN
    edge (see :func:`_edge_biased_forward`), reconstruct a smooth gradient and
    compare against the SAME rule run once over the whole image — the ideal a
    seamless tiling should approach, where the bias only ever appears at the
    true image border.

    Away from the true image border, error near a former window edge must not
    be anywhere near the injected bias's magnitude — that is exactly what the
    down-weighted border ring plus a neighbour's full-weight interior are for.
    (Right at the true image edge the bias is intrinsic — even a non-tiled
    forward pass sees it there — so that margin is excluded from both masks.)

    Sanity-checked against a deliberately broken comparison (no window overlap
    at all, i.e. no blending): that configuration leaves ~30-40% of the raw
    bias at these same pixels, roughly 3-8x worse than the assertions below
    allow, confirming this test would actually catch a broken border/overlap
    invariant rather than passing vacuously.
    """
    window, height, width = 64, 200, 200
    rgb = _gradient_rgb(height, width)

    denoised = tiling.denoise_image_tiled(
        rgb, forward_fn=_edge_biased_forward, to_tensor_fn=_identity_to_tensor,
        window=window, device="cpu",
    )

    whole = _identity_to_tensor(rgb).unsqueeze(0)
    expected = _edge_biased_forward(whole)[0, 0].numpy()
    err = np.abs(denoised - expected)

    # Classify every pixel by its distance to the nearest FORMER window edge
    # (any tile origin or origin + window) on each axis.
    step = tiling.step_for(window)
    border = tiling.border_for(window)
    y_origins = tiling.tile_origins(height, window, step)
    x_origins = tiling.tile_origins(width, window, step)

    def edge_distance(size: int, origins: list[int]) -> np.ndarray:
        coords = np.arange(size)
        edges = np.array(sorted({o for o in origins} | {o + window for o in origins}))
        return np.min(np.abs(coords[:, None] - edges[None, :]), axis=1)

    near_window_edge = (edge_distance(height, y_origins)[:, None] <= border) | (
        edge_distance(width, x_origins)[None, :] <= border
    )

    margin = border + 2  # true image border, excluded from both masks
    away_from_image_border = np.zeros((height, width), dtype=bool)
    away_from_image_border[margin : height - margin, margin : width - margin] = True  # noqa: E203

    boundary_mask = near_window_edge & away_from_image_border
    interior_mask = (~near_window_edge) & away_from_image_border
    assert boundary_mask.sum() > 0 and interior_mask.sum() > 0  # the test geometry actually covers both

    boundary_err = float(err[boundary_mask].mean())
    interior_err = float(err[interior_mask].mean())

    # Interior pixels (far from every window's own edge) reconstruct exactly:
    # nothing there ever saw the bias in the first place.
    assert interior_err < 1e-3
    # Boundary pixels DID see up to _EDGE_BIAS_MAX of injected bias from at
    # least one contributing window, but blending must suppress the vast
    # majority of it — see the module docstring's measured comparison above.
    assert boundary_err < 0.15 * _EDGE_BIAS_MAX


def test_denoise_image_tiled_crops_padding_for_a_small_image() -> None:
    rgb = _gradient_rgb(30, 20)

    denoised = tiling.denoise_image_tiled(
        rgb, forward_fn=lambda batch: batch.mean(dim=1, keepdim=True),
        to_tensor_fn=_identity_to_tensor, window=64, device="cpu",
    )

    assert denoised.shape == (30, 20)


def test_denoise_image_tiled_cancellation_returns_none() -> None:
    rgb = _gradient_rgb(200, 200)

    denoised = tiling.denoise_image_tiled(
        rgb, forward_fn=lambda batch: batch.mean(dim=1, keepdim=True),
        to_tensor_fn=_identity_to_tensor, window=64, device="cpu", cancel_cb=lambda: True,
    )

    assert denoised is None
