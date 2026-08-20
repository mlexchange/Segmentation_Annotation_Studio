"""Tests for the classical denoising filters in ``denoise.py``.

These pin down the properties the UI and the bake job depend on: shape/dtype
are preserved exactly (the bake writes back into a same-dtype Tiled array), the
noise estimator is accurate without PyWavelets, and each method actually does
the thing its menu entry claims — which is NOT the same claim for every method.

Note the deliberate asymmetry in the noise-reduction assertions: on a
step-edge phantom, edge-preserving methods (TV/NLM/bilateral) raise global SNR,
while a plain Gaussian LOWERS it because the edge blur costs more than the
noise it removes. So blur methods are asserted to reduce noise in FLAT regions
(what they genuinely do) rather than globally (what they don't).
"""

from __future__ import annotations

import numpy as np
import pytest

import denoise

# ---------------------------------------------------------------------------
# Phantoms
# ---------------------------------------------------------------------------

SIGMA = 250.0
LEVEL = 3000.0


def _clean(h: int = 96, w: int = 96) -> np.ndarray:
    """Step-edge phantom: a bright square on a dark field, like a grain slice."""
    img = np.zeros((h, w), dtype=np.float64)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = LEVEL
    return img


def _noisy(clean: np.ndarray, sigma: float = SIGMA, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 65535).astype(np.uint16)


def _global_snr(out: np.ndarray, clean: np.ndarray) -> float:
    return float(np.std(clean) / np.std(out.astype(np.float64) - clean))


def _flat_noise(out: np.ndarray, clean: np.ndarray) -> float:
    """Residual std inside the phantom's flat interior only (no edges)."""
    h, w = clean.shape
    interior = (slice(h // 4 + 6, 3 * h // 4 - 6), slice(w // 4 + 6, 3 * w // 4 - 6))
    return float(np.std(out.astype(np.float64)[interior] - clean[interior]))


# ---------------------------------------------------------------------------
# Contract: shape, dtype, passthrough
# ---------------------------------------------------------------------------

USABLE_2D = [m for m in denoise.METHODS_2D if m in denoise.available_methods()]


@pytest.mark.parametrize("method", USABLE_2D)
def test_every_2d_method_preserves_shape_and_dtype(method: str) -> None:
    """The bake job writes results back into a same-dtype Tiled array, so a
    filter that silently returned float64 would change the stored dataset."""
    noisy = _noisy(_clean())

    out = denoise.denoise_slice(noisy, method, 0.5)

    assert out.shape == noisy.shape
    assert out.dtype == noisy.dtype


@pytest.mark.parametrize("method", denoise.METHODS_3D)
def test_every_3d_method_preserves_shape_and_dtype(method: str) -> None:
    clean = np.repeat(_clean(64, 64)[None, :, :], 7, axis=0)
    noisy = _noisy(clean)

    out = denoise.denoise_stack(noisy, method, 0.5)

    assert out.shape == noisy.shape
    assert out.dtype == noisy.dtype


def test_none_returns_the_input_untouched() -> None:
    noisy = _noisy(_clean())
    assert denoise.denoise_slice(noisy, "none") is noisy


def test_a_flat_slice_is_returned_unchanged_rather_than_dividing_by_zero() -> None:
    """A constant slice has zero dynamic range; the [0,1] rescale would divide
    by zero, so it must short-circuit instead."""
    flat = np.full((32, 32), 500, dtype=np.uint16)

    for method in USABLE_2D:
        out = denoise.denoise_slice(flat, method, 0.8)
        assert np.array_equal(out, flat), method


def test_uint16_output_never_leaves_the_dtype_range() -> None:
    """Filters run in float; the cast back must clip, not wrap around."""
    extreme = np.full((32, 32), 65535, dtype=np.uint16)
    extreme[::2, ::2] = 0

    out = denoise.denoise_slice(extreme, "gaussian", 1.0)

    assert out.min() >= 0
    assert out.max() <= 65535


def test_float32_input_stays_float32() -> None:
    noisy = _noisy(_clean()).astype(np.float32)
    out = denoise.denoise_slice(noisy, "tv", 0.5)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Behaviour: each method does what its menu entry claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["tv", "nlm", "bilateral"])
def test_edge_preserving_methods_raise_global_snr(method: str) -> None:
    """These are the methods recommended for piecewise-constant CT data, so
    they must improve the whole image, edges included."""
    clean = _clean()
    noisy = _noisy(clean)

    out = denoise.denoise_slice(noisy, method, 0.5)

    assert _global_snr(out, clean) > _global_snr(noisy, clean)


@pytest.mark.parametrize("method", ["gaussian", "median"])
def test_blur_methods_reduce_noise_in_flat_regions(method: str) -> None:
    """A plain blur genuinely removes noise but pays for it at edges — asserted
    where it actually helps, not globally (see the module docstring)."""
    clean = _clean()
    noisy = _noisy(clean)

    out = denoise.denoise_slice(noisy, method, 0.5)

    assert _flat_noise(out, clean) < _flat_noise(noisy, clean)


def test_tv_preserves_the_edge_that_gaussian_destroys() -> None:
    """The load-bearing distinction behind recommending TV over Gaussian for
    this data: compare the recovered gradient magnitude across the boundary."""
    clean = _clean()
    noisy = _noisy(clean)
    row = clean.shape[0] // 2

    tv = denoise.denoise_slice(noisy, "tv", 0.6).astype(np.float64)[row]
    gauss = denoise.denoise_slice(noisy, "gaussian", 0.6).astype(np.float64)[row]

    # Steepest single-pixel step along the profile: the true edge is a cliff.
    assert np.abs(np.diff(tv)).max() > np.abs(np.diff(gauss)).max()


def test_median_removes_impulse_spikes_that_gaussian_only_smears() -> None:
    """Zingers / dead pixels are a real CT artifact class; this is why median
    is in the menu alongside the smoothing filters."""
    clean = _clean()
    spiked = clean.copy()
    rng = np.random.default_rng(3)
    ys = rng.integers(30, 66, 40)
    xs = rng.integers(30, 66, 40)
    spiked[ys, xs] = 60000.0  # isolated hot pixels inside the bright square
    spiked16 = spiked.astype(np.uint16)

    med = denoise.denoise_slice(spiked16, "median", 0.3).astype(np.float64)
    gauss = denoise.denoise_slice(spiked16, "gaussian", 0.3).astype(np.float64)

    # Worst remaining excursion above the true plateau.
    assert np.abs(med[ys, xs] - LEVEL).max() < np.abs(gauss[ys, xs] - LEVEL).max()


def test_3d_filtering_beats_its_2d_counterpart_on_the_same_frame() -> None:
    """The point of the 3-D methods: adjacent slices share structure but not
    noise, so averaging along z removes noise more cheaply than in-plane blur.
    This is the training-free version of the Noise2Noise idea."""
    clean = np.repeat(_clean(64, 64)[None, :, :], 7, axis=0)
    noisy = _noisy(clean)
    mid = noisy.shape[0] // 2

    out2d = denoise.denoise_slice(noisy[mid], "gaussian", 0.5)
    out3d = denoise.denoise_stack(noisy, "gaussian3d", 0.5)[mid]

    frame_clean = clean[mid]
    assert _global_snr(out3d, frame_clean) > _global_snr(out2d, frame_clean)


def test_3d_result_actually_depends_on_the_neighbouring_slices() -> None:
    """Guards against a 3-D method that silently degenerates to a per-slice
    filter (which would make the z-window reads pure waste)."""
    clean = np.repeat(_clean(48, 48)[None, :, :], 5, axis=0)
    noisy = _noisy(clean)
    mid = noisy.shape[0] // 2

    baseline = denoise.denoise_stack(noisy, "gaussian3d", 0.5)[mid]
    perturbed = noisy.copy()
    perturbed[mid + 1] = 0  # change ONLY a neighbour, never the target frame
    changed = denoise.denoise_stack(perturbed, "gaussian3d", 0.5)[mid]

    assert not np.array_equal(baseline, changed)


def test_stronger_strength_smooths_more() -> None:
    clean = _clean()
    noisy = _noisy(clean)

    light = denoise.denoise_slice(noisy, "gaussian", 0.1)
    heavy = denoise.denoise_slice(noisy, "gaussian", 0.9)

    assert _flat_noise(heavy, clean) < _flat_noise(light, clean)


def test_extra_overrides_the_strength_mapping() -> None:
    """Power users (and the bake job replaying stored params) need exact control."""
    noisy = _noisy(_clean())

    mapped = denoise.denoise_slice(noisy, "gaussian", 1.0)
    overridden = denoise.denoise_slice(noisy, "gaussian", 1.0, {"sigma_spatial": 0.1})

    assert not np.array_equal(mapped, overridden)


# ---------------------------------------------------------------------------
# Noise estimation (PyWavelets-free)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [10.0, 40.0, 80.0, 200.0])
def test_noise_sigma_estimate_is_accurate_across_two_orders_of_magnitude(sigma: float) -> None:
    """``skimage.restoration.estimate_sigma`` needs PyWavelets, which is not
    installed, so this module ships Immerkaer's estimator instead. It has to be
    trustworthy: every noise-scaled parameter (TV weight, NLM h, bilateral
    sigma_color) is derived from it."""
    rng = np.random.default_rng(1)
    noisy = _clean(256, 256) + rng.normal(0, sigma, (256, 256))

    estimated = denoise.estimate_noise_sigma(noisy)

    assert estimated == pytest.approx(sigma, rel=0.15)


def test_noise_estimate_is_not_fooled_by_a_structureless_image() -> None:
    rng = np.random.default_rng(2)
    pure_noise = rng.normal(500, 300, (256, 256))

    assert denoise.estimate_noise_sigma(pure_noise) == pytest.approx(300, rel=0.15)


def test_noise_estimate_degrades_gracefully_on_a_tiny_image() -> None:
    """The analytic gain divides by (W-2)(H-2), so anything under 3x3 must
    return a sentinel rather than divide by zero."""
    assert denoise.estimate_noise_sigma(np.ones((2, 2))) == 0.0


# ---------------------------------------------------------------------------
# Auto strength
# ---------------------------------------------------------------------------

def test_auto_strength_rises_with_measured_noise() -> None:
    clean = _clean(128, 128)
    quiet = denoise.auto_strength(_noisy(clean, sigma=30), "tv")
    loud = denoise.auto_strength(_noisy(clean, sigma=600), "tv")

    assert 0.0 <= quiet < loud <= 1.0


def test_auto_strength_is_damped_hardest_for_the_edge_destroying_filters() -> None:
    """"Auto" must never hand back a setting that makes the image worse. At the
    same measured noise level, Gaussian is pushed far less than TV/NLM."""
    noisy = _noisy(_clean(128, 128), sigma=400)

    assert denoise.auto_strength(noisy, "gaussian") < denoise.auto_strength(noisy, "tv")
    assert denoise.auto_strength(noisy, "gaussian") < denoise.auto_strength(noisy, "nlm")


def test_auto_strength_for_none_is_zero() -> None:
    assert denoise.auto_strength(_noisy(_clean()), "none") == 0.0


def test_auto_strength_accepts_a_3d_stack_and_samples_its_centre() -> None:
    stack = _noisy(np.repeat(_clean(64, 64)[None, :, :], 5, axis=0))
    assert 0.0 <= denoise.auto_strength(stack, "gaussian3d") <= 1.0


# ---------------------------------------------------------------------------
# Validation and capability reporting
# ---------------------------------------------------------------------------

def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown denoise method"):
        denoise.denoise_slice(_noisy(_clean()), "definitely-not-a-filter")


def test_a_3d_method_passed_to_denoise_slice_says_to_use_denoise_stack() -> None:
    with pytest.raises(ValueError, match="denoise_stack"):
        denoise.denoise_slice(_noisy(_clean()), "gaussian3d")


def test_a_2d_method_passed_to_denoise_stack_is_rejected() -> None:
    stack = _noisy(np.repeat(_clean(32, 32)[None, :, :], 3, axis=0))
    with pytest.raises(ValueError, match="not a 3-D denoise method"):
        denoise.denoise_stack(stack, "tv")


def test_wrong_dimensionality_is_rejected_by_both_entry_points() -> None:
    with pytest.raises(ValueError, match="2-D slice"):
        denoise.denoise_slice(np.zeros((3, 8, 8), dtype=np.uint16), "gaussian")
    with pytest.raises(ValueError, match=r"\(z, y, x\)"):
        denoise.denoise_stack(np.zeros((8, 8), dtype=np.uint16), "gaussian3d")


def test_wavelet_is_excluded_from_available_methods_without_pywavelets() -> None:
    """PyWavelets is an optional dependency that skimage imports lazily, so
    ``denoise_wavelet`` raises at CALL time rather than import time. The menu
    must reflect what can actually run."""
    available = denoise.available_methods()

    assert ("wavelet" in available) == denoise.wavelet_available()
    if not denoise.wavelet_available():
        with pytest.raises(ValueError, match="PyWavelets"):
            denoise.denoise_slice(_noisy(_clean()), "wavelet")


def test_z_radius_is_zero_for_2d_methods_and_positive_for_3d() -> None:
    for method in denoise.METHODS_2D:
        assert denoise.z_radius_for(method) == 0, method
    for method in denoise.METHODS_3D:
        assert denoise.z_radius_for(method) > 0, method


def test_describe_methods_covers_every_method_with_ui_metadata() -> None:
    described = denoise.describe_methods()

    assert {d["method"] for d in described} == set(denoise.ALL_METHODS)
    for entry in described:
        assert entry["cost"] in {"cheap", "moderate", "slow"}
        assert entry["description"]
        assert entry["available"] == (entry["method"] in denoise.available_methods())
