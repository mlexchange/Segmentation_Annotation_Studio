"""Tests for the classical denoise filters.

Ported alongside :mod:`denoise` itself. The properties pinned here are the ones
a user would notice if they broke: that a filter actually reduces noise without
destroying the edges this app exists to annotate, that shape and dtype survive
the round trip through normalized units, and that "Auto" never hands back a
setting that makes the image worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import denoise  # noqa: E402


def step_phantom(h=192, w=192, noise=12.0, seed=3):
    """A hard vertical edge plus additive Gaussian noise.

    The standard shape for asking "did it denoise, or did it just blur?" — the
    edge is what a bad filter destroys.
    """
    rng = np.random.default_rng(seed)
    clean = np.zeros((h, w), np.float32)
    clean[:, w // 2:] = 100.0
    return clean, clean + rng.normal(0, noise, clean.shape).astype(np.float32)


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    return float(10 * np.log10(clean.var() / ((test - clean) ** 2).mean()))


class TestDenoiseSlice:
    @pytest.mark.parametrize("method", ["gaussian", "median", "bilateral", "tv", "nlm"])
    def test_improves_snr_on_a_noisy_edge(self, method):
        clean, noisy = step_phantom()
        out = denoise.denoise_slice(noisy, method, 0.5).astype(np.float32)
        assert snr_db(clean, out) > snr_db(clean, noisy)

    @pytest.mark.parametrize("method", ["gaussian", "median", "tv"])
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_preserves_shape_and_dtype(self, method, dtype):
        # Filtering happens in normalized [0,1] floats; the round trip back to
        # the source dtype is where clipping and rounding bugs would live.
        rng = np.random.default_rng(1)
        arr = (rng.random((48, 64)) * 200).astype(dtype)
        out = denoise.denoise_slice(arr, method, 0.5)
        assert out.shape == arr.shape
        assert out.dtype == arr.dtype

    def test_none_is_an_identity(self):
        _, noisy = step_phantom(h=32, w=32)
        assert denoise.denoise_slice(noisy, "none", 1.0) is noisy

    def test_flat_slice_is_returned_untouched(self):
        # Zero dynamic range: normalizing would divide by zero.
        flat = np.full((32, 32), 7, np.uint16)
        assert np.array_equal(denoise.denoise_slice(flat, "tv", 0.5), flat)

    def test_stronger_settings_smooth_more(self):
        _, noisy = step_phantom()
        weak = denoise.denoise_slice(noisy, "gaussian", 0.1).astype(np.float32)
        strong = denoise.denoise_slice(noisy, "gaussian", 0.9).astype(np.float32)
        assert strong.std() < weak.std()

    def test_rejects_a_3d_method(self):
        _, noisy = step_phantom(h=16, w=16)
        with pytest.raises(ValueError, match="denoise_stack"):
            denoise.denoise_slice(noisy, "gaussian3d", 0.5)

    def test_rejects_an_unknown_method(self):
        _, noisy = step_phantom(h=16, w=16)
        with pytest.raises(ValueError, match="unknown"):
            denoise.denoise_slice(noisy, "bogus", 0.5)

    def test_rejects_a_3d_array(self):
        with pytest.raises(ValueError, match="2-D"):
            denoise.denoise_slice(np.zeros((3, 8, 8), np.float32), "tv", 0.5)


class TestDenoiseStack:
    def test_uses_z_neighbours_to_beat_the_2d_equivalent(self):
        # The whole argument for the 3-D filters: adjacent slices share
        # structure while their noise is independent, so averaging along z buys
        # noise reduction at a far lower cost in real detail than in-plane blur.
        rng = np.random.default_rng(5)
        clean2d = np.zeros((96, 96), np.float32)
        clean2d[:, 48:] = 100.0
        clean = np.repeat(clean2d[None], 5, axis=0)
        noisy = clean + rng.normal(0, 15, clean.shape).astype(np.float32)

        out3d = denoise.denoise_stack(noisy, "gaussian3d", 0.5).astype(np.float32)[2]
        out2d = denoise.denoise_slice(noisy[2], "gaussian", 0.5).astype(np.float32)
        assert snr_db(clean2d, out3d) > snr_db(clean2d, out2d)

    @pytest.mark.parametrize("method", ["gaussian3d", "median3d"])
    def test_preserves_shape_and_dtype(self, method):
        rng = np.random.default_rng(2)
        stack = (rng.random((5, 32, 32)) * 500).astype(np.uint16)
        out = denoise.denoise_stack(stack, method, 0.5)
        assert out.shape == stack.shape
        assert out.dtype == stack.dtype

    def test_rejects_a_2d_method(self):
        with pytest.raises(ValueError, match="not a 3-D"):
            denoise.denoise_stack(np.zeros((3, 8, 8), np.float32), "tv", 0.5)

    def test_rejects_a_2d_array(self):
        with pytest.raises(ValueError, match=r"\(z, y, x\)"):
            denoise.denoise_stack(np.zeros((8, 8), np.float32), "gaussian3d", 0.5)


class TestNoiseEstimate:
    def test_tracks_the_true_sigma(self):
        # Immerkaer's estimator, on a flat field where the answer is known.
        rng = np.random.default_rng(9)
        for sigma in (0.01, 0.05, 0.1):
            field = rng.normal(0.5, sigma, (256, 256))
            estimate = denoise.estimate_noise_sigma(field)
            assert abs(estimate - sigma) / sigma < 0.1

    def test_is_not_fooled_by_structure(self):
        # A clean step edge is structure, not noise; the estimate must stay low
        # or "Auto" would recommend smoothing a noiseless image.
        clean, _ = step_phantom(noise=0.0)
        unit, _, span = denoise._to_unit(clean)
        assert denoise.estimate_noise_sigma(unit) < 0.02

    def test_degenerate_input_returns_zero(self):
        assert denoise.estimate_noise_sigma(np.zeros((2, 2))) == 0.0


class TestAutoStrength:
    def test_suggests_more_for_a_noisier_image(self):
        _, quiet = step_phantom(noise=2.0)
        _, loud = step_phantom(noise=30.0)
        assert denoise.auto_strength(loud, "tv") > denoise.auto_strength(quiet, "tv")

    def test_never_makes_a_clean_image_worse(self):
        # The failure this guards: "Auto" on a clean slice returning a strength
        # that visibly softens it.
        clean, _ = step_phantom(noise=0.0)
        assert denoise.auto_strength(clean, "tv") == pytest.approx(0.0, abs=0.05)

    def test_damps_gaussian_below_the_edge_preserving_methods(self):
        # Measured: at equal nominal strength a plain Gaussian can LOWER SNR
        # where TV raises it, so Auto must not push it as hard.
        _, noisy = step_phantom(noise=25.0)
        assert denoise.auto_strength(noisy, "gaussian") < denoise.auto_strength(noisy, "tv")

    def test_none_is_zero(self):
        _, noisy = step_phantom()
        assert denoise.auto_strength(noisy, "none") == 0.0

    def test_stays_in_range(self):
        for noise in (0.0, 1.0, 50.0, 500.0):
            _, noisy = step_phantom(noise=noise)
            for method in denoise.available_methods():
                assert 0.0 <= denoise.auto_strength(noisy, method) <= 1.0


class TestCapabilities:
    def test_describes_every_method(self):
        described = {m["method"] for m in denoise.describe_methods()}
        assert described == set(denoise.ALL_METHODS)

    def test_availability_is_probed_not_assumed(self):
        # denoise_wavelet imports fine without PyWavelets and only fails when
        # called, so the menu has to probe rather than trust the import.
        for entry in denoise.describe_methods():
            if entry["method"] == "wavelet":
                assert entry["available"] == denoise.wavelet_available()

    def test_only_3d_methods_need_z_neighbours(self):
        for entry in denoise.describe_methods():
            expected = entry["method"] in denoise.METHODS_3D
            assert (entry["z_radius"] > 0) is expected

    def test_unavailable_methods_are_excluded_from_available_methods(self):
        usable = set(denoise.available_methods())
        assert usable <= set(denoise.ALL_METHODS)
        if not denoise.wavelet_available():
            assert "wavelet" not in usable
