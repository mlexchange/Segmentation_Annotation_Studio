"""Classical denoising filters for tomography slices.

Ported from Alex's denoiser branch (``Segmentation_Annotation_Studio-feat-denoiser``)
essentially unchanged. The parameter mappings and the ``_AUTO_DAMPING`` table
below were arrived at by measurement, not taste — rewriting them would throw
that away.

Used by two callers that must agree exactly: the display preview
(``GET /api/image/slice``'s optional ``denoise_*`` params) and the batch
"denoise & save as a new dataset" job (``denoise_bake.py``). Keeping one
implementation here is the whole point — a JS copy for the preview would
inevitably drift from whatever the bake writes.

Denoising runs on the RAW slice, before ``images.normalize_scalar_unit``:
noise statistics live in the source's own intensity units, not in the 8-bit
display range.

Design notes
------------
* **One ``strength`` knob (0..1) per method.** The UI shows a single slider
  regardless of method; each method maps it onto its own native parameter.
  Callers wanting exact control pass ``extra``.
* **Parameters are scaled by the image's own estimated noise level** for the
  methods that need a noise scale (bilateral / TV / NLM / wavelet). Raw CT is
  uint16 with ranges in the thousands, so a hardcoded TV weight or NLM ``h``
  would be meaningless on one dataset and destructive on the next.
* **Filtering happens in a normalized [0, 1] copy**, then rescales back to the
  source range and dtype. This makes every tuned constant below dimensionless
  and portable across dtypes, and avoids skimage's various assumptions about
  float images being in [0, 1].
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 2-D methods, plus the two 3-D ones that filter ACROSS slices — the cheap,
# training-free way to exploit the fact that adjacent tomographic slices share
# structure while their noise is independent.
METHODS_2D = ("gaussian", "median", "bilateral", "tv", "nlm", "wavelet")
METHODS_3D = ("gaussian3d", "median3d")
ALL_METHODS = ("none",) + METHODS_2D + METHODS_3D

# How many z-neighbours each 3-D method needs on EACH side of the target slice.
_Z_RADIUS = {"gaussian3d": 2, "median3d": 1}

# Immerkaer's Laplacian mask. Its noise gain is analytically known, which is
# what lets `estimate_noise_sigma` be both robust and dependency-free.
_IMMERKAER_MASK = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])


def wavelet_available() -> bool:
    """True if wavelet denoising can actually run.

    ``skimage.restoration.denoise_wavelet`` imports fine without PyWavelets but
    raises at CALL time, so availability has to be probed rather than assumed.
    (Same for ``estimate_sigma``, which is why this module ships its own
    noise estimator instead — see :func:`estimate_noise_sigma`.)
    """
    try:
        import pywt  # noqa: F401,PLC0415
    except Exception:
        return False
    return True


def available_methods() -> tuple[str, ...]:
    """Methods that can actually run in this environment."""
    if wavelet_available():
        return ALL_METHODS
    return tuple(m for m in ALL_METHODS if m != "wavelet")


def z_radius_for(method: str) -> int:
    """Z-neighbours needed on each side of the target slice (0 for 2-D methods)."""
    return _Z_RADIUS.get(method, 0)


def estimate_noise_sigma(arr: np.ndarray) -> float:
    """Estimate additive noise sigma via Immerkaer's fast Laplacian method.

    ``skimage.restoration.estimate_sigma`` would be the obvious choice but it
    requires PyWavelets, which is not installed here. Immerkaer's estimator
    needs only a convolution: the mask above responds almost entirely to noise
    rather than to structure, and its gain is known analytically, so

        sigma = sqrt(pi/2) * sum(|mask * image|) / (6 * (W-2) * (H-2))

    Verified accurate to ~2% for sigma from 10 to 200 on a step-edge phantom,
    and on a pure-noise image (i.e. it isn't fooled by having no structure).

    Returns 0.0 for degenerate (too small) inputs so callers can fall back.
    """
    from scipy.ndimage import convolve  # noqa: PLC0415

    data = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    h, w = data.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    conv = convolve(data, _IMMERKAER_MASK, mode="reflect")
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def _odd(value: float, minimum: int = 3) -> int:
    """Nearest odd integer >= minimum (rank filters need an odd window)."""
    size = int(round(value))
    if size % 2 == 0:
        size += 1
    return max(minimum, size)


def _to_unit(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Scale to [0, 1]. Returns ``(unit, offset, span)`` for the inverse."""
    data = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(data.min())
    hi = float(data.max())
    span = hi - lo
    if span <= 0:
        return np.zeros_like(data), lo, 0.0
    return (data - lo) / span, lo, span


def _from_unit(unit: np.ndarray, offset: float, span: float, dtype: np.dtype) -> np.ndarray:
    """Inverse of :func:`_to_unit`, clipped into *dtype*'s range."""
    if span <= 0:
        restored = np.full_like(unit, offset, dtype=np.float64)
    else:
        restored = unit * span + offset
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        restored = np.clip(np.round(restored), info.min, info.max)
    return restored.astype(dtype)


def denoise_slice(
    arr: np.ndarray,
    method: str,
    strength: float = 0.5,
    extra: dict[str, Any] | None = None,
) -> np.ndarray:
    """Denoise one 2-D slice, preserving shape and dtype.

    Args:
        arr: 2-D slice in its native dtype (uint16, float32, ...).
        method: One of :data:`METHODS_2D`, or ``"none"`` (returns *arr*).
        strength: 0..1, mapped onto the method's native parameter.
        extra: Per-method overrides, bypassing the ``strength`` mapping.

    Raises:
        ValueError: unknown *method*, a 3-D-only method (use
            :func:`denoise_stack`), or wavelet without PyWavelets installed.
    """
    if method == "none":
        return arr
    if method in METHODS_3D:
        raise ValueError(f"{method!r} needs z-neighbours — call denoise_stack()")
    if method not in METHODS_2D:
        raise ValueError(f"unknown denoise method {method!r}")
    if arr.ndim != 2:
        raise ValueError(f"denoise_slice expects a 2-D slice, got shape {arr.shape}")

    opts = dict(extra or {})
    s = float(np.clip(strength, 0.0, 1.0))
    unit, offset, span = _to_unit(arr)
    if span <= 0:
        return arr  # flat slice — nothing to denoise
    # Noise scale in the SAME normalized units the filters below work in.
    sigma_n = opts.get("sigma") or estimate_noise_sigma(unit) or 0.01

    if method == "gaussian":
        from scipy.ndimage import gaussian_filter  # noqa: PLC0415

        out = gaussian_filter(unit, sigma=opts.get("sigma_spatial", 0.3 + s * 3.0))
    elif method == "median":
        from scipy.ndimage import median_filter  # noqa: PLC0415

        out = median_filter(unit, size=opts.get("size", _odd(3 + s * 6)))
    elif method == "bilateral":
        from skimage.restoration import denoise_bilateral  # noqa: PLC0415

        out = denoise_bilateral(
            unit,
            sigma_color=opts.get("sigma_color", sigma_n * (0.5 + s * 2.5)),
            sigma_spatial=opts.get("sigma_spatial", 1.0 + s * 4.0),
        )
    elif method == "tv":
        from skimage.restoration import denoise_tv_chambolle  # noqa: PLC0415

        out = denoise_tv_chambolle(unit, weight=opts.get("weight", sigma_n * (0.3 + s * 3.0)))
    elif method == "nlm":
        from skimage.restoration import denoise_nl_means  # noqa: PLC0415

        out = denoise_nl_means(
            unit,
            h=opts.get("h", sigma_n * (0.4 + s * 1.6)),
            sigma=opts.get("noise_sigma", sigma_n),
            patch_size=opts.get("patch_size", 5),
            patch_distance=opts.get("patch_distance", 6),
            fast_mode=True,
            channel_axis=None,
        )
    else:  # wavelet
        if not wavelet_available():
            raise ValueError(
                "wavelet denoising needs PyWavelets — install it (pip install PyWavelets) "
                "or pick another method"
            )
        from skimage.restoration import denoise_wavelet  # noqa: PLC0415

        out = denoise_wavelet(
            unit,
            sigma=opts.get("noise_sigma", sigma_n * (0.5 + s)),
            mode="soft",
            method="BayesShrink",
            rescale_sigma=True,
        )

    return _from_unit(np.asarray(out, dtype=np.float64), offset, span, arr.dtype)


def denoise_stack(
    frames: np.ndarray,
    method: str,
    strength: float = 0.5,
    extra: dict[str, Any] | None = None,
) -> np.ndarray:
    """Denoise a ``(z, y, x)`` stack with a 3-D filter, preserving shape/dtype.

    This is the training-free way to exploit slice-to-slice correlation:
    adjacent tomographic slices show nearly the same structure while their
    noise is independent, so averaging along z suppresses noise at a far lower
    cost in real detail than the equivalent in-plane blur.

    Callers previewing a single slice pass a z-window and keep the centre frame
    (see :func:`z_radius_for`); the bake job passes larger chunks.

    Raises:
        ValueError: *method* is not one of :data:`METHODS_3D`.
    """
    if method not in METHODS_3D:
        raise ValueError(f"{method!r} is not a 3-D denoise method")
    if frames.ndim != 3:
        raise ValueError(f"denoise_stack expects (z, y, x), got shape {frames.shape}")

    opts = dict(extra or {})
    s = float(np.clip(strength, 0.0, 1.0))
    unit, offset, span = _to_unit(frames)
    if span <= 0:
        return frames

    if method == "gaussian3d":
        from scipy.ndimage import gaussian_filter  # noqa: PLC0415

        sigma_xy = opts.get("sigma_spatial", 0.3 + s * 2.0)
        # z gets less smoothing than xy by default: slice spacing is usually
        # coarser than pixel pitch, so equal sigma would blur genuine
        # through-plane structure more than it blurs noise.
        sigma_z = opts.get("sigma_z", sigma_xy * 0.6)
        out = gaussian_filter(unit, sigma=(sigma_z, sigma_xy, sigma_xy))
    else:  # median3d
        size_xy = opts.get("size", _odd(3 + s * 4))
        size_z = opts.get("size_z", 3)
        out = _median3d(unit, size_z, size_xy)

    return _from_unit(np.asarray(out, dtype=np.float64), offset, span, frames.dtype)


def _median3d(unit: np.ndarray, size_z: int, size_xy: int) -> np.ndarray:
    from scipy.ndimage import median_filter  # noqa: PLC0415

    size_z = min(_odd(size_z), unit.shape[0] if unit.shape[0] % 2 == 1 else unit.shape[0] - 1)
    size_z = max(1, size_z)
    return median_filter(unit, size=(size_z, size_xy, size_xy))


# How far to push each method at a given measured noise level. Edge-preserving
# methods can be driven hard because their strength buys noise reduction
# without eating boundaries; a plain Gaussian trades the two off directly, so
# the same nominal strength that helps TV visibly destroys edges here (measured:
# on a step-edge phantom at strength 0.5, TV and NLM RAISE SNR while Gaussian
# LOWERS it below the noisy input). "Auto" must not hand the user a setting
# that makes the image worse.
_AUTO_DAMPING = {
    "gaussian": 0.35,
    "gaussian3d": 0.5,
    "median": 0.6,
    "median3d": 0.6,
    "bilateral": 0.9,
    "wavelet": 0.9,
    "tv": 1.0,
    "nlm": 1.0,
}


def auto_strength(arr: np.ndarray, method: str) -> float:
    """Suggest a ``strength`` for *arr* from its own estimated noise level.

    Powers the UI's "Auto" button. Maps the measured noise-to-range ratio onto
    0..1 with a gentle curve, then damps it per method (see
    :data:`_AUTO_DAMPING`). Deliberately conservative — over-smoothing destroys
    the very boundaries this app exists to annotate, and the user can always
    push the slider further.
    """
    if method == "none":
        return 0.0
    unit, _, span = _to_unit(arr if arr.ndim == 2 else arr[arr.shape[0] // 2])
    if span <= 0:
        return 0.0
    sigma_n = estimate_noise_sigma(unit)
    # sigma_n is a fraction of the full dynamic range. ~0.5% reads as clean,
    # ~5%+ as heavily noisy; sqrt keeps the low end from collapsing to zero.
    ratio = float(np.clip((sigma_n - 0.005) / 0.045, 0.0, 1.0))
    return round(float(np.sqrt(ratio)) * 0.8 * _AUTO_DAMPING.get(method, 0.8), 3)


def describe_methods() -> list[dict[str, Any]]:
    """Method metadata for the capability endpoint / UI menu.

    ``cost`` drives whether the frontend warns about latency before requesting
    a full-resolution preview.
    """
    info = [
        ("none", "None", "cheap", "No denoising."),
        ("gaussian", "Gaussian", "cheap",
         "Simple blur. Fast baseline, but softens edges — usually the weakest choice here."),
        ("median", "Median", "cheap",
         "Removes salt-and-pepper speckle, zingers and dead pixels that blurring cannot."),
        ("bilateral", "Bilateral", "moderate",
         "Edge-preserving smoothing: averages only over similar-intensity neighbours."),
        ("tv", "Total variation", "moderate",
         "Edge-preserving; excellent on piecewise-constant material regions."),
        ("nlm", "Non-local means", "slow",
         "Averages similar patches from across the slice. Best detail preservation, slowest."),
        ("wavelet", "Wavelet", "cheap",
         "Wavelet-shrinkage denoising (needs PyWavelets installed)."),
        ("gaussian3d", "Gaussian 3D", "moderate",
         "Smooths across neighbouring slices too — uses slice-to-slice correlation, no training."),
        ("median3d", "Median 3D", "slow",
         "Median across neighbouring slices — strong on speckle while keeping in-plane edges."),
    ]
    usable = set(available_methods())
    return [
        {"method": m, "label": label, "cost": cost, "description": desc,
         "available": m in usable, "z_radius": z_radius_for(m)}
        for m, label, cost, desc in info
    ]
