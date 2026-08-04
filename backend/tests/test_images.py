"""Numerical-reference tests for slice rendering normalisation.

Regression coverage for a domain-mismatch bug: global-normalised log/symlog
renders used to clamp already-transformed pixel data against untransformed
raw global bounds, collapsing the image toward black. See images.render_slice.
"""

from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from images import render_slice

_DEFAULT_OPTS = {"vmin_pct": 1.0, "vmax_pct": 99.0, "cmap": "gray"}


def _opts(norm: str, scale: str) -> dict:
    return {**_DEFAULT_OPTS, "norm": norm, "scale": scale}


def test_global_log_uses_full_output_range() -> None:
    """A slice spanning the full sampled global range should use black-to-white."""
    arr = np.array([[0.0, 10_000.0]], dtype=np.float64)
    global_range = (0.0, 10_000.0)  # matches the slice's own raw min/max

    rgb = render_slice(arr, _opts("global", "log"), global_range=global_range)

    assert rgb[0, 0, 0] == 0  # darkest pixel maps to the bottom of the range
    assert rgb[0, 1, 0] == 255  # brightest pixel maps to the top of the range


def test_global_symlog_uses_full_output_range() -> None:
    """Symmetric negative/positive extremes should also span black-to-white."""
    arr = np.array([[-5_000.0, 5_000.0]], dtype=np.float64)
    global_range = (-5_000.0, 5_000.0)

    rgb = render_slice(arr, _opts("global", "symlog"), global_range=global_range)

    assert rgb[0, 0, 0] == 0
    assert rgb[0, 1, 0] == 255


def test_global_log_mid_value_is_not_collapsed_to_black() -> None:
    """A mid-range value under a wide global range must render mid-brightness,
    not near-black — the failure mode of comparing transformed pixels against
    untransformed bounds."""
    arr = np.array([[3_000.0]], dtype=np.float64)
    global_range = (0.0, 10_000.0)

    rgb = render_slice(arr, _opts("global", "log"), global_range=global_range)

    # log1p(3000) / log1p(10000) ≈ 0.865 — should be bright, not clamped dark.
    assert rgb[0, 0, 0] > 200


def test_global_linear_matches_previous_behavior() -> None:
    """Linear + global was already correct; the fix must not change it."""
    arr = np.array([[0.0, 50.0, 100.0]], dtype=np.float64)
    global_range = (0.0, 100.0)

    rgb = render_slice(arr, _opts("global", "linear"), global_range=global_range)

    assert rgb[0, 0, 0] == 0
    assert rgb[0, 2, 0] == 255
    assert 120 <= rgb[0, 1, 0] <= 135


def test_slice_log_normalisation_is_unchanged() -> None:
    """Slice-scoped log normalisation (own min/max) is untouched by the fix."""
    arr = np.array([[10.0, 1_000.0]], dtype=np.float64)

    rgb = render_slice(arr, _opts("slice", "log"), global_range=None)

    assert rgb[0, 0, 0] == 0
    assert rgb[0, 1, 0] == 255


def test_global_stats_skips_unreadable_sibling_slice(monkeypatch) -> None:
    """A "stack" container can group array children of different shapes (e.g.
    two unrelated single-image samples living side by side) — read_slice
    rejects the mismatched sibling. Sampling for global contrast must skip
    that sibling rather than failing the whole request over a slice nobody
    asked to view. Regression for a bug where opening slice 0 of such a stack
    422'd because sampling also touched the mismatched slice 1."""
    import arrays
    from images import _sample_global_stats, _stats_cache

    def fake_read_slice(node, meta, idx):
        if idx == 1:
            raise HTTPException(422, "Slice 1 shape (562, 780) does not match the stack's declared shape (906, 1038)")
        # 5x5 so the `[::4, ::4]` spatial subsample (real images are large; this
        # mimics it) still keeps both corners: (0,0)=0.0 and (4,4)=10.0.
        arr = np.zeros((5, 5))
        arr[4, 4] = 10.0
        return arr

    monkeypatch.setattr(arrays, "read_slice", fake_read_slice)
    _stats_cache.clear()

    node = object()  # unique id() so this test doesn't hit another test's cache entry
    vmin, vmax = _sample_global_stats(node, {"n_slices": 2})

    assert (vmin, vmax) == (0.0, 10.0)


def test_global_stats_raises_when_every_slice_is_unreadable(monkeypatch) -> None:
    """If NO sample slice is readable, still fail loudly instead of returning
    a bogus range."""
    import arrays
    from images import _sample_global_stats, _stats_cache

    def fake_read_slice(node, meta, idx):
        raise HTTPException(422, "bad slice")

    monkeypatch.setattr(arrays, "read_slice", fake_read_slice)
    _stats_cache.clear()

    node = object()
    try:
        _sample_global_stats(node, {"n_slices": 2})
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 422
