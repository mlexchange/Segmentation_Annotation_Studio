"""Tests for multiscale feature labels + CLAHE stack (skimage)."""

from __future__ import annotations

import numpy as np
import pytest

from multiscale_features import (
    compute_feature_stack,
    feature_channel_labels,
    list_sigmas,
)


def test_list_sigmas_powers_of_two() -> None:
    assert list(list_sigmas(1.0, 8.0)) == pytest.approx([1.0, 2.0, 4.0, 8.0])


def test_label_count_matches_feature_kinds() -> None:
    sigmas = list_sigmas(1.0, 8.0)
    labels = feature_channel_labels(
        sigma_min=1.0, sigma_max=8.0, intensity=True, edges=True, texture=True
    )
    # Per σ: intensity + edges + 2 texture eigenvalues
    assert len(labels) == len(sigmas) * 4
    assert labels[0] == "intensity σ=1"
    assert labels[1] == "edges σ=1"
    assert "texture λ− σ=1" in labels[2]
    assert labels[4].startswith("intensity σ=2")


def test_compute_stack_shape_and_uint8_range() -> None:
    rng = np.random.default_rng(0)
    gray = rng.random((48, 48), dtype=np.float64)
    stack, labels = compute_feature_stack(
        gray,
        sigma_min=1.0,
        sigma_max=4.0,
        intensity=True,
        edges=True,
        texture=True,
        clahe=True,
    )
    assert stack.dtype == np.uint8
    assert stack.shape == (48, 48, len(labels))
    assert stack.min() >= 0
    assert stack.max() <= 255
    # CLAHE should use most of the dynamic range on structured noise
    assert int(stack.max()) - int(stack.min()) > 50


def test_intensity_only_fewer_channels() -> None:
    gray = np.linspace(0, 1, 32 * 32, dtype=np.float64).reshape(32, 32)
    stack, labels = compute_feature_stack(
        gray,
        sigma_min=1.0,
        sigma_max=2.0,
        intensity=True,
        edges=False,
        texture=False,
        clahe=False,
    )
    assert len(labels) == 2
    assert stack.shape[-1] == 2
