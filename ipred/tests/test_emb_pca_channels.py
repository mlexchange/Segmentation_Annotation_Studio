"""Encoder PCA → full-res heatmap channels."""

from __future__ import annotations

import numpy as np

from ipred import sam_embed


def test_emb_grid_to_pca_channels_shape() -> None:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(8, 8, 16)).astype(np.float32)
    up, labels, info = sam_embed.emb_grid_to_pca_channels(
        emb, out_h=32, out_w=40, n_components=8
    )
    assert up.shape == (32, 40, 8)
    assert labels == [f"pca{i}" for i in range(8)]
    assert info["pca_dims"] == 8
    assert info["baked_into_float_stack"] is True


def test_emb_grid_pca_caps_at_rank() -> None:
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(4, 4, 5)).astype(np.float32)
    up, labels, info = sam_embed.emb_grid_to_pca_channels(
        emb, out_h=16, out_w=16, n_components=64
    )
    assert up.shape[-1] == 5
    assert len(labels) == 5
    assert info["pca_dims"] == 5
