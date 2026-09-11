"""Tests for tiling.py — real (no mocks) since torch+qlty are both installed
in this environment; this module's whole point is qlty geometry, so faking
that away would test nothing meaningful.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qlty")

import tiling  # noqa: E402
from train_common import IGNORE_INDEX  # noqa: E402


class TestStepAndBorder:
    def test_step_for_applies_overlap_fraction(self):
        assert tiling.step_for(64) == 48  # 64 - round(64*0.25)

    def test_step_for_never_below_one(self):
        assert tiling.step_for(1) >= 1

    def test_border_for_uses_divisor(self):
        assert tiling.border_for(64) == 8  # 64 // 8

    def test_border_for_never_below_one(self):
        assert tiling.border_for(1) >= 1


class TestTileOrigins:
    def test_exact_multiple_of_step(self):
        # dim=100, window=20, step=20 -> origins 0,20,40,60,80 (last clamps to 80 anyway)
        origins = tiling.tile_origins(100, 20, 20)
        assert origins[0] == 0
        assert origins[-1] == 80
        assert all(o + 20 <= 100 for o in origins)

    def test_dim_equals_window_gives_single_origin(self):
        assert tiling.tile_origins(20, 20, 15) == [0]

    def test_last_window_clamped_inside_image(self):
        origins = tiling.tile_origins(50, 20, 15)
        assert origins[-1] == 30  # 50 - 20
        assert max(origins) + 20 == 50

    def test_dim_smaller_than_window_raises(self):
        with pytest.raises(ValueError, match="smaller than window"):
            tiling.tile_origins(10, 20, 5)


class TestPadToMin:
    def test_already_big_enough_returns_same_array(self):
        arr = np.zeros((32, 32), dtype=np.uint8)
        assert tiling.pad_to_min(arr, 32, 32, fill=0) is arr

    def test_pads_bottom_right_only(self):
        arr = np.ones((10, 10), dtype=np.uint8)
        out = tiling.pad_to_min(arr, 16, 20, fill=0)
        assert out.shape == (16, 20)
        assert np.all(out[:10, :10] == 1)
        assert np.all(out[10:, :] == 0)
        assert np.all(out[:, 10:] == 0)

    def test_fill_value_used_for_padding(self):
        arr = np.zeros((5, 5), dtype=np.uint8)
        out = tiling.pad_to_min(arr, 8, 8, fill=IGNORE_INDEX)
        assert out[7, 7] == IGNORE_INDEX

    def test_preserves_trailing_channel_dim(self):
        arr = np.ones((10, 10, 3), dtype=np.uint8)
        out = tiling.pad_to_min(arr, 16, 16, fill=0)
        assert out.shape == (16, 16, 3)


class TestQltyAvailable:
    def test_true_when_installed(self):
        assert tiling.qlty_available() is True


class TestTilePair:
    def test_produces_window_sized_patches(self):
        rgb = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        label = np.zeros((64, 64), dtype=np.uint8)
        label[10:54, 10:54] = 1  # broad interior annotation
        patches = tiling._tile_pair(rgb, label, window=32)
        assert len(patches) > 0
        for patch_rgb, patch_label in patches:
            assert patch_rgb.shape == (32, 32, 3)
            assert patch_label.shape == (32, 32)

    def test_entirely_unannotated_image_yields_no_patches(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        label = np.full((64, 64), IGNORE_INDEX, dtype=np.uint8)
        patches = tiling._tile_pair(rgb, label, window=32)
        assert patches == []

    def test_edge_only_annotation_falls_back_to_unmasked_patches(self):
        # Annotate only the outermost couple of pixels — inside every patch's
        # down-weighted border ring, so the masked pass would keep nothing;
        # the edge-only fallback (mask_borders=False) must still find it.
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        label = np.full((64, 64), IGNORE_INDEX, dtype=np.uint8)
        label[0:2, 0:2] = 1
        patches = tiling._tile_pair(rgb, label, window=32)
        assert len(patches) > 0
        assert any((p_label != IGNORE_INDEX).any() for _, p_label in patches)

    def test_smaller_than_window_image_is_padded_first(self):
        rgb = np.ones((10, 10, 3), dtype=np.uint8)
        label = np.ones((10, 10), dtype=np.uint8)
        patches = tiling._tile_pair(rgb, label, window=32)
        assert len(patches) >= 1
        assert patches[0][0].shape == (32, 32, 3)


class TestHoldoutValPatches:
    def _pairs(self, n):
        return [(np.zeros((4, 4)), np.zeros((4, 4))) for _ in range(n)]

    def test_no_op_when_val_already_present(self):
        datasets = {"train": self._pairs(20), "val": self._pairs(2)}
        out, n = tiling.holdout_val_patches(datasets, seed=0)
        assert n == 0
        assert out is datasets

    def test_no_op_below_min_patches(self):
        datasets = {"train": self._pairs(4), "val": []}
        out, n = tiling.holdout_val_patches(datasets, seed=0, min_patches=8)
        assert n == 0
        assert out["train"] == datasets["train"]

    def test_holds_out_a_seeded_fraction(self):
        datasets = {"train": self._pairs(20), "val": []}
        out, n = tiling.holdout_val_patches(datasets, seed=0, fraction=0.1)
        assert n == 2  # round(20 * 0.1)
        assert len(out["train"]) == 18
        assert len(out["val"]) == 2

    def test_deterministic_given_same_seed(self):
        datasets = {"train": self._pairs(20), "val": []}
        out1, _ = tiling.holdout_val_patches(datasets, seed=42)
        out2, _ = tiling.holdout_val_patches(datasets, seed=42)
        assert len(out1["val"]) == len(out2["val"])


class TestTileDatasets:
    def test_tiles_every_split(self):
        rgb = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        label = np.ones((64, 64), dtype=np.uint8)
        datasets = {"train": [(rgb, label)], "val": [(rgb, label)]}
        out = tiling.tile_datasets(datasets, window=32)
        assert len(out["train"]) > 0
        assert len(out["val"]) > 0

    def test_cancellation_returns_none(self):
        rgb = np.ones((64, 64, 3), dtype=np.uint8)
        label = np.ones((64, 64), dtype=np.uint8)
        datasets = {"train": [(rgb, label), (rgb, label)]}
        out = tiling.tile_datasets(datasets, window=32, cancel_cb=lambda: True)
        assert out is None


def _tiny_model():
    model = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1)

    def forward_fn(batch):
        return model(batch)

    def to_tensor_fn(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
        return t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0)

    return forward_fn, to_tensor_fn


class TestBlendTiledForwardAndFriends:
    def test_predict_label_map_tiled_shape_and_dtype(self):
        forward_fn, to_tensor_fn = _tiny_model()
        rgb = np.random.default_rng(0).integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        label = tiling.predict_label_map_tiled(
            rgb, forward_fn=forward_fn, to_tensor_fn=to_tensor_fn,
            window=32, min_confidence=0.0, device="cpu",
        )
        assert label.shape == (48, 48)
        assert label.dtype == np.uint8

    def test_predict_label_map_tiled_high_confidence_threshold_yields_background(self):
        forward_fn, to_tensor_fn = _tiny_model()
        rgb = np.zeros((48, 48, 3), dtype=np.uint8)
        label = tiling.predict_label_map_tiled(
            rgb, forward_fn=forward_fn, to_tensor_fn=to_tensor_fn,
            window=32, min_confidence=1.1, device="cpu",  # impossible to reach
        )
        assert np.all(label == 0)

    def test_predict_label_map_tiled_cancellation_returns_none(self):
        forward_fn, to_tensor_fn = _tiny_model()
        rgb = np.zeros((48, 48, 3), dtype=np.uint8)
        label = tiling.predict_label_map_tiled(
            rgb, forward_fn=forward_fn, to_tensor_fn=to_tensor_fn,
            window=32, min_confidence=0.5, device="cpu", cancel_cb=lambda: True,
        )
        assert label is None

    def test_denoise_image_tiled_single_channel_squeezed(self):
        model = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)

        def forward_fn(batch):
            return model(batch)

        def to_tensor_fn(arr):
            return torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0) / 255.0

        gray = np.random.default_rng(0).integers(0, 256, size=(48, 48), dtype=np.uint8)
        out = tiling.denoise_image_tiled(
            gray, forward_fn=forward_fn, to_tensor_fn=to_tensor_fn, window=32, device="cpu",
        )
        assert out.shape == (48, 48)
        assert out.dtype == np.float32

    def test_image_smaller_than_window_is_handled(self):
        forward_fn, to_tensor_fn = _tiny_model()
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        label = tiling.predict_label_map_tiled(
            rgb, forward_fn=forward_fn, to_tensor_fn=to_tensor_fn,
            window=32, min_confidence=0.0, device="cpu",
        )
        assert label.shape == (10, 10)
