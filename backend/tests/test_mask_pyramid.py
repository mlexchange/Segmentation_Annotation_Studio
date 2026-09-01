"""Unit tests for mask_pyramid's pure downsampling (no Tiled I/O)."""
import numpy as np

from mask_pyramid import build_mask_pyramid, majority_downsample


def test_majority_downsample_picks_the_more_frequent_class():
    # A 2x4x4 volume, factor (1,2,2): each output voxel covers a 1x2x2 block.
    # Top-left block is all class 1 except one class-2 voxel — 1 wins.
    block = np.zeros((1, 4, 4), dtype=np.uint8)
    block[0, 0:2, 0:2] = 1
    block[0, 0, 0] = 2  # one dissenting voxel inside that block
    out = majority_downsample(block, [1, 2, 2])
    assert out.shape == (1, 2, 2)
    assert out[0, 0, 0] == 1


def test_majority_downsample_uniform_block():
    volume = np.full((2, 4, 4), 3, dtype=np.uint8)
    out = majority_downsample(volume, [2, 2, 2])
    assert out.shape == (1, 2, 2)
    assert np.all(out == 3)


def test_majority_downsample_never_averages():
    # Two classes, 50/50 split within a block: averaging would invent a value
    # (e.g. 1.5 -> rounds to 2), which must never happen — the winner must be
    # one of the actual present class ids.
    block = np.zeros((1, 2, 2), dtype=np.uint8)
    block[0, 0, 0] = 5
    block[0, 0, 1] = 9
    out = majority_downsample(block, [1, 1, 2])
    assert out[0, 0, 0] in (5, 9)


def test_majority_downsample_tie_favors_lower_class_id():
    block = np.zeros((1, 1, 2), dtype=np.uint8)
    block[0, 0, 0] = 7
    block[0, 0, 1] = 3
    out = majority_downsample(block, [1, 1, 2])
    assert out[0, 0, 0] == 3


def test_majority_downsample_rejects_factor_below_one():
    import pytest

    with pytest.raises(ValueError):
        majority_downsample(np.zeros((2, 2, 2), dtype=np.uint8), [0, 1, 1])


def test_build_mask_pyramid_scale0_is_the_original_array():
    semantic = np.zeros((4, 8, 8), dtype=np.uint8)
    semantic[:, 2:6, 2:6] = 1
    levels, generated = build_mask_pyramid(semantic)
    assert "scale0" in levels
    assert np.array_equal(levels["scale0"], semantic)
    # 8x8 is already tiny — pyramid_plan should generate nothing further.
    assert generated == []


def test_build_mask_pyramid_generates_coarser_levels_for_a_large_volume():
    semantic = np.zeros((4, 4096, 4096), dtype=np.uint8)
    semantic[:, :2048, :2048] = 1
    levels, generated = build_mask_pyramid(semantic)
    assert len(generated) > 0
    for level in generated:
        assert level["path"] in levels
        assert levels[level["path"]].dtype == np.uint8
        # Downsampled level must only contain class ids actually present.
        assert set(np.unique(levels[level["path"]])).issubset({0, 1})
