"""Tests for shared image utilities."""

import numpy as np
import pytest

from peeklet.utils.image import (
    compute_block_grid,
    crop_region,
    dhash_64,
    ensure_rgb_uint8,
    hamming_distance,
)
from peeklet.utils.types import Region


class TestEnsureRgbUint8:
    def test_passthrough_valid_rgb(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_convert_rgba_to_rgb(self) -> None:
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        result = ensure_rgb_uint8(frame)
        assert result.shape == (100, 100, 3)

    def test_convert_grayscale_to_rgb(self) -> None:
        frame = np.zeros((100, 100), dtype=np.uint8)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (100, 100, 3)

    def test_convert_float_to_uint8(self) -> None:
        frame = np.ones((100, 100, 3), dtype=np.float64) * 0.5
        result = ensure_rgb_uint8(frame)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 127 or result[0, 0, 0] == 128

    def test_reject_invalid_ndim(self) -> None:
        frame = np.zeros((100,), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            ensure_rgb_uint8(frame)

    def test_reject_invalid_channels(self) -> None:
        frame = np.zeros((100, 100, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 1, 3, or 4 channels"):
            ensure_rgb_uint8(frame)


class TestCropRegion:
    def test_crop_extracts_correct_area(self) -> None:
        frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        region = Region(x=10, y=20, w=30, h=40)
        cropped = crop_region(frame, region)
        assert cropped.shape == (40, 30, 3)
        np.testing.assert_array_equal(cropped, frame[20:60, 10:40])

    def test_crop_clamps_to_bounds(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        region = Region(x=80, y=80, w=50, h=50)
        cropped = crop_region(frame, region)
        assert cropped.shape == (20, 20, 3)


class TestComputeBlockGrid:
    def test_grid_dimensions(self) -> None:
        rows, cols = compute_block_grid(1080, 1920, block_size=32)
        assert rows == 1080 // 32
        assert cols == 1920 // 32

    def test_grid_with_non_divisible(self) -> None:
        rows, cols = compute_block_grid(100, 100, block_size=32)
        assert rows == 3
        assert cols == 3

    def test_block_size_larger_than_image(self) -> None:
        rows, cols = compute_block_grid(16, 16, block_size=32)
        assert rows == 0
        assert cols == 0


class TestDhash64:
    def test_identical_frames_produce_identical_hash(self) -> None:
        frame = np.random.default_rng(0).integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        assert dhash_64(frame) == dhash_64(frame.copy())

    def test_different_frames_produce_different_hash(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        b = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        assert dhash_64(a) != dhash_64(b)

    def test_hash_is_64_bit_integer(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        h = dhash_64(frame)
        assert isinstance(h, int)
        assert 0 <= h < (1 << 64)

    def test_handles_grayscale_input(self) -> None:
        frame = np.zeros((50, 50), dtype=np.uint8)
        assert dhash_64(frame) == 0  # uniform frame → all zero diffs


class TestHammingDistance:
    def test_identical_hashes_distance_zero(self) -> None:
        assert hamming_distance(0xDEADBEEF, 0xDEADBEEF) == 0

    def test_single_bit_flip(self) -> None:
        assert hamming_distance(0b1010, 0b1011) == 1

    def test_fully_different(self) -> None:
        assert hamming_distance(0, 0xFFFFFFFFFFFFFFFF) == 64
