"""Edge and corner case tests for all Peeklet modules."""

import numpy as np
import pytest

from peeklet.core.comparator import ComparisonResult, compare_frames
from peeklet.core.hasher import compute_phash, hashes_match
from peeklet.core.masking import AdaptiveMask
from peeklet.utils.image import compute_block_grid, crop_region, ensure_rgb_uint8
from peeklet.utils.types import Region

# ── Image utility edge cases ──────────────────────────────────────────────


class TestEnsureRgbUint8EdgeCases:
    def test_single_channel_3d(self) -> None:
        frame = np.full((50, 50, 1), 128, dtype=np.uint8)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (50, 50, 3)
        assert np.all(result == 128)

    def test_float_out_of_range_clips(self) -> None:
        frame = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
        result = ensure_rgb_uint8(frame)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 255  # clipped to 1.0 * 255
        assert result[0, 0, 1] == 0  # clipped to 0.0

    def test_float_zero_and_one(self) -> None:
        frame = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float64)
        result = ensure_rgb_uint8(frame)
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 255

    def test_4d_array_raises(self) -> None:
        frame = np.zeros((1, 10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            ensure_rgb_uint8(frame)

    def test_2_channel_raises(self) -> None:
        frame = np.zeros((10, 10, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 1, 3, or 4 channels"):
            ensure_rgb_uint8(frame)

    def test_rgba_strips_alpha(self) -> None:
        frame = np.zeros((10, 10, 4), dtype=np.uint8)
        frame[:, :, 0] = 100
        frame[:, :, 3] = 200  # alpha should be dropped
        result = ensure_rgb_uint8(frame)
        assert result.shape == (10, 10, 3)
        assert result[0, 0, 0] == 100

    def test_grayscale_float(self) -> None:
        frame = np.full((10, 10), 0.5, dtype=np.float64)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8
        assert 127 <= result[0, 0, 0] <= 128

    def test_1x1_pixel(self) -> None:
        frame = np.array([[[42, 43, 44]]], dtype=np.uint8)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (1, 1, 3)
        np.testing.assert_array_equal(result[0, 0], [42, 43, 44])

    def test_uint16_converted(self) -> None:
        frame = np.full((10, 10, 3), 30000, dtype=np.uint16)
        result = ensure_rgb_uint8(frame)
        assert result.dtype == np.uint8


class TestCropRegionEdgeCases:
    def test_negative_coordinates_clamped(self) -> None:
        frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        region = Region(x=-10, y=-10, w=30, h=30)
        cropped = crop_region(frame, region)
        assert cropped.shape == (20, 20, 3)

    def test_region_entirely_outside(self) -> None:
        frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        region = Region(x=100, y=100, w=10, h=10)
        cropped = crop_region(frame, region)
        assert cropped.shape[0] == 0 or cropped.shape[1] == 0

    def test_zero_size_region(self) -> None:
        frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        region = Region(x=10, y=10, w=0, h=0)
        cropped = crop_region(frame, region)
        assert cropped.shape[0] == 0 or cropped.shape[1] == 0

    def test_region_covers_entire_frame(self) -> None:
        frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        region = Region(x=0, y=0, w=50, h=50)
        cropped = crop_region(frame, region)
        np.testing.assert_array_equal(cropped, frame)


class TestComputeBlockGridEdgeCases:
    def test_zero_height(self) -> None:
        rows, cols = compute_block_grid(0, 100, block_size=32)
        assert rows == 0

    def test_zero_width(self) -> None:
        rows, cols = compute_block_grid(100, 0, block_size=32)
        assert cols == 0

    def test_exact_multiple(self) -> None:
        rows, cols = compute_block_grid(64, 96, block_size=32)
        assert rows == 2
        assert cols == 3

    def test_very_small_block_size(self) -> None:
        rows, cols = compute_block_grid(10, 10, block_size=1)
        assert rows == 10
        assert cols == 10

    def test_1x1_image(self) -> None:
        rows, cols = compute_block_grid(1, 1, block_size=1)
        assert rows == 1
        assert cols == 1


# ── Loader edge cases ─────────────────────────────────────────────────────


# ── Hasher edge cases ─────────────────────────────────────────────────────


class TestHasherEdgeCases:
    def test_uniform_white_frame(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        h = compute_phash(frame)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_uniform_black_frame(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        h = compute_phash(frame)
        assert isinstance(h, str)

    def test_solid_colors_same_hash(self) -> None:
        """Different solid colors may produce same hash (no structural detail)."""
        red = np.full((100, 100, 3), 0, dtype=np.uint8)
        red[:, :, 0] = 255
        green = np.full((100, 100, 3), 0, dtype=np.uint8)
        green[:, :, 1] = 255
        # Both are uniform => same phash (this is expected; per-channel means disambiguate)
        h1 = compute_phash(red)
        h2 = compute_phash(green)
        assert h1 == h2  # proves need for mean disambiguation

    def test_1x1_frame(self) -> None:
        frame = np.array([[[128, 64, 32]]], dtype=np.uint8)
        h = compute_phash(frame)
        assert isinstance(h, str)

    def test_very_wide_frame(self) -> None:
        frame = np.full((10, 1000, 3), 128, dtype=np.uint8)
        h = compute_phash(frame)
        assert len(h) == 16

    def test_hashes_match_identical(self) -> None:
        assert hashes_match("abcdef0123456789", "abcdef0123456789", tolerance=0)

    def test_hashes_match_tolerance_boundary(self) -> None:
        # 0x0 vs 0x1 = 1 bit difference
        assert hashes_match("0000000000000000", "0000000000000001", tolerance=1)
        assert not hashes_match("0000000000000000", "0000000000000001", tolerance=0)


# ── Comparator edge cases ─────────────────────────────────────────────────


class TestComparatorEdgeCases:
    def test_identical_frames_zero_change(self) -> None:
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = compare_frames(frame, frame.copy(), block_size=32)
        assert result.ssim_score > 0.999
        assert result.change_score < 0.001
        assert result.changed_pct == 0.0

    def test_inverted_frame_maximal_change(self) -> None:
        frame = np.full((64, 64, 3), 0, dtype=np.uint8)
        inverted = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = compare_frames(frame, inverted, block_size=32)
        assert result.ssim_score < 0.1
        assert result.changed_pct == 100.0

    def test_block_size_larger_than_frame(self) -> None:
        """Block size > frame size => 0 blocks, no crash."""
        frame = np.full((16, 16, 3), 128, dtype=np.uint8)
        different = np.full((16, 16, 3), 0, dtype=np.uint8)
        result = compare_frames(frame, different, block_size=32)
        assert result.changed_pct == 0.0  # 0 blocks
        assert result.changed_regions == []

    def test_block_change_threshold_boundary(self) -> None:
        """Block with mean diff exactly at threshold."""
        frame_a = np.full((32, 32, 3), 100, dtype=np.uint8)
        frame_b = np.full((32, 32, 3), 110, dtype=np.uint8)  # diff = 10.0 exactly
        result = compare_frames(frame_a, frame_b, block_size=32, block_change_threshold=10.0)
        # mean diff = 10.0, threshold checks > 10.0, so should NOT be changed
        assert len(result.changed_regions) == 0

    def test_block_change_above_threshold(self) -> None:
        frame_a = np.full((32, 32, 3), 100, dtype=np.uint8)
        frame_b = np.full((32, 32, 3), 111, dtype=np.uint8)  # diff = 11.0
        result = compare_frames(frame_a, frame_b, block_size=32, block_change_threshold=10.0)
        assert len(result.changed_regions) == 1

    def test_small_block_size(self) -> None:
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        different = np.full((10, 10, 3), 0, dtype=np.uint8)
        result = compare_frames(frame, different, block_size=2)
        assert result.changed_pct == 100.0

    def test_change_score_is_complement_of_ssim(self) -> None:
        frame_a = np.full((64, 64, 3), 100, dtype=np.uint8)
        frame_b = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = compare_frames(frame_a, frame_b, block_size=32)
        assert abs(result.change_score - (1.0 - result.ssim_score)) < 1e-6


class TestComparisonResultEdgeCases:
    def test_empty_changed_regions(self) -> None:
        result = ComparisonResult(
            ssim_score=1.0, change_score=0.0, changed_pct=0.0, changed_regions=[]
        )
        assert len(result.changed_regions) == 0

    def test_frozen_dataclass(self) -> None:
        result = ComparisonResult(
            ssim_score=0.5, change_score=0.5, changed_pct=50.0, changed_regions=[]
        )
        with pytest.raises(AttributeError):
            result.ssim_score = 0.9  # type: ignore[misc]


# ── Masking edge cases ────────────────────────────────────────────────────


class TestAdaptiveMaskEdgeCases:
    def test_frame_smaller_than_block_size(self) -> None:
        """Frame smaller than block_size => 0 grid blocks, returns unmasked."""
        mask = AdaptiveMask(block_size=100, window_size=5, noise_threshold=0.8)
        frame = np.full((50, 50, 3), 128, dtype=np.uint8)
        for _ in range(10):
            result, regions = mask.apply(frame)
        assert regions == []
        np.testing.assert_array_equal(result, frame)

    def test_window_size_1(self) -> None:
        """Degenerate window_size=1 should still work."""
        mask = AdaptiveMask(block_size=50, window_size=1, noise_threshold=0.5)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask.apply(frame)  # first frame
        # Second frame with change
        frame2 = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame2[:50, :50] = 200
        mask.apply(frame2)  # second frame, should not mask yet (history < 2 won't happen
        # actually window_size=1 means deque maxlen=1, so len(history) never reaches 2
        # let's just ensure it doesn't crash
        result, regions = mask.apply(frame)
        # With window_size=1, change_history can only hold 1 item => never >= 2
        assert regions == []

    def test_noise_threshold_0(self) -> None:
        """With threshold 0, any change frequency > 0 should mask."""
        mask = AdaptiveMask(block_size=50, window_size=3, noise_threshold=0.0)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask.apply(frame)  # first frame

        for i in range(5):
            frame2 = frame.copy()
            frame2[:50, :50] = 100 + i * 10
            result, regions = mask.apply(frame2)

        # After enough history, any changing block should be masked
        assert len(regions) > 0

    def test_noise_threshold_1(self) -> None:
        """With threshold 1.0, nothing should ever be masked (freq must exceed 1.0)."""
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=1.0)
        for i in range(20):
            frame = np.full((100, 100, 3), i * 10, dtype=np.uint8)
            _, regions = mask.apply(frame)
        assert regions == []

    def test_dimension_change_resets_state(self) -> None:
        """Changing frame dimensions should reset masking state."""
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        for i in range(10):
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
            frame[:50, :50] = i * 25
            mask.apply(frame)

        # Change dimensions
        frame_new = np.full((200, 200, 3), 128, dtype=np.uint8)
        result, regions = mask.apply(frame_new)
        assert regions == []  # reset, treats as first frame

    def test_very_large_block_returns_empty(self) -> None:
        """Block size much larger than frame => no blocks."""
        mask = AdaptiveMask(block_size=1000, window_size=5, noise_threshold=0.8)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        for _ in range(10):
            result, regions = mask.apply(frame)
        assert regions == []

    def test_single_pixel_block(self) -> None:
        """block_size=1 — every pixel is its own block."""
        mask = AdaptiveMask(block_size=1, window_size=3, noise_threshold=0.5)
        frame = np.full((5, 5, 3), 128, dtype=np.uint8)
        mask.apply(frame)
        frame2 = np.full((5, 5, 3), 200, dtype=np.uint8)
        mask.apply(frame2)
        frame3 = np.full((5, 5, 3), 50, dtype=np.uint8)
        _, regions = mask.apply(frame3)
        # All pixels changed every frame, should be masked
        assert len(regions) == 25  # 5x5 blocks


# ── Config edge cases ─────────────────────────────────────────────────────


# ── Exporter edge cases ───────────────────────────────────────────────────


# ── Types edge cases ──────────────────────────────────────────────────────


class TestTypesEdgeCases:
    def test_region_zero_area(self) -> None:
        r = Region(x=5, y=5, w=0, h=0)
        assert r.area == 0

    def test_region_to_dict(self) -> None:
        r = Region(x=1, y=2, w=3, h=4)
        assert r.to_dict() == {"x": 1, "y": 2, "w": 3, "h": 4}

    def test_region_frozen(self) -> None:
        r = Region(x=0, y=0, w=10, h=10)
        with pytest.raises(AttributeError):
            r.x = 5  # type: ignore[misc]


# ── Pipeline integration edge cases ───────────────────────────────────────
