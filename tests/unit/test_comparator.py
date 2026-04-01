"""Tests for SSIM comparison and block-level diff."""

import numpy as np

from peeklet.core.comparator import ComparisonResult, compare_frames


class TestCompareFrames:
    def test_identical_frames_high_ssim(self, sample_frame: np.ndarray) -> None:
        result = compare_frames(sample_frame, sample_frame.copy(), block_size=32)
        assert result.ssim_score > 0.99
        assert result.changed_pct == 0.0
        assert result.changed_regions == []

    def test_completely_different_frames_low_ssim(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        result = compare_frames(sample_frame, sample_frame_big_change, block_size=32)
        assert result.ssim_score < 0.5
        assert result.changed_pct > 50.0

    def test_small_change_moderate_ssim(
        self, sample_frame: np.ndarray, sample_frame_small_change: np.ndarray
    ) -> None:
        result = compare_frames(sample_frame, sample_frame_small_change, block_size=10)
        assert 0.5 < result.ssim_score < 1.0

    def test_change_score_inverse_of_ssim(self, sample_frame: np.ndarray) -> None:
        different = sample_frame.copy()
        different[:50, :50] = 255 - different[:50, :50]
        result = compare_frames(sample_frame, different, block_size=32)
        assert abs(result.change_score - (1.0 - result.ssim_score)) < 0.01

    def test_changed_regions_have_valid_bounds(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        result = compare_frames(sample_frame, sample_frame_big_change, block_size=32)
        h, w = sample_frame.shape[:2]
        for region in result.changed_regions:
            assert region.x >= 0
            assert region.y >= 0
            assert region.x + region.w <= w + 32
            assert region.y + region.h <= h + 32

    def test_changed_pct_between_0_and_100(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        result = compare_frames(sample_frame, sample_frame_big_change, block_size=32)
        assert 0.0 <= result.changed_pct <= 100.0


class TestComparisonResult:
    def test_dataclass_fields(self) -> None:
        result = ComparisonResult(
            ssim_score=0.85,
            change_score=0.15,
            changed_pct=12.5,
            changed_regions=[],
        )
        assert result.ssim_score == 0.85
