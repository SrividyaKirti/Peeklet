"""Tests for adaptive masking, perceptual hashing, and SSIM comparison."""

import numpy as np

from peeklet.change import (
    AdaptiveMask,
    ComparisonResult,
    compare_frames,
    compute_phash,
    hashes_match,
)


class TestAdaptiveMask:
    def test_init_creates_empty_state(self) -> None:
        mask = AdaptiveMask(block_size=32, window_size=5, noise_threshold=0.8)
        assert mask.block_size == 32
        assert mask.window_size == 5

    def test_first_frame_returns_unmasked(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result, regions = mask.apply(frame)
        np.testing.assert_array_equal(result, frame)
        assert regions == []

    def test_static_frames_no_mask(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        for _ in range(10):
            result, regions = mask.apply(frame)
        assert regions == []
        np.testing.assert_array_equal(result, frame)

    def test_constantly_changing_block_gets_masked(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        for i in range(10):
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
            frame[:50, :50] = i * 25
            result, regions = mask.apply(frame)
        assert len(regions) > 0
        assert np.all(result[:50, :50] == 0)
        assert np.all(result[50:, 50:] == 128)

    def test_noisy_region_returns_correct_regions(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        for i in range(10):
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
            frame[:50, :50] = i * 25
            _, regions = mask.apply(frame)
        assert any(r.x == 0 and r.y == 0 for r in regions)

    def test_window_slides(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        for i in range(10):
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
            frame[:50, :50] = i * 25
            mask.apply(frame)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        for _ in range(10):
            _, regions = mask.apply(frame)
        assert regions == []

    def test_all_blocks_noisy(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        for i in range(10):
            frame = np.full((100, 100, 3), i * 25, dtype=np.uint8)
            result, regions = mask.apply(frame)
        assert np.all(result == 0)


class TestComputePhash:
    def test_returns_string(self, sample_frame: np.ndarray) -> None:
        h = compute_phash(sample_frame)
        assert isinstance(h, str)
        assert len(h) == 16  # 64-bit hash as 16-char hex

    def test_identical_frames_same_hash(self, sample_frame: np.ndarray) -> None:
        h1 = compute_phash(sample_frame)
        h2 = compute_phash(sample_frame.copy())
        assert h1 == h2

    def test_small_change_same_hash(
        self, sample_frame: np.ndarray, sample_frame_small_change: np.ndarray
    ) -> None:
        h1 = compute_phash(sample_frame)
        h2 = compute_phash(sample_frame_small_change)
        assert h1 == h2

    def test_big_change_different_hash(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        h1 = compute_phash(sample_frame)
        h2 = compute_phash(sample_frame_big_change)
        assert h1 != h2

    def test_deterministic(self, sample_frame: np.ndarray) -> None:
        results = [compute_phash(sample_frame) for _ in range(5)]
        assert len(set(results)) == 1


class TestHashesMatch:
    def test_identical_hashes_match(self, sample_frame: np.ndarray) -> None:
        h = compute_phash(sample_frame)
        assert hashes_match(h, h)

    def test_different_hashes_no_match(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        h1 = compute_phash(sample_frame)
        h2 = compute_phash(sample_frame_big_change)
        assert not hashes_match(h1, h2)

    def test_similar_hashes_match_with_tolerance(self) -> None:
        assert hashes_match("0000000000000000", "0000000000000001", tolerance=4)

    def test_similar_hashes_no_match_without_tolerance(self) -> None:
        assert not hashes_match("0000000000000000", "ffffffffffffffff", tolerance=0)


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
