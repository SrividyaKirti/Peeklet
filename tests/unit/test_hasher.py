"""Tests for perceptual hashing."""

import numpy as np

from peeklet.core.hasher import compute_phash, hashes_match


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
