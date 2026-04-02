"""Tests for perceptual hashing."""

import numpy as np

from peeklet.core.hasher import compute_phash, compute_phash_tiled, hashes_match, tiled_hashes_match


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


class TestComputePhashTiled:
    def test_returns_list_of_hashes(self) -> None:
        # Tall image: 100 wide, 300 tall -> tile_height=100 -> 3 tiles
        frame = np.zeros((300, 100, 3), dtype=np.uint8)
        frame[:100] = [255, 0, 0]
        frame[100:200] = [0, 255, 0]
        frame[200:] = [0, 0, 255]
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        assert isinstance(hashes, list)
        assert len(hashes) == 3
        assert all(isinstance(h, str) for h in hashes)

    def test_different_tiles_get_different_hashes(self) -> None:
        # Create tiles with different visual content to ensure different hashes
        frame = np.zeros((300, 100, 3), dtype=np.uint8)
        # Tile 1: gradient pattern
        for i in range(100):
            frame[i, :, 0] = int(255 * i / 100)
        # Tile 2: different gradient
        for i in range(100, 200):
            frame[i, :, 1] = int(255 * (i - 100) / 100)
        # Tile 3: checkerboard pattern
        for i in range(200, 300):
            for j in range(100):
                if (i + j) % 20 == 0:
                    frame[i, j] = [255, 255, 255]
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        assert len(set(hashes)) > 1

    def test_single_tile_when_image_not_tall_enough(self) -> None:
        frame = np.full((80, 100, 3), 128, dtype=np.uint8)
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        assert len(hashes) == 1

    def test_last_tile_handles_remainder(self) -> None:
        frame = np.full((250, 100, 3), 128, dtype=np.uint8)
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        assert len(hashes) == 3

    def test_identical_frames_same_tiled_hashes(self) -> None:
        frame = np.full((300, 100, 3), 128, dtype=np.uint8)
        h1 = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        h2 = compute_phash_tiled(frame.copy(), hash_size=8, tile_height=100)
        assert h1 == h2

    def test_substantial_change_in_one_tile_detected(self) -> None:
        # Create frames with pattern so phash can detect differences
        frame_a = np.full((300, 100, 3), 100, dtype=np.uint8)
        # Add some pattern to make hashing work
        for i in range(300):
            frame_a[i, :50] = [50, 50, 50]

        frame_b = frame_a.copy()
        # Make significant change to second tile (rows 100-200)
        frame_b[100:200] = [200, 200, 200]

        h_a = compute_phash_tiled(frame_a, hash_size=8, tile_height=100)
        h_b = compute_phash_tiled(frame_b, hash_size=8, tile_height=100)
        # Tile 0 should be same
        assert h_a[0] == h_b[0]
        # Tile 1 should be different due to major change
        assert h_a[1] != h_b[1]
        # Tile 2 should be same
        assert h_a[2] == h_b[2]


class TestTiledHashesMatch:
    def test_identical_tiled_hashes_match(self) -> None:
        hashes = ["abcdef0123456789", "1234567890abcdef"]
        assert tiled_hashes_match(hashes, hashes, tolerance=0)

    def test_one_tile_differs_no_match(self) -> None:
        hashes_a = ["abcdef0123456789", "1234567890abcdef"]
        hashes_b = ["abcdef0123456789", "ffffffffffffffff"]
        assert not tiled_hashes_match(hashes_a, hashes_b, tolerance=0)

    def test_different_tile_count_no_match(self) -> None:
        hashes_a = ["abcdef0123456789", "1234567890abcdef"]
        hashes_b = ["abcdef0123456789"]
        assert not tiled_hashes_match(hashes_a, hashes_b, tolerance=0)

    def test_tolerance_applied_per_tile(self) -> None:
        hashes_a = ["0000000000000000", "0000000000000000"]
        hashes_b = ["0000000000000001", "0000000000000001"]
        assert tiled_hashes_match(hashes_a, hashes_b, tolerance=4)

    def test_empty_lists_match(self) -> None:
        assert tiled_hashes_match([], [], tolerance=0)
