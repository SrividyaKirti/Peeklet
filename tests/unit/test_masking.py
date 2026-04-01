"""Tests for adaptive block-based masking."""

import numpy as np

from peeklet.core.masking import AdaptiveMask


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
