"""Adaptive block-based masking for high-frequency change regions."""

from __future__ import annotations

from collections import deque

import numpy as np

from peeklet.utils.image import compute_block_grid
from peeklet.utils.types import Region


class AdaptiveMask:
    """Detects and masks screen regions that change on nearly every frame."""

    def __init__(
        self, block_size: int = 32, window_size: int = 15, noise_threshold: float = 0.8
    ) -> None:
        self.block_size = block_size
        self.window_size = window_size
        self.noise_threshold = noise_threshold
        self._prev_blocks: np.ndarray | None = None
        self._change_history: deque[np.ndarray] = deque(maxlen=window_size)

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, list[Region]]:
        """Apply adaptive mask. Returns masked frame and list of masked regions."""
        h, w = frame.shape[:2]
        rows, cols = compute_block_grid(h, w, self.block_size)
        if rows == 0 or cols == 0:
            return frame.copy(), []

        current_blocks = self._compute_block_means(frame, rows, cols)
        if self._prev_blocks is None:
            self._prev_blocks = current_blocks
            return frame.copy(), []

        change_map = np.abs(current_blocks - self._prev_blocks).mean(axis=-1) > 5.0
        self._change_history.append(change_map)
        self._prev_blocks = current_blocks

        if len(self._change_history) < 2:
            return frame.copy(), []

        history = np.stack(list(self._change_history), axis=0)
        noise_freq = history.mean(axis=0)
        noisy_blocks = noise_freq > self.noise_threshold

        masked = frame.copy()
        regions: list[Region] = []
        for r in range(rows):
            for c in range(cols):
                if noisy_blocks[r, c]:
                    y = r * self.block_size
                    x = c * self.block_size
                    masked[y : y + self.block_size, x : x + self.block_size] = 0
                    regions.append(Region(x=x, y=y, w=self.block_size, h=self.block_size))
        return masked, regions

    def _compute_block_means(self, frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
        bs = self.block_size
        means = np.zeros((rows, cols, 3), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                block = frame[r * bs : (r + 1) * bs, c * bs : (c + 1) * bs]
                means[r, c] = block.mean(axis=(0, 1))
        return means
