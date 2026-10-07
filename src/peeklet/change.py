"""Frame change detection: adaptive masking, perceptual hashing, and SSIM comparison."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from peeklet.image_utils import compute_block_grid
from peeklet.types import Region


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
        if self._prev_blocks is None or current_blocks.shape != self._prev_blocks.shape:
            self._prev_blocks = current_blocks
            self._change_history.clear()
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


def compute_phash(
    frame: np.ndarray[tuple[int, ...], np.dtype[np.uint8]], hash_size: int = 8
) -> str:
    """Compute a perceptual hash of an RGB uint8 frame.

    Returns a hex string representation of the 64-bit hash.
    """
    img = Image.fromarray(frame)
    h = imagehash.phash_simple(img, hash_size=hash_size)
    return str(h)


def hashes_match(hash_a: str, hash_b: str, tolerance: int = 0) -> bool:
    """Check if two perceptual hashes are similar within a Hamming distance tolerance."""
    h1 = imagehash.hex_to_hash(hash_a)
    h2 = imagehash.hex_to_hash(hash_b)
    return (h1 - h2) <= tolerance


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    ssim_score: float
    change_score: float
    changed_pct: float
    changed_regions: list[Region] = field(default_factory=list)


def compare_frames(
    current: np.ndarray,
    reference: np.ndarray,
    block_size: int = 32,
    block_change_threshold: float = 10.0,
) -> ComparisonResult:
    gray_current = _to_grayscale(current)
    gray_reference = _to_grayscale(reference)
    ssim_score = float(
        structural_similarity(gray_reference, gray_current, data_range=255)  # type: ignore[no-untyped-call]
    )
    change_score = 1.0 - ssim_score
    changed_regions = _compute_block_diff(current, reference, block_size, block_change_threshold)
    h, w = current.shape[:2]
    rows, cols = compute_block_grid(h, w, block_size)
    total_blocks = max(rows * cols, 1)
    changed_pct = (len(changed_regions) / total_blocks) * 100.0
    return ComparisonResult(
        ssim_score=ssim_score,
        change_score=change_score,
        changed_pct=changed_pct,
        changed_regions=changed_regions,
    )


def _compute_block_diff(
    current: np.ndarray,
    reference: np.ndarray,
    block_size: int,
    threshold: float,
) -> list[Region]:
    h, w = current.shape[:2]
    rows, cols = compute_block_grid(h, w, block_size)
    bs = block_size
    diff = np.abs(current.astype(np.float32) - reference.astype(np.float32))
    regions: list[Region] = []
    for r in range(rows):
        for c in range(cols):
            block_diff = diff[r * bs : (r + 1) * bs, c * bs : (c + 1) * bs]
            if block_diff.mean() > threshold:
                regions.append(Region(x=c * bs, y=r * bs, w=bs, h=bs))
    return regions


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    return (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(
        np.uint8
    )
