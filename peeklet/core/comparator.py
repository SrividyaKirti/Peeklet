"""SSIM comparison and block-level change detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.metrics import structural_similarity

from peeklet.utils.image import compute_block_grid
from peeklet.utils.types import Region


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
    ssim_score = float(structural_similarity(gray_reference, gray_current, data_range=255))
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
    return (
        0.2989 * frame[:, :, 0]
        + 0.5870 * frame[:, :, 1]
        + 0.1140 * frame[:, :, 2]
    ).astype(np.uint8)
