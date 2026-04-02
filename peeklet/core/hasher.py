"""Perceptual hashing for screenshot deduplication."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import imagehash
from PIL import Image

if TYPE_CHECKING:
    import numpy as np


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


def compute_phash_tiled(
    frame: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    hash_size: int = 8,
    tile_height: int = 1080,
) -> list[str]:
    """Compute perceptual hashes for vertical tiles of a frame.

    Splits the frame into tiles of `tile_height` rows each (last tile may be shorter).
    Returns one hash per tile.
    """
    h = frame.shape[0]
    n_tiles = max(1, math.ceil(h / tile_height))
    hashes = []
    for i in range(n_tiles):
        y_start = i * tile_height
        y_end = min((i + 1) * tile_height, h)
        tile = frame[y_start:y_end]
        hashes.append(compute_phash(tile, hash_size=hash_size))
    return hashes


def tiled_hashes_match(hashes_a: list[str], hashes_b: list[str], tolerance: int = 0) -> bool:
    """Check if all corresponding tile hashes match.

    Returns False if tile counts differ or any tile pair exceeds tolerance.
    """
    if len(hashes_a) != len(hashes_b):
        return False
    return all(
        hashes_match(ha, hb, tolerance=tolerance) for ha, hb in zip(hashes_a, hashes_b, strict=True)
    )
