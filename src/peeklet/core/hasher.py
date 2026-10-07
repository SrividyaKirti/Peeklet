"""Perceptual hashing for screenshot deduplication."""

from __future__ import annotations

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
