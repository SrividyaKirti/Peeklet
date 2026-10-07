"""Shared image operations for Peeklet."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from peeklet.types import Region


def ensure_rgb_uint8(frame: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert any image array to RGB uint8 (H, W, 3).

    Handles: grayscale, RGBA, float [0,1], and passthrough for valid RGB uint8.
    """
    if frame.ndim == 2:
        if frame.dtype != np.uint8:
            frame = _to_uint8(frame)
        return np.stack([frame, frame, frame], axis=-1)

    if frame.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {frame.ndim}D")

    channels = frame.shape[2]

    if channels == 4:
        frame = frame[:, :, :3]
    elif channels == 1:
        frame = np.concatenate([frame, frame, frame], axis=-1)
    elif channels != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got {channels}")

    if frame.dtype != np.uint8:
        frame = _to_uint8(frame)

    return frame


def crop_region(
    frame: np.ndarray[Any, np.dtype[Any]], region: Region
) -> np.ndarray[Any, np.dtype[Any]]:
    """Crop a region from a frame, clamping to image bounds."""
    h, w = frame.shape[:2]
    x1 = max(0, region.x)
    y1 = max(0, region.y)
    x2 = min(w, region.x + region.w)
    y2 = min(h, region.y + region.h)
    return frame[y1:y2, x1:x2]


def compute_block_grid(height: int, width: int, block_size: int) -> tuple[int, int]:
    """Compute the number of block rows and columns for a given image size."""
    rows = height // block_size
    cols = width // block_size
    return rows, cols


def _to_uint8(frame: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert float or other dtype arrays to uint8."""
    if np.issubdtype(frame.dtype, np.floating):
        return (frame * 255).clip(0, 255).astype(np.uint8)
    return frame.astype(np.uint8)


def dhash_64(frame: np.ndarray[Any, np.dtype[Any]]) -> int:
    """Compute a 64-bit difference hash (dHash) of a frame.

    Resize to 9x8 grayscale and compare each pixel to its right neighbor,
    packing 64 bits into a Python int. Identical frames hash equal; similar
    frames have small Hamming distance.
    """
    from PIL import Image

    rgb = ensure_rgb_uint8(frame)
    gray = (0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]).astype(np.uint8)
    img = Image.fromarray(gray).resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten().astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two non-negative integers."""
    return (a ^ b).bit_count()


def downscale_to_max_dim(frame: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize so the longest edge is at most max_dim; returns the input if already small."""
    from PIL import Image

    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR))


def encode_jpeg(frame: np.ndarray, quality: int = 90, max_edge: int | None = None) -> bytes:
    """Encode an RGB uint8 frame as JPEG, optionally shrinking the long edge first."""
    from PIL import Image

    if max_edge is not None:
        frame = downscale_to_max_dim(frame, max_edge)
    buf = io.BytesIO()
    Image.fromarray(ensure_rgb_uint8(frame)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to an RGB uint8 array."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))
