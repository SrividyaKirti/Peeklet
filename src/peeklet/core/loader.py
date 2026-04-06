"""Load images from any supported format into normalized numpy arrays."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from peeklet.utils.image import ensure_rgb_uint8


def load_frame(
    source: str | Path | bytes | np.ndarray[Any, np.dtype[Any]] | Image.Image,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Normalize any supported input to an RGB uint8 numpy array (H, W, 3)."""
    if isinstance(source, np.ndarray):
        return ensure_rgb_uint8(source)

    if isinstance(source, Image.Image):
        return ensure_rgb_uint8(np.asarray(source))

    if isinstance(source, (str, Path)):
        return _load_from_path(Path(source))

    if isinstance(source, (bytes, bytearray)):
        return _load_from_bytes(source)

    raise TypeError(f"Unsupported source type: {type(source).__name__}")


def _load_from_path(path: Path) -> np.ndarray[Any, np.dtype[Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        img = Image.open(path)
        img.load()
        return ensure_rgb_uint8(np.asarray(img.convert("RGB")))
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"Could not load image from {path}: {e}") from e


def _load_from_bytes(data: bytes | bytearray) -> np.ndarray[Any, np.dtype[Any]]:
    try:
        img = Image.open(BytesIO(data))
        img.load()
        return ensure_rgb_uint8(np.asarray(img.convert("RGB")))
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"Could not load image from bytes: {e}") from e
