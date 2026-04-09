"""Demo-mode frame filtering: sparse OCR sweep + transcript-anchored selection.

Activated via the CLI ``--demo-mode`` flag. See the design spec at
``docs/superpowers/specs/2026-04-08-demo-mode-frame-filtering-design.md``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peeklet.config import DemoFilterConfig  # noqa: F401  (used in later tasks)
    from peeklet.core.audio import TranscriptSegment  # noqa: F401  (used in later tasks)
    from peeklet.core.video import VideoDecoder  # noqa: F401  (used in later tasks)
    from peeklet.utils.types import (  # noqa: F401  (used in later tasks)
        ContentSegment,
        FrameResult,
    )

logger = logging.getLogger(__name__)

# Imported lazily so test files can patch ``demo_filter.pytesseract``.
try:
    import pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised in install-error path
    pytesseract = None  # type: ignore[assignment]

_MIN_WORD_LENGTH = 2
_MIN_WORD_CONFIDENCE = 30


def _downscale_for_ocr(frame: np.ndarray, downscale_dim: int) -> np.ndarray:
    """Resize ``frame`` so its longest edge equals ``downscale_dim``.

    Returns the original frame if it is already smaller. Uses Pillow for
    a high-quality resize without pulling in extra dependencies.
    """
    from PIL import Image

    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= downscale_dim:
        return frame
    scale = downscale_dim / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(frame).resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(img)


def _count_words_in_frame(frame: np.ndarray, downscale_dim: int) -> int:
    """Run Tesseract and return the number of words above the noise floor.

    Words are counted only if they have at least ``_MIN_WORD_CONFIDENCE``
    confidence and at least ``_MIN_WORD_LENGTH`` characters. Returns 0 on
    any Tesseract error so the caller can treat the frame as non-demo.
    """
    if pytesseract is None:
        return 0

    downscaled = _downscale_for_ocr(frame, downscale_dim)
    try:
        data = pytesseract.image_to_data(downscaled, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        logger.warning("OCR failed on frame: %s", exc)
        return 0

    texts = data.get("text", [])
    confs = data.get("conf", [])
    count = 0
    for text, conf in zip(texts, confs, strict=False):
        if not text or len(text.strip()) < _MIN_WORD_LENGTH:
            continue
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_val < _MIN_WORD_CONFIDENCE:
            continue
        count += 1
    return count
