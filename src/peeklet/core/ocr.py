"""OCR text extraction for PII detection on keyframe images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from peeklet.utils.types import Region

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class OcrResult:
    """A single OCR detection: extracted text and its bounding box in frame coordinates."""

    text: str
    region: Region
    confidence: float


def _get_reader():  # type: ignore[no-untyped-def]
    """Lazy-load easyocr reader (singleton)."""
    import easyocr  # optional dep — ImportError if not installed

    if not hasattr(_get_reader, "_reader"):
        _get_reader._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _get_reader._reader


def _bbox_to_region(bbox: list[list[int]]) -> Region:
    """Convert easyocr bbox ([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]) to Region."""
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return Region(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def extract_text_regions(
    frame: np.ndarray,
    min_confidence: float = 0.3,
) -> list[OcrResult]:
    """Run OCR on an entire frame, return text + bounding boxes.

    Args:
        frame: RGB uint8 numpy array.
        min_confidence: Discard detections below this threshold.

    Returns:
        List of OcrResult with text, region, and confidence.
    """
    reader = _get_reader()
    detections = reader.readtext(frame)
    results: list[OcrResult] = []
    for bbox, text, confidence in detections:
        if confidence >= min_confidence:
            results.append(
                OcrResult(text=text, region=_bbox_to_region(bbox), confidence=confidence),
            )
    return results


def extract_text_from_regions(
    frame: np.ndarray,
    regions: list[Region],
    min_confidence: float = 0.3,
) -> list[OcrResult]:
    """Run OCR only on specific regions of a frame, returning results in full-frame coordinates.

    Args:
        frame: Full RGB uint8 numpy array.
        regions: List of Region bounding boxes to OCR.
        min_confidence: Discard detections below this threshold.

    Returns:
        List of OcrResult with coordinates mapped back to full-frame space.
    """
    reader = _get_reader()
    h, w = frame.shape[:2]
    results: list[OcrResult] = []
    for region in regions:
        # Clamp crop to frame bounds
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(w, region.x + region.w)
        y2 = min(h, region.y + region.h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        detections = reader.readtext(crop)
        for bbox, text, confidence in detections:
            if confidence >= min_confidence:
                local = _bbox_to_region(bbox)
                # Offset back to full-frame coordinates
                results.append(
                    OcrResult(
                        text=text,
                        region=Region(x=x1 + local.x, y=y1 + local.y, w=local.w, h=local.h),
                        confidence=confidence,
                    )
                )
    return results
