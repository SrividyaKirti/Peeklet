"""OCR word-box helpers and the low-information frame rejector."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from peeklet.config import DemoFilterConfig

logger = logging.getLogger(__name__)

# Imported lazily so test files can patch ``demo_filter.pytesseract``.
try:
    import pytesseract
except ImportError:  # pragma: no cover - exercised in install-error path
    pytesseract = None

_MIN_WORD_LENGTH = 2
_MIN_WORD_CONFIDENCE = 30


class WordBox(NamedTuple):
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int


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
    img = Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.asarray(img)


def _ocr_word_boxes(frame: np.ndarray, downscale_dim: int) -> list[WordBox]:
    """Run Tesseract once and return accepted word boxes.

    A word is accepted if its confidence >= ``_MIN_WORD_CONFIDENCE`` and its
    stripped text length >= ``_MIN_WORD_LENGTH``. Coordinates are in the
    downscaled frame's pixel space so all layout signals share one
    coordinate system from a single OCR call.
    """
    if pytesseract is None:
        return []

    downscaled = _downscale_for_ocr(frame, downscale_dim)
    try:
        data = pytesseract.image_to_data(downscaled, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        logger.warning("OCR failed on frame: %s", exc)
        return []

    texts = data.get("text", [])
    confs = data.get("conf", [])
    n = len(texts)
    # Geometry fields may be absent in mocked OCR responses; default to zeros
    # so legacy callers that only populate text+conf still produce word counts.
    lefts = data.get("left") or [0] * n
    tops = data.get("top") or [0] * n
    widths = data.get("width") or [0] * n
    heights = data.get("height") or [0] * n
    boxes: list[WordBox] = []
    for text, conf, x, y, w, h in zip(texts, confs, lefts, tops, widths, heights, strict=False):
        if not text or len(text.strip()) < _MIN_WORD_LENGTH:
            continue
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_val < _MIN_WORD_CONFIDENCE:
            continue
        boxes.append(
            WordBox(
                text=text,
                conf=conf_val,
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
            )
        )
    return boxes


def _count_text_lines(boxes: list[WordBox]) -> int:
    """Cluster word boxes by y-center into distinct text lines.

    Uses a tolerance of half the median box height so a single typographic
    row stays one line even when words have minor y jitter from Tesseract.
    """
    if not boxes:
        return 0
    median_h = float(np.median([b.h for b in boxes]))
    tolerance = max(1.0, median_h / 2.0)
    centers = sorted(b.y + b.h / 2.0 for b in boxes)
    lines = 1
    current = centers[0]
    for c in centers[1:]:
        if c - current > tolerance:
            lines += 1
            current = c
    return lines


_GRID_DIM = 8


def _count_occupied_grid_cells(boxes: list[WordBox], frame_shape: tuple[int, ...]) -> int:
    """Count distinct cells in an 8x8 grid that contain a word-box center.

    Frame shape follows numpy convention (H, W, ...). Out-of-bounds centers
    are clamped to the nearest edge cell so an OCR box that extends beyond
    the downscaled frame still counts once.
    """
    if not boxes:
        return 0
    h, w = frame_shape[:2]
    cell_h = max(1, h // _GRID_DIM)
    cell_w = max(1, w // _GRID_DIM)
    occupied: set[tuple[int, int]] = set()
    for b in boxes:
        cx = b.x + b.w / 2.0
        cy = b.y + b.h / 2.0
        col = min(_GRID_DIM - 1, max(0, int(cx // cell_w)))
        row = min(_GRID_DIM - 1, max(0, int(cy // cell_h)))
        occupied.add((row, col))
    return len(occupied)


_EDGE_MAGNITUDE_THRESHOLD = 30.0


def _edge_pixel_ratio(frame: np.ndarray) -> float:
    """Fraction of pixels whose |∂x|+|∂y| gradient exceeds a fixed threshold.

    Uses ``np.gradient`` on the grayscale frame — numpy-only, no cv2
    dependency. Gallery views (flat colored tiles) land near zero;
    dashboard UIs with chrome sit well above 0.02.
    """
    if frame.ndim == 3:
        gray = (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(
            np.float32
        )
    else:
        gray = frame.astype(np.float32)
    gy, gx = np.gradient(gray)
    mag = np.abs(gx) + np.abs(gy)
    edge_pixels = int((mag > _EDGE_MAGNITUDE_THRESHOLD).sum())
    total = gray.size
    return edge_pixels / total if total else 0.0


def _is_low_info_frame(frame: np.ndarray, config: DemoFilterConfig) -> bool:
    """Composite low-information rejector (triple-AND).

    Rejects a frame only if **all three** independent signals fall under
    their thresholds. A legit minimalist UI will pass on at least one axis.
    """
    boxes = _ocr_word_boxes(frame, config.gallery_ocr_min_dim)
    num_lines = _count_text_lines(boxes)
    num_cells = _count_occupied_grid_cells(boxes, frame.shape)
    edge_ratio = _edge_pixel_ratio(frame)
    if edge_ratio >= config.min_edge_ratio:
        return False
    if num_lines >= config.min_text_lines:
        return False
    if num_cells >= config.min_grid_cells:
        return False
    logger.info(
        "Low-info frame rejected: lines=%d cells=%d edge_ratio=%.4f",
        num_lines,
        num_cells,
        edge_ratio,
    )
    return True
