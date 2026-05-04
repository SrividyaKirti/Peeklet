"""Demo-mode frame filtering: transcript-driven LLM moment picking + linear-pass dedup.

The LLM picks the moments from the transcript; Peeklet captures a frame at
each moment timestamp, runs the universal layout/info-density quality gate
(with bounded ±N-second fallback), computes a content-addressable fingerprint,
and deduplicates against screens already seen this run.

See the design spec at
``docs/superpowers/specs/2026-04-30-content-addressable-screen-dedup-design.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from peeklet.core.llm import build_llm_client
from peeklet.utils.types import Moment, MomentEntry, Screen

if TYPE_CHECKING:
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.video import VideoDecoder

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


def _count_words_in_frame(frame: np.ndarray, downscale_dim: int) -> int:
    """Thin wrapper: number of accepted OCR word boxes.

    Preserved for call sites and tests that still target the word-count
    name. All new code should consume :func:`_ocr_word_boxes` directly.
    """
    return len(_ocr_word_boxes(frame, downscale_dim))


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


def _quality_capture(
    decoder: VideoDecoder,
    t: float,
    config: DemoFilterConfig,
) -> tuple[np.ndarray, float, int] | None:
    """Capture a frame at ``t`` that passes the quality gate.

    Tries ``t``, then ``t±step``, ``t±2*step``, ... in alternating
    ahead/behind order, up to ``quality_fallback_max_attempts`` attempts
    within ``±quality_fallback_half_window_seconds``. Returns the first
    frame that passes the layout/info-density gate, or ``None`` if every
    attempt fails (caller emits ``image_unavailable``).

    Negative timestamps are skipped silently so a moment near t=0 still
    gets the chance to try later neighbors.
    """
    half = config.quality_fallback_half_window_seconds
    step = config.quality_fallback_step_seconds
    max_attempts = config.quality_fallback_max_attempts

    deltas: list[float] = [0.0]
    n = 1
    while len(deltas) < max_attempts and n * step <= half + 1e-9:
        deltas.append(+n * step)
        if len(deltas) < max_attempts:
            deltas.append(-n * step)
        n += 1

    for delta in deltas:
        target = t + delta
        if target < 0:
            continue
        try:
            frame, ts, fnum = decoder.extract_frame_at(target)
        except Exception as exc:  # pragma: no cover - decoder failures are rare
            logger.warning("Quality-fallback decode failed at %.2fs: %s", target, exc)
            continue
        if not _is_low_info_frame(frame, config):
            return frame, ts, fnum

    return None


def _ocr_joined_text(boxes: list[WordBox]) -> str:
    """Plain-joined OCR text for the screens[].ocr_text sidecar field."""
    return " ".join(b.text for b in boxes)


def apply_demo_filter(
    decoder: VideoDecoder,
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
    output_dir: Path,
    transcript_text: str = "",
) -> tuple[list[Screen], list[MomentEntry]]:
    """Top-level demo-mode entry point.

    Parses Fathom anchors from ``transcript_text``, asks the LLM to
    pick complementary moments, then runs a single linear pass:

      1. Capture frame at moment timestamp.
      2. Universal quality gate (with bounded ±N-second fallback).
      3. Compute fingerprint.
      4. Lookup; on hit reuse screen_id; on miss save image + register.

    Returns ``(screens, moments)``. Anchor moments use ``type="action_item"``;
    LLM picks use ``type="llm"``. Quality-gate failures emit
    ``image_unavailable=True`` with ``screen_id=None``.
    """
    from peeklet.core.audio import parse_fathom_anchors
    from peeklet.core.exporter import save_keyframe
    from peeklet.core.fingerprint import (
        FingerprintIndex,
        compute_fingerprint,
    )

    if pytesseract is None:
        raise RuntimeError(
            "Demo mode requires pytesseract for OCR-based frame scoring "
            "and gallery detection. Install with: pip install peeklet[demo]"
        )

    output_dir = Path(output_dir)
    meta = decoder.get_metadata()
    client = build_llm_client(provider=config.llm_provider, model=config.llm_model)

    anchors = parse_fathom_anchors(transcript_text) if transcript_text else []
    llm_picks = client.pick_moments(transcript, meta.duration, anchors=anchors)
    logger.info("Demo dedup: %d anchors + %d LLM picks", len(anchors), len(llm_picks))

    # Tag moments with type and merge in time order. Tail-skip applies
    # to LLM picks only (anchors are guaranteed by Fathom).
    tail_cutoff: float | None = None
    if config.tail_skip_ratio > 0.0 and meta.duration > 0.0:
        tail_cutoff = meta.duration * (1.0 - config.tail_skip_ratio)

    annotated: list[tuple[Moment, str]] = []
    for a in anchors:
        annotated.append((a, "action_item"))
    for p in llm_picks:
        if tail_cutoff is not None and p.timestamp >= tail_cutoff:
            logger.info(
                "LLM pick at %.2fs in tail (cutoff %.2fs) — skipping.",
                p.timestamp,
                tail_cutoff,
            )
            continue
        annotated.append((p, "llm"))
    annotated.sort(key=lambda pair: pair[0].timestamp)

    if not annotated:
        logger.warning("no anchors and no LLM picks — demo mode produced 0 outputs")
        return [], []

    index = FingerprintIndex(
        phash_threshold=config.phash_threshold,
        ocr_field_min_chars=config.ocr_field_min_chars,
    )
    screens: list[Screen] = []
    moments: list[MomentEntry] = []
    next_screen_index = 1

    for moment, mtype in annotated:
        captured = _quality_capture(decoder, t=moment.timestamp, config=config)
        if captured is None:
            moments.append(
                MomentEntry(
                    timestamp_ms=int(round(moment.timestamp * 1000)),
                    caption=moment.visual_context_goal or moment.textual_anchor,
                    type=mtype,
                    screen_id=None,
                    image_unavailable=True,
                )
            )
            continue

        frame, ts, _fnum = captured
        boxes = _ocr_word_boxes(frame, config.gallery_ocr_min_dim)
        fp = compute_fingerprint(frame, boxes)  # type: ignore[arg-type]
        existing_id = index.lookup(fp)
        if existing_id is not None:
            moments.append(
                MomentEntry(
                    timestamp_ms=int(round(moment.timestamp * 1000)),
                    caption=moment.visual_context_goal or moment.textual_anchor,
                    type=mtype,
                    screen_id=existing_id,
                    image_unavailable=False,
                )
            )
            continue

        screen_id = f"screen_{next_screen_index:03d}"
        ts_ms = int(round(ts * 1000))
        frame_id = f"demo_{next_screen_index:04d}_{ts_ms:08d}ms"
        path = save_keyframe(frame, output_dir, frame_id, fmt="jpg")
        index.register(fp, screen_id=screen_id)
        screens.append(
            Screen(
                screen_id=screen_id,
                image_path=path.name,
                first_seen_ms=ts_ms,
                fingerprint=fp,
                ocr_text=_ocr_joined_text(boxes),
            )
        )
        moments.append(
            MomentEntry(
                timestamp_ms=int(round(moment.timestamp * 1000)),
                caption=moment.visual_context_goal or moment.textual_anchor,
                type=mtype,
                screen_id=screen_id,
                image_unavailable=False,
            )
        )
        next_screen_index += 1

    return screens, moments
