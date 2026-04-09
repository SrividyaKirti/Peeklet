"""Demo-mode frame filtering: transcript-driven LLM moment picking + frame selection.

The LLM picks the moments from the transcript, Peeklet picks the exact frame
at each moment using forward-search bidirectional SSIM stability and an OCR
gallery check. See the design spec at
``docs/superpowers/specs/2026-04-09-transcript-driven-demo-mode-design.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peeklet.core.exporter import save_keyframe
from peeklet.utils.types import EventType, FrameResult

if TYPE_CHECKING:
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.video import VideoDecoder
    from peeklet.utils.types import Moment

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


def _build_search_window(
    moment_ts: float,
    segment: TranscriptSegment,
    max_window_sec: float,
) -> tuple[float, float]:
    """Compute the (start, end) timestamps for the forward-search window.

    The window starts at the LLM's moment timestamp and ends at the earlier of:
    - The end of the transcript segment the moment falls into.
    - ``moment_ts + max_window_sec``.
    """
    end = min(segment.end, moment_ts + max_window_sec)
    return moment_ts, end


def _is_stable(
    frame: np.ndarray,
    prev: np.ndarray,
    nxt: np.ndarray,
    threshold: float,
) -> bool:
    """Bidirectional SSIM check: a frame is stable if both neighbors are similar.

    Short-circuits the second SSIM call when the first one already disqualifies
    the frame, since SSIM is in the Stage B hot path.
    """
    from peeklet.core.comparator import compare_frames

    if compare_frames(frame, prev).ssim_score <= threshold:
        return False
    return compare_frames(frame, nxt).ssim_score > threshold


def _is_gallery_frame(frame: np.ndarray, downscale_dim: int, min_words: int) -> bool:
    """Return True if the frame has too few visible words to be demo content."""
    return _count_words_in_frame(frame, downscale_dim) < min_words


def _find_segment_for_timestamp(
    ts: float, transcript: list[TranscriptSegment]
) -> TranscriptSegment | None:
    """Return the transcript segment containing ``ts``, or None."""
    for seg in transcript:
        if seg.start <= ts <= seg.end:
            return seg
    return None


def _sample_window_frames(
    decoder: VideoDecoder,
    start: float,
    end: float,
    step: float,
) -> list[tuple[np.ndarray, float, int]]:
    """Decode a small set of frames from the search window."""
    if end <= start:
        try:
            return [decoder.extract_frame_at(start)]
        except Exception as exc:
            logger.warning("Failed to extract frame at %.2fs: %s", start, exc)
            return []
    timestamps: list[float] = []
    t = start
    while t <= end + 1e-6:
        timestamps.append(round(t, 6))
        t += step
    samples: list[tuple[np.ndarray, float, int]] = []
    for ts in timestamps:
        try:
            samples.append(decoder.extract_frame_at(ts))
        except Exception as exc:
            logger.warning("Failed to extract frame at %.2fs: %s", ts, exc)
    return samples


def _pick_stable_index(
    samples: list[tuple[np.ndarray, float, int]],
    threshold: float,
) -> int:
    """Return the index of the first sample whose two neighbors are SSIM-similar.

    Falls back to index 0 if no sample passes the bidirectional check (or if
    the window has fewer than 3 samples to compare).
    """
    if len(samples) < 3:
        return 0
    for i in range(1, len(samples) - 1):
        frame, _, _ = samples[i]
        prev_frame, _, _ = samples[i - 1]
        next_frame, _, _ = samples[i + 1]
        if _is_stable(frame, prev_frame, next_frame, threshold):
            return i
    return 0


def select_frames_for_moments(
    decoder: VideoDecoder,
    moments: list[Moment],
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
    output_dir: Path,
) -> list[FrameResult]:
    """Stage B: for each LLM-picked moment, find the best actual frame.

    Walks each moment, builds a forward search window inside the current
    transcript segment, samples frames at ``forward_search_step_sec`` intervals,
    picks the first bidirectionally-stable frame, runs the gallery check, and
    saves the surviving frame as a keyframe.
    """
    output_dir = Path(output_dir)
    meta = decoder.get_metadata()
    results: list[FrameResult] = []

    for idx, moment in enumerate(moments, start=1):
        seg = _find_segment_for_timestamp(moment.timestamp, transcript)
        if seg is None:
            logger.warning(
                "LLM moment at %.2fs ('%s') has no matching transcript segment, skipping",
                moment.timestamp,
                moment.caption,
            )
            continue

        win_start, win_end = _build_search_window(
            moment_ts=moment.timestamp,
            segment=seg,
            max_window_sec=config.forward_search_window_max_sec,
        )
        samples = _sample_window_frames(
            decoder=decoder,
            start=win_start,
            end=win_end,
            step=config.forward_search_step_sec,
        )
        if not samples:
            logger.warning(
                "No frames could be extracted for moment at %.2fs, skipping",
                moment.timestamp,
            )
            continue

        picked_idx = _pick_stable_index(samples, config.ssim_stability_threshold)
        picked_frame, picked_ts, picked_frame_num = samples[picked_idx]

        if _is_gallery_frame(
            picked_frame,
            downscale_dim=config.frame_search_resolution,
            min_words=config.gallery_min_words,
        ):
            logger.warning(
                "LLM picked moment at %.2fs ('%s') but the frame is gallery-view "
                "or blank — no demo content visible. Skipping.",
                moment.timestamp,
                moment.caption,
            )
            continue

        frame_id = f"demo_{idx:04d}_{int(picked_ts):04d}s"
        asset_path = save_keyframe(picked_frame, output_dir, frame_id, fmt="jpg")

        results.append(
            FrameResult(
                frame_id=frame_id,
                event_type=EventType.KEYFRAME,
                is_keyframe=True,
                perceptual_hash="",  # not computed in demo mode
                frame_width=picked_frame.shape[1],
                frame_height=picked_frame.shape[0],
                source_format="video",
                asset_path=str(asset_path),
                trigger_type="transcript_trigger",
                source_video=meta.filename,
                video_timestamp=picked_ts,
                video_frame_number=picked_frame_num,
                video_duration=meta.duration,
                llm_caption=moment.caption,
                llm_reason=moment.reason,
                keyframe_index=idx,
            )
        )

    # Backfill total_keyframes
    for r in results:
        r.total_keyframes = len(results)
    return results
