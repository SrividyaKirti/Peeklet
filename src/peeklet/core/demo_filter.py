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
from peeklet.core.llm import build_llm_client
from peeklet.utils.types import EventType, FrameResult, Moment

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


def _pick_best_content_index(
    samples: list[tuple[np.ndarray, float, int]],
    downscale_dim: int,
) -> int:
    """Return the index of the frame with the most OCR-readable text.

    Scores each sampled frame by OCR word count and returns the index
    with the highest count. Ties are broken by picking the latest frame
    (higher index — more likely to be fully loaded). Returns 0 when all
    frames score zero or the sample list has a single entry.
    """
    if len(samples) <= 1:
        return 0

    best_idx = 0
    best_count = -1
    for i, (frame, _ts, _fnum) in enumerate(samples):
        count = _count_words_in_frame(frame, downscale_dim)
        if count > best_count or (count == best_count and count > 0):
            best_count = count
            best_idx = i

    return best_idx


def merge_moments(
    anchors: list[Moment],
    llm_picks: list[Moment],
    proximity_sec: float = 5.0,
) -> list[Moment]:
    """Merge anchor and LLM-picked moments.

    All anchors are kept unconditionally. LLM picks within
    +/-proximity_sec of any anchor are dropped. Result is sorted
    by timestamp.
    """
    anchor_timestamps = [a.timestamp for a in anchors]
    filtered_llm: list[Moment] = []
    for pick in llm_picks:
        if any(abs(pick.timestamp - at) <= proximity_sec for at in anchor_timestamps):
            logger.info(
                "LLM pick at %.2fs dropped — within %.1fs of an anchor",
                pick.timestamp,
                proximity_sec,
            )
            continue
        filtered_llm.append(pick)

    combined = list(anchors) + filtered_llm
    combined.sort(key=lambda m: m.timestamp)
    return combined


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
    from peeklet.core.comparator import compare_frames

    output_dir = Path(output_dir)
    meta = decoder.get_metadata()
    results: list[FrameResult] = []
    last_saved_frame: np.ndarray | None = None

    tail_cutoff: float | None = None
    if config.tail_skip_ratio > 0.0 and meta.duration > 0.0:
        tail_cutoff = meta.duration * (1.0 - config.tail_skip_ratio)

    for idx, moment in enumerate(moments, start=1):
        is_anchor = moment.source == "anchor"

        if not is_anchor and tail_cutoff is not None and moment.timestamp >= tail_cutoff:
            logger.info(
                "Moment at %.2fs ('%s') falls in the final %.1f%% of the video "
                "(cutoff %.2fs), skipping as meeting-end noise.",
                moment.timestamp,
                moment.visual_context_goal,
                config.tail_skip_ratio * 100.0,
                tail_cutoff,
            )
            continue

        seg = _find_segment_for_timestamp(moment.timestamp, transcript)
        if seg is None:
            logger.warning(
                "LLM moment at %.2fs ('%s') has no matching transcript segment, skipping",
                moment.timestamp,
                moment.visual_context_goal,
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

        picked_idx = _pick_best_content_index(samples, config.gallery_ocr_min_dim)
        picked_frame, picked_ts, picked_frame_num = samples[picked_idx]

        if _is_gallery_frame(
            picked_frame,
            downscale_dim=config.gallery_ocr_min_dim,
            min_words=config.gallery_min_words,
        ):
            logger.warning(
                "LLM picked moment at %.2fs ('%s') but the frame is gallery-view "
                "or blank — no demo content visible. Skipping.",
                moment.timestamp,
                moment.visual_context_goal,
            )
            continue

        if not is_anchor and last_saved_frame is not None and config.dedup_ssim_threshold < 1.0:
            dedup_score = compare_frames(picked_frame, last_saved_frame).ssim_score
            if dedup_score > config.dedup_ssim_threshold:
                logger.info(
                    "Moment at %.2fs ('%s') is a near-duplicate of the previous "
                    "keyframe (ssim=%.3f > %.3f), skipping.",
                    moment.timestamp,
                    moment.visual_context_goal,
                    dedup_score,
                    config.dedup_ssim_threshold,
                )
                continue

        frame_id = f"demo_{idx:04d}_{int(picked_ts * 1000):08d}ms"
        asset_path = save_keyframe(picked_frame, output_dir, frame_id, fmt="jpg")
        last_saved_frame = picked_frame

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
                visual_context_goal=moment.visual_context_goal,
                textual_anchor=moment.textual_anchor,
                downstream_utility=moment.downstream_utility,
                moment_source=moment.source,
                keyframe_index=idx,
            )
        )

    # Backfill total_keyframes
    for r in results:
        r.total_keyframes = len(results)
    return results


def apply_demo_filter(
    decoder: VideoDecoder,
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
    output_dir: Path,
    transcript_text: str = "",
) -> list[FrameResult]:
    """Top-level demo-mode entry point.

    Builds the LLM client, asks it to pick screenshot-worthy moments from the
    transcript, parses Fathom ACTION ITEM anchors from the raw transcript text,
    merges anchors with LLM picks, then runs Stage B (forward-search + stability
    + gallery check) to pick the actual frames.
    """
    from peeklet.core.audio import parse_fathom_anchors

    if pytesseract is None:
        raise RuntimeError(
            "Demo mode requires pytesseract for OCR-based frame scoring and "
            "gallery detection. Install with: pip install peeklet[demo]"
        )

    meta = decoder.get_metadata()
    client = build_llm_client(provider=config.llm_provider, model=config.llm_model)

    llm_picks = client.pick_moments(transcript, meta.duration)
    logger.info("LLM picked %d screenshot-worthy moments", len(llm_picks))

    anchors = parse_fathom_anchors(transcript_text) if transcript_text else []
    if anchors:
        logger.info("Parsed %d ACTION ITEM anchors from transcript", len(anchors))

    moments = merge_moments(anchors, llm_picks, proximity_sec=5.0)

    if not moments:
        logger.warning("No screenshot-worthy moments found (LLM + anchors).")
        return []

    return select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=config,
        output_dir=output_dir,
    )
