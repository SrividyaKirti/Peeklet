"""Demo-mode frame filtering: transcript-driven LLM moment picking + frame selection.

The LLM picks the moments from the transcript, Peeklet picks the exact frame
at each moment using forward-search bidirectional SSIM stability and an OCR
gallery check. See the design spec at
``docs/superpowers/specs/2026-04-09-transcript-driven-demo-mode-design.md``.
"""

from __future__ import annotations

import logging
import re
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peeklet.core.exporter import save_keyframe
from peeklet.core.llm import build_llm_client
from peeklet.utils.image import dhash_64, hamming_distance
from peeklet.utils.types import AnchorRef, EventType, FrameResult, Moment

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


class SaveDecision(Enum):
    """Outcome of the unified `_should_save_frame` predicate."""

    SAVE = "save"
    DROP = "drop"
    MERGE_INTO_PREV = "merge_into_prev"


@dataclass
class DedupState:
    """Shared dedup state threaded through every frame-save path.

    Kept explicit (not a closure) so that gap-fill and main-loop paths
    both take the same state by reference — forgetting to thread it
    becomes a type error instead of a silent regression.
    """

    last_saved_tokens: set[str] | None = None
    last_saved_dhash: int | None = None
    last_saved_result: FrameResult | None = None

    def update(
        self,
        tokens: set[str],
        dhash: int,
        result: FrameResult,
    ) -> None:
        self.last_saved_tokens = tokens
        self.last_saved_dhash = dhash
        self.last_saved_result = result


_MIN_WORD_LENGTH = 2
_MIN_WORD_CONFIDENCE = 30

WordBox = namedtuple("WordBox", ["text", "conf", "x", "y", "w", "h"])


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
        try:
            frame_id = f"{dhash_64(frame):016x}"
        except Exception:
            frame_id = "<hash_failed>"
        logger.warning("OCR failed on frame %s: %s", frame_id, exc)
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
    try:
        rows = list(zip(texts, confs, lefts, tops, widths, heights, strict=True))
    except ValueError as exc:
        logger.warning(
            "Tesseract column-length mismatch on frame (%s); OCR result discarded.",
            exc,
        )
        return []
    for text, conf, x, y, w, h in rows:
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


_TOKEN_RE = re.compile(r"[a-z0-9]+")
# English stopwords that would otherwise manufacture false-positive overlaps
# between a generic caption ("the dashboard") and any frame with common chrome.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "or",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "into",
        "onto",
        "as",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "we",
        "they",
        "he",
        "she",
        "i",
        "you",
        "our",
        "your",
        "their",
        "my",
        "show",
        "shows",
        "showing",
        "view",
        "viewing",
        "see",
        "click",
        "page",
        "screen",
        "panel",
        "section",
        "tab",
        "window",
    }
)


def _normalize_tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens from ``text`` with stopwords removed.

    Used for caption/image alignment: both the picked frame's OCR and the
    moment's caption go through this so overlap is computed on a stable
    canonical form. Short tokens (<3 chars) are dropped — they're mostly
    single letters that introduce noise.
    """
    tokens = {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3}
    return tokens - _STOPWORDS


def _ocr_text_and_tokens(frame: np.ndarray, downscale_dim: int) -> tuple[str, set[str]]:
    """Joined OCR text and normalized token set from one OCR pass.

    Runs :func:`_ocr_word_boxes` once and returns both the raw joined
    word text (useful as a sidecar for downstream MLLM grounding) and
    the normalized token set used for caption/image alignment. Sharing
    one OCR call keeps the alignment validation path single-pass even
    when callers need both representations.
    """
    boxes = _ocr_word_boxes(frame, downscale_dim)
    if not boxes:
        return "", set()
    joined = " ".join(b.text for b in boxes)
    return joined, _normalize_tokens(joined)


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


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity over two token sets. Returns 0 when either is empty."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _should_save_frame(
    frame: np.ndarray,
    frame_tokens: set[str],
    frame_dhash: int,
    moment: Moment,
    state: DedupState,
    config: DemoFilterConfig,
) -> SaveDecision:
    """Unified save/drop/merge predicate — the single source of truth for
    whether a candidate frame joins the keyframe stream.

    Primary dedup signal is OCR-token Jaccard (robust to pixel noise in
    screen-share recordings); dHash Hamming distance is the fallback when
    either frame has fewer than ``config.min_ocr_tokens_for_jaccard``
    tokens. Low-info frames are dropped unless the moment is a Fathom
    anchor (we trust the semantic marker over the pixels). Anchors
    never silently DROP on a dedup collision — they return MERGE_INTO_PREV
    so the caller can preserve the transcript evidence on the prior frame.

    Callers must pre-compute ``frame_tokens`` via ``_ocr_text_and_tokens``
    and ``frame_dhash`` via ``dhash_64`` so this predicate stays pure and
    free of side effects.
    """
    is_anchor = moment.source == "anchor"

    if not is_anchor and _is_low_info_frame(frame, config):
        return SaveDecision.DROP

    if state.last_saved_tokens is None and state.last_saved_dhash is None:
        return SaveDecision.SAVE

    prev_tokens = state.last_saved_tokens or set()
    can_use_jaccard = (
        len(frame_tokens) >= config.min_ocr_tokens_for_jaccard
        and len(prev_tokens) >= config.min_ocr_tokens_for_jaccard
    )
    if can_use_jaccard:
        score = _jaccard(frame_tokens, prev_tokens)
        if score >= config.dedup_jaccard_threshold:
            return SaveDecision.MERGE_INTO_PREV if is_anchor else SaveDecision.DROP
        return SaveDecision.SAVE

    if state.last_saved_dhash is not None:
        dist = hamming_distance(frame_dhash, state.last_saved_dhash)
        if dist <= config.dhash_hamming_threshold:
            return SaveDecision.MERGE_INTO_PREV if is_anchor else SaveDecision.DROP

    return SaveDecision.SAVE


def _build_search_window(
    moment_ts: float,
    segment: TranscriptSegment,
    max_window_sec: float,
    lookback_sec: float = 0.0,
) -> tuple[float, float]:
    """Compute the (start, end) timestamps for the forward-search window.

    The window is biased ``lookback_sec`` before ``moment_ts`` so frames
    the speaker was referencing *before* naming an action are still in
    range, and clamped to the containing transcript segment. Total
    window width stays ``max_window_sec``.
    """
    start = max(segment.start, moment_ts - lookback_sec)
    end = min(segment.end, moment_ts + max_window_sec - lookback_sec)
    if end < start:
        end = start
    return start, end


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


def _pick_frame_for_moment(
    decoder: VideoDecoder,
    moment: Moment,
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
    extra_window_sec: float = 0.0,
) -> tuple[np.ndarray, float, int, tuple[float, float]] | None:
    """Run the search-window + scoring + low-info gates for one moment.

    Returns ``(frame, timestamp, frame_number, (win_start, win_end))`` for
    the picked frame, or ``None`` if no transcript segment matches, no
    frames can be extracted, or the picked frame fails the low-info gate.
    Dedup and tail-skip are the caller's responsibility. ``extra_window_sec``
    widens the search window symmetrically (still clamped to segment
    bounds) — used by the caption/image alignment retry path.
    """
    seg = _find_segment_for_timestamp(moment.timestamp, transcript)
    if seg is None:
        logger.warning(
            "Moment at %.2fs ('%s') has no matching transcript segment, skipping",
            moment.timestamp,
            moment.visual_context_goal,
        )
        return None

    win_start, win_end = _build_search_window(
        moment_ts=moment.timestamp,
        segment=seg,
        max_window_sec=config.forward_search_window_max_sec + extra_window_sec,
        lookback_sec=config.search_window_lookback_sec + extra_window_sec,
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
        return None

    picked_idx = _pick_best_content_index(samples, config.gallery_ocr_min_dim)
    picked_frame, picked_ts, picked_frame_num = samples[picked_idx]

    if _is_low_info_frame(picked_frame, config):
        logger.warning(
            "Moment at %.2fs ('%s') picked a low-information frame "
            "(gallery view or blank) — skipping.",
            moment.timestamp,
            moment.visual_context_goal,
        )
        return None

    return picked_frame, picked_ts, picked_frame_num, (win_start, win_end)


def _fill_coverage_gaps(
    decoder: VideoDecoder,
    results: list[FrameResult],
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
    output_dir: Path,
    state: DedupState,
) -> list[FrameResult]:
    """Inject synthetic ``gap_fill`` frames wherever consecutive keyframes
    are more than ``max_seconds_between_keyframes`` apart.

    Walks the sorted results; when a gap exceeds the threshold, builds a
    synthetic :class:`Moment` at the midpoint and pushes it through the
    same ``_should_save_frame`` predicate used by the main loop. The
    shared ``DedupState`` ensures a gap-fill that duplicates the most
    recently saved keyframe is dropped rather than silently forced in.
    If the midpoint is rejected, the gap is accepted — forcing a bad
    frame would defeat the point of the low-info gate. Iterates until
    no gaps remain or the safety cap is hit.
    """
    if config.max_seconds_between_keyframes <= 0.0 or len(results) < 2:
        return results

    max_gap = config.max_seconds_between_keyframes
    meta = decoder.get_metadata()
    filled = list(results)
    skip_midpoints: set[float] = set()

    for _ in range(50):
        filled.sort(key=lambda r: r.video_timestamp or 0.0)
        inserted = False
        for i in range(len(filled) - 1):
            prev_ts = filled[i].video_timestamp or 0.0
            next_ts = filled[i + 1].video_timestamp or 0.0
            gap = next_ts - prev_ts
            if gap <= max_gap:
                continue
            midpoint = round((prev_ts + next_ts) / 2.0, 3)
            if midpoint in skip_midpoints:
                continue
            synthetic = Moment(
                timestamp=midpoint,
                visual_context_goal="Coverage gap fill",
                textual_anchor="",
                downstream_utility=("Maintain temporal coverage between triggered moments."),
                source="gap_fill",
            )
            picked = _pick_frame_for_moment(decoder, synthetic, transcript, config)
            if picked is None:
                skip_midpoints.add(midpoint)
                logger.info(
                    "Gap fill at %.2fs rejected or unavailable; accepting gap.",
                    midpoint,
                )
                continue
            picked_frame, picked_ts, picked_frame_num, _win = picked
            gap_ocr_text, gap_ocr_tokens = _ocr_text_and_tokens(
                picked_frame, config.gallery_ocr_min_dim
            )
            gap_dhash = dhash_64(picked_frame)
            decision = _should_save_frame(
                frame=picked_frame,
                frame_tokens=gap_ocr_tokens,
                frame_dhash=gap_dhash,
                moment=synthetic,
                state=state,
                config=config,
            )
            if decision is not SaveDecision.SAVE:
                # Gap-fill moments are never anchors, so MERGE cannot occur.
                # DROP means the midpoint duplicates a prior keyframe —
                # give up on this gap and mark the midpoint skipped so we
                # don't retry it.
                skip_midpoints.add(midpoint)
                logger.info(
                    "Gap fill at %.2fs dropped by _should_save_frame; accepting gap.",
                    midpoint,
                )
                continue

            frame_id = f"demo_gapfill_{int(picked_ts * 1000):08d}ms"
            asset_path = save_keyframe(picked_frame, output_dir, frame_id, fmt="jpg")
            fr = FrameResult(
                frame_id=frame_id,
                event_type=EventType.KEYFRAME,
                is_keyframe=True,
                perceptual_hash="",
                frame_width=picked_frame.shape[1],
                frame_height=picked_frame.shape[0],
                source_format="video",
                asset_path=str(asset_path),
                trigger_type="transcript_trigger",
                source_video=meta.filename,
                video_timestamp=picked_ts,
                video_frame_number=picked_frame_num,
                video_duration=meta.duration,
                visual_context_goal=synthetic.visual_context_goal,
                textual_anchor=synthetic.textual_anchor,
                downstream_utility=synthetic.downstream_utility,
                moment_source="gap_fill",
                alignment_confidence="temporal_only",
                ocr_text=gap_ocr_text,
                ocr_tokens=sorted(gap_ocr_tokens),
            )
            filled.append(fr)
            state.update(tokens=gap_ocr_tokens, dhash=gap_dhash, result=fr)
            inserted = True
            break
        if not inserted:
            break

    filled.sort(key=lambda r: r.video_timestamp or 0.0)
    for i, r in enumerate(filled, start=1):
        r.keyframe_index = i
    return filled


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
    picks the best-content frame, runs the unified dedup predicate, and saves
    surviving frames as keyframes. Anchors that collide with the prior saved
    frame are merged into that frame's ``anchors`` list rather than dropped.
    """
    output_dir = Path(output_dir)
    meta = decoder.get_metadata()
    results: list[FrameResult] = []
    state = DedupState()

    tail_cutoff: float | None = None
    if config.tail_skip_ratio > 0.0 and meta.duration > 0.0:
        tail_cutoff = meta.duration * (1.0 - config.tail_skip_ratio)

    for moment in moments:
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

        picked = _pick_frame_for_moment(decoder, moment, transcript, config)
        if picked is None:
            continue
        picked_frame, picked_ts, picked_frame_num, picked_window = picked

        # --- Caption/image alignment block ---
        caption_tokens = _normalize_tokens(f"{moment.visual_context_goal} {moment.textual_anchor}")
        alignment_confidence: str = "content"
        frame_ocr_text, frame_ocr_tokens = _ocr_text_and_tokens(
            picked_frame, config.gallery_ocr_min_dim
        )
        if caption_tokens:
            frame_tokens = frame_ocr_tokens
            if not (frame_tokens & caption_tokens):
                seg = _find_segment_for_timestamp(moment.timestamp, transcript)
                # Skip retry if the window already spans the whole segment (nothing to widen into).
                already_exhausted = seg is not None and (
                    picked_window[0] <= seg.start and picked_window[1] >= seg.end
                )
                if not already_exhausted:
                    retry = _pick_frame_for_moment(
                        decoder, moment, transcript, config, extra_window_sec=5.0
                    )
                    if retry is not None:
                        retry_frame, retry_ts, retry_fnum, _ = retry
                        retry_text, retry_tokens = _ocr_text_and_tokens(
                            retry_frame, config.gallery_ocr_min_dim
                        )
                        if retry_tokens & caption_tokens:
                            picked_frame, picked_ts, picked_frame_num = (
                                retry_frame,
                                retry_ts,
                                retry_fnum,
                            )
                            frame_ocr_text, frame_ocr_tokens = retry_text, retry_tokens
                            logger.info(
                                "Moment at %.2fs: widened window recovered a "
                                "caption-aligned frame at %.2fs.",
                                moment.timestamp,
                                retry_ts,
                            )
                        else:
                            alignment_confidence = "temporal_only"
                    else:
                        alignment_confidence = "temporal_only"
                else:
                    alignment_confidence = "temporal_only"
                if alignment_confidence == "temporal_only":
                    logger.info(
                        "Moment at %.2fs ('%s'): kept temporal_only — no OCR "
                        "token overlap with caption.",
                        moment.timestamp,
                        moment.visual_context_goal,
                    )
        # --- End alignment block ---

        frame_dhash = dhash_64(picked_frame)
        decision = _should_save_frame(
            frame=picked_frame,
            frame_tokens=frame_ocr_tokens,
            frame_dhash=frame_dhash,
            moment=moment,
            state=state,
            config=config,
        )

        if decision is SaveDecision.DROP:
            logger.info(
                "Moment at %.2fs ('%s') dropped by _should_save_frame.",
                moment.timestamp,
                moment.visual_context_goal,
            )
            continue

        if decision is SaveDecision.MERGE_INTO_PREV:
            if state.last_saved_result is not None:
                state.last_saved_result.anchors.append(
                    AnchorRef(
                        timestamp=moment.timestamp,
                        visual_context_goal=moment.visual_context_goal,
                        textual_anchor=moment.textual_anchor,
                    )
                )
                logger.info(
                    "Anchor at %.2fs merged into prior saved frame (dedup collision).",
                    moment.timestamp,
                )
            continue

        # SAVE path
        next_idx = len(results) + 1
        frame_id = f"demo_{next_idx:04d}_{int(picked_ts * 1000):08d}ms"
        asset_path = save_keyframe(picked_frame, output_dir, frame_id, fmt="jpg")

        fr = FrameResult(
            frame_id=frame_id,
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="",
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
            keyframe_index=next_idx,
            alignment_confidence=alignment_confidence,
            ocr_text=frame_ocr_text,
            ocr_tokens=sorted(frame_ocr_tokens),
        )
        results.append(fr)
        state.update(tokens=frame_ocr_tokens, dhash=frame_dhash, result=fr)

    results = _fill_coverage_gaps(decoder, results, transcript, config, output_dir, state)

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
