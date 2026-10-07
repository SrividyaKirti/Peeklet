"""annotate(): video + transcript -> annotated transcript with screenshots."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from peeklet.align import align_lines, screen_at
from peeklet.config import PeekletConfig
from peeklet.cues import verbal_cue_checkpoints
from peeklet.image_utils import decode_jpeg, encode_jpeg
from peeklet.llm.base import (
    JudgeRequest,
    JudgmentCache,
    build_llm_client,
    has_credentials,
    judge_screens,
)
from peeklet.render import (
    Entries,
    RunStats,
    build_entries,
    image_filenames,
    write_debug,
    write_outputs,
)
from peeklet.score import compute_signals, rank_screens, score_screens, select_screens
from peeklet.screen import ocr_word_boxes, require_tesseract
from peeklet.screens import Sample, build_screens
from peeklet.speech import detect_speech_segments, speech_onset_checkpoints
from peeklet.transcript import action_item_checkpoints, parse_transcript, split_long_lines
from peeklet.types import format_hms
from peeklet.video import VideoDecoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peeklet.llm.base import LLMClient
    from peeklet.screens import OcrFn
    from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment

logger = logging.getLogger(__name__)


def annotate(
    video: str | Path,
    transcript: str | Path,
    out_dir: str | Path,
    *,
    max_images: int | None = None,
    config: PeekletConfig | None = None,
    use_llm: bool | None = None,
    debug: bool = False,
    llm_client: LLMClient | None = None,
    ocr: OcrFn | None = None,
) -> Entries:
    """Run the full pipeline and write transcript.json + frame images to out_dir."""
    t0 = time.monotonic()
    cfg = (config or PeekletConfig()).model_copy(deep=True)
    if max_images is not None:
        cfg.max_images = max_images
    if use_llm is not None:
        cfg.use_llm = use_llm
    video, transcript, out_dir = Path(video), Path(transcript), Path(out_dir)
    warnings: list[str] = []

    def warn(msg: str) -> None:
        logger.warning(msg)
        warnings.append(msg)

    if ocr is None:
        require_tesseract()

    # 1. Transcript
    lines = split_long_lines(parse_transcript(transcript), cfg.max_line_seconds)

    # 2. Checkpoints
    checkpoints: list[Checkpoint] = list(action_item_checkpoints(transcript))
    checkpoints += verbal_cue_checkpoints(lines, cfg.cue_threshold)
    try:
        ranges = detect_speech_segments(video, silence_threshold_dbfs=cfg.silence_threshold_dbfs)
        checkpoints += speech_onset_checkpoints(ranges, cfg.min_pause_seconds)
    except Exception as exc:  # no audio track, undecodable audio, missing ffmpeg
        warn(f"speech detection skipped ({exc}); no speech-onset checkpoints")

    # 3–4. Screens
    if ocr is None:

        def ocr_fn(frame: Any) -> Any:
            return ocr_word_boxes(frame, cfg.ocr_max_dim)

        ocr = ocr_fn
    decoder = VideoDecoder(video)
    meta = decoder.get_metadata()
    if lines and lines[-1].end > meta.duration + 1.0:
        warn(f"transcript runs past the end of the video ({format_hms(meta.duration)})")
    samples = (
        Sample(t=ts, frame=frame)
        for frame, ts, _ in decoder.iter_coarse_frames_seek(cfg.sample_fps)
    )
    t_screens = time.monotonic()
    sp = build_screens(samples, checkpoints, cfg, ocr, video_end=meta.duration)
    screen_seconds = time.monotonic() - t_screens
    for w in sp.warnings:
        warn(w)
    screens = sp.screens
    if not screens:
        warn("no usable screens found (only low-information frames)")

    # 5–6. Align and score
    line_screens = align_lines(lines, screens, cfg.lead_seconds)
    signals = compute_signals(screens, lines, line_screens, checkpoints)
    scores = score_screens(signals, cfg.score_weights, cfg.anchor_bonus)

    # 7. LLM judge
    judgments: dict[str, ScreenJudgment | None] | None = None
    shortlist: list[Screen] = []
    if cfg.use_llm and cfg.max_images > 0 and screens:
        client = llm_client
        if client is None:
            if has_credentials(cfg.llm_provider):
                client = build_llm_client(cfg.llm_provider, cfg.llm_model)
            else:
                warn(f"no credentials for LLM provider {cfg.llm_provider!r}; running as --no-llm")
        if client is not None:
            k = math.ceil(cfg.shortlist_factor * cfg.max_images)
            shortlist = rank_screens(screens, scores)[:k]
            requests = _build_requests(shortlist, lines, line_screens, checkpoints, cfg)
            cache = JudgmentCache(Path(cfg.llm_cache_dir), client.provider, client.model)
            judgments, llm_warnings = judge_screens(client, requests, cache, cfg.llm_concurrency)
            for w in llm_warnings:
                warnings.append(w)
            if judgments and all(v is None for v in judgments.values()):
                warn("every LLM call failed; running as --no-llm")
                judgments = None

    # 8–9. Select and render
    kept = select_screens(screens, scores, signals, judgments, cfg.max_images)
    files = image_filenames(kept)
    action_items = [c for c in checkpoints if c.kind == "action_item"]
    entries = build_entries(lines, line_screens, files, judgments or {}, action_items)
    write_outputs(entries, kept, files, out_dir)
    stats = RunStats(
        lines=len(lines),
        screens_found=len(screens),
        shortlisted=len(shortlist),
        kept=len(kept),
        warnings=warnings,
    )
    if debug:
        write_debug(
            out_dir / "debug.json",
            _debug_payload(
                screens,
                signals,
                scores,
                shortlist,
                judgments,
                kept,
                lines,
                line_screens,
                checkpoints,
                warnings,
                {"seconds_total": time.monotonic() - t0, "seconds_screen_pass": screen_seconds},
            ),
        )
    return Entries(items=entries, out_dir=out_dir, stats=stats, max_image_edge=cfg.max_image_edge)


def _build_requests(
    shortlist: Sequence[Screen],
    lines: Sequence[Line],
    line_screens: Sequence[str | None],
    checkpoints: Sequence[Checkpoint],
    cfg: PeekletConfig,
) -> list[JudgeRequest]:
    requests: list[JudgeRequest] = []
    for s in shortlist:
        idx = [i for i, sid in enumerate(line_screens) if sid == s.id]
        has_cp = {
            i
            for i in idx
            if any(lines[i].start <= c.t <= lines[i].end + cfg.lead_seconds for c in checkpoints)
        }
        chosen = sorted(sorted(idx, key=lambda i: (i not in has_cp, i))[: cfg.llm_max_lines])
        rendered = tuple(
            f"[{format_hms(lines[i].start)}] {lines[i].speaker or 'Speaker'}: {lines[i].text}"
            for i in chosen
        )
        items = tuple(
            c.label for c in checkpoints if c.kind == "action_item" and screen_at([s], c.t)
        )
        image = encode_jpeg(decode_jpeg(s.image_jpeg), quality=90, max_edge=cfg.max_image_edge)
        requests.append(JudgeRequest(s.id, image, s.ocr_text, rendered, items))
    return requests


def _judgment_dict(j: ScreenJudgment | None) -> dict[str, Any] | None:
    return None if j is None else {"include": j.include, "reason": j.reason}


def _debug_payload(
    screens: Sequence[Screen],
    signals: dict[str, Any],
    scores: dict[str, float],
    shortlist: Sequence[Screen],
    judgments: dict[str, ScreenJudgment | None] | None,
    kept: Sequence[Screen],
    lines: Sequence[Line],
    line_screens: Sequence[str | None],
    checkpoints: Sequence[Checkpoint],
    warnings: list[str],
    timing: dict[str, float],
) -> dict[str, Any]:
    short_ids = {s.id for s in shortlist}
    kept_ids = {s.id for s in kept}
    return {
        "screens": [
            {
                "id": s.id,
                "frame_t": s.frame_t,
                "occurrences": s.occurrences,
                "word_count": s.word_count,
                "signals": signals[s.id].as_dict(),
                "score": scores[s.id],
                "shortlisted": s.id in short_ids,
                "judgment": _judgment_dict((judgments or {}).get(s.id)),
                "kept": s.id in kept_ids,
            }
            for s in screens
        ],
        "lines": [
            {"start": ln.start, "end": ln.end, "screen_id": sid}
            for ln, sid in zip(lines, line_screens, strict=True)
        ],
        "checkpoints": [{"t": c.t, "kind": c.kind, "label": c.label} for c in checkpoints],
        "warnings": warnings,
        "timing": timing,
    }
