"""JSON and Markdown context export for LLM consumption."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

_WATCH_URL_RE = re.compile(r"\[WATCH\]\((https?://[^)]+)\)")

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment
    from peeklet.utils.types import FrameResult, MomentEntry, Screen


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def timestamp_filename(seconds: float) -> str:
    """Convert seconds to screenshot_HH_MM_SS_mmm filename (no extension)."""
    ts = format_timestamp(seconds)
    name = ts.replace(":", "_").replace(".", "_")
    return f"screenshot_{name}"


def build_context(
    filename: str,
    duration_s: float,
    results: list[FrameResult],
    segments: list[TranscriptSegment],
) -> dict[str, Any]:
    """Build the context data structure with bidirectional linking."""
    keyframes = [r for r in results if r.is_keyframe]

    screenshots: list[dict[str, Any]] = []
    prev_ts: float | None = None
    for i, r in enumerate(keyframes):
        ts = r.video_timestamp or 0.0
        since_prev = (ts - prev_ts) if prev_ts is not None else None
        # Derive filename from the actual asset_path (handles png/jpg/etc)
        file_name = Path(r.asset_path).name if r.asset_path else f"{timestamp_filename(ts)}.jpg"
        screenshots.append(
            {
                "id": i + 1,
                "file": file_name,
                "timestamp_s": ts,
                "timestamp": format_timestamp(ts),
                "trigger": r.trigger_type,
                "change_magnitude": r.change_magnitude,
                "seconds_since_prev_screenshot": since_prev,
                "transcript_ids": [],
                "visual_context_goal": r.visual_context_goal or "",
                "textual_anchor": r.textual_anchor or "",
                "downstream_utility": r.downstream_utility or "",
                "moment_source": r.moment_source or "",
                "alignment_confidence": r.alignment_confidence or "content",
                "ocr_text": r.ocr_text or "",
                "ocr_tokens": list(r.ocr_tokens or []),
            }
        )
        prev_ts = ts

    transcript: list[dict[str, Any]] = []
    for j, seg in enumerate(segments):
        transcript.append(
            {
                "id": j + 1,
                "start_s": seg.start,
                "end_s": seg.end,
                "start": format_timestamp(seg.start),
                "end": format_timestamp(seg.end),
                "text": seg.text,
                "speaker": seg.speaker,
                "screenshot_ids": [],
            }
        )

    for s_entry in screenshots:
        ts = s_entry["timestamp_s"]
        for t_entry in transcript:
            seg = segments[t_entry["id"] - 1]
            if seg.start <= ts <= seg.end:
                s_entry["transcript_ids"].append(t_entry["id"])
                t_entry["screenshot_ids"].append(s_entry["id"])

    return {
        "video": {
            "filename": filename,
            "duration_s": duration_s,
            "total_screenshots": len(screenshots),
        },
        "screenshots": screenshots,
        "transcript": transcript,
    }


def write_context_json(ctx: dict[str, Any], path: Path | str) -> None:
    """Write context data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ctx, indent=2, ensure_ascii=False) + "\n")


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    parts: list[str] = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    parts.append(f"{s:.1f}s")
    return " ".join(parts)


def _screenshot_block(s: dict[str, Any]) -> list[str]:
    """Render a screenshot annotation block as Markdown lines.

    Compacted form: bolded header + image only. The ``[guaranteed]`` tag
    is appended to the header when the screenshot originated from a
    Fathom action-item anchor. The blockquoted ``textual_anchor`` and
    the ``downstream_utility`` caption are intentionally omitted — both
    duplicated information already present in the transcript or header.
    """
    goal = s.get("visual_context_goal") or (s.get("trigger") or "visual_change").replace("_", " ")
    header = f"**Screenshot {s['id']} ({s['timestamp']}) — {goal}**"
    if s.get("moment_source") == "anchor":
        header += " [guaranteed]"
    return [
        header,
        f"![Screenshot]({s['file']})",
        "",
    ]


def _format_short_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS for display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def write_context_markdown(ctx: dict[str, Any], path: Path | str) -> None:
    """Write context data as Markdown — chronologically interleaved transcript
    and screenshot blocks optimized for downstream MLLM consumption.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    video = ctx["video"]
    lines: list[str] = [
        f"# Meeting Context: {video['filename']}",
        "",
        f"Duration: {_format_duration(video['duration_s'])}"
        f" | Screenshots: {video['total_screenshots']}",
        "",
    ]

    screenshots = sorted(ctx["screenshots"], key=lambda s: s["timestamp_s"])

    # Visual Table of Contents
    if screenshots:
        lines.append("## Visual Table of Contents")
        lines.append("")
        lines.append("| # | Time | Visual Context |")
        lines.append("|---|------|---------------|")
        for s in screenshots:
            short_ts = _format_short_timestamp(s["timestamp_s"])
            goal = s.get("visual_context_goal") or "\u2014"
            lines.append(f"| {s['id']} | {short_ts} | {goal} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Timeline")
    lines.append("")

    transcript = sorted(ctx["transcript"], key=lambda t: t["start_s"])

    seg_idx = 0
    shot_idx = 0
    inf = float("inf")
    while seg_idx < len(transcript) or shot_idx < len(screenshots):
        next_seg_time = transcript[seg_idx]["start_s"] if seg_idx < len(transcript) else inf
        next_shot_time = (
            screenshots[shot_idx]["timestamp_s"] if shot_idx < len(screenshots) else inf
        )

        if next_seg_time <= next_shot_time:
            seg = transcript[seg_idx]
            speaker = seg.get("speaker")
            if speaker:
                lines.append(f"**[{seg['start']} \u2192 {seg['end']}] {speaker}**")
            else:
                lines.append(f"**[{seg['start']} \u2192 {seg['end']}]**")
            lines.append(seg["text"])
            lines.append("")
            seg_idx += 1
        else:
            lines.extend(_screenshot_block(screenshots[shot_idx]))
            shot_idx += 1

    references: list[tuple[int, str]] = []
    for s in screenshots:
        if s.get("moment_source") != "anchor":
            continue
        match = _WATCH_URL_RE.search(s.get("textual_anchor") or "")
        if match:
            references.append((s["id"], match.group(1)))

    if references:
        lines.append("## References")
        lines.append("")
        for shot_id, url in references:
            lines.append(f"- Screenshot {shot_id}: {url}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def _format_mm_ss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_demo_context(
    filename: str,
    duration_s: float,
    screens: list[Screen],
    moments: list[MomentEntry],
) -> dict[str, Any]:
    """Build the demo context.json structure (screens + moments)."""
    return {
        "video": {
            "filename": filename,
            "duration_s": duration_s,
            "unique_screens": len(screens),
            "total_moments": len(moments),
        },
        "screens": [
            {
                "id": s.screen_id,
                "image_path": s.image_path,
                "first_seen_ms": s.first_seen_ms,
                "fingerprint": {
                    "url": s.fingerprint.url,
                    "heading": s.fingerprint.heading,
                    "sidebar_text": s.fingerprint.sidebar_text,
                    "header_phash": f"{s.fingerprint.header_phash:016x}",
                },
                "ocr_text": s.ocr_text,
            }
            for s in screens
        ],
        "moments": [
            {
                "timestamp_ms": m.timestamp_ms,
                "caption": m.caption,
                "type": m.type,
                "screen_id": m.screen_id,
                "image_unavailable": m.image_unavailable,
            }
            for m in moments
        ],
    }


def write_demo_context_json(ctx: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ctx, indent=2, ensure_ascii=False) + "\n")


def write_demo_context_markdown(
    *,
    path: Path | str,
    video_filename: str,
    duration_s: float,
    screens: list[Screen],
    moments: list[MomentEntry],
) -> None:
    """Render the demo context.md format with shared image paths + backrefs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    screen_index = {s.screen_id: s for s in screens}
    first_seen_at_mmss: dict[str, str] = {}

    lines: list[str] = [
        f"# Demo Context: {video_filename}",
        "",
        f"Duration: {_format_duration(duration_s)} | Unique screens: "
        f"{len(screens)} | Moments: {len(moments)}",
        "",
        "---",
        "",
    ]

    for m in moments:
        ts_s = m.timestamp_ms / 1000.0
        mmss = _format_mm_ss(ts_s)
        type_label = "ACTION ITEM" if m.type == "action_item" else "LLM"
        lines.append(f"### {mmss} — {type_label}")
        lines.append(f'> "{m.caption}"')
        lines.append("")
        if m.image_unavailable or m.screen_id is None:
            lines.append("_(image unavailable)_")
        else:
            screen = screen_index[m.screen_id]
            lines.append(f"![Screen]({screen.image_path})")
            if m.screen_id in first_seen_at_mmss:
                lines.append(f"↳ same screen as {first_seen_at_mmss[m.screen_id]}")
            else:
                first_seen_at_mmss[m.screen_id] = mmss
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines))
