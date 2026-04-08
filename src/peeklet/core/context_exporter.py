"""JSON and Markdown context export for LLM consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment
    from peeklet.utils.types import FrameResult


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
    """Render a screenshot annotation block as Markdown lines."""
    trigger_label = (s["trigger"] or "visual_change").replace("_", " ")
    since = s["seconds_since_prev_screenshot"]
    since_str = f"{since:.1f}s" if since is not None else "\u2014"
    return [
        f"## Screenshot {s['id']} ({s['timestamp']}) — {trigger_label}",
        f"![{s['file']}]({s['file']})",
        f"**Change:** {s['change_magnitude']} | **Since prev:** {since_str}",
        "",
    ]


def write_context_markdown(ctx: dict[str, Any], path: Path | str) -> None:
    """Write context data as Markdown — the original transcript with
    screenshot annotation blocks injected inline at their timestamp positions.

    The transcript is the primary document. Screenshots are inserted between
    transcript segments at the chronological position matching their
    ``timestamp_s``. If there is no transcript, screenshots are listed in
    chronological order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    video = ctx["video"]
    lines: list[str] = [
        f"# Video Summary: {video['filename']}",
        f"Duration: {_format_duration(video['duration_s'])}"
        f" | Screenshots: {video['total_screenshots']}",
        "",
        "---",
        "",
    ]

    transcript = sorted(ctx["transcript"], key=lambda t: t["start_s"])
    screenshots = sorted(ctx["screenshots"], key=lambda s: s["timestamp_s"])

    # Merge transcript and screenshots in chronological order. A screenshot
    # is emitted right before the next transcript segment that starts after
    # the screenshot's timestamp, so it appears immediately after the line
    # being spoken when the visual change occurred.
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
            lines.append(f"**[{seg['start']} \u2192 {seg['end']}]** {seg['text']}")
            lines.append("")
            seg_idx += 1
        else:
            lines.extend(_screenshot_block(screenshots[shot_idx]))
            shot_idx += 1

    path.write_text("\n".join(lines) + "\n")
