"""Transcript parsing (SRT, VTT, Fathom markdown) and Fathom action items."""

from __future__ import annotations

import re
from pathlib import Path

from peeklet.types import Checkpoint, Line


class TranscriptError(ValueError):
    """The transcript is missing, unreadable, or contains no timestamped lines."""


def parse_transcript(path: Path) -> list[Line]:
    """Parse .srt, .vtt or Fathom .md into lines. Raises TranscriptError on bad input."""
    path = Path(path)
    if not path.is_file():
        raise TranscriptError(f"Transcript not found: {path}")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise TranscriptError(f"Could not read transcript {path}: {exc}") from exc
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        lines = _parse_vtt(text)
    elif suffix == ".md":
        lines = _parse_fathom_md(text)
    else:
        lines = _parse_srt(text)
    if not lines:
        raise TranscriptError(f"Transcript {path} has no timestamped lines")
    return lines


_SENTENCE_BREAK_RE = re.compile(r"(?<=[.?!])\s+")


def split_long_lines(lines: list[Line], max_seconds: float) -> list[Line]:
    """Split lines longer than max_seconds at sentence boundaries, timing pieces by length."""
    out: list[Line] = []
    for ln in lines:
        duration = ln.end - ln.start
        parts = [p.strip() for p in _SENTENCE_BREAK_RE.split(ln.text) if p.strip()]
        if duration <= max_seconds or len(parts) < 2:
            out.append(ln)
            continue
        total = sum(len(p) for p in parts)
        t = ln.start
        for i, part in enumerate(parts):
            end = ln.end if i == len(parts) - 1 else round(t + duration * len(part) / total, 3)
            out.append(Line(start=t, end=end, text=part, speaker=ln.speaker))
            t = end
    return out


def parse_fathom_action_items(text: str) -> list[Checkpoint]:
    """Extract Fathom ACTION ITEM ... [WATCH](...?timestamp=N) markers, deduplicated."""
    seen: set[tuple[float, str]] = set()
    found: list[Checkpoint] = []
    for match in _FATHOM_ACTION_RE.finditer(text):
        label = match.group(1).strip()
        t = float(match.group(2))
        if (t, label) in seen:
            continue
        seen.add((t, label))
        found.append(Checkpoint(t=t, kind="action_item", label=label))
    found.sort(key=lambda c: c.t)
    return found


def action_item_checkpoints(path: Path) -> list[Checkpoint]:
    """Action-item checkpoints for Fathom markdown transcripts; [] for other formats."""
    path = Path(path)
    if path.suffix.lower() != ".md":
        return []
    return parse_fathom_action_items(path.read_text(encoding="utf-8"))


def _parse_timestamp(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


_TIMESTAMP_RE = re.compile(r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})")


def _parse_srt(text: str) -> list[Line]:
    """Parse SRT format."""
    segments: list[Line] = []
    blocks = re.split(r"\n\n+", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        match = None
        timestamp_line_idx = -1
        for i, line in enumerate(lines):
            match = _TIMESTAMP_RE.search(line)
            if match:
                timestamp_line_idx = i
                break
        if match is None or timestamp_line_idx == -1:
            continue
        start = _parse_timestamp(match.group(1))
        end = _parse_timestamp(match.group(2))
        text_lines = lines[timestamp_line_idx + 1 :]
        content = " ".join(line.strip() for line in text_lines if line.strip())
        if content:
            segments.append(Line(start=start, end=end, text=content))
    return segments


# Fathom-style markdown timestamp line:
#   ++[@0:03](https://fathom.video/calls/123?timestamp=3.0)++ - **Speaker Name**
# We anchor at the start of the line and require the trailing speaker block,
# so embedded ``[WATCH](...?timestamp=...)`` markers inside speech text don't
# falsely split segments.
_FATHOM_TS_LINE_RE = re.compile(
    r"^\s*\+\+\[@\d+:\d+\]\([^)]*\?timestamp=(\d+(?:\.\d+)?)\)\+\+\s*-\s*\*\*([^*]+)\*\*\s*$"
)

_FATHOM_ACTION_RE = re.compile(
    r"\*\*ACTION ITEM:\s*(.+?)\s*-\s*"
    r"\+\+\[WATCH\]\([^?]*\?timestamp=(\d+(?:\.\d+)?)\)\+\+\*\*"
)


def _parse_fathom_md(text: str) -> list[Line]:
    """Parse a Fathom-style markdown transcript.

    Each segment looks like::

        ++[@0:03](https://fathom.video/calls/123?timestamp=3.0)++ - **Speaker**
        Spoken text here, possibly across
        multiple lines until a blank line or the next timestamp marker.

    The numeric ``?timestamp=`` value is used for the start time (more
    precise than the visible ``MM:SS``). The end time of each segment is
    inferred from the start of the next segment.
    """
    raw_segments: list[tuple[float, str | None, list[str]]] = []
    current_start: float | None = None
    current_speaker: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = _FATHOM_TS_LINE_RE.match(line)
        if match:
            ts = float(match.group(1))
            # Skip duplicate timestamp lines (Fathom often emits two in a row)
            if current_start is not None and ts == current_start and not current_lines:
                continue
            # Flush previous segment
            if current_start is not None and current_lines:
                raw_segments.append((current_start, current_speaker, current_lines))
            current_start = ts
            current_speaker = match.group(2).strip()
            current_lines = []
            continue
        if current_start is None:
            continue  # skip frontmatter before the first timestamp
        stripped = line.strip()
        if stripped and (not current_lines or current_lines[-1] != stripped):
            current_lines.append(stripped)

    if current_start is not None and current_lines:
        raw_segments.append((current_start, current_speaker, current_lines))

    segments: list[Line] = []
    for i, (start, speaker, lines) in enumerate(raw_segments):
        end = raw_segments[i + 1][0] if i + 1 < len(raw_segments) else start + 5.0
        content = " ".join(lines).strip()
        if content:
            segments.append(Line(start=start, end=end, text=content, speaker=speaker))
    return segments


def _parse_vtt(text: str) -> list[Line]:
    """Parse WebVTT format."""
    lines = text.splitlines()
    body_start = 0
    if lines and lines[0].startswith("WEBVTT"):
        for i in range(1, len(lines)):
            if lines[i].strip() == "":
                body_start = i + 1
                break
        else:
            body_start = len(lines)

    body = "\n".join(lines[body_start:])
    return _parse_srt(body)
