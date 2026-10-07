"""Transcript parsing (SRT, VTT, Fathom markdown) and Fathom action anchors."""

from __future__ import annotations

import re
from pathlib import Path

from peeklet.types import Line


def parse_transcript(path: Path) -> list[Line]:
    """Parse a transcript file into timestamped segments.

    Auto-detects format by file extension:
    - ``.srt`` — SubRip
    - ``.vtt`` — WebVTT
    - ``.md`` — Fathom-style markdown (lines like
      ``++[@MM:SS](url?timestamp=N.NN)++ - **Speaker**`` followed by spoken text)
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return _parse_vtt(text)
    if suffix == ".md":
        return _parse_fathom_md(text)
    return _parse_srt(text)


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


def parse_fathom_anchors(text: str) -> list:
    """Extract ACTION ITEM...WATCH markers as privileged anchor Moments.

    Fathom inlines these as::

        **ACTION ITEM: <desc> - ++[WATCH](https://fathom.video/...?timestamp=<s>)++**

    Each marker often appears twice on consecutive lines; this function
    deduplicates by (timestamp, description) before returning.

    Returns Moments sorted by timestamp with ``source="anchor"``.
    """
    from peeklet.types import Moment

    seen: set[tuple[float, str]] = set()
    anchors: list[Moment] = []

    for match in _FATHOM_ACTION_RE.finditer(text):
        description = match.group(1).strip()
        timestamp = float(match.group(2))
        key = (timestamp, description)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            Moment(
                timestamp=timestamp,
                visual_context_goal=description,
                textual_anchor=match.group(0),
                downstream_utility="Action item flagged by meeting tool — guaranteed capture",
                source="anchor",
            )
        )

    anchors.sort(key=lambda m: m.timestamp)
    return anchors


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
