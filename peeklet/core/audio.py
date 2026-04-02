"""Audio analysis — transcript parsing and speech/silence detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A timestamped segment from an SRT or VTT transcript."""

    start: float  # seconds
    end: float  # seconds
    text: str


def parse_transcript(path: Path) -> list[TranscriptSegment]:
    """Parse an SRT or VTT file into timestamped segments.

    Auto-detects format by file extension (.srt or .vtt).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return _parse_vtt(text)
    return _parse_srt(text)


def _parse_timestamp(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


_TIMESTAMP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})"
)


def _parse_srt(text: str) -> list[TranscriptSegment]:
    """Parse SRT format."""
    segments: list[TranscriptSegment] = []
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
            segments.append(TranscriptSegment(start=start, end=end, text=content))
    return segments


def _parse_vtt(text: str) -> list[TranscriptSegment]:
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


def align_transcript(timestamp: float, segments: list[TranscriptSegment]) -> str | None:
    """Find transcript segment(s) overlapping the given timestamp.

    Returns joined text if multiple segments overlap, or None if no match.
    """
    matches = [s for s in segments if s.start <= timestamp <= s.end]
    if not matches:
        return None
    return " | ".join(s.text for s in matches)
