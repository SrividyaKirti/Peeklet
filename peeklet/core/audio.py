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


def _check_audio_deps() -> None:
    """Raise a clear error if audio dependencies are not installed."""
    try:
        import os

        import imageio_ffmpeg

        os.environ.setdefault("FFMPEG_BINARY", imageio_ffmpeg.get_ffmpeg_exe())
        from pydub import AudioSegment

        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    try:
        import pydub  # noqa: F401
    except ImportError:
        raise ImportError(
            "Audio detection requires additional dependencies. "
            "Install with: pip install peeklet[video]"
        ) from None


def detect_speech_segments(
    audio_path: Path,
    chunk_ms: int = 500,
    silence_threshold_dbfs: float = -40.0,
) -> list[TranscriptSegment]:
    """Detect speech segments using RMS energy thresholds.

    Divides audio into chunks and classifies each as speech or silence
    based on dBFS level. Merges consecutive speech chunks into segments.

    Returns list of TranscriptSegment with text="[speech]" for detected speech.
    """
    _check_audio_deps()
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    duration_s = len(audio) / 1000.0

    speech_ranges: list[tuple[float, float]] = []
    current_start: float | None = None

    for chunk_start_ms in range(0, len(audio), chunk_ms):
        chunk_end_ms = min(chunk_start_ms + chunk_ms, len(audio))
        chunk = audio[chunk_start_ms:chunk_end_ms]

        is_speech = chunk.dBFS > silence_threshold_dbfs

        start_s = chunk_start_ms / 1000.0

        if is_speech:
            if current_start is None:
                current_start = start_s
        else:
            if current_start is not None:
                speech_ranges.append((current_start, start_s))
                current_start = None

    # Close any trailing speech segment
    if current_start is not None:
        speech_ranges.append((current_start, duration_s))

    return [
        TranscriptSegment(start=s, end=e, text="[speech]")
        for s, e in speech_ranges
    ]


def get_audio_activity(
    timestamp: float, speech_segments: list[TranscriptSegment]
) -> str:
    """Return 'speech' or 'silence' for a given timestamp."""
    for seg in speech_segments:
        if seg.start <= timestamp <= seg.end:
            return "speech"
    return "silence"
