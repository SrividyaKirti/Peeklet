"""Speech/silence detection from a media file's audio track."""

from __future__ import annotations

from typing import TYPE_CHECKING

from peeklet.types import Checkpoint

if TYPE_CHECKING:
    from pathlib import Path


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
        raise ImportError("Speech detection requires pydub. Reinstall peeklet.") from None


def detect_speech_segments(
    media_path: Path,
    chunk_ms: int = 500,
    silence_threshold_dbfs: float = -40.0,
) -> list[tuple[float, float]]:
    """Detect speech segments using RMS energy thresholds.

    Divides audio into chunks and classifies each as speech or silence
    based on dBFS level. Merges consecutive speech chunks into segments.

    Returns list of (start, end) tuples in seconds.
    """
    _check_audio_deps()
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(media_path))
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

    return speech_ranges


def speech_onset_checkpoints(
    ranges: list[tuple[float, float]], min_pause_seconds: float
) -> list[Checkpoint]:
    """Onsets of speech that follow at least min_pause_seconds of silence.

    The first speech range always counts (silence before the recording starts).
    """
    out: list[Checkpoint] = []
    prev_end: float | None = None
    for start, end in ranges:
        if prev_end is None or start - prev_end >= min_pause_seconds:
            out.append(Checkpoint(t=start, kind="speech_onset"))
        prev_end = end
    return out
