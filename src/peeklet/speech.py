"""Speech/silence detection from a media file's audio track."""

from __future__ import annotations

from typing import TYPE_CHECKING

from peeklet.types import Line

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
        raise ImportError(
            "Audio detection requires additional dependencies. "
            "Install with: pip install peeklet[video]"
        ) from None


def detect_speech_segments(
    audio_path: Path,
    chunk_ms: int = 500,
    silence_threshold_dbfs: float = -40.0,
) -> list[Line]:
    """Detect speech segments using RMS energy thresholds.

    Divides audio into chunks and classifies each as speech or silence
    based on dBFS level. Merges consecutive speech chunks into segments.

    Returns list of Line with text="[speech]" for detected speech.
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

    return [Line(start=s, end=e, text="[speech]") for s, e in speech_ranges]
