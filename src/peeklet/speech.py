"""Speech/silence detection from a media file's audio track."""

from __future__ import annotations

import math
import subprocess
from typing import TYPE_CHECKING

import imageio_ffmpeg
import numpy as np

from peeklet.types import Checkpoint

if TYPE_CHECKING:
    from pathlib import Path


_SAMPLE_RATE = 16000


def _decode_audio(media_path: Path) -> np.ndarray:
    """Decode a media file's audio to mono 16 kHz int16 samples via bundled ffmpeg."""
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [
            exe, "-v", "error", "-i", str(media_path),
            "-vn", "-ac", "1", "-ar", str(_SAMPLE_RATE), "-f", "s16le", "-",
        ],
        capture_output=True,
        check=False,
    )  # fmt: skip
    samples = np.frombuffer(proc.stdout[: len(proc.stdout) // 2 * 2], dtype=np.int16)
    if proc.returncode != 0 or samples.size == 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        reason = stderr.splitlines()[0] if stderr else "no audio stream"
        raise RuntimeError(f"could not decode audio from {media_path}: {reason}")
    return samples


def detect_speech_segments(
    media_path: Path,
    chunk_ms: int = 500,
    silence_threshold_dbfs: float = -40.0,
) -> list[tuple[float, float]]:
    """Detect speech segments using RMS energy thresholds.

    Decodes the audio with the ffmpeg binary bundled by imageio-ffmpeg (no system
    ffmpeg/ffprobe needed), divides it into chunks and classifies each as speech or
    silence by dBFS level. Merges consecutive speech chunks into segments.

    Returns list of (start, end) tuples in seconds. Raises RuntimeError if the
    audio cannot be decoded.
    """
    samples = _decode_audio(media_path)
    chunk_len = max(1, _SAMPLE_RATE * chunk_ms // 1000)
    duration_s = samples.size / _SAMPLE_RATE

    speech_ranges: list[tuple[float, float]] = []
    current_start: float | None = None

    for offset in range(0, samples.size, chunk_len):
        chunk = samples[offset : offset + chunk_len].astype(np.float64)
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        dbfs = 20 * math.log10(rms / 32768) if rms > 0 else -math.inf
        start_s = offset / _SAMPLE_RATE

        if dbfs > silence_threshold_dbfs:
            if current_start is None:
                current_start = start_s
        elif current_start is not None:
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
