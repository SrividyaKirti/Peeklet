"""Tests for speech/silence detection."""

import wave
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from peeklet.speech import detect_speech_segments

SR = 16000


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def test_detect_speech_in_audio(tmp_path: Path) -> None:
    t = np.arange(2 * SR) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sil = np.zeros(SR)
    path = tmp_path / "tone.wav"
    _write_wav(path, np.concatenate([sil, tone, sil]))

    [(start, end)] = detect_speech_segments(path)
    assert start == pytest.approx(1.0, abs=0.5)
    assert end == pytest.approx(3.0, abs=0.5)


def test_trailing_speech_closes_at_duration(tmp_path: Path) -> None:
    t = np.arange(SR) / SR
    path = tmp_path / "t.wav"
    _write_wav(path, np.concatenate([np.zeros(SR), 0.5 * np.sin(2 * np.pi * 440 * t)]))
    [(start, end)] = detect_speech_segments(path)
    assert start == pytest.approx(1.0, abs=0.5)
    assert end == pytest.approx(2.0, abs=0.01)


def test_silence_only_audio(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    _write_wav(path, np.zeros(2 * SR))
    assert detect_speech_segments(path) == []


def test_video_without_audio_raises(tmp_path: Path) -> None:
    path = tmp_path / "silent.mp4"
    frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    iio.imwrite(path, frames, plugin="pyav", fps=5, codec="libx264")
    with pytest.raises(RuntimeError, match="could not decode audio"):
        detect_speech_segments(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="could not decode audio"):
        detect_speech_segments(tmp_path / "nope.mp4")


def test_first_range_is_an_onset() -> None:
    from peeklet.speech import speech_onset_checkpoints

    [cp] = speech_onset_checkpoints([(0.5, 3.0)], min_pause_seconds=1.5)
    assert (cp.t, cp.kind) == (0.5, "speech_onset")


def test_onset_requires_minimum_pause() -> None:
    from peeklet.speech import speech_onset_checkpoints

    ranges = [(0.0, 2.0), (2.5, 4.0), (6.0, 8.0)]
    cps = speech_onset_checkpoints(ranges, min_pause_seconds=1.5)
    assert [c.t for c in cps] == [0.0, 6.0]


def test_pause_exactly_at_threshold_counts() -> None:
    from peeklet.speech import speech_onset_checkpoints

    cps = speech_onset_checkpoints([(0.0, 1.0), (2.5, 3.0)], min_pause_seconds=1.5)
    assert [c.t for c in cps] == [0.0, 2.5]


def test_no_ranges() -> None:
    from peeklet.speech import speech_onset_checkpoints

    assert speech_onset_checkpoints([], min_pause_seconds=1.5) == []
