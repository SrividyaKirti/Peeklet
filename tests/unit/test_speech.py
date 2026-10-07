"""Tests for speech/silence detection."""

from pathlib import Path

import pytest

try:
    import pydub  # noqa: F401

    _has_pydub = True
except ImportError:
    _has_pydub = False


@pytest.mark.skipif(not _has_pydub, reason="requires peeklet[video]")
class TestSpeechSilenceDetection:
    def test_detect_speech_in_audio(self, tmp_path: Path) -> None:
        from pydub import AudioSegment
        from pydub.generators import Sine

        # Generate 3 seconds: 1s silence, 1s tone (speech proxy), 1s silence
        silence = AudioSegment.silent(duration=1000)
        tone = Sine(440).to_audio_segment(duration=1000).apply_gain(-10)
        audio = silence + tone + silence
        audio_path = tmp_path / "test.wav"
        audio.export(str(audio_path), format="wav")

        from peeklet.speech import detect_speech_segments

        speech_segments = detect_speech_segments(audio_path)
        # Should detect speech roughly in the 1-2 second range
        assert len(speech_segments) >= 1
        has_speech_in_middle = any(s.start < 2.0 and s.end > 1.0 for s in speech_segments)
        assert has_speech_in_middle

    def test_silence_only_audio(self, tmp_path: Path) -> None:
        from pydub import AudioSegment

        silence = AudioSegment.silent(duration=2000)
        audio_path = tmp_path / "test.wav"
        silence.export(str(audio_path), format="wav")

        from peeklet.speech import detect_speech_segments

        speech_segments = detect_speech_segments(audio_path)
        assert speech_segments == []
