"""Tests for the audio module — transcript parsing and speech detection."""

from pathlib import Path

import pytest

from peeklet.core.audio import TranscriptSegment, align_transcript, parse_transcript

try:
    import pydub  # noqa: F401

    _has_pydub = True
except ImportError:
    _has_pydub = False


class TestParseSrt:
    def test_parse_basic_srt(self, tmp_path: Path) -> None:
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:03,500\n"
            "Hello world\n"
            "\n"
            "2\n"
            "00:00:05,000 --> 00:00:08,200\n"
            "Click the button\n"
            "\n"
        )
        segments = parse_transcript(srt_file)
        assert len(segments) == 2
        assert segments[0] == TranscriptSegment(start=1.0, end=3.5, text="Hello world")
        assert segments[1] == TranscriptSegment(start=5.0, end=8.2, text="Click the button")

    def test_parse_multiline_srt(self, tmp_path: Path) -> None:
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:04,000\nLine one\nLine two\n\n")
        segments = parse_transcript(srt_file)
        assert len(segments) == 1
        assert segments[0].text == "Line one Line two"

    def test_parse_empty_srt(self, tmp_path: Path) -> None:
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("")
        segments = parse_transcript(srt_file)
        assert segments == []


class TestParseVtt:
    def test_parse_basic_vtt(self, tmp_path: Path) -> None:
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(
            "WEBVTT\n"
            "\n"
            "00:00:01.000 --> 00:00:03.500\n"
            "Hello world\n"
            "\n"
            "00:00:05.000 --> 00:00:08.200\n"
            "Click the button\n"
            "\n"
        )
        segments = parse_transcript(vtt_file)
        assert len(segments) == 2
        assert segments[0] == TranscriptSegment(start=1.0, end=3.5, text="Hello world")
        assert segments[1] == TranscriptSegment(start=5.0, end=8.2, text="Click the button")

    def test_vtt_with_header_metadata(self, tmp_path: Path) -> None:
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(
            "WEBVTT\nKind: captions\nLanguage: en\n\n00:00:01.000 --> 00:00:03.000\nFirst line\n\n"
        )
        segments = parse_transcript(vtt_file)
        assert len(segments) == 1
        assert segments[0].text == "First line"


class TestParseFathomMd:
    def test_parse_basic_fathom_md(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text(
            "## Some Title\n"
            "\n"
            "++[@0:00](https://fathom.video/calls/1?timestamp=0.56)++ - **Alice**  \n"
            "Welcome everyone.  \n"
            "\n"
            "++[@0:03](https://fathom.video/calls/1?timestamp=3.0)++ - **Bob**  \n"
            "Thanks for having me.  \n"
            "\n"
        )
        segments = parse_transcript(md)
        assert len(segments) == 2
        assert segments[0].start == 0.56
        assert segments[0].end == 3.0
        assert segments[0].text == "Welcome everyone."
        assert segments[1].start == 3.0
        assert segments[1].text == "Thanks for having me."

    def test_duplicate_timestamp_lines_dedupe(self, tmp_path: Path) -> None:
        # Fathom often emits the speaker line twice in a row
        md = tmp_path / "test.md"
        md.write_text(
            "++[@0:00](https://fathom.video/calls/1?timestamp=0.5)++ - **Alice**  \n"
            "++[@0:00](https://fathom.video/calls/1?timestamp=0.5)++ - **Alice**  \n"
            "Hello world.  \n"
            "\n"
            "++[@0:05](https://fathom.video/calls/1?timestamp=5.2)++ - **Bob**  \n"
            "Yes, indeed.  \n"
            "\n"
        )
        segments = parse_transcript(md)
        assert len(segments) == 2
        assert segments[0].text == "Hello world."
        assert segments[1].text == "Yes, indeed."

    def test_multiline_segment(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text(
            "++[@0:00](https://fathom.video/calls/1?timestamp=0.0)++ - **Alice**  \n"
            "Line one of the segment.  \n"
            "Line two of the same segment.  \n"
            "\n"
            "++[@0:10](https://fathom.video/calls/1?timestamp=10.0)++ - **Bob**  \n"
            "Next.  \n"
        )
        segments = parse_transcript(md)
        assert len(segments) == 2
        assert "Line one of the segment." in segments[0].text
        assert "Line two of the same segment." in segments[0].text

    def test_inline_watch_marker_does_not_split(self, tmp_path: Path) -> None:
        # An action-item line embeds [WATCH](...?timestamp=...) inside speech.
        # That must not split the segment.
        md = tmp_path / "test.md"
        watch_line = (
            "**ACTION ITEM: do thing - "
            "++[WATCH](https://fathom.video/calls/1?timestamp=361.99)++**  \n"
        )
        md.write_text(
            "++[@0:00](https://fathom.video/calls/1?timestamp=0.0)++ - **Alice**  \n"
            "Some speech that mentions a " + watch_line + "and continues.  \n"
            "\n"
            "++[@0:10](https://fathom.video/calls/1?timestamp=10.0)++ - **Bob**  \n"
            "Next.  \n"
        )
        segments = parse_transcript(md)
        assert len(segments) == 2
        assert segments[0].start == 0.0
        assert "ACTION ITEM" in segments[0].text


class TestAlignTranscript:
    def test_align_finds_overlapping_segment(self) -> None:
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Hello"),
            TranscriptSegment(start=5.0, end=8.0, text="Click here"),
            TranscriptSegment(start=10.0, end=12.0, text="Done"),
        ]
        assert align_transcript(6.0, segments) == "Click here"

    def test_align_returns_none_for_gap(self) -> None:
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Hello"),
            TranscriptSegment(start=5.0, end=8.0, text="Click here"),
        ]
        assert align_transcript(4.0, segments) is None

    def test_align_joins_multiple_overlapping(self) -> None:
        segments = [
            TranscriptSegment(start=0.0, end=5.0, text="First part"),
            TranscriptSegment(start=4.0, end=8.0, text="Second part"),
        ]
        assert align_transcript(4.5, segments) == "First part | Second part"

    def test_align_empty_segments(self) -> None:
        assert align_transcript(1.0, []) is None


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

        from peeklet.core.audio import detect_speech_segments

        speech_segments = detect_speech_segments(audio_path)
        # Should detect speech roughly in the 1-2 second range
        assert len(speech_segments) >= 1
        has_speech_in_middle = any(s.start < 2.0 and s.end > 1.0 for s in speech_segments)
        assert has_speech_in_middle

    def test_get_audio_activity_at_timestamp(self, tmp_path: Path) -> None:
        from pydub import AudioSegment
        from pydub.generators import Sine

        silence = AudioSegment.silent(duration=1000)
        tone = Sine(440).to_audio_segment(duration=1000).apply_gain(-10)
        audio = silence + tone + silence
        audio_path = tmp_path / "test.wav"
        audio.export(str(audio_path), format="wav")

        from peeklet.core.audio import detect_speech_segments, get_audio_activity

        speech_segments = detect_speech_segments(audio_path)
        assert get_audio_activity(0.5, speech_segments) == "silence"
        assert get_audio_activity(1.5, speech_segments) == "speech"
        assert get_audio_activity(2.5, speech_segments) == "silence"

    def test_silence_only_audio(self, tmp_path: Path) -> None:
        from pydub import AudioSegment

        silence = AudioSegment.silent(duration=2000)
        audio_path = tmp_path / "test.wav"
        silence.export(str(audio_path), format="wav")

        from peeklet.core.audio import detect_speech_segments

        speech_segments = detect_speech_segments(audio_path)
        assert speech_segments == []
