"""Tests for the audio module — transcript parsing and speech detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from peeklet.core.audio import parse_transcript, TranscriptSegment, align_transcript


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
        srt_file.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "Line one\n"
            "Line two\n"
            "\n"
        )
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
            "WEBVTT\n"
            "Kind: captions\n"
            "Language: en\n"
            "\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "First line\n"
            "\n"
        )
        segments = parse_transcript(vtt_file)
        assert len(segments) == 1
        assert segments[0].text == "First line"


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
