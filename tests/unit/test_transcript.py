"""Tests for transcript parsing and Fathom anchors."""

from pathlib import Path

import pytest

from peeklet.transcript import (
    TranscriptError,
    action_item_checkpoints,
    parse_fathom_action_items,
    parse_transcript,
    split_long_lines,
)
from peeklet.types import Line


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
        assert segments[0] == Line(start=1.0, end=3.5, text="Hello world")
        assert segments[1] == Line(start=5.0, end=8.2, text="Click the button")

    def test_parse_multiline_srt(self, tmp_path: Path) -> None:
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:04,000\nLine one\nLine two\n\n")
        segments = parse_transcript(srt_file)
        assert len(segments) == 1
        assert segments[0].text == "Line one Line two"

    def test_parse_empty_srt(self, tmp_path: Path) -> None:
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("")
        with pytest.raises(TranscriptError):
            parse_transcript(srt_file)


class TestParseVtt:
    def test_vtt_without_hours(self, tmp_path: Path) -> None:
        f = tmp_path / "t.vtt"
        f.write_text("WEBVTT\n\n00:01.000 --> 00:04.500\nOne\n\n01:05.250 --> 01:07.000\nTwo\n")
        segs = parse_transcript(f)
        assert segs == [Line(1.0, 4.5, "One"), Line(65.25, 67.0, "Two")]

    def test_vtt_mixed_hours(self, tmp_path: Path) -> None:
        f = tmp_path / "t.vtt"
        f.write_text(
            "WEBVTT\n\n00:01.000 --> 00:04.500\nOne\n\n01:00:05.000 --> 01:00:07.000\nTwo\n"
        )
        segs = parse_transcript(f)
        assert [(s.start, s.end) for s in segs] == [(1.0, 4.5), (3605.0, 3607.0)]

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
        assert segments[0] == Line(start=1.0, end=3.5, text="Hello world")
        assert segments[1] == Line(start=5.0, end=8.2, text="Click the button")

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

    def test_duplicate_action_item_line_dedup(self, tmp_path: Path) -> None:
        # Fathom emits the same ACTION ITEM line twice on consecutive lines
        # inside a speech segment. Accumulated verbatim this duplicates the
        # anchor text in the joined segment content.
        md = tmp_path / "test.md"
        action_line = (
            "**ACTION ITEM: Add Tasks link - "
            "++[WATCH](https://fathom.video/calls/1?timestamp=133.99)++**  \n"
        )
        md.write_text(
            "++[@0:00](https://fathom.video/calls/1?timestamp=0.0)++ - **Alice**  \n"
            "Preamble speech.  \n" + action_line + action_line + "\n"
            "++[@0:10](https://fathom.video/calls/1?timestamp=10.0)++ - **Bob**  \n"
            "Next.  \n"
        )
        segments = parse_transcript(md)
        assert len(segments) == 2
        assert segments[0].text.count("ACTION ITEM: Add Tasks link") == 1

    def test_consecutive_duplicate_lines_collapsed(self, tmp_path: Path) -> None:
        # Any consecutive identical non-empty line is collapsed — Fathom
        # frequently emits duplicate anchor lines, and legitimate prose
        # doesn't repeat verbatim across line breaks.
        md = tmp_path / "test.md"
        md.write_text(
            "++[@0:00](https://fathom.video/calls/1?timestamp=0.0)++ - **Alice**  \n"
            "Same sentence here.  \n"
            "Same sentence here.  \n"
            "\n"
        )
        segments = parse_transcript(md)
        assert len(segments) == 1
        assert segments[0].text == "Same sentence here."

    def test_speaker_extracted(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text(
            "++[@0:00](https://fathom.video/calls/1?timestamp=0.56)++ - **Alice**  \n"
            "Welcome everyone.  \n"
            "\n"
            "++[@0:03](https://fathom.video/calls/1?timestamp=3.0)++ - **Bob Smith**  \n"
            "Thanks for having me.  \n"
            "\n"
        )
        segments = parse_transcript(md)
        assert len(segments) == 2
        assert segments[0].speaker == "Alice"
        assert segments[1].speaker == "Bob Smith"

    def test_srt_has_no_speaker(self, tmp_path: Path) -> None:
        srt_file = tmp_path / "test.srt"
        srt_file.write_text("1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n")
        segments = parse_transcript(srt_file)
        assert segments[0].speaker is None


class TestParseFathomActionItems:
    def test_extracts_action_items_with_watch_timestamps(self) -> None:
        text = (
            "**ACTION ITEM: Fix missing assistant prompt - "
            "++[WATCH](https://fathom.video/calls/123?timestamp=232.9999)++**\n"
            "**ACTION ITEM: Fix missing assistant prompt - "
            "++[WATCH](https://fathom.video/calls/123?timestamp=232.9999)++**\n"
            "Some other text\n"
            "**ACTION ITEM: Investigate Policy Health guard-flag issue; fix - "
            "++[WATCH](https://fathom.video/calls/123?timestamp=455.9999)++**\n"
        )
        cps = parse_fathom_action_items(text)

        assert len(cps) == 2  # deduped consecutive duplicate
        assert cps[0].t == pytest.approx(232.9999)
        assert cps[0].kind == "action_item"
        assert "Fix missing assistant prompt" in cps[0].label
        assert cps[1].t == pytest.approx(455.9999)

    def test_returns_empty_on_no_action_items(self) -> None:
        text = "Just some regular transcript text with no action items.\n"
        assert parse_fathom_action_items(text) == []

    def test_sorted_by_timestamp(self) -> None:
        text = (
            "**ACTION ITEM: Second - "
            "++[WATCH](https://fathom.video/calls/1?timestamp=500.0)++**\n"
            "**ACTION ITEM: First - "
            "++[WATCH](https://fathom.video/calls/1?timestamp=100.0)++**\n"
        )
        assert [c.t for c in parse_fathom_action_items(text)] == [100.0, 500.0]


class TestTranscriptErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(TranscriptError, match="not found"):
            parse_transcript(tmp_path / "missing.vtt")

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "t.vtt"
        p.write_text("WEBVTT\n\n")
        with pytest.raises(TranscriptError, match="no timestamped lines"):
            parse_transcript(p)

    def test_undecodable_file(self, tmp_path: Path) -> None:
        p = tmp_path / "t.srt"
        p.write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(TranscriptError):
            parse_transcript(p)


class TestSplitLongLines:
    def test_short_lines_unchanged(self) -> None:
        lines = [Line(0.0, 5.0, "One. Two.", "A")]
        assert split_long_lines(lines, 8.0) == lines

    def test_long_line_without_boundary_unchanged(self) -> None:
        lines = [Line(0.0, 20.0, "no sentence end here at all", None)]
        assert split_long_lines(lines, 8.0) == lines

    def test_splits_proportionally_and_keeps_speaker(self) -> None:
        text = "Short one. " + "This second sentence is much longer than the first!"
        out = split_long_lines([Line(10.0, 30.0, text, "Alice")], 8.0)
        assert [ln.text for ln in out] == [
            "Short one.",
            "This second sentence is much longer than the first!",
        ]
        assert all(ln.speaker == "Alice" for ln in out)
        assert out[0].start == 10.0
        assert out[-1].end == 30.0
        assert out[0].end == out[1].start
        n0, n1 = len(out[0].text), len(out[1].text)
        assert out[0].end == pytest.approx(10.0 + 20.0 * n0 / (n0 + n1), abs=1e-3)

    def test_question_and_exclamation_are_boundaries(self) -> None:
        out = split_long_lines([Line(0.0, 30.0, "Why? Because! Done.", None)], 8.0)
        assert [ln.text for ln in out] == ["Why?", "Because!", "Done."]


def test_action_item_checkpoints_only_for_markdown(tmp_path: Path) -> None:
    vtt = tmp_path / "t.vtt"
    vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n")
    assert action_item_checkpoints(vtt) == []
    md = tmp_path / "t.md"
    md.write_text(
        "**ACTION ITEM: Fix login - ++[WATCH](https://fathom.video/x?timestamp=42.5)++**\n"
    )
    [cp] = action_item_checkpoints(md)
    assert (cp.t, cp.kind, cp.label) == (42.5, "action_item", "Fix login")


def test_parse_fathom_action_items_dedupes() -> None:
    line = "**ACTION ITEM: Fix login - ++[WATCH](https://fathom.video/x?timestamp=42.5)++**\n"
    assert len(parse_fathom_action_items(line + line)) == 1
