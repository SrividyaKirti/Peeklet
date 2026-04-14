"""Tests for JSON + Markdown context export."""

import json
from pathlib import Path

from peeklet.core.audio import TranscriptSegment
from peeklet.core.context_exporter import (
    _screenshot_block,
    build_context,
    format_timestamp,
    timestamp_filename,
    write_context_json,
    write_context_markdown,
)
from peeklet.utils.types import EventType, FrameResult


class TestFormatTimestamp:
    def test_zero(self) -> None:
        assert format_timestamp(0.0) == "00:00:00.000"

    def test_simple_seconds(self) -> None:
        assert format_timestamp(5.2) == "00:00:05.200"

    def test_minutes_and_seconds(self) -> None:
        assert format_timestamp(65.5) == "00:01:05.500"

    def test_hours(self) -> None:
        assert format_timestamp(3661.123) == "01:01:01.123"


class TestTimestampFilename:
    def test_basic(self) -> None:
        assert timestamp_filename(5.2) == "screenshot_00_00_05_200"

    def test_with_minutes(self) -> None:
        assert timestamp_filename(65.5) == "screenshot_00_01_05_500"


class TestBuildContext:
    def _make_keyframe_result(
        self,
        frame_id: str,
        timestamp: float,
        trigger_type: str = "visual_change",
        change_magnitude: str = "major",
    ) -> FrameResult:
        return FrameResult(
            frame_id=frame_id,
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abc123",
            frame_width=1920,
            frame_height=1080,
            video_timestamp=timestamp,
            trigger_type=trigger_type,
            change_magnitude=change_magnitude,
            asset_path=f"output/{frame_id}.png",
            visual_context_goal="Test screen",
            textual_anchor="test anchor",
            downstream_utility="test utility",
            moment_source="llm",
        )

    def test_bidirectional_linking(self) -> None:
        results = [
            self._make_keyframe_result("s1", 5.0),
            self._make_keyframe_result("s2", 12.0, trigger_type="transcript_trigger"),
        ]
        segments = [
            TranscriptSegment(start=0.0, end=4.0, text="Welcome"),
            TranscriptSegment(start=4.5, end=7.0, text="Look at this dashboard"),
            TranscriptSegment(start=10.0, end=14.0, text="Click here to open settings"),
        ]
        ctx = build_context("demo.mp4", 60.0, results, segments)

        assert ctx["video"]["filename"] == "demo.mp4"
        assert ctx["video"]["total_screenshots"] == 2

        assert ctx["screenshots"][0]["transcript_ids"] == [2]
        assert ctx["screenshots"][1]["transcript_ids"] == [3]

        assert ctx["transcript"][1]["screenshot_ids"] == [1]
        assert ctx["transcript"][2]["screenshot_ids"] == [2]
        assert ctx["transcript"][0]["screenshot_ids"] == []

        assert ctx["screenshots"][0]["visual_context_goal"] == "Test screen"
        assert ctx["screenshots"][0]["moment_source"] == "llm"
        assert ctx["transcript"][0]["speaker"] is None  # SRT segments have no speaker

    def test_seconds_since_prev_screenshot(self) -> None:
        results = [
            self._make_keyframe_result("s1", 5.0),
            self._make_keyframe_result("s2", 12.5),
        ]
        ctx = build_context("demo.mp4", 60.0, results, [])

        assert ctx["screenshots"][0]["seconds_since_prev_screenshot"] is None
        assert ctx["screenshots"][1]["seconds_since_prev_screenshot"] == 7.5

    def test_empty_results(self) -> None:
        ctx = build_context("demo.mp4", 60.0, [], [])
        assert ctx["screenshots"] == []
        assert ctx["transcript"] == []
        assert ctx["video"]["total_screenshots"] == 0

    def test_no_transcript(self) -> None:
        results = [self._make_keyframe_result("s1", 5.0)]
        ctx = build_context("demo.mp4", 60.0, results, [])
        assert ctx["screenshots"][0]["transcript_ids"] == []

    def test_ocr_fields_propagate_to_screenshot_entries(self) -> None:
        r = self._make_keyframe_result("s1", 5.0)
        r.ocr_text = "Invoice Dashboard Submit"
        r.ocr_tokens = ["invoice", "dashboard", "submit"]
        ctx = build_context("demo.mp4", 60.0, [r], [])
        entry = ctx["screenshots"][0]
        assert entry["ocr_text"] == "Invoice Dashboard Submit"
        assert entry["ocr_tokens"] == ["invoice", "dashboard", "submit"]

    def test_ocr_fields_default_to_empty(self) -> None:
        r = self._make_keyframe_result("s1", 5.0)
        ctx = build_context("demo.mp4", 60.0, [r], [])
        entry = ctx["screenshots"][0]
        assert entry["ocr_text"] == ""
        assert entry["ocr_tokens"] == []

    def test_filters_non_keyframes(self) -> None:
        """build_context should ignore FrameResult entries with is_keyframe=False."""
        keyframe = self._make_keyframe_result("kf", 5.0)
        skipped = FrameResult(
            frame_id="skipped",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="abc",
            frame_width=1920,
            frame_height=1080,
            video_timestamp=10.0,
        )
        ctx = build_context("demo.mp4", 60.0, [keyframe, skipped], [])
        assert ctx["video"]["total_screenshots"] == 1
        assert len(ctx["screenshots"]) == 1
        assert ctx["screenshots"][0]["id"] == 1


class TestWriteContextJson:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 60.0, "total_screenshots": 0},
            "screenshots": [],
            "transcript": [],
        }
        out_path = tmp_path / "context.json"
        write_context_json(ctx, out_path)

        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["video"]["filename"] == "demo.mp4"


class TestWriteContextMarkdown:
    def test_writes_markdown_with_visual_toc_and_new_header(self, tmp_path: Path) -> None:
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 65.5, "total_screenshots": 1},
            "screenshots": [
                {
                    "id": 1,
                    "file": "demo_0001_00005200ms.jpg",
                    "timestamp_s": 5.2,
                    "timestamp": "00:00:05.200",
                    "trigger": "transcript_trigger",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [1],
                    "visual_context_goal": "Dashboard with cost breakdown",
                    "textual_anchor": "Let me show you the estimated cost",
                    "downstream_utility": "To extract specific cost values",
                    "moment_source": "llm",
                }
            ],
            "transcript": [
                {
                    "id": 1,
                    "start_s": 4.5,
                    "end_s": 7.0,
                    "start": "00:00:04.500",
                    "end": "00:00:07.000",
                    "text": "Look at this chart",
                    "screenshot_ids": [1],
                    "speaker": "Alice",
                }
            ],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)

        md = out_path.read_text()
        assert "# Meeting Context: demo.mp4" in md
        assert "Screenshots: 1" in md
        assert "Visual Table of Contents" in md
        assert "Dashboard with cost breakdown" in md
        assert "Alice" in md
        # Boilerplate downstream_utility line and anchor blockquote are
        # no longer emitted — they duplicated info already in the header
        # or transcript.
        assert "*To extract specific cost values*" not in md
        assert "> Let me show you the estimated cost" not in md

    def test_screenshots_injected_inline_with_transcript(self, tmp_path: Path) -> None:
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 30.0, "total_screenshots": 2},
            "screenshots": [
                {
                    "id": 1,
                    "file": "demo_0001.jpg",
                    "timestamp_s": 6.0,
                    "timestamp": "00:00:06.000",
                    "trigger": "transcript_trigger",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [1],
                    "visual_context_goal": "First screen",
                    "textual_anchor": "",
                    "downstream_utility": "",
                    "moment_source": "llm",
                },
                {
                    "id": 2,
                    "file": "demo_0002.jpg",
                    "timestamp_s": 15.0,
                    "timestamp": "00:00:15.000",
                    "trigger": "transcript_trigger",
                    "change_magnitude": "minor",
                    "seconds_since_prev_screenshot": 9.0,
                    "transcript_ids": [],
                    "visual_context_goal": "Second screen",
                    "textual_anchor": "",
                    "downstream_utility": "",
                    "moment_source": "anchor",
                },
            ],
            "transcript": [
                {
                    "id": 1,
                    "start_s": 0.0,
                    "end_s": 5.0,
                    "start": "00:00:00.000",
                    "end": "00:00:05.000",
                    "text": "First line spoken",
                    "screenshot_ids": [],
                    "speaker": None,
                },
                {
                    "id": 2,
                    "start_s": 10.0,
                    "end_s": 14.0,
                    "start": "00:00:10.000",
                    "end": "00:00:14.000",
                    "text": "Second line spoken",
                    "screenshot_ids": [],
                    "speaker": "Bob",
                },
                {
                    "id": 3,
                    "start_s": 20.0,
                    "end_s": 25.0,
                    "start": "00:00:20.000",
                    "end": "00:00:25.000",
                    "text": "Third line spoken",
                    "screenshot_ids": [],
                    "speaker": None,
                },
            ],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)

        md = out_path.read_text()
        first_line = md.find("First line spoken")
        shot1 = md.find("Screenshot 1")
        second_line = md.find("Second line spoken")
        shot2 = md.find("Screenshot 2")
        third_line = md.find("Third line spoken")
        assert -1 < first_line < shot1 < second_line < shot2 < third_line

    def test_guaranteed_tag_only_on_anchor_source(self, tmp_path: Path) -> None:
        anchor_shot = {
            "id": 1,
            "file": "a.jpg",
            "timestamp_s": 1.0,
            "timestamp": "00:00:01.000",
            "trigger": "transcript_trigger",
            "change_magnitude": "major",
            "seconds_since_prev_screenshot": None,
            "transcript_ids": [],
            "visual_context_goal": "Click Save",
            "textual_anchor": (
                "**ACTION ITEM: Click Save - "
                "++[WATCH](https://fathom.video/calls/1?timestamp=1.0)++**"
            ),
            "downstream_utility": "",
            "moment_source": "anchor",
        }
        llm_shot = {
            **anchor_shot,
            "id": 2,
            "timestamp_s": 5.0,
            "timestamp": "00:00:05.000",
            "file": "b.jpg",
            "moment_source": "llm",
            "textual_anchor": "",
        }
        block_anchor = "\n".join(_screenshot_block(anchor_shot))
        block_llm = "\n".join(_screenshot_block(llm_shot))
        assert "[guaranteed]" in block_anchor
        assert "[guaranteed]" not in block_llm

    def test_screenshot_block_drops_utility_and_anchor_lines(self) -> None:
        shot = {
            "id": 1,
            "file": "a.jpg",
            "timestamp_s": 1.0,
            "timestamp": "00:00:01.000",
            "trigger": "transcript_trigger",
            "change_magnitude": "major",
            "seconds_since_prev_screenshot": None,
            "transcript_ids": [],
            "visual_context_goal": "Dashboard",
            "textual_anchor": "some transcript line",
            "downstream_utility": "boilerplate utility text",
            "moment_source": "llm",
        }
        block = "\n".join(_screenshot_block(shot))
        assert "boilerplate utility text" not in block
        assert "> some transcript line" not in block
        assert "**Screenshot 1" in block
        assert "![Screenshot](a.jpg)" in block

    def test_references_section_lists_watch_urls(self, tmp_path: Path) -> None:
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 10.0, "total_screenshots": 2},
            "screenshots": [
                {
                    "id": 1,
                    "file": "a.jpg",
                    "timestamp_s": 1.0,
                    "timestamp": "00:00:01.000",
                    "trigger": "transcript_trigger",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [],
                    "visual_context_goal": "Click Save",
                    "textual_anchor": (
                        "**ACTION ITEM: Click Save - "
                        "++[WATCH](https://fathom.video/calls/1?timestamp=1.0)++**"
                    ),
                    "downstream_utility": "",
                    "moment_source": "anchor",
                },
                {
                    "id": 2,
                    "file": "b.jpg",
                    "timestamp_s": 5.0,
                    "timestamp": "00:00:05.000",
                    "trigger": "transcript_trigger",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": 4.0,
                    "transcript_ids": [],
                    "visual_context_goal": "LLM pick",
                    "textual_anchor": "a spoken line",
                    "downstream_utility": "",
                    "moment_source": "llm",
                },
            ],
            "transcript": [],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)
        md = out_path.read_text()
        # Footnote section exists and contains the WATCH URL for the
        # anchor screenshot only.
        assert "## References" in md
        assert "https://fathom.video/calls/1?timestamp=1.0" in md
        # URL must not be duplicated inline as a blockquote under the
        # screenshot block.
        assert md.count("https://fathom.video/calls/1?timestamp=1.0") == 1
        # References section must come after the timeline.
        assert md.find("## References") > md.find("## Timeline")

    def test_screenshots_only_when_no_transcript(self, tmp_path: Path) -> None:
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 10.0, "total_screenshots": 1},
            "screenshots": [
                {
                    "id": 1,
                    "file": "demo_0001.jpg",
                    "timestamp_s": 2.0,
                    "timestamp": "00:00:02.000",
                    "trigger": "visual_change",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [],
                    "visual_context_goal": "Some screen",
                    "textual_anchor": "",
                    "downstream_utility": "",
                    "moment_source": "llm",
                },
            ],
            "transcript": [],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)

        md = out_path.read_text()
        assert "Screenshot 1" in md
