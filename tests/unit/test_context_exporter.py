"""Tests for JSON + Markdown context export."""

import json
from pathlib import Path

from peeklet.core.audio import TranscriptSegment
from peeklet.core.context_exporter import (
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
    def test_writes_markdown_with_header(self, tmp_path: Path) -> None:
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 65.5, "total_screenshots": 1},
            "screenshots": [
                {
                    "id": 1,
                    "file": "screenshot_00_00_05_200.png",
                    "timestamp_s": 5.2,
                    "timestamp": "00:00:05.200",
                    "trigger": "visual_change",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [1],
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
                }
            ],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)

        md = out_path.read_text()
        assert "# Video Summary: demo.mp4" in md
        assert "Duration: 1m 5.5s" in md
        assert "screenshot_00_00_05_200.png" in md
        assert "visual change" in md.lower() or "visual_change" in md
        assert "Look at this chart" in md

    def test_screenshots_injected_inline_with_transcript(self, tmp_path: Path) -> None:
        """Screenshots should be inserted between transcript segments at the
        chronological position matching their timestamp."""
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 30.0, "total_screenshots": 2},
            "screenshots": [
                {
                    "id": 1,
                    "file": "screenshot_00_00_06_000.jpg",
                    "timestamp_s": 6.0,
                    "timestamp": "00:00:06.000",
                    "trigger": "visual_change",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [1],
                },
                {
                    "id": 2,
                    "file": "screenshot_00_00_15_000.jpg",
                    "timestamp_s": 15.0,
                    "timestamp": "00:00:15.000",
                    "trigger": "visual_change",
                    "change_magnitude": "minor",
                    "seconds_since_prev_screenshot": 9.0,
                    "transcript_ids": [],
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
                },
                {
                    "id": 2,
                    "start_s": 10.0,
                    "end_s": 14.0,
                    "start": "00:00:10.000",
                    "end": "00:00:14.000",
                    "text": "Second line spoken",
                    "screenshot_ids": [],
                },
                {
                    "id": 3,
                    "start_s": 20.0,
                    "end_s": 25.0,
                    "start": "00:00:20.000",
                    "end": "00:00:25.000",
                    "text": "Third line spoken",
                    "screenshot_ids": [],
                },
            ],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)

        md = out_path.read_text()
        # Order should be: seg1 -> shot1 -> seg2 -> shot2 -> seg3
        first_line = md.find("First line spoken")
        shot1 = md.find("Screenshot 1")
        second_line = md.find("Second line spoken")
        shot2 = md.find("Screenshot 2")
        third_line = md.find("Third line spoken")

        assert -1 < first_line < shot1 < second_line < shot2 < third_line, (
            f"unexpected order in markdown:\n{md}"
        )

    def test_screenshots_only_when_no_transcript(self, tmp_path: Path) -> None:
        """With no transcript, screenshots should still be emitted in order."""
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 10.0, "total_screenshots": 2},
            "screenshots": [
                {
                    "id": 1,
                    "file": "screenshot_00_00_02_000.jpg",
                    "timestamp_s": 2.0,
                    "timestamp": "00:00:02.000",
                    "trigger": "visual_change",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [],
                },
                {
                    "id": 2,
                    "file": "screenshot_00_00_07_000.jpg",
                    "timestamp_s": 7.0,
                    "timestamp": "00:00:07.000",
                    "trigger": "visual_change",
                    "change_magnitude": "minor",
                    "seconds_since_prev_screenshot": 5.0,
                    "transcript_ids": [],
                },
            ],
            "transcript": [],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)

        md = out_path.read_text()
        s1 = md.find("Screenshot 1")
        s2 = md.find("Screenshot 2")
        assert -1 < s1 < s2
