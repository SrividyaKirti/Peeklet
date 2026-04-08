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
