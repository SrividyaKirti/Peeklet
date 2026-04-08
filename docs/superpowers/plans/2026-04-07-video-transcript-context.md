# Video + Transcript Context Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance Peeklet's video mode to extract keyframes on both visual changes and transcript trigger words, then output JSON + Markdown with bidirectional screenshot/transcript linking.

**Architecture:** Add transcript trigger detection as a separate module (`core/transcript_trigger.py`). Feed forced timestamps into the existing `process_video()` two-pass pipeline. Add new JSON/Markdown exporters (`core/context_exporter.py`). Modify `FrameResult` with `trigger_type` field. Update CLI to wire it together.

**Tech Stack:** Python 3.10+, existing Peeklet modules (audio.py, video.py, pipeline.py), pytest for TDD.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/peeklet/core/transcript_trigger.py` | Create | Weighted keyword scoring on transcript segments |
| `src/peeklet/core/context_exporter.py` | Create | JSON + Markdown output with bidirectional linking |
| `src/peeklet/utils/types.py` | Modify | Add `trigger_type` field to `FrameResult` |
| `src/peeklet/core/exporter.py` | Modify | Add `trigger_type` to Parquet schema |
| `src/peeklet/core/video.py` | Modify | Accept forced timestamps, merge triggers, timestamp-based naming |
| `src/peeklet/cli.py` | Modify | Wire trigger detection + context export for video mode |
| `tests/unit/test_transcript_trigger.py` | Create | Unit tests for keyword scoring |
| `tests/unit/test_context_exporter.py` | Create | Unit tests for JSON/Markdown output |
| `tests/integration/test_video_pipeline.py` | Modify | Add integration tests for forced timestamps + context export |

---

### Task 1: Add `trigger_type` to `FrameResult` and Parquet schema

**Files:**
- Modify: `src/peeklet/utils/types.py:48-82`
- Modify: `src/peeklet/core/exporter.py:17-76` (schema) and `100-133` (append)
- Test: `tests/unit/test_types.py` (existing, verify no breakage)

- [ ] **Step 1: Add `trigger_type` field to `FrameResult`**

In `src/peeklet/utils/types.py`, add after the `visual_reason` field (line 68):

```python
    visual_reason: str | None = None
    trigger_type: str | None = None  # "visual_change", "transcript_trigger", "both"
    prev_keyframe_id: str | None = None
```

- [ ] **Step 2: Add `trigger_type` to Parquet schema**

In `src/peeklet/core/exporter.py`, add after the `visual_reason` field in `MANIFEST_SCHEMA` (line 60):

```python
        pa.field("visual_reason", pa.string(), nullable=True),
        pa.field("trigger_type", pa.string(), nullable=True),
        pa.field("prev_keyframe_id", pa.string(), nullable=True),
```

In the `append()` method, add after `visual_reason` (line 117):

```python
                "visual_reason": result.visual_reason,
                "trigger_type": result.trigger_type,
                "prev_keyframe_id": result.prev_keyframe_id,
```

- [ ] **Step 3: Run existing tests to verify no breakage**

Run: `pytest tests/unit/test_types.py tests/unit/test_exporter.py -v`
Expected: All tests PASS (new field defaults to None)

- [ ] **Step 4: Commit**

```bash
git add src/peeklet/utils/types.py src/peeklet/core/exporter.py
git commit -m "feat: add trigger_type field to FrameResult and Parquet schema"
```

---

### Task 2: Create transcript trigger detection module

**Files:**
- Create: `src/peeklet/core/transcript_trigger.py`
- Test: `tests/unit/test_transcript_trigger.py`

- [ ] **Step 1: Write failing tests for trigger detection**

Create `tests/unit/test_transcript_trigger.py`:

```python
"""Tests for transcript trigger word detection."""

from __future__ import annotations

import pytest

from peeklet.core.audio import TranscriptSegment
from peeklet.core.transcript_trigger import TriggerResult, detect_triggers


class TestDetectTriggers:
    def test_high_plus_medium_triggers(self) -> None:
        """High-signal noun + medium-signal verb = trigger."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Now look at this dashboard"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 1
        assert triggers[0].timestamp == pytest.approx(1.5)  # midpoint of segment
        assert triggers[0].segment == segments[0]

    def test_two_high_signal_triggers(self) -> None:
        """Two high-signal words alone = trigger."""
        segments = [
            TranscriptSegment(start=5.0, end=8.0, text="The chart and graph show growth"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 1

    def test_single_medium_no_trigger(self) -> None:
        """Single medium word alone = no trigger (too noisy)."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="I see what you mean"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 0

    def test_no_keywords_no_trigger(self) -> None:
        """No keywords at all = no trigger."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Welcome to the presentation"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 0

    def test_empty_segments(self) -> None:
        triggers = detect_triggers([])
        assert triggers == []

    def test_multiple_segments_multiple_triggers(self) -> None:
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Hello everyone"),
            TranscriptSegment(start=5.0, end=8.0, text="Click this button here"),
            TranscriptSegment(start=10.0, end=13.0, text="The results are in"),
            TranscriptSegment(start=15.0, end=18.0, text="Notice the sidebar menu"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 2
        timestamps = [t.timestamp for t in triggers]
        assert pytest.approx(6.5) in timestamps  # "Click this button here"
        assert pytest.approx(16.5) in timestamps  # "Notice the sidebar menu"

    def test_trigger_timestamp_is_segment_midpoint(self) -> None:
        segments = [
            TranscriptSegment(start=10.0, end=20.0, text="Look at this chart"),
        ]
        triggers = detect_triggers(segments)
        assert triggers[0].timestamp == pytest.approx(15.0)

    def test_case_insensitive(self) -> None:
        """Keywords should match regardless of case."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="CLICK THIS BUTTON"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_transcript_trigger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'peeklet.core.transcript_trigger'`

- [ ] **Step 3: Implement transcript trigger detection**

Create `src/peeklet/core/transcript_trigger.py`:

```python
"""Transcript trigger word detection for screen-reference keyframes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment

HIGH_SIGNAL_WORDS: frozenset[str] = frozenset(
    {
        "chart",
        "graph",
        "dashboard",
        "button",
        "menu",
        "sidebar",
        "diagram",
        "screen",
        "page",
        "modal",
        "dropdown",
        "tab",
        "panel",
        "table",
        "form",
        "icon",
        "window",
        "popup",
        "field",
        "toolbar",
        "flowchart",
        "heatmap",
    }
)

MEDIUM_SIGNAL_WORDS: frozenset[str] = frozenset(
    {
        "here",
        "this",
        "notice",
        "look",
        "see",
        "click",
        "shown",
        "display",
        "hover",
        "scroll",
        "select",
        "drag",
        "toggle",
        "zoom",
        "highlight",
        "observe",
        "watch",
    }
)


@dataclass(frozen=True, slots=True)
class TriggerResult:
    """A detected transcript trigger — forces a keyframe at this timestamp."""

    timestamp: float  # seconds (midpoint of the segment)
    segment: TranscriptSegment


def _is_trigger(text: str) -> bool:
    """Check if text contains enough signal words to trigger a keyframe."""
    words = set(text.lower().split())
    has_high = bool(words & HIGH_SIGNAL_WORDS)
    has_medium = bool(words & MEDIUM_SIGNAL_WORDS)
    high_count = len(words & HIGH_SIGNAL_WORDS)

    # High + medium = trigger
    if has_high and has_medium:
        return True
    # 2+ high-signal words alone = trigger
    if high_count >= 2:
        return True
    return False


def detect_triggers(segments: list[TranscriptSegment]) -> list[TriggerResult]:
    """Scan transcript segments for screen-reference trigger words.

    Returns a list of TriggerResult with timestamps where keyframes
    should be forced due to transcript content.
    """
    triggers: list[TriggerResult] = []
    for segment in segments:
        if _is_trigger(segment.text):
            midpoint = (segment.start + segment.end) / 2
            triggers.append(TriggerResult(timestamp=midpoint, segment=segment))
    return triggers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_transcript_trigger.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run lint**

Run: `ruff check src/peeklet/core/transcript_trigger.py tests/unit/test_transcript_trigger.py && ruff format --check src/peeklet/core/transcript_trigger.py tests/unit/test_transcript_trigger.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/core/transcript_trigger.py tests/unit/test_transcript_trigger.py
git commit -m "feat: add transcript trigger word detection module"
```

---

### Task 3: Create context exporter module (JSON + Markdown)

**Files:**
- Create: `src/peeklet/core/context_exporter.py`
- Test: `tests/unit/test_context_exporter.py`

- [ ] **Step 1: Write failing tests for timestamp formatting helper**

Create `tests/unit/test_context_exporter.py`:

```python
"""Tests for JSON + Markdown context export."""

from __future__ import annotations

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

        # Screenshot 1 at t=5.0 overlaps segment 2 (4.5–7.0)
        assert ctx["screenshots"][0]["transcript_ids"] == [2]
        # Screenshot 2 at t=12.0 overlaps segment 3 (10.0–14.0)
        assert ctx["screenshots"][1]["transcript_ids"] == [3]

        # Segment 2 links back to screenshot 1
        assert ctx["transcript"][1]["screenshot_ids"] == [1]
        # Segment 3 links back to screenshot 2
        assert ctx["transcript"][2]["screenshot_ids"] == [2]
        # Segment 1 has no screenshots
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_context_exporter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement context exporter**

Create `src/peeklet/core/context_exporter.py`:

```python
"""JSON and Markdown context export for LLM consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment
    from peeklet.utils.types import FrameResult


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def timestamp_filename(seconds: float) -> str:
    """Convert seconds to screenshot_HH_MM_SS_mmm filename (no extension)."""
    ts = format_timestamp(seconds)
    # "00:01:05.500" -> "00_01_05_500"
    name = ts.replace(":", "_").replace(".", "_")
    return f"screenshot_{name}"


def build_context(
    filename: str,
    duration_s: float,
    results: list[FrameResult],
    segments: list[TranscriptSegment],
) -> dict[str, Any]:
    """Build the context data structure with bidirectional linking.

    Args:
        filename: Video filename.
        duration_s: Video duration in seconds.
        results: Keyframe FrameResults only (filtered by caller or here).
        segments: Full transcript segments.

    Returns:
        Dict matching the context.json schema.
    """
    keyframes = [r for r in results if r.is_keyframe]

    # Build screenshot entries
    screenshots: list[dict[str, Any]] = []
    prev_ts: float | None = None
    for i, r in enumerate(keyframes):
        ts = r.video_timestamp or 0.0
        since_prev = (ts - prev_ts) if prev_ts is not None else None
        screenshots.append(
            {
                "id": i + 1,
                "file": f"{timestamp_filename(ts)}.png",
                "timestamp_s": ts,
                "timestamp": format_timestamp(ts),
                "trigger": r.trigger_type or "visual_change",
                "change_magnitude": r.change_magnitude or "major",
                "seconds_since_prev_screenshot": since_prev,
                "transcript_ids": [],  # filled below
            }
        )
        prev_ts = ts

    # Build transcript entries
    transcript: list[dict[str, Any]] = []
    for j, seg in enumerate(segments):
        transcript.append(
            {
                "id": j + 1,
                "start": format_timestamp(seg.start),
                "end": format_timestamp(seg.end),
                "text": seg.text,
                "screenshot_ids": [],  # filled below
            }
        )

    # Bidirectional linking: screenshot timestamp falls within segment [start, end]
    for s_entry in screenshots:
        ts = s_entry["timestamp_s"]
        for t_entry in transcript:
            seg = segments[t_entry["id"] - 1]
            if seg.start <= ts <= seg.end:
                s_entry["transcript_ids"].append(t_entry["id"])
                t_entry["screenshot_ids"].append(s_entry["id"])

    return {
        "video": {
            "filename": filename,
            "duration_s": duration_s,
            "total_screenshots": len(screenshots),
        },
        "screenshots": screenshots,
        "transcript": transcript,
    }


def write_context_json(ctx: dict[str, Any], path: Path) -> None:
    """Write context data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ctx, indent=2, ensure_ascii=False))


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    parts: list[str] = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    parts.append(f"{s:.1f}s")
    return " ".join(parts)


def write_context_markdown(ctx: dict[str, Any], path: Path) -> None:
    """Write context data as Markdown with inline image references."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    video = ctx["video"]
    lines: list[str] = [
        f"# Video Summary: {video['filename']}",
        f"Duration: {_format_duration(video['duration_s'])}"
        f" | Screenshots: {video['total_screenshots']}",
        "",
        "---",
        "",
    ]

    # Build a lookup: screenshot_id -> list of transcript texts
    transcript_by_id: dict[int, str] = {
        t["id"]: t["text"] for t in ctx["transcript"]
    }

    for s in ctx["screenshots"]:
        trigger_label = s["trigger"].replace("_", " ")
        lines.append(f"## Screenshot {s['id']} ({s['timestamp']}) — {trigger_label}")
        lines.append(f"![{s['file']}]({s['file']})")

        since = s["seconds_since_prev_screenshot"]
        since_str = f"{since:.1f}s" if since is not None else "\u2014"
        lines.append(
            f"**Change:** {s['change_magnitude']} | **Since prev:** {since_str}"
        )
        lines.append("")

        # Add overlapping transcript quotes
        for tid in s["transcript_ids"]:
            text = transcript_by_id.get(tid, "")
            if text:
                lines.append(f"> \"{text}\"")
                lines.append("")

        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_context_exporter.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run lint**

Run: `ruff check src/peeklet/core/context_exporter.py tests/unit/test_context_exporter.py && ruff format --check src/peeklet/core/context_exporter.py tests/unit/test_context_exporter.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/core/context_exporter.py tests/unit/test_context_exporter.py
git commit -m "feat: add JSON and Markdown context exporters"
```

---

### Task 4: Integrate forced timestamps into video pipeline

**Files:**
- Modify: `src/peeklet/core/video.py:151-304`
- Test: `tests/unit/test_video_trigger_integration.py`

- [ ] **Step 1: Write failing tests for forced timestamp integration**

Create `tests/unit/test_video_trigger_integration.py`:

```python
"""Tests for forced timestamp integration in video pipeline."""

from __future__ import annotations

from peeklet.core.video import _should_force_keyframe, _merge_trigger_type


class TestShouldForceKeyframe:
    def test_within_tolerance(self) -> None:
        forced = [5.0, 12.0, 20.0]
        assert _should_force_keyframe(5.3, forced, tolerance=0.5) is True

    def test_outside_tolerance(self) -> None:
        forced = [5.0, 12.0, 20.0]
        assert _should_force_keyframe(6.0, forced, tolerance=0.5) is False

    def test_empty_forced_list(self) -> None:
        assert _should_force_keyframe(5.0, [], tolerance=0.5) is False

    def test_exact_match(self) -> None:
        forced = [10.0]
        assert _should_force_keyframe(10.0, forced, tolerance=0.5) is True

    def test_boundary_tolerance(self) -> None:
        forced = [10.0]
        assert _should_force_keyframe(10.5, forced, tolerance=0.5) is True
        assert _should_force_keyframe(10.6, forced, tolerance=0.5) is False


class TestMergeTriggerType:
    def test_visual_only(self) -> None:
        assert _merge_trigger_type(is_visual=True, is_transcript=False) == "visual_change"

    def test_transcript_only(self) -> None:
        assert _merge_trigger_type(is_visual=False, is_transcript=True) == "transcript_trigger"

    def test_both(self) -> None:
        assert _merge_trigger_type(is_visual=True, is_transcript=True) == "both"

    def test_neither(self) -> None:
        assert _merge_trigger_type(is_visual=False, is_transcript=False) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_video_trigger_integration.py -v`
Expected: FAIL — `ImportError: cannot import name '_should_force_keyframe'`

- [ ] **Step 3: Add helper functions to video.py**

In `src/peeklet/core/video.py`, add before the `process_video` function (before line 151):

```python
def _should_force_keyframe(
    timestamp: float, forced_timestamps: list[float], tolerance: float = 0.5
) -> bool:
    """Check if a frame timestamp is close enough to a forced timestamp."""
    return any(abs(timestamp - ft) <= tolerance for ft in forced_timestamps)


def _merge_trigger_type(is_visual: bool, is_transcript: bool) -> str | None:
    """Determine trigger_type from visual and transcript flags."""
    if is_visual and is_transcript:
        return "both"
    if is_visual:
        return "visual_change"
    if is_transcript:
        return "transcript_trigger"
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_video_trigger_integration.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/core/video.py tests/unit/test_video_trigger_integration.py
git commit -m "feat: add forced keyframe helpers to video module"
```

---

### Task 5: Modify `process_video()` to use forced timestamps and timestamp-based naming

**Files:**
- Modify: `src/peeklet/core/video.py:151-304`

- [ ] **Step 1: Update `process_video()` signature and imports**

In `src/peeklet/core/video.py`, update the imports at the top (lines 11-19) to add `context_exporter`:

```python
from peeklet.core.audio import (
    align_transcript,
    detect_speech_segments,
    get_audio_activity,
    parse_transcript,
)
from peeklet.core.context_exporter import (
    build_context,
    timestamp_filename,
    write_context_json,
    write_context_markdown,
)
from peeklet.core.exporter import ManifestWriter
from peeklet.pipeline import Pipeline
from peeklet.utils.image import ensure_rgb_uint8
```

Update the `process_video()` signature (line 151) to accept forced timestamps:

```python
def process_video(
    path: Path,
    config: PeekletConfig,
    writer: ManifestWriter | None = None,
    forced_timestamps: list[float] | None = None,
) -> list[FrameResult]:
```

- [ ] **Step 2: Add forced keyframe logic in the final pass**

In `process_video()`, after the final pipeline processes each frame (around line 237), add forced keyframe logic. Replace the final pass section (lines 222-268) with:

```python
    # --- Final pass: process frames, enrich metadata, write manifest ---
    final_config = config.model_copy(deep=True)
    final_pipeline = Pipeline(final_config)
    results: list[FrameResult] = []
    keyframe_count = 0
    prev_keyframe_ts: float | None = None
    forced_ts = forced_timestamps or []

    for frame, ts, frame_num in all_frames:
        frame_id = f"frame_{frame_num:06d}"
        result = final_pipeline.process_frame(
            frame,
            frame_id=frame_id,
            source_format="video",
        )

        # Check if this timestamp should force a keyframe
        is_forced = _should_force_keyframe(ts, forced_ts)

        if is_forced and not result.is_keyframe:
            # Force this frame as a keyframe
            from peeklet.core.exporter import save_keyframe as _save_kf

            asset_path = _save_kf(
                frame,
                Path(config.exporter.output_dir),
                frame_id,
                fmt=config.exporter.keyframe_format,
            )
            result.is_keyframe = True
            result.event_type = EventType.KEYFRAME
            result.asset_path = str(asset_path)
            result.visual_reason = "Transcript trigger"

        # Set trigger_type on keyframes
        if result.is_keyframe:
            was_visual = result.visual_reason != "Transcript trigger"
            result.trigger_type = _merge_trigger_type(
                is_visual=was_visual,
                is_transcript=is_forced,
            )

        # Enrich with video metadata
        result.source_video = meta.filename
        result.video_timestamp = ts
        result.video_frame_number = frame_num
        result.video_duration = meta.duration

        if result.is_keyframe:
            keyframe_count += 1
            # Timestamp-based naming
            ts_name = timestamp_filename(ts)
            if result.asset_path:
                old_path = Path(result.asset_path)
                new_path = old_path.with_name(ts_name + old_path.suffix)
                if old_path.exists():
                    old_path.rename(new_path)
                result.asset_path = str(new_path)
            result.frame_id = ts_name
            result.keyframe_index = keyframe_count
            result.change_magnitude = _change_magnitude(result)
            if prev_keyframe_ts is not None:
                result.time_since_prev_keyframe = ts - prev_keyframe_ts
            prev_keyframe_ts = ts

        results.append(result)

    # Backfill total_keyframes on all keyframe results
    for r in results:
        if r.is_keyframe:
            r.total_keyframes = keyframe_count
```

- [ ] **Step 3: Add context export after audio enrichment**

After the audio enrichment section and before writing to the manifest (before line 298), add context export:

```python
    # --- Context export (JSON + Markdown) ---
    output_dir = Path(config.exporter.output_dir)
    transcript_for_context = transcript_segments or []
    ctx = build_context(meta.filename, meta.duration, results, transcript_for_context)
    write_context_json(ctx, output_dir / "context.json")
    write_context_markdown(ctx, output_dir / "context.md")

    # Write fully-enriched results to the manifest
```

Also add the `EventType` import at the top of the file. In the imports section, add:

```python
from peeklet.utils.types import EventType
```

alongside the existing `ensure_rgb_uint8` import line.

- [ ] **Step 4: Run existing video tests**

Run: `pytest tests/integration/test_video_pipeline.py -v` (requires peeklet[video])
Expected: All tests PASS. If video deps not available, run: `pytest tests/unit/test_video_trigger_integration.py -v`

- [ ] **Step 5: Run lint**

Run: `ruff check src/peeklet/core/video.py && ruff format --check src/peeklet/core/video.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/core/video.py
git commit -m "feat: integrate forced timestamps and context export into video pipeline"
```

---

### Task 6: Wire CLI to use trigger detection

**Files:**
- Modify: `src/peeklet/cli.py:116-153`

- [ ] **Step 1: Update `_run_video_mode` to detect triggers and pass forced timestamps**

In `src/peeklet/cli.py`, replace the `_run_video_mode` function (lines 116-153):

```python
def _run_video_mode(input_path: Path, config: peeklet.config.PeekletConfig) -> None:
    """Process video file(s)."""
    from peeklet.core.audio import parse_transcript
    from peeklet.core.exporter import ManifestWriter
    from peeklet.core.transcript_trigger import detect_triggers
    from peeklet.core.video import process_video

    # Detect transcript triggers if transcript is provided
    forced_timestamps: list[float] = []
    if config.video.transcript_path:
        segments = parse_transcript(Path(config.video.transcript_path))
        triggers = detect_triggers(segments)
        forced_timestamps = [t.timestamp for t in triggers]
        if triggers:
            click.echo(f"Found {len(triggers)} transcript trigger(s)")

    if input_path.is_file():
        video_files = [input_path]
    else:
        video_files = sorted(
            f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

    if not video_files:
        click.echo(f"No video files found in {input_path}")
        return

    click.echo(f"Processing {len(video_files)} video(s)")

    output_dir = Path(config.exporter.output_dir)
    writer = ManifestWriter(
        path=output_dir / "manifest.parquet",
        compression=config.exporter.parquet_compression,
    )

    total_keyframes = 0
    for vf in video_files:
        click.echo(f"  Processing: {vf.name}")
        results = process_video(vf, config, writer=writer, forced_timestamps=forced_timestamps)
        kf_count = sum(1 for r in results if r.is_keyframe)
        total_keyframes += kf_count
        click.echo(f"    {kf_count} keyframes extracted")

    writer.flush()

    output_dir = Path(config.exporter.output_dir)
    click.echo(
        f"Done: {total_keyframes} total keyframes. Manifest: {output_dir / 'manifest.parquet'}"
    )
    click.echo(f"Context: {output_dir / 'context.json'}, {output_dir / 'context.md'}")
```

- [ ] **Step 2: Run existing CLI tests**

Run: `pytest tests/unit/test_cli.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run lint**

Run: `ruff check src/peeklet/cli.py && ruff format --check src/peeklet/cli.py`
Expected: All checks passed

- [ ] **Step 4: Commit**

```bash
git add src/peeklet/cli.py
git commit -m "feat: wire transcript trigger detection into CLI video mode"
```

---

### Task 7: Update README and pyproject.toml

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update project description in pyproject.toml**

In `pyproject.toml`, update the description (line 4):

```toml
description = "Smart screenshot change detection — filters noise, exports structured keyframe manifests with LLM-ready context"
```

- [ ] **Step 2: Update README features and usage**

In `README.md`, update the tagline (line 7):

```markdown
Smart screenshot change detection. Filters noise, exports structured Parquet manifests of keyframes with LLM-ready context output.
```

In the Features section, add after the "Video support" bullet:

```markdown
- **LLM-ready context export** -- JSON + Markdown output with bidirectional transcript/screenshot linking
- **Transcript trigger detection** -- automatically captures screenshots when presenter references the screen
```

In the Quick Start CLI section, add an example:

```markdown
# Process video with transcript (produces JSON + Markdown context)
peeklet --input recording.mp4 --output ./output --transcript captions.srt
```

- [ ] **Step 3: Commit**

```bash
git add README.md pyproject.toml
git commit -m "docs: update README and description for context export feature"
```

---

### Task 8: Full integration test

**Files:**
- Modify: `tests/integration/test_video_pipeline.py`

- [ ] **Step 1: Add integration test for video + transcript context export**

Add to `tests/integration/test_video_pipeline.py`:

```python
class TestVideoContextExport:
    def test_video_with_transcript_produces_context_files(self, tmp_path: Path) -> None:
        """Video + transcript -> context.json + context.md + timestamped screenshots."""
        frames = [_solid_frame((0, 0, 0))] * 45 + [_solid_frame((255, 255, 255))] * 45
        video_path = _make_test_video(tmp_path / "demo.mp4", frames, fps=30)

        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Welcome to the demo\n"
            "\n"
            "2\n"
            "00:00:01,000 --> 00:00:02,500\n"
            "Look at this dashboard here\n"
            "\n"
        )

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.transcript_path = str(srt_path)

        from peeklet.core.audio import parse_transcript
        from peeklet.core.transcript_trigger import detect_triggers

        segments = parse_transcript(srt_path)
        triggers = detect_triggers(segments)
        forced_timestamps = [t.timestamp for t in triggers]

        results = process_video(video_path, config, forced_timestamps=forced_timestamps)

        output_dir = tmp_path / "output"

        # Context files exist
        assert (output_dir / "context.json").exists()
        assert (output_dir / "context.md").exists()

        # JSON is valid and has expected structure
        import json
        ctx = json.loads((output_dir / "context.json").read_text())
        assert ctx["video"]["filename"] == "demo.mp4"
        assert ctx["video"]["total_screenshots"] >= 1
        assert len(ctx["screenshots"]) >= 1

        # Screenshots use timestamp-based naming
        for s in ctx["screenshots"]:
            assert s["file"].startswith("screenshot_")
            assert (output_dir / s["file"]).exists()

        # Markdown references screenshots
        md = (output_dir / "context.md").read_text()
        assert "# Video Summary: demo.mp4" in md

    def test_video_without_transcript_still_produces_context(self, tmp_path: Path) -> None:
        """Video without transcript still produces context.json + context.md."""
        frames = [_solid_frame((0, 0, 0))] * 30 + [_solid_frame((255, 255, 255))] * 30
        video_path = _make_test_video(tmp_path / "demo.mp4", frames, fps=30)

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")

        process_video(video_path, config)

        output_dir = tmp_path / "output"
        assert (output_dir / "context.json").exists()
        assert (output_dir / "context.md").exists()

        import json
        ctx = json.loads((output_dir / "context.json").read_text())
        assert ctx["transcript"] == []
        assert ctx["video"]["total_screenshots"] >= 1
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/integration/test_video_pipeline.py -v` (requires peeklet[video])
Expected: All tests PASS (existing + new)

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/datasets`
Expected: All tests PASS

- [ ] **Step 4: Run lint on everything**

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: All checks passed

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_video_pipeline.py
git commit -m "test: add integration tests for video context export"
```
