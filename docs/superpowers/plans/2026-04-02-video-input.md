# Video Input Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept video files (MP4, MOV, WebM) as input, extract meaningful keyframes using smart sampling, and enrich the Parquet manifest with video/audio metadata for downstream LLM consumption.

**Architecture:** A new `core/video.py` module wraps `imageio` for frame extraction with a two-pass smart sampling strategy (coarse at 1fps, backfill around transitions). A new `core/audio.py` module handles speech/silence detection via `pydub` RMS energy analysis and SRT/VTT transcript alignment. Both modules feed into the existing `Pipeline` unchanged — they are frame sources, not processing logic. New fields are added to `FrameResult` and the Parquet schema.

**Tech Stack:** imageio[ffmpeg], pydub, existing Peeklet pipeline

**Spec:** `docs/superpowers/specs/2026-04-02-video-input-design.md`

---

### Task 1: Add Video Fields to FrameResult

**Files:**
- Modify: `peeklet/utils/types.py:48-71`
- Test: `tests/unit/test_types.py`

- [ ] **Step 1: Write failing test for new FrameResult fields**

In `tests/unit/test_types.py`, add:

```python
class TestFrameResultVideoFields:
    def test_video_fields_default_to_none(self) -> None:
        result = FrameResult(
            frame_id="f1",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abc123",
            frame_width=1920,
            frame_height=1080,
        )
        assert result.source_video is None
        assert result.video_timestamp is None
        assert result.video_frame_number is None
        assert result.time_since_prev_keyframe is None
        assert result.audio_activity is None
        assert result.transcript_segment is None
        assert result.keyframe_index is None
        assert result.total_keyframes is None
        assert result.video_duration is None
        assert result.change_magnitude is None

    def test_video_fields_set_explicitly(self) -> None:
        result = FrameResult(
            frame_id="f1",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abc123",
            frame_width=1920,
            frame_height=1080,
            source_video="demo.mp4",
            video_timestamp=8.234,
            video_frame_number=247,
            time_since_prev_keyframe=8.234,
            audio_activity="speech",
            transcript_segment="Click the sign up button",
            keyframe_index=2,
            total_keyframes=5,
            video_duration=90.0,
            change_magnitude="major",
        )
        assert result.source_video == "demo.mp4"
        assert result.video_timestamp == 8.234
        assert result.video_frame_number == 247
        assert result.time_since_prev_keyframe == 8.234
        assert result.audio_activity == "speech"
        assert result.transcript_segment == "Click the sign up button"
        assert result.keyframe_index == 2
        assert result.total_keyframes == 5
        assert result.video_duration == 90.0
        assert result.change_magnitude == "major"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_types.py::TestFrameResultVideoFields -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'source_video'`

- [ ] **Step 3: Add new fields to FrameResult**

In `peeklet/utils/types.py`, add these fields to the `FrameResult` dataclass after `prev_keyframe_path`:

```python
    # Video-specific fields (all None for image-sourced frames)
    source_video: str | None = None
    video_timestamp: float | None = None
    video_frame_number: int | None = None
    time_since_prev_keyframe: float | None = None
    audio_activity: str | None = None
    transcript_segment: str | None = None
    keyframe_index: int | None = None
    total_keyframes: int | None = None
    video_duration: float | None = None
    change_magnitude: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_types.py::TestFrameResultVideoFields -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check no regressions**

Run: `pytest tests/unit/ -v`
Expected: All existing tests still pass (new fields default to None)

- [ ] **Step 6: Commit**

```bash
git add peeklet/utils/types.py tests/unit/test_types.py
git commit -m "feat: add video-specific fields to FrameResult"
```

---

### Task 2: Extend Parquet Schema and ManifestWriter

**Files:**
- Modify: `peeklet/core/exporter.py:17-66` (schema), `peeklet/core/exporter.py:83-113` (append method)
- Test: `tests/unit/test_exporter.py`

- [ ] **Step 1: Write failing test for new Parquet columns**

In `tests/unit/test_exporter.py`, add:

```python
class TestManifestVideoColumns:
    def test_video_columns_in_manifest(self, tmp_path: Path) -> None:
        writer = ManifestWriter(tmp_path / "manifest.parquet")
        result = FrameResult(
            frame_id="step_001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcdef1234567890",
            frame_width=1920,
            frame_height=1080,
            source_video="demo.mp4",
            video_timestamp=8.234,
            video_frame_number=247,
            time_since_prev_keyframe=8.234,
            audio_activity="speech",
            transcript_segment="Click the button",
            keyframe_index=2,
            total_keyframes=5,
            video_duration=90.0,
            change_magnitude="major",
        )
        writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_path / "manifest.parquet")
        assert "source_video" in table.column_names
        assert "video_timestamp" in table.column_names
        assert "video_frame_number" in table.column_names
        assert "time_since_prev_keyframe" in table.column_names
        assert "audio_activity" in table.column_names
        assert "transcript_segment" in table.column_names
        assert "keyframe_index" in table.column_names
        assert "total_keyframes" in table.column_names
        assert "video_duration" in table.column_names
        assert "change_magnitude" in table.column_names

        row = table.to_pydict()
        assert row["source_video"][0] == "demo.mp4"
        assert row["video_timestamp"][0] == pytest.approx(8.234)
        assert row["video_frame_number"][0] == 247
        assert row["keyframe_index"][0] == 2
        assert row["total_keyframes"][0] == 5
        assert row["change_magnitude"][0] == "major"

    def test_video_columns_null_for_image_frames(self, tmp_path: Path) -> None:
        writer = ManifestWriter(tmp_path / "manifest.parquet")
        result = FrameResult(
            frame_id="img_001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcdef1234567890",
            frame_width=1920,
            frame_height=1080,
        )
        writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_path / "manifest.parquet")
        row = table.to_pydict()
        assert row["source_video"][0] is None
        assert row["video_timestamp"][0] is None
        assert row["change_magnitude"][0] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_exporter.py::TestManifestVideoColumns -v`
Expected: FAIL — `source_video` not in column_names

- [ ] **Step 3: Add new columns to MANIFEST_SCHEMA**

In `peeklet/core/exporter.py`, add these fields to `MANIFEST_SCHEMA` after the `prev_keyframe_path` field:

```python
        pa.field("source_video", pa.string(), nullable=True),
        pa.field("video_timestamp", pa.float64(), nullable=True),
        pa.field("video_frame_number", pa.int32(), nullable=True),
        pa.field("time_since_prev_keyframe", pa.float64(), nullable=True),
        pa.field("audio_activity", pa.string(), nullable=True),
        pa.field("transcript_segment", pa.string(), nullable=True),
        pa.field("keyframe_index", pa.int32(), nullable=True),
        pa.field("total_keyframes", pa.int32(), nullable=True),
        pa.field("video_duration", pa.float64(), nullable=True),
        pa.field("change_magnitude", pa.string(), nullable=True),
```

- [ ] **Step 4: Update ManifestWriter.append() to include new fields**

In the `append` method's dict, add after `prev_keyframe_path`:

```python
                "source_video": result.source_video,
                "video_timestamp": result.video_timestamp,
                "video_frame_number": result.video_frame_number,
                "time_since_prev_keyframe": result.time_since_prev_keyframe,
                "audio_activity": result.audio_activity,
                "transcript_segment": result.transcript_segment,
                "keyframe_index": result.keyframe_index,
                "total_keyframes": result.total_keyframes,
                "video_duration": result.video_duration,
                "change_magnitude": result.change_magnitude,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_exporter.py::TestManifestVideoColumns -v`
Expected: PASS

- [ ] **Step 6: Run full exporter tests to check no regressions**

Run: `pytest tests/unit/test_exporter.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add peeklet/core/exporter.py tests/unit/test_exporter.py
git commit -m "feat: extend Parquet schema with video/audio metadata columns"
```

---

### Task 3: Add VideoConfig to PeekletConfig

**Files:**
- Modify: `peeklet/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test for VideoConfig**

In `tests/unit/test_config.py`, add:

```python
class TestVideoConfig:
    def test_defaults(self) -> None:
        config = PeekletConfig()
        assert config.video.sample_fps == 1.0
        assert config.video.formats == ["mp4", "mov", "webm"]
        assert config.video.audio_detection is True
        assert config.video.transcript_path is None

    def test_custom_video_config_from_dict(self) -> None:
        config = PeekletConfig.model_validate({
            "video": {
                "sample_fps": 2.0,
                "formats": ["mp4"],
                "audio_detection": False,
                "transcript_path": "/path/to/transcript.srt",
            }
        })
        assert config.video.sample_fps == 2.0
        assert config.video.formats == ["mp4"]
        assert config.video.audio_detection is False
        assert config.video.transcript_path == "/path/to/transcript.srt"

    def test_sample_fps_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PeekletConfig.model_validate({"video": {"sample_fps": 0}})

    def test_load_video_config_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("video:\n  sample_fps: 0.5\n  audio_detection: false\n")
        config = load_config(config_file)
        assert config.video.sample_fps == 0.5
        assert config.video.audio_detection is False
```

Note: Import `ValidationError` from `pydantic` at the top of the test file if not already imported.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::TestVideoConfig -v`
Expected: FAIL — `AttributeError: 'PeekletConfig' object has no attribute 'video'`

- [ ] **Step 3: Add VideoConfig model and wire into PeekletConfig**

In `peeklet/config.py`, add after the `InputConfig` class:

```python
class VideoConfig(BaseModel):
    """Video input settings."""

    sample_fps: float = Field(default=1.0, gt=0.0)
    formats: List[str] = Field(default_factory=lambda: ["mp4", "mov", "webm"])
    audio_detection: bool = True
    transcript_path: Optional[str] = None
```

And add to `PeekletConfig`:

```python
    video: VideoConfig = Field(default_factory=VideoConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py::TestVideoConfig -v`
Expected: PASS

- [ ] **Step 5: Run full config tests**

Run: `pytest tests/unit/test_config.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add peeklet/config.py tests/unit/test_config.py
git commit -m "feat: add VideoConfig to PeekletConfig"
```

---

### Task 4: Add `peeklet[video]` Optional Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add video optional dependency group**

In `pyproject.toml`, add after the `ocr` group in `[project.optional-dependencies]`:

```toml
video = [
    "imageio[ffmpeg]>=2.31",
    "pydub>=0.25",
]
```

- [ ] **Step 2: Add imageio and pydub to mypy overrides**

In the `[[tool.mypy.overrides]]` section, add `"imageio.*"`, `"pydub.*"` to the `module` list:

```toml
[[tool.mypy.overrides]]
module = [
    "pyarrow.*",
    "imagehash.*",
    "skimage.*",
    "imageio.*",
    "pydub.*",
]
ignore_missing_imports = true
disallow_untyped_calls = false
```

- [ ] **Step 3: Install the new dependencies**

Run: `pip install -e ".[video,dev]"`
Expected: imageio, imageio-ffmpeg, and pydub install successfully

- [ ] **Step 4: Verify imports work**

Run: `python -c "import imageio.v3 as iio; print('imageio OK')" && python -c "from pydub import AudioSegment; print('pydub OK')"`
Expected: Both print OK

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add peeklet[video] optional dependency (imageio, pydub)"
```

---

### Task 5: Implement SRT/VTT Transcript Parser

**Files:**
- Create: `peeklet/core/audio.py`
- Test: `tests/unit/test_audio.py`

This task implements transcript parsing only. Speech/silence detection is Task 8.

- [ ] **Step 1: Write failing tests for SRT parsing**

Create `tests/unit/test_audio.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_audio.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_transcript' from 'peeklet.core.audio'`

- [ ] **Step 3: Implement transcript parser**

Create `peeklet/core/audio.py`:

```python
"""Audio analysis — transcript parsing and speech/silence detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A timestamped segment from an SRT or VTT transcript."""

    start: float  # seconds
    end: float  # seconds
    text: str


def parse_transcript(path: Path) -> list[TranscriptSegment]:
    """Parse an SRT or VTT file into timestamped segments.

    Auto-detects format by file extension (.srt or .vtt).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return _parse_vtt(text)
    return _parse_srt(text)


def _parse_timestamp(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


_TIMESTAMP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})"
)


def _parse_srt(text: str) -> list[TranscriptSegment]:
    """Parse SRT format."""
    segments: list[TranscriptSegment] = []
    blocks = re.split(r"\n\n+", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        match = None
        timestamp_line_idx = -1
        for i, line in enumerate(lines):
            match = _TIMESTAMP_RE.search(line)
            if match:
                timestamp_line_idx = i
                break
        if match is None or timestamp_line_idx == -1:
            continue
        start = _parse_timestamp(match.group(1))
        end = _parse_timestamp(match.group(2))
        text_lines = lines[timestamp_line_idx + 1 :]
        content = " ".join(line.strip() for line in text_lines if line.strip())
        if content:
            segments.append(TranscriptSegment(start=start, end=end, text=content))
    return segments


def _parse_vtt(text: str) -> list[TranscriptSegment]:
    """Parse WebVTT format."""
    # Strip WEBVTT header and any metadata lines before the first blank line
    lines = text.splitlines()
    body_start = 0
    if lines and lines[0].startswith("WEBVTT"):
        # Skip until first blank line after header
        for i in range(1, len(lines)):
            if lines[i].strip() == "":
                body_start = i + 1
                break
        else:
            body_start = len(lines)

    body = "\n".join(lines[body_start:])
    return _parse_srt(body)


def align_transcript(timestamp: float, segments: list[TranscriptSegment]) -> str | None:
    """Find transcript segment(s) overlapping the given timestamp.

    Returns joined text if multiple segments overlap, or None if no match.
    """
    matches = [s for s in segments if s.start <= timestamp <= s.end]
    if not matches:
        return None
    return " | ".join(s.text for s in matches)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_audio.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add peeklet/core/audio.py tests/unit/test_audio.py
git commit -m "feat: add SRT/VTT transcript parser with timestamp alignment"
```

---

### Task 6: Implement VideoDecoder — Metadata and Coarse Sampling

**Files:**
- Create: `peeklet/core/video.py`
- Test: `tests/unit/test_video.py`

- [ ] **Step 1: Write failing tests for VideoDecoder metadata and coarse frame extraction**

Create `tests/unit/test_video.py`:

```python
"""Tests for the video decoder module."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from peeklet.core.video import VideoDecoder, VideoMeta


def _make_test_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> Path:
    """Create a synthetic video file for testing."""
    with iio.imopen(path, "w", plugin="pyav") as out:
        out.init_video_stream("libx264", fps=fps)
        for frame in frames:
            out.write_frame(frame)
    return path


def _solid_frame(color: tuple[int, int, int], w: int = 320, h: int = 240) -> np.ndarray:
    """Create a solid-color RGB frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class TestVideoMeta:
    def test_metadata_from_video(self, tmp_path: Path) -> None:
        frames = [_solid_frame((255, 0, 0))] * 30  # 1 second of red
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)

        decoder = VideoDecoder(video_path)
        meta = decoder.get_metadata()

        assert isinstance(meta, VideoMeta)
        assert meta.filename == "test.mp4"
        assert meta.fps == pytest.approx(30.0, abs=1.0)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.duration == pytest.approx(1.0, abs=0.2)

    def test_invalid_file_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.mp4"
        bad_file.write_bytes(b"not a video")
        with pytest.raises(ValueError, match="Could not open video"):
            VideoDecoder(bad_file)

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            VideoDecoder(Path("/nonexistent/video.mp4"))


class TestCoarseExtraction:
    def test_coarse_pass_extracts_at_sample_fps(self, tmp_path: Path) -> None:
        # 3 seconds of video at 30fps = 90 frames
        # First 30 red, next 30 green, last 30 blue
        frames = (
            [_solid_frame((255, 0, 0))] * 30
            + [_solid_frame((0, 255, 0))] * 30
            + [_solid_frame((0, 0, 255))] * 30
        )
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)

        decoder = VideoDecoder(video_path)
        coarse_frames = list(decoder.extract_coarse_frames(sample_fps=1.0))

        # At 1fps over ~3 seconds, expect ~3 samples
        assert len(coarse_frames) >= 2
        # Each frame should be (numpy_array, timestamp_seconds, frame_number)
        for frame, ts, frame_num in coarse_frames:
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8
            assert isinstance(ts, float)
            assert isinstance(frame_num, int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_video.py -v`
Expected: FAIL — `ImportError: cannot import name 'VideoDecoder' from 'peeklet.core.video'`

- [ ] **Step 3: Implement VideoDecoder with metadata and coarse extraction**

Create `peeklet/core/video.py`:

```python
"""Video frame extraction with smart sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from peeklet.utils.image import ensure_rgb_uint8


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """Metadata about a video file."""

    filename: str
    duration: float  # seconds
    fps: float
    width: int
    height: int
    total_frames: int


def _check_video_deps() -> None:
    """Raise a clear error if video dependencies are not installed."""
    try:
        import imageio.v3  # noqa: F401
    except ImportError:
        raise ImportError(
            "Video support requires additional dependencies. "
            "Install with: pip install peeklet[video]"
        ) from None


class VideoDecoder:
    """Decodes video files and extracts frames with smart sampling.

    Wraps imageio for frame extraction. Provides coarse sampling
    at a configurable fps and backfill extraction around transitions.
    """

    def __init__(self, path: Path) -> None:
        _check_video_deps()
        import imageio.v3 as iio

        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        try:
            props = iio.improps(self._path, plugin="pyav")
            meta = iio.immeta(self._path, plugin="pyav")
        except Exception as e:
            raise ValueError(f"Could not open video: {self._path}: {e}") from e

        self._fps: float = meta.get("fps", 30.0)
        self._duration: float = meta.get("duration", 0.0)
        h, w = props.shape[:2]
        self._width: int = w
        self._height: int = h
        self._total_frames: int = int(self._fps * self._duration)

    def get_metadata(self) -> VideoMeta:
        """Return metadata about the video file."""
        return VideoMeta(
            filename=self._path.name,
            duration=self._duration,
            fps=self._fps,
            width=self._width,
            height=self._height,
            total_frames=self._total_frames,
        )

    def extract_coarse_frames(
        self, sample_fps: float = 1.0
    ) -> Iterator[tuple[np.ndarray, float, int]]:
        """Extract frames at a coarse sample rate.

        Yields (frame_rgb, timestamp_seconds, frame_number) tuples.
        """
        import imageio.v3 as iio

        frame_interval = max(1, int(self._fps / sample_fps))

        for idx, frame in enumerate(iio.imiter(self._path, plugin="pyav")):
            if idx % frame_interval != 0:
                continue
            timestamp = idx / self._fps
            rgb = ensure_rgb_uint8(np.asarray(frame))
            yield rgb, timestamp, idx

    def extract_frame_range(
        self, start_frame: int, end_frame: int
    ) -> Iterator[tuple[np.ndarray, float, int]]:
        """Extract all frames in a range [start_frame, end_frame).

        Used for backfill around detected transitions.
        Yields (frame_rgb, timestamp_seconds, frame_number) tuples.
        """
        import imageio.v3 as iio

        for idx, frame in enumerate(iio.imiter(self._path, plugin="pyav")):
            if idx < start_frame:
                continue
            if idx >= end_frame:
                break
            timestamp = idx / self._fps
            rgb = ensure_rgb_uint8(np.asarray(frame))
            yield rgb, timestamp, idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_video.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add peeklet/core/video.py tests/unit/test_video.py
git commit -m "feat: add VideoDecoder with metadata extraction and coarse sampling"
```

---

### Task 7: Implement Smart Sampling with Backfill

**Files:**
- Modify: `peeklet/core/video.py`
- Test: `tests/unit/test_video.py`

- [ ] **Step 1: Write failing test for backfill logic**

In `tests/unit/test_video.py`, add:

```python
from peeklet.config import PeekletConfig
from peeklet.core.video import process_video


class TestSmartSampling:
    def test_backfill_finds_transition(self, tmp_path: Path) -> None:
        # 3 seconds: 1.5s red, then 1.5s green (sharp transition at frame 45)
        frames = (
            [_solid_frame((255, 0, 0))] * 45
            + [_solid_frame((0, 255, 0))] * 45
        )
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.sample_fps = 1.0

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # Should find at least 2 keyframes: first frame + the transition
        assert len(keyframes) >= 2
        # First keyframe is always frame 0
        assert keyframes[0].video_frame_number == 0
        # Second keyframe should be near frame 45 (the transition),
        # not at frame 60 (the next coarse sample)
        assert keyframes[1].video_frame_number is not None
        assert keyframes[1].video_frame_number < 60

    def test_no_backfill_when_no_transitions(self, tmp_path: Path) -> None:
        # 2 seconds of solid red — no transitions
        frames = [_solid_frame((255, 0, 0))] * 60
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.sample_fps = 1.0

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # Only the first frame should be a keyframe
        assert len(keyframes) == 1

    def test_results_have_video_metadata(self, tmp_path: Path) -> None:
        frames = (
            [_solid_frame((255, 0, 0))] * 30
            + [_solid_frame((0, 255, 0))] * 30
        )
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        for kf in keyframes:
            assert kf.source_video == "test.mp4"
            assert kf.video_timestamp is not None
            assert kf.video_frame_number is not None
            assert kf.video_duration is not None
            assert kf.video_duration == pytest.approx(2.0, abs=0.5)
            assert kf.keyframe_index is not None
            assert kf.change_magnitude is not None

    def test_time_since_prev_keyframe(self, tmp_path: Path) -> None:
        frames = (
            [_solid_frame((255, 0, 0))] * 30
            + [_solid_frame((0, 255, 0))] * 30
        )
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # First keyframe has no previous
        assert keyframes[0].time_since_prev_keyframe is None
        # Second keyframe should have a time gap
        if len(keyframes) > 1:
            assert keyframes[1].time_since_prev_keyframe is not None
            assert keyframes[1].time_since_prev_keyframe > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_video.py::TestSmartSampling -v`
Expected: FAIL — `ImportError: cannot import name 'process_video'`

- [ ] **Step 3: Implement process_video with smart sampling and backfill**

In `peeklet/core/video.py`, add the `process_video` function and a helper to derive change magnitude. Add these imports at the top:

```python
from peeklet.config import PeekletConfig
from peeklet.pipeline import Pipeline
from peeklet.utils.types import FrameResult
```

Then add after the `VideoDecoder` class:

```python
def _change_magnitude(result: FrameResult) -> str:
    """Derive change magnitude from SSIM and block count."""
    if result.ssim_score is None:
        return "major"  # first frame or dimension change
    n_blocks = len(result.changed_regions) if result.changed_regions else 0
    if result.ssim_score < 0.70 or n_blocks > 15:
        return "major"
    if result.ssim_score < 0.90 or n_blocks > 5:
        return "moderate"
    return "minor"


def process_video(path: Path, config: PeekletConfig) -> list[FrameResult]:
    """Process a video file through smart sampling + Peeklet pipeline.

    1. Coarse pass at config.video.sample_fps
    2. Backfill around SKIP->KEYFRAME transitions at native fps
    3. Enrich results with video-specific metadata

    Returns list of FrameResult for all processed frames (keyframes + skips).
    """
    decoder = VideoDecoder(path)
    meta = decoder.get_metadata()
    pipeline = Pipeline(config)
    sample_fps = config.video.sample_fps

    # --- Pass 1: Coarse sampling ---
    coarse_samples: list[tuple[np.ndarray, float, int, FrameResult]] = []
    for frame, ts, frame_num in decoder.extract_coarse_frames(sample_fps):
        frame_id = f"step_{len(coarse_samples):03d}"
        result = pipeline.process_frame(
            frame, frame_id=frame_id, source_format="video",
        )
        coarse_samples.append((frame, ts, frame_num, result))

    # --- Pass 2: Identify transitions and backfill ---
    # Reset pipeline for the backfill pass so we re-evaluate from scratch
    # with the backfill frames inserted at the right positions
    pipeline_final = Pipeline(config)
    all_frames: list[tuple[np.ndarray, float, int]] = []

    # Collect frames that need processing: coarse + backfill gaps
    for i, (frame, ts, frame_num, result) in enumerate(coarse_samples):
        if i > 0:
            prev_result = coarse_samples[i - 1][3]
            prev_frame_num = coarse_samples[i - 1][2]
            # SKIP -> KEYFRAME transition: backfill the gap
            if not prev_result.is_keyframe and result.is_keyframe:
                for bf_frame, bf_ts, bf_num in decoder.extract_frame_range(
                    prev_frame_num + 1, frame_num
                ):
                    all_frames.append((bf_frame, bf_ts, bf_num))
        all_frames.append((frame, ts, frame_num))

    # --- Final pass: process all frames in order and enrich ---
    results: list[FrameResult] = []
    keyframe_count = 0
    prev_keyframe_ts: float | None = None

    for frame, ts, frame_num in all_frames:
        frame_id = f"step_{keyframe_count:03d}" if keyframe_count == 0 else f"frame_{frame_num:06d}"
        result = pipeline_final.process_frame(
            frame, frame_id=frame_id, source_format="video",
        )

        # Enrich with video metadata
        result.source_video = meta.filename
        result.video_timestamp = ts
        result.video_frame_number = frame_num
        result.video_duration = meta.duration

        if result.is_keyframe:
            keyframe_count += 1
            result.frame_id = f"step_{keyframe_count:03d}"
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

    pipeline_final.finalize()
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_video.py::TestSmartSampling -v`
Expected: PASS

- [ ] **Step 5: Run all video tests**

Run: `pytest tests/unit/test_video.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add peeklet/core/video.py tests/unit/test_video.py
git commit -m "feat: implement smart sampling with backfill around transitions"
```

---

### Task 8: Implement Speech/Silence Detection

**Files:**
- Modify: `peeklet/core/audio.py`
- Test: `tests/unit/test_audio.py`

- [ ] **Step 1: Write failing tests for speech/silence detection**

In `tests/unit/test_audio.py`, add:

```python
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
        # At least one segment should overlap the 1.0-2.0s range
        has_speech_in_middle = any(
            s.start < 2.0 and s.end > 1.0 for s in speech_segments
        )
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_audio.py::TestSpeechSilenceDetection -v`
Expected: FAIL — `ImportError: cannot import name 'detect_speech_segments'`

- [ ] **Step 3: Implement speech/silence detection**

In `peeklet/core/audio.py`, add:

```python
def _check_audio_deps() -> None:
    """Raise a clear error if audio dependencies are not installed."""
    try:
        import pydub  # noqa: F401
    except ImportError:
        raise ImportError(
            "Audio detection requires additional dependencies. "
            "Install with: pip install peeklet[video]"
        ) from None


def detect_speech_segments(
    audio_path: Path,
    chunk_ms: int = 500,
    silence_threshold_dbfs: float = -40.0,
) -> list[TranscriptSegment]:
    """Detect speech segments using RMS energy thresholds.

    Divides audio into chunks and classifies each as speech or silence
    based on dBFS level. Merges consecutive speech chunks into segments.

    Returns list of TranscriptSegment with text="[speech]" for detected speech.
    """
    _check_audio_deps()
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    duration_s = len(audio) / 1000.0

    speech_ranges: list[tuple[float, float]] = []
    current_start: float | None = None

    for chunk_start_ms in range(0, len(audio), chunk_ms):
        chunk_end_ms = min(chunk_start_ms + chunk_ms, len(audio))
        chunk = audio[chunk_start_ms:chunk_end_ms]

        is_speech = chunk.dBFS > silence_threshold_dbfs

        start_s = chunk_start_ms / 1000.0
        end_s = chunk_end_ms / 1000.0

        if is_speech:
            if current_start is None:
                current_start = start_s
        else:
            if current_start is not None:
                speech_ranges.append((current_start, start_s))
                current_start = None

    # Close any trailing speech segment
    if current_start is not None:
        speech_ranges.append((current_start, duration_s))

    return [
        TranscriptSegment(start=s, end=e, text="[speech]")
        for s, e in speech_ranges
    ]


def get_audio_activity(
    timestamp: float, speech_segments: list[TranscriptSegment]
) -> str:
    """Return 'speech' or 'silence' for a given timestamp."""
    for seg in speech_segments:
        if seg.start <= timestamp <= seg.end:
            return "speech"
    return "silence"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_audio.py::TestSpeechSilenceDetection -v`
Expected: PASS

- [ ] **Step 5: Run all audio tests**

Run: `pytest tests/unit/test_audio.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add peeklet/core/audio.py tests/unit/test_audio.py
git commit -m "feat: add speech/silence detection via pydub RMS energy"
```

---

### Task 9: Wire Audio into Video Processing

**Files:**
- Modify: `peeklet/core/video.py`
- Test: `tests/unit/test_video.py`

- [ ] **Step 1: Write failing test for audio enrichment**

In `tests/unit/test_video.py`, add:

```python
class TestAudioEnrichment:
    def test_audio_activity_populated_when_enabled(self, tmp_path: Path) -> None:
        frames = (
            [_solid_frame((255, 0, 0))] * 30
            + [_solid_frame((0, 255, 0))] * 30
        )
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.audio_detection = True

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        for kf in keyframes:
            assert kf.audio_activity in ("speech", "silence")

    def test_audio_activity_none_when_disabled(self, tmp_path: Path) -> None:
        frames = [_solid_frame((255, 0, 0))] * 30
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.audio_detection = False

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        for kf in keyframes:
            assert kf.audio_activity is None

    def test_transcript_alignment(self, tmp_path: Path) -> None:
        frames = (
            [_solid_frame((255, 0, 0))] * 60
            + [_solid_frame((0, 255, 0))] * 60
        )
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)

        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(
            "1\n"
            "00:00:00,000 --> 00:00:01,500\n"
            "Welcome to the demo\n"
            "\n"
            "2\n"
            "00:00:02,000 --> 00:00:03,500\n"
            "Now click here\n"
            "\n"
        )

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.transcript_path = str(srt_path)

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # First keyframe at t=0 should match first transcript segment
        assert keyframes[0].transcript_segment is not None
        assert "Welcome" in keyframes[0].transcript_segment
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_video.py::TestAudioEnrichment -v`
Expected: FAIL — `audio_activity` is None even when enabled (not yet wired)

- [ ] **Step 3: Wire audio detection and transcript alignment into process_video**

In `peeklet/core/video.py`, add these imports at the top:

```python
from peeklet.core.audio import (
    align_transcript,
    detect_speech_segments,
    get_audio_activity,
    parse_transcript,
    TranscriptSegment,
)
```

Then modify `process_video` to add audio enrichment after the final pass. Insert this block before `pipeline_final.finalize()`:

```python
    # --- Audio enrichment ---
    transcript_segments: list[TranscriptSegment] | None = None
    speech_segments: list[TranscriptSegment] | None = None

    if config.video.transcript_path:
        transcript_segments = parse_transcript(Path(config.video.transcript_path))

    if config.video.audio_detection:
        try:
            speech_segments = detect_speech_segments(path)
        except Exception:
            speech_segments = None  # Audio extraction may fail for some videos

    for r in results:
        if not r.is_keyframe or r.video_timestamp is None:
            continue

        ts = r.video_timestamp

        # Transcript alignment (takes priority for audio_activity too)
        if transcript_segments:
            r.transcript_segment = align_transcript(ts, transcript_segments)
            r.audio_activity = "speech" if r.transcript_segment else "silence"
        elif speech_segments is not None:
            r.audio_activity = get_audio_activity(ts, speech_segments)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_video.py::TestAudioEnrichment -v`
Expected: PASS

- [ ] **Step 5: Run all video tests**

Run: `pytest tests/unit/test_video.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add peeklet/core/video.py tests/unit/test_video.py
git commit -m "feat: wire audio detection and transcript alignment into video processing"
```

---

### Task 10: Update CLI for Video Input

**Files:**
- Modify: `peeklet/cli.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI video mode**

In `tests/unit/test_cli.py`, add (using Click's test runner):

```python
from click.testing import CliRunner

from peeklet.cli import main


class TestVideoCliDetection:
    def test_video_file_input(self, tmp_path: Path) -> None:
        """CLI accepts a video file as --input."""
        import imageio.v3 as iio
        import numpy as np

        # Create a tiny test video
        video_path = tmp_path / "demo.mp4"
        frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
        with iio.imopen(video_path, "w", plugin="pyav") as out:
            out.init_video_stream("libx264", fps=10)
            for f in frames:
                out.write_frame(f)

        runner = CliRunner()
        result = runner.invoke(main, [
            "--input", str(video_path),
            "--output", str(tmp_path / "output"),
            "--no-redact",
        ])
        assert result.exit_code == 0
        assert "keyframe" in result.output.lower()

    def test_directory_with_only_videos(self, tmp_path: Path) -> None:
        """CLI processes directory of video files."""
        import imageio.v3 as iio
        import numpy as np

        for name in ["a.mp4", "b.mp4"]:
            video_path = tmp_path / name
            frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
            with iio.imopen(video_path, "w", plugin="pyav") as out:
                out.init_video_stream("libx264", fps=10)
                for f in frames:
                    out.write_frame(f)

        runner = CliRunner()
        result = runner.invoke(main, [
            "--input", str(tmp_path),
            "--output", str(tmp_path / "output"),
            "--no-redact",
        ])
        assert result.exit_code == 0

    def test_mixed_directory_without_mode_errors(self, tmp_path: Path) -> None:
        """CLI errors on mixed directory without --mode."""
        import imageio.v3 as iio
        import numpy as np
        from PIL import Image

        # Create one video and one image
        video_path = tmp_path / "demo.mp4"
        frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
        with iio.imopen(video_path, "w", plugin="pyav") as out:
            out.init_video_stream("libx264", fps=10)
            for f in frames:
                out.write_frame(f)

        img = Image.new("RGB", (80, 60))
        img.save(tmp_path / "shot.png")

        runner = CliRunner()
        result = runner.invoke(main, [
            "--input", str(tmp_path),
            "--output", str(tmp_path / "output"),
        ])
        assert result.exit_code != 0
        assert "--mode" in result.output

    def test_mixed_directory_with_mode_video(self, tmp_path: Path) -> None:
        """CLI processes only videos when --mode video is specified."""
        import imageio.v3 as iio
        import numpy as np
        from PIL import Image

        video_path = tmp_path / "demo.mp4"
        frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
        with iio.imopen(video_path, "w", plugin="pyav") as out:
            out.init_video_stream("libx264", fps=10)
            for f in frames:
                out.write_frame(f)

        img = Image.new("RGB", (80, 60))
        img.save(tmp_path / "shot.png")

        runner = CliRunner()
        result = runner.invoke(main, [
            "--input", str(tmp_path),
            "--output", str(tmp_path / "output"),
            "--mode", "video",
            "--no-redact",
        ])
        assert result.exit_code == 0

    def test_transcript_flag(self, tmp_path: Path) -> None:
        """CLI accepts --transcript flag."""
        import imageio.v3 as iio
        import numpy as np

        video_path = tmp_path / "demo.mp4"
        frames = [np.zeros((60, 80, 3), dtype=np.uint8)] * 10
        with iio.imopen(video_path, "w", plugin="pyav") as out:
            out.init_video_stream("libx264", fps=10)
            for f in frames:
                out.write_frame(f)

        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "--input", str(video_path),
            "--output", str(tmp_path / "output"),
            "--no-redact",
            "--transcript", str(srt_path),
        ])
        assert result.exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_cli.py::TestVideoCliDetection -v`
Expected: FAIL — CLI doesn't accept video file as `--input` (currently `file_okay=False`)

- [ ] **Step 3: Rewrite CLI to support both video and image modes**

Replace the contents of `peeklet/cli.py` with:

```python
"""CLI entry point for Peeklet."""

from __future__ import annotations

from pathlib import Path

import click

import peeklet
from peeklet.config import load_config
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}


def _detect_mode(
    input_path: Path, mode: str | None, image_extensions: set[str]
) -> str:
    """Detect whether input is video or image mode.

    Returns 'video' or 'image'.
    Raises click.UsageError for ambiguous cases.
    """
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            return "video"
        return "image"

    # Directory — scan contents
    has_videos = any(
        f.suffix.lower() in VIDEO_EXTENSIONS
        for f in input_path.iterdir() if f.is_file()
    )
    has_images = any(
        f.suffix.lower() in image_extensions
        for f in input_path.iterdir() if f.is_file()
    )

    if mode:
        return mode

    if has_videos and has_images:
        raise click.UsageError(
            "Directory contains both image and video files. "
            "Select a mode: --mode video or --mode image"
        )

    if has_videos:
        return "video"
    return "image"


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Video file, or directory containing images or videos.",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default="./output",
    help="Directory for keyframes and manifest.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config file (JSON or YAML).",
)
@click.option("--no-redact", is_flag=True, default=False, help="Disable PII redaction.")
@click.option("--no-audio", is_flag=True, default=False, help="Disable audio detection for video.")
@click.option(
    "--mode",
    type=click.Choice(["video", "image"]),
    default=None,
    help="Force video or image mode (required for mixed directories).",
)
@click.option(
    "--transcript",
    "transcript_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to SRT or VTT transcript file.",
)
@click.version_option(version=peeklet.__version__, prog_name="peeklet")
def main(
    input_path: Path,
    output_dir: Path,
    config_path: Path | None,
    no_redact: bool,
    no_audio: bool,
    mode: str | None,
    transcript_path: Path | None,
) -> None:
    """Smart screenshot change detection.

    Filters noise from screenshot sequences or video recordings,
    redacts PII, and exports a structured Parquet manifest of keyframes.
    """
    config = load_config(config_path)
    if no_redact:
        config.redactor.enabled = False
    if no_audio:
        config.video.audio_detection = False
    if transcript_path:
        config.video.transcript_path = str(transcript_path)
    config.exporter.output_dir = str(output_dir)

    image_extensions = {f".{fmt}" for fmt in config.input.supported_formats}
    detected_mode = _detect_mode(input_path, mode, image_extensions)

    if detected_mode == "video":
        _run_video_mode(input_path, config)
    else:
        _run_image_mode(input_path, config, image_extensions)


def _run_video_mode(input_path: Path, config: "peeklet.config.PeekletConfig") -> None:
    """Process video file(s)."""
    from peeklet.core.video import process_video

    if input_path.is_file():
        video_files = [input_path]
    else:
        video_files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

    if not video_files:
        click.echo(f"No video files found in {input_path}")
        return

    click.echo(f"Processing {len(video_files)} video(s)")

    total_keyframes = 0
    for vf in video_files:
        click.echo(f"  Processing: {vf.name}")
        results = process_video(vf, config)
        kf_count = sum(1 for r in results if r.is_keyframe)
        total_keyframes += kf_count
        click.echo(f"    {kf_count} keyframes extracted")

    output_dir = Path(config.exporter.output_dir)
    click.echo(
        f"Done: {total_keyframes} total keyframes. "
        f"Manifest: {output_dir / 'manifest.parquet'}"
    )


def _run_image_mode(
    input_dir: Path,
    config: "peeklet.config.PeekletConfig",
    extensions: set[str],
) -> None:
    """Process image directory (existing behavior)."""
    pipeline = Pipeline(config)
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not files:
        click.echo(f"No supported images found in {input_dir}")
        return

    click.echo(f"Processing {len(files)} frames from {input_dir}")

    keyframe_count = 0
    for f in files:
        frame = load_frame(f)
        result = pipeline.process_frame(
            frame, frame_id=f.stem, source_format=f.suffix.lstrip(".")
        )
        if result.is_keyframe:
            keyframe_count += 1

    pipeline.finalize()
    output_dir = Path(config.exporter.output_dir)
    click.echo(
        f"Done: {keyframe_count} keyframes from {len(files)} frames "
        f"({100 * keyframe_count / len(files):.1f}%). "
        f"Manifest: {output_dir / 'manifest.parquet'}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_cli.py::TestVideoCliDetection -v`
Expected: PASS

- [ ] **Step 5: Run full CLI and unit tests**

Run: `pytest tests/unit/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add peeklet/cli.py tests/unit/test_cli.py
git commit -m "feat: update CLI to support video input with --mode, --transcript, --no-audio"
```

---

### Task 11: Integration Test — End-to-End Video Processing

**Files:**
- Create: `tests/integration/test_video_pipeline.py`

- [ ] **Step 1: Write end-to-end integration test**

Create `tests/integration/test_video_pipeline.py`:

```python
"""Integration tests for video input end-to-end."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import pytest

from peeklet.config import PeekletConfig
from peeklet.core.video import process_video


def _solid_frame(color: tuple[int, int, int], w: int = 320, h: int = 240) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _make_test_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> Path:
    with iio.imopen(path, "w", plugin="pyav") as out:
        out.init_video_stream("libx264", fps=fps)
        for frame in frames:
            out.write_frame(frame)
    return path


class TestVideoEndToEnd:
    def test_full_pipeline_produces_manifest_and_keyframes(self, tmp_path: Path) -> None:
        """Video → process_video → manifest.parquet + keyframe images."""
        frames = (
            [_solid_frame((255, 0, 0))] * 45
            + [_solid_frame((0, 255, 0))] * 45
            + [_solid_frame((0, 0, 255))] * 45
        )
        video_path = _make_test_video(tmp_path / "demo.mp4", frames, fps=30)

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = False

        results = process_video(video_path, config)

        # Check results structure
        keyframes = [r for r in results if r.is_keyframe]
        assert len(keyframes) >= 2  # At least first frame + one transition

        # Check manifest was written
        manifest_path = tmp_path / "output" / "manifest.parquet"
        assert manifest_path.exists()
        table = pq.read_table(manifest_path)

        # Verify video-specific columns exist and are populated
        kf_rows = table.filter(table.column("is_keyframe").to_pylist())
        row = table.to_pydict()
        keyframe_indices = [
            i for i, is_kf in enumerate(row["is_keyframe"]) if is_kf
        ]

        for idx in keyframe_indices:
            assert row["source_video"][idx] == "demo.mp4"
            assert row["video_timestamp"][idx] is not None
            assert row["video_frame_number"][idx] is not None
            assert row["keyframe_index"][idx] is not None
            assert row["total_keyframes"][idx] is not None
            assert row["video_duration"][idx] is not None
            assert row["change_magnitude"][idx] in ("minor", "moderate", "major")

        # Check keyframe images exist
        output_dir = tmp_path / "output"
        keyframe_images = list(output_dir.glob("step_*.png"))
        assert len(keyframe_images) >= 2

    def test_video_with_transcript(self, tmp_path: Path) -> None:
        """Video + transcript → keyframes with transcript_segment populated."""
        frames = (
            [_solid_frame((255, 0, 0))] * 60
            + [_solid_frame((0, 255, 0))] * 60
        )
        video_path = _make_test_video(tmp_path / "demo.mp4", frames, fps=30)

        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(
            "1\n"
            "00:00:00,000 --> 00:00:01,500\n"
            "Welcome to the demo\n"
            "\n"
            "2\n"
            "00:00:02,000 --> 00:00:03,500\n"
            "Now click here\n"
            "\n"
        )

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = False
        config.video.transcript_path = str(srt_path)

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # First keyframe at t=0 should have transcript
        assert keyframes[0].transcript_segment is not None

    def test_multi_video_directory(self, tmp_path: Path) -> None:
        """Multiple videos in a directory produce unified manifest."""
        for name, color in [("a.mp4", (255, 0, 0)), ("b.mp4", (0, 255, 0))]:
            frames = [_solid_frame(color)] * 30
            _make_test_video(tmp_path / name, frames, fps=30)

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = False

        # Process both videos
        for vf in sorted(tmp_path.glob("*.mp4")):
            process_video(vf, config)

        manifest_path = tmp_path / "output" / "manifest.parquet"
        assert manifest_path.exists()
        table = pq.read_table(manifest_path)
        videos = set(table.to_pydict()["source_video"])
        # Filter out None values (shouldn't be any, but defensive)
        videos = {v for v in videos if v is not None}
        assert "a.mp4" in videos
        assert "b.mp4" in videos
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/integration/test_video_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v -m "not datasets and not mind2web and not showui and not webui and not benchmark"`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_video_pipeline.py
git commit -m "test: add end-to-end integration tests for video pipeline"
```

---

### Task 12: Lint, Type Check, and Final Verification

**Files:**
- All modified files

- [ ] **Step 1: Run ruff linter**

Run: `ruff check peeklet/ tests/`
Expected: No errors. Fix any issues if found.

- [ ] **Step 2: Run mypy type checker**

Run: `mypy peeklet/`
Expected: No errors. Fix any issues if found.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v -m "not datasets and not mind2web and not showui and not webui and not benchmark"`
Expected: All pass

- [ ] **Step 4: Commit any lint/type fixes**

```bash
git add -u
git commit -m "chore: fix lint and type check issues for video input feature"
```

(Skip this commit if no fixes were needed.)
