# Demo-Mode Frame Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--demo-mode` flag that filters video keyframes down to ~100-200 high-signal frames using a sparse OCR sweep + transcript-anchored selection, in under 60 seconds for a 60-min video.

**Architecture:** A new `demo_filter.py` module sits on top of the existing video pipeline. Stage 1 runs Tesseract on a sparse time-grid (1 sample per 30s) to classify segments as demo / non-demo. Stages 2-3 drop non-demo keyframes and pick at most one keyframe per transcript segment plus any "major change" frames. Existing video processing is unchanged unless `--demo-mode` is passed.

**Tech Stack:** Python 3.10+, pytesseract, PyAV, NumPy, Pillow, pydantic, click, pytest.

**Spec:** `docs/superpowers/specs/2026-04-08-demo-mode-frame-filtering-design.md`

---

## File Structure

**Create:**
- `src/peeklet/core/demo_filter.py` — all demo-mode filtering logic (3 public functions + helpers)
- `tests/unit/test_demo_filter.py` — unit tests for `demo_filter.py`
- `tests/integration/test_demo_filter_pipeline.py` — end-to-end pipeline test with synthetic video

**Modify:**
- `src/peeklet/utils/types.py` — add `ContentSegment` dataclass and 2 fields on `FrameResult`
- `src/peeklet/config.py` — add `DemoFilterConfig` and wire into `PeekletConfig`
- `src/peeklet/core/video.py` — add `VideoDecoder.extract_frame_at()` and call `apply_demo_filter()` from `process_video()`
- `src/peeklet/cli.py` — add `--demo-mode` flag, validate activation rules, propagate to config
- `src/peeklet/core/audio.py` — no changes (read only)
- `pyproject.toml` — add `pytesseract` to the `video` optional-dependency group
- `tests/unit/test_cli.py` — add 3 CLI tests for `--demo-mode`
- `tests/unit/test_config.py` — add test for new `DemoFilterConfig` defaults

**No deletes.** All existing video behavior is preserved when `--demo-mode` is not set.

---

## Task 1: Add `pytesseract` dependency and runtime guard

**Files:**
- Modify: `pyproject.toml:36-40`

- [ ] **Step 1: Add pytesseract to the video extras**

Edit `pyproject.toml`. Replace:

```toml
video = [
    "imageio[ffmpeg]>=2.31",
    "av>=14.0",
    "pydub>=0.25",
]
```

with:

```toml
video = [
    "imageio[ffmpeg]>=2.31",
    "av>=14.0",
    "pydub>=0.25",
    "pytesseract>=0.3.10",
]
```

- [ ] **Step 2: Add the mypy override for pytesseract**

In the `[[tool.mypy.overrides]]` `module` list (around line 84), add `"pytesseract.*"`:

```toml
module = [
    "pyarrow.*",
    "imagehash.*",
    "skimage.*",
    "av.*",
    "imageio.*",
    "imageio_ffmpeg.*",
    "pydub.*",
    "pytesseract.*",
]
```

- [ ] **Step 3: Sync the lockfile**

Run: `uv sync --extra video --extra dev`
Expected: `pytesseract` appears in `uv.lock` with no errors.

- [ ] **Step 4: Verify import**

Run: `uv run python -c "import pytesseract; print(pytesseract.__version__)"`
Expected: prints a version string (e.g. `0.3.10`).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pytesseract to video extras for demo-mode OCR"
```

---

## Task 2: Add `ContentSegment` and new `FrameResult` fields

**Files:**
- Modify: `src/peeklet/utils/types.py`
- Test: `tests/unit/test_types.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_types.py`:

```python
def test_content_segment_basic():
    from peeklet.utils.types import ContentSegment

    seg = ContentSegment(start_sec=0.0, end_sec=30.0, is_demo=True, text_density=12.5)
    assert seg.start_sec == 0.0
    assert seg.end_sec == 30.0
    assert seg.is_demo is True
    assert seg.text_density == 12.5


def test_frame_result_demo_fields_default_none():
    from datetime import datetime  # noqa: F401  (imported for type completeness)

    from peeklet.utils.types import EventType, FrameResult

    r = FrameResult(
        frame_id="f",
        event_type=EventType.SKIPPED,
        is_keyframe=False,
        perceptual_hash="0" * 16,
        frame_width=10,
        frame_height=10,
    )
    assert r.content_type is None
    assert r.selection_reason is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_types.py::test_content_segment_basic tests/unit/test_types.py::test_frame_result_demo_fields_default_none -v`
Expected: FAIL with `ImportError: cannot import name 'ContentSegment'` and `AttributeError` for the `content_type` / `selection_reason` access.

- [ ] **Step 3: Implement**

Edit `src/peeklet/utils/types.py`. After the existing `Region` dataclass (around line 35), add:

```python
@dataclass(frozen=True, slots=True)
class ContentSegment:
    """A time range classified as demo screen-share or non-demo content."""

    start_sec: float
    end_sec: float
    is_demo: bool
    text_density: float  # average words detected per sampled frame in this range
```

In the `FrameResult` dataclass, after the existing `change_magnitude: str | None = None` field (line 82), add:

```python
    content_type: Literal["demo", "non-demo"] | None = None
    selection_reason: Literal["transcript_anchor", "major_change"] | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_types.py::test_content_segment_basic tests/unit/test_types.py::test_frame_result_demo_fields_default_none -v`
Expected: PASS.

- [ ] **Step 5: Run the full test file**

Run: `uv run pytest tests/unit/test_types.py -v`
Expected: all tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/utils/types.py tests/unit/test_types.py
git commit -m "types: add ContentSegment and demo-mode FrameResult fields"
```

---

## Task 3: Add `DemoFilterConfig` to config

**Files:**
- Modify: `src/peeklet/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config.py`:

```python
def test_demo_filter_config_defaults():
    from peeklet.config import PeekletConfig

    cfg = PeekletConfig()
    assert cfg.demo_filter.enabled is False
    assert cfg.demo_filter.ocr_sample_interval_sec == 30.0
    assert cfg.demo_filter.ocr_downscale_dim == 360
    assert cfg.demo_filter.ocr_min_words == 5
    assert cfg.demo_filter.major_change_ssim == 0.70
    assert cfg.demo_filter.major_change_blocks == 15
    assert cfg.demo_filter.visual_change_weight == 0.7
    assert cfg.demo_filter.text_density_weight == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py::test_demo_filter_config_defaults -v`
Expected: FAIL with `AttributeError: 'PeekletConfig' object has no attribute 'demo_filter'`.

- [ ] **Step 3: Implement**

Edit `src/peeklet/config.py`. After the `VideoConfig` class (around line 81), add:

```python
class DemoFilterConfig(BaseModel):
    """Demo-mode frame filtering settings.

    Activated via the CLI ``--demo-mode`` flag. When enabled, runs a sparse
    OCR sweep to classify content type and uses the transcript to anchor
    keyframe selection. See the design spec for full details.
    """

    enabled: bool = False
    ocr_sample_interval_sec: float = Field(default=30.0, gt=0.0)
    ocr_downscale_dim: int = Field(default=360, gt=0)
    ocr_min_words: int = Field(default=5, ge=0)
    major_change_ssim: float = Field(default=0.70, ge=0.0, le=1.0)
    major_change_blocks: int = Field(default=15, ge=0)
    visual_change_weight: float = Field(default=0.7, ge=0.0)
    text_density_weight: float = Field(default=0.3, ge=0.0)
```

In `PeekletConfig` (around line 92), add `demo_filter` to the field list:

```python
class PeekletConfig(BaseModel):
    """Root configuration for Peeklet."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    hasher: HasherConfig = Field(default_factory=HasherConfig)
    comparator: ComparatorConfig = Field(default_factory=ComparatorConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    demo_filter: DemoFilterConfig = Field(default_factory=DemoFilterConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py::test_demo_filter_config_defaults -v`
Expected: PASS.

- [ ] **Step 5: Run the full config test file**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/config.py tests/unit/test_config.py
git commit -m "config: add DemoFilterConfig with default tuning values"
```

---

## Task 4: Add `VideoDecoder.extract_frame_at()` helper

**Files:**
- Modify: `src/peeklet/core/video.py:172-202`
- Test: `tests/unit/test_video.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_video.py`:

```python
def test_extract_frame_at_returns_frame_near_timestamp(tmp_path):
    """extract_frame_at should return a frame near the requested timestamp."""
    import numpy as np

    from peeklet.core.video import VideoDecoder
    from tests.unit.helpers_video import write_synthetic_video

    # 5-second video, 10 fps, solid color frames that change every second
    video_path = tmp_path / "synthetic.mp4"
    write_synthetic_video(video_path, duration_sec=5, fps=10, width=64, height=64)

    decoder = VideoDecoder(video_path)
    frame, ts, frame_num = decoder.extract_frame_at(2.5)

    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3  # HxWxC RGB
    assert 2.0 <= ts <= 3.0  # within ±0.5s of request
    assert frame_num >= 0
```

- [ ] **Step 2: Create the synthetic video helper if missing**

Check whether `tests/unit/helpers_video.py` already exists with a `write_synthetic_video` function:

Run: `uv run python -c "from tests.unit.helpers_video import write_synthetic_video"`
Expected: either succeeds (helper exists, skip to Step 3) or fails with `ModuleNotFoundError` / `ImportError`.

If it does NOT exist, create `tests/unit/helpers_video.py`:

```python
"""Test helpers for generating tiny synthetic videos."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_synthetic_video(
    path: Path,
    duration_sec: int = 5,
    fps: int = 10,
    width: int = 64,
    height: int = 64,
) -> Path:
    """Write a tiny solid-color video that changes color every second.

    Used by demo-filter and video tests. Color cycles through R/G/B/Y/M
    so each second is visually distinct.
    """
    import imageio.v3 as iio

    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    frames = []
    for sec in range(duration_sec):
        color = palette[sec % len(palette)]
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        for _ in range(fps):
            frames.append(frame)
    iio.imwrite(path, np.stack(frames), plugin="pyav", fps=fps, codec="libx264")
    return path
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_video.py::test_extract_frame_at_returns_frame_near_timestamp -v`
Expected: FAIL with `AttributeError: 'VideoDecoder' object has no attribute 'extract_frame_at'`.

- [ ] **Step 4: Implement**

Edit `src/peeklet/core/video.py`. After the `extract_frame_range` method (line 202), add:

```python
    def extract_frame_at(
        self, timestamp_sec: float
    ) -> tuple[np.ndarray, float, int]:
        """Extract a single frame at (or just after) ``timestamp_sec``.

        Seeks to the nearest keyframe before the requested timestamp via
        libav and decodes forward until a frame at-or-past the target is
        found. Returns ``(frame_rgb, actual_timestamp, frame_number)``.

        Used by demo-mode's sparse OCR sweep where we only need a few
        frames spread across the video, not a continuous range.

        Raises:
            RuntimeError: if no frame can be decoded at or after the
                requested timestamp (e.g., timestamp past end of video).
        """
        import av

        container = av.open(str(self._path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            target_ts = int(timestamp_sec * av.time_base)
            container.seek(target_ts)

            for frame in container.decode(stream):
                if frame.pts is None or stream.time_base is None:
                    continue
                ts = float(frame.pts * stream.time_base)
                if ts + 1e-6 < timestamp_sec:
                    continue
                arr = frame.to_ndarray(format="rgb24")
                rgb = ensure_rgb_uint8(np.asarray(arr))
                frame_num = int(round(ts * self._fps))
                return rgb, ts, frame_num

            raise RuntimeError(
                f"No frame found at or after timestamp {timestamp_sec}s "
                f"in {self._path.name} (duration={self._duration}s)"
            )
        finally:
            container.close()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_video.py::test_extract_frame_at_returns_frame_near_timestamp -v`
Expected: PASS.

- [ ] **Step 6: Run the full video test file**

Run: `uv run pytest tests/unit/test_video.py -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/core/video.py tests/unit/test_video.py tests/unit/helpers_video.py
git commit -m "video: add VideoDecoder.extract_frame_at for sparse seeking"
```

---

## Task 5: Implement `_count_words_in_frame()` (OCR text-presence helper)

**Files:**
- Create: `src/peeklet/core/demo_filter.py`
- Test: `tests/unit/test_demo_filter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_demo_filter.py`:

```python
"""Unit tests for demo_filter.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_frame(h: int = 360, w: int = 360) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_count_words_filters_low_confidence_and_short_tokens():
    from peeklet.core.demo_filter import _count_words_in_frame

    fake_data = {
        "text": ["Settings", "x", "Save", "", "Login", "."],
        "conf": ["95", "92", "80", "-1", "20", "99"],
    }
    with patch("peeklet.core.demo_filter.pytesseract") as mock_pt:
        mock_pt.image_to_data.return_value = fake_data
        mock_pt.Output.DICT = "dict"
        count = _count_words_in_frame(_make_frame(), downscale_dim=360)

    # "Settings" (95, len 8) ✓
    # "x" (92, len 1) ✗ short
    # "Save" (80, len 4) ✓
    # ""  (-1, empty) ✗
    # "Login" (20, len 5) ✗ low conf
    # "." (99, len 1) ✗ short
    assert count == 2


def test_count_words_handles_tesseract_exception():
    from peeklet.core.demo_filter import _count_words_in_frame

    with patch("peeklet.core.demo_filter.pytesseract") as mock_pt:
        mock_pt.image_to_data.side_effect = RuntimeError("tesseract crashed")
        mock_pt.Output.DICT = "dict"
        count = _count_words_in_frame(_make_frame(), downscale_dim=360)

    assert count == 0  # graceful fallback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_demo_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peeklet.core.demo_filter'`.

- [ ] **Step 3: Create the module skeleton + helper**

Create `src/peeklet/core/demo_filter.py`:

```python
"""Demo-mode frame filtering: sparse OCR sweep + transcript-anchored selection.

Activated via the CLI ``--demo-mode`` flag. See the design spec at
``docs/superpowers/specs/2026-04-08-demo-mode-frame-filtering-design.md``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.video import VideoDecoder
    from peeklet.utils.types import ContentSegment, FrameResult

logger = logging.getLogger(__name__)

# Imported lazily so test files can patch ``demo_filter.pytesseract``.
try:
    import pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised in install-error path
    pytesseract = None  # type: ignore[assignment]

_MIN_WORD_LENGTH = 2
_MIN_WORD_CONFIDENCE = 30


def _downscale_for_ocr(frame: np.ndarray, downscale_dim: int) -> np.ndarray:
    """Resize ``frame`` so its longest edge equals ``downscale_dim``.

    Returns the original frame if it is already smaller. Uses Pillow for
    a high-quality resize without pulling in extra dependencies.
    """
    from PIL import Image

    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= downscale_dim:
        return frame
    scale = downscale_dim / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(frame).resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(img)


def _count_words_in_frame(frame: np.ndarray, downscale_dim: int) -> int:
    """Run Tesseract and return the number of words above the noise floor.

    Words are counted only if they have at least ``_MIN_WORD_CONFIDENCE``
    confidence and at least ``_MIN_WORD_LENGTH`` characters. Returns 0 on
    any Tesseract error so the caller can treat the frame as non-demo.
    """
    if pytesseract is None:
        return 0

    downscaled = _downscale_for_ocr(frame, downscale_dim)
    try:
        data = pytesseract.image_to_data(downscaled, output_type=pytesseract.Output.DICT)
    except Exception as exc:  # pragma: no cover - mocked in tests
        logger.warning("OCR failed on frame: %s", exc)
        return 0

    texts = data.get("text", [])
    confs = data.get("conf", [])
    count = 0
    for text, conf in zip(texts, confs, strict=False):
        if not text or len(text.strip()) < _MIN_WORD_LENGTH:
            continue
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_val < _MIN_WORD_CONFIDENCE:
            continue
        count += 1
    return count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_demo_filter.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py
git commit -m "demo_filter: add OCR-based word count helper"
```

---

## Task 6: Implement `classify_content_segments()` (Stage 1)

**Files:**
- Modify: `src/peeklet/core/demo_filter.py`
- Test: `tests/unit/test_demo_filter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_demo_filter.py`:

```python
def _make_decoder(duration: float):
    decoder = MagicMock()
    decoder.get_metadata.return_value = MagicMock(duration=duration)
    decoder.extract_frame_at = MagicMock()
    return decoder


def test_classify_content_segments_smooths_isolated_sample():
    """A single non-demo sample sandwiched between demo samples is flipped."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import classify_content_segments

    cfg = DemoFilterConfig(ocr_sample_interval_sec=10.0, ocr_min_words=5)
    decoder = _make_decoder(duration=40.0)

    # 5 samples at t=0,10,20,30,40 with word counts: 10, 12, 0, 11, 13
    decoder.extract_frame_at.side_effect = [
        (np.zeros((10, 10, 3), dtype=np.uint8), float(t), int(t * 10))
        for t in [0, 10, 20, 30, 40]
    ]
    word_counts = iter([10, 12, 0, 11, 13])

    with patch(
        "peeklet.core.demo_filter._count_words_in_frame",
        side_effect=lambda *_a, **_kw: next(word_counts),
    ):
        segments = classify_content_segments(decoder, cfg)

    # All 5 samples should be smoothed to a single demo segment.
    assert len(segments) == 1
    assert segments[0].is_demo is True
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec >= 40.0


def test_classify_content_segments_finds_real_transition():
    """First half is non-demo (gallery), second half is demo (screen share)."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import classify_content_segments

    cfg = DemoFilterConfig(ocr_sample_interval_sec=10.0, ocr_min_words=5)
    decoder = _make_decoder(duration=60.0)
    decoder.extract_frame_at.side_effect = [
        (np.zeros((10, 10, 3), dtype=np.uint8), float(t), int(t * 10))
        for t in [0, 10, 20, 30, 40, 50, 60]
    ]
    # 0,10,20 = gallery (0 words); 30,40,50,60 = demo (15 words each)
    word_counts = iter([0, 0, 0, 15, 15, 15, 15])

    with patch(
        "peeklet.core.demo_filter._count_words_in_frame",
        side_effect=lambda *_a, **_kw: next(word_counts),
    ):
        segments = classify_content_segments(decoder, cfg)

    assert len(segments) == 2
    assert segments[0].is_demo is False
    assert segments[1].is_demo is True
    # Boundary should be midpoint between sample t=20 and sample t=30 → 25.0
    assert abs(segments[1].start_sec - 25.0) < 0.01


def test_classify_content_segments_short_video_uses_three_samples():
    """Videos shorter than the sample interval get exactly 3 samples."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import classify_content_segments

    cfg = DemoFilterConfig(ocr_sample_interval_sec=30.0, ocr_min_words=5)
    decoder = _make_decoder(duration=10.0)
    decoder.extract_frame_at.side_effect = [
        (np.zeros((10, 10, 3), dtype=np.uint8), float(t), int(t * 10))
        for t in [0.0, 5.0, 10.0]
    ]
    with patch(
        "peeklet.core.demo_filter._count_words_in_frame",
        return_value=20,
    ):
        segments = classify_content_segments(decoder, cfg)

    assert decoder.extract_frame_at.call_count == 3
    assert all(s.is_demo for s in segments)


def test_classify_content_segments_all_ocr_fail_raises():
    """If every sample yields 0 words AND tesseract is unavailable, raise."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import classify_content_segments

    cfg = DemoFilterConfig(ocr_sample_interval_sec=10.0)
    decoder = _make_decoder(duration=20.0)
    decoder.extract_frame_at.side_effect = [
        (np.zeros((10, 10, 3), dtype=np.uint8), float(t), int(t * 10))
        for t in [0, 10, 20]
    ]
    with (
        patch("peeklet.core.demo_filter.pytesseract", None),
        pytest.raises(RuntimeError, match="OCR pipeline failed entirely"),
    ):
        classify_content_segments(decoder, cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_demo_filter.py -v`
Expected: 4 new tests FAIL with `ImportError: cannot import name 'classify_content_segments'`.

- [ ] **Step 3: Implement**

Append to `src/peeklet/core/demo_filter.py`:

```python
def _generate_sample_timestamps(
    duration: float, interval: float
) -> list[float]:
    """Pick the timestamps to OCR-sample.

    Long videos: one sample every ``interval`` seconds, plus a final
    sample at the end. Short videos (< ``interval``): three samples
    evenly spaced (start, middle, end).
    """
    if duration <= 0:
        return [0.0]
    if duration < interval:
        return [0.0, duration / 2.0, duration]
    samples = [float(i) for i in range(0, int(duration), int(interval))]
    if samples[-1] < duration:
        samples.append(float(duration))
    return samples


def _smooth_isolated_flips(flags: list[bool]) -> list[bool]:
    """Flip any single sample that disagrees with both neighbors.

    e.g. [True, True, False, True, True] -> [True, True, True, True, True].
    Endpoints are left untouched.
    """
    if len(flags) < 3:
        return list(flags)
    smoothed = list(flags)
    for i in range(1, len(flags) - 1):
        if flags[i - 1] == flags[i + 1] and flags[i] != flags[i - 1]:
            smoothed[i] = flags[i - 1]
    return smoothed


def classify_content_segments(
    decoder: VideoDecoder,
    config: DemoFilterConfig,
) -> list[ContentSegment]:
    """Stage 1: sparse OCR sweep across the video.

    Samples one frame every ``config.ocr_sample_interval_sec`` seconds,
    runs Tesseract on a downscaled copy, and groups consecutive samples
    of the same classification into ``ContentSegment`` ranges. Boundaries
    are placed at the midpoint between adjacent samples.

    Raises:
        RuntimeError: if OCR is completely unusable (no tesseract binary
            and zero word counts across all samples).
    """
    from peeklet.utils.types import ContentSegment

    meta = decoder.get_metadata()
    timestamps = _generate_sample_timestamps(meta.duration, config.ocr_sample_interval_sec)

    sample_word_counts: list[tuple[float, int]] = []
    for ts in timestamps:
        try:
            frame, actual_ts, _ = decoder.extract_frame_at(ts)
        except Exception as exc:
            logger.warning("Frame extraction failed at t=%.2fs: %s", ts, exc)
            continue
        words = _count_words_in_frame(frame, config.ocr_downscale_dim)
        sample_word_counts.append((actual_ts, words))

    if not sample_word_counts:
        raise RuntimeError(
            "OCR pipeline failed entirely — could not extract or analyze "
            "any sample frames. Check tesseract installation."
        )

    # If every sample is 0 words AND tesseract is unavailable, fail loudly.
    if pytesseract is None and all(w == 0 for _, w in sample_word_counts):
        raise RuntimeError(
            "OCR pipeline failed entirely — pytesseract is not installed. "
            "Install with: pip install peeklet[video]"
        )

    flags = [w >= config.ocr_min_words for _, w in sample_word_counts]
    smoothed = _smooth_isolated_flips(flags)

    # Walk samples and merge consecutive same-flag runs into segments.
    segments: list[ContentSegment] = []
    run_start_idx = 0
    for i in range(1, len(sample_word_counts) + 1):
        if i == len(sample_word_counts) or smoothed[i] != smoothed[run_start_idx]:
            run_first_ts = sample_word_counts[run_start_idx][0]
            run_last_ts = sample_word_counts[i - 1][0]

            # Boundary alignment: each segment starts at the midpoint
            # between this run's first sample and the previous sample.
            if run_start_idx == 0:
                start_sec = 0.0
            else:
                prev_sample_ts = sample_word_counts[run_start_idx - 1][0]
                start_sec = (prev_sample_ts + run_first_ts) / 2.0

            if i == len(sample_word_counts):
                end_sec = max(run_last_ts, meta.duration)
            else:
                next_sample_ts = sample_word_counts[i][0]
                end_sec = (run_last_ts + next_sample_ts) / 2.0

            run_words = [w for _, w in sample_word_counts[run_start_idx:i]]
            avg_density = sum(run_words) / len(run_words) if run_words else 0.0

            segments.append(
                ContentSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    is_demo=smoothed[run_start_idx],
                    text_density=avg_density,
                )
            )
            run_start_idx = i

    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_demo_filter.py -v`
Expected: all 6 tests in the file PASS.

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py
git commit -m "demo_filter: implement Stage 1 sparse OCR content classification"
```

---

## Task 7: Implement `select_demo_keyframes()` (Stages 2 + 3)

**Files:**
- Modify: `src/peeklet/core/demo_filter.py`
- Test: `tests/unit/test_demo_filter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_demo_filter.py`:

```python
def _make_keyframe(
    *, ts: float, magnitude: str = "moderate", ssim: float = 0.85, blocks: int = 5
):
    from peeklet.utils.types import EventType, FrameResult, Region

    return FrameResult(
        frame_id=f"f_{int(ts * 1000)}",
        event_type=EventType.KEYFRAME,
        is_keyframe=True,
        perceptual_hash="0" * 16,
        frame_width=720,
        frame_height=480,
        video_timestamp=ts,
        change_magnitude=magnitude,
        ssim_score=ssim,
        changed_regions=[Region(0, 0, 1, 1)] * blocks,
    )


def test_select_demo_keyframes_drops_non_demo_segments():
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_demo_keyframes
    from peeklet.utils.types import ContentSegment

    segments = [
        ContentSegment(0.0, 30.0, is_demo=False, text_density=0.0),
        ContentSegment(30.0, 60.0, is_demo=True, text_density=20.0),
    ]
    keyframes = [
        _make_keyframe(ts=10.0),  # in non-demo, dropped
        _make_keyframe(ts=20.0),  # in non-demo, dropped
        _make_keyframe(ts=40.0),  # in demo, considered
    ]
    transcript = [TranscriptSegment(start=35.0, end=45.0, text="hello")]
    cfg = DemoFilterConfig()

    out = select_demo_keyframes(keyframes, segments, transcript, cfg)
    assert len(out) == 1
    assert out[0].video_timestamp == 40.0
    assert out[0].content_type == "demo"
    assert out[0].selection_reason == "transcript_anchor"


def test_select_demo_keyframes_picks_highest_score_per_transcript_segment():
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_demo_keyframes
    from peeklet.utils.types import ContentSegment

    segments = [ContentSegment(0.0, 60.0, is_demo=True, text_density=10.0)]
    keyframes = [
        _make_keyframe(ts=12.0, magnitude="minor"),
        _make_keyframe(ts=14.0, magnitude="moderate"),
        _make_keyframe(ts=16.0, magnitude="minor"),
    ]
    transcript = [TranscriptSegment(start=10.0, end=20.0, text="opening settings")]
    cfg = DemoFilterConfig()

    out = select_demo_keyframes(keyframes, segments, transcript, cfg)
    # The "moderate" frame at 14.0s should win over the two "minor" frames.
    assert len(out) == 1
    assert out[0].video_timestamp == 14.0
    assert out[0].selection_reason == "transcript_anchor"


def test_select_demo_keyframes_major_change_floor_keeps_second_major():
    """Two major frames in one transcript segment: anchor + floor both survive."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_demo_keyframes
    from peeklet.utils.types import ContentSegment

    segments = [ContentSegment(0.0, 60.0, is_demo=True, text_density=10.0)]
    keyframes = [
        # Two majors inside the same transcript window.
        _make_keyframe(ts=12.0, magnitude="major", ssim=0.50, blocks=25),
        # Mid-sentence modal: also a major change.
        _make_keyframe(ts=16.0, magnitude="major", ssim=0.40, blocks=30),
        _make_keyframe(ts=18.0, magnitude="minor", ssim=0.95, blocks=2),
    ]
    transcript = [TranscriptSegment(start=10.0, end=20.0, text="and now click here")]
    cfg = DemoFilterConfig()

    out = select_demo_keyframes(keyframes, segments, transcript, cfg)
    by_ts = {r.video_timestamp: r.selection_reason for r in out}

    # Both majors must survive; the minor is dropped.
    assert 12.0 in by_ts
    assert 16.0 in by_ts
    assert 18.0 not in by_ts
    # Exactly one of the two majors is the anchor; the other is major_change.
    reasons = sorted(by_ts.values())
    assert reasons == ["major_change", "transcript_anchor"]


def test_select_demo_keyframes_anchor_only_when_no_other_majors():
    """A single major frame is the anchor, no extras from the floor."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_demo_keyframes
    from peeklet.utils.types import ContentSegment

    segments = [ContentSegment(0.0, 60.0, is_demo=True, text_density=10.0)]
    keyframes = [
        _make_keyframe(ts=12.0, magnitude="moderate", ssim=0.85, blocks=5),
        _make_keyframe(ts=15.0, magnitude="major", ssim=0.50, blocks=25),
        _make_keyframe(ts=18.0, magnitude="minor", ssim=0.95, blocks=2),
    ]
    transcript = [TranscriptSegment(start=10.0, end=20.0, text="and now click here")]
    cfg = DemoFilterConfig()

    out = select_demo_keyframes(keyframes, segments, transcript, cfg)
    assert len(out) == 1
    assert out[0].video_timestamp == 15.0
    assert out[0].selection_reason == "transcript_anchor"


def test_select_demo_keyframes_drops_silence_segments():
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import select_demo_keyframes
    from peeklet.utils.types import ContentSegment

    segments = [ContentSegment(0.0, 60.0, is_demo=True, text_density=10.0)]
    keyframes = [
        _make_keyframe(ts=10.0, magnitude="moderate"),
        _make_keyframe(ts=20.0, magnitude="moderate"),
    ]
    cfg = DemoFilterConfig()

    out = select_demo_keyframes(keyframes, segments, transcript=[], config=cfg)
    # No transcript segments → no keyframes selected (silence drops all).
    assert out == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_demo_filter.py -v -k "select_demo_keyframes"`
Expected: 5 tests FAIL with `ImportError: cannot import name 'select_demo_keyframes'`.

- [ ] **Step 3: Implement**

Append to `src/peeklet/core/demo_filter.py`:

```python
_CHANGE_MAGNITUDE_SCORE: dict[str | None, float] = {
    "major": 1.0,
    "moderate": 0.6,
    "minor": 0.2,
    None: 0.0,
}


def _segment_for_timestamp(
    ts: float, segments: list[ContentSegment]
) -> ContentSegment | None:
    """Return the ContentSegment containing ``ts``, or None."""
    for seg in segments:
        if seg.start_sec <= ts < seg.end_sec:
            return seg
    # Final-frame edge case: include ts equal to last segment's end.
    if segments and ts == segments[-1].end_sec:
        return segments[-1]
    return None


def _is_major_change(
    keyframe: FrameResult, config: DemoFilterConfig
) -> bool:
    """Apply the major-change floor rule from the spec."""
    if keyframe.ssim_score is not None and keyframe.ssim_score < config.major_change_ssim:
        return True
    n_blocks = len(keyframe.changed_regions) if keyframe.changed_regions else 0
    return n_blocks > config.major_change_blocks


def _score_keyframe(
    keyframe: FrameResult,
    segment_text_density: float,
    max_text_density: float,
    config: DemoFilterConfig,
) -> float:
    """Composite relevance score: visual change weight + text density weight."""
    change_score = _CHANGE_MAGNITUDE_SCORE.get(keyframe.change_magnitude, 0.0)
    if max_text_density > 0:
        normalized_density = segment_text_density / max_text_density
    else:
        normalized_density = 0.0
    return (
        config.visual_change_weight * change_score
        + config.text_density_weight * normalized_density
    )


def select_demo_keyframes(
    keyframes: list[FrameResult],
    segments: list[ContentSegment],
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
) -> list[FrameResult]:
    """Stages 2 & 3: drop non-demo keyframes, then transcript-anchored selection.

    See the design spec for the full algorithm. In brief:
    1. Drop any keyframe whose timestamp falls in a non-demo segment.
    2. For each transcript segment, pick the highest-scoring candidate.
    3. Additionally keep any candidate that meets the major-change floor.
    4. Empty transcript (silence) drops all candidates from that range.
    """
    # --- Stage 2: drop non-demo keyframes ---
    demo_keyframes: list[FrameResult] = []
    for kf in keyframes:
        if kf.video_timestamp is None:
            continue
        seg = _segment_for_timestamp(kf.video_timestamp, segments)
        if seg is None or not seg.is_demo:
            continue
        kf.content_type = "demo"
        demo_keyframes.append(kf)

    if not transcript:
        return []

    max_text_density = max((s.text_density for s in segments if s.is_demo), default=0.0)

    # --- Stage 3: transcript-anchored selection + major-change floor ---
    selected_by_id: dict[str, FrameResult] = {}

    for tseg in transcript:
        candidates = [
            kf
            for kf in demo_keyframes
            if kf.video_timestamp is not None
            and tseg.start <= kf.video_timestamp <= tseg.end
        ]
        if not candidates:
            continue

        # Best-scoring candidate becomes the transcript anchor.
        scored = []
        for kf in candidates:
            assert kf.video_timestamp is not None
            seg = _segment_for_timestamp(kf.video_timestamp, segments)
            seg_density = seg.text_density if seg is not None else 0.0
            score = _score_keyframe(kf, seg_density, max_text_density, config)
            scored.append((score, kf))
        scored.sort(key=lambda x: x[0], reverse=True)
        anchor = scored[0][1]
        anchor.selection_reason = "transcript_anchor"
        selected_by_id[anchor.frame_id] = anchor

        # Major-change floor: any other candidate that qualifies survives.
        for kf in candidates:
            if kf.frame_id == anchor.frame_id:
                continue
            if _is_major_change(kf, config):
                kf.selection_reason = "major_change"
                selected_by_id.setdefault(kf.frame_id, kf)

    # Preserve original keyframe order.
    selected_ids = set(selected_by_id.keys())
    return [kf for kf in demo_keyframes if kf.frame_id in selected_ids]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_demo_filter.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py
git commit -m "demo_filter: implement Stages 2-3 transcript-anchored selection"
```

---

## Task 8: Implement `apply_demo_filter()` entry point with tesseract guard

**Files:**
- Modify: `src/peeklet/core/demo_filter.py`
- Test: `tests/unit/test_demo_filter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_demo_filter.py`:

```python
def test_apply_demo_filter_raises_when_tesseract_missing():
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = MagicMock()
    cfg = DemoFilterConfig()

    with (
        patch("peeklet.core.demo_filter.pytesseract", None),
        pytest.raises(RuntimeError, match="tesseract binary"),
    ):
        apply_demo_filter(decoder, keyframes=[], transcript=[], config=cfg)


def test_apply_demo_filter_raises_when_tesseract_version_check_fails():
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = MagicMock()
    cfg = DemoFilterConfig()

    with patch("peeklet.core.demo_filter.pytesseract") as mock_pt:
        mock_pt.get_tesseract_version.side_effect = Exception("not found")
        with pytest.raises(RuntimeError, match="tesseract binary"):
            apply_demo_filter(decoder, keyframes=[], transcript=[], config=cfg)


def test_apply_demo_filter_calls_classify_then_select():
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import apply_demo_filter
    from peeklet.utils.types import ContentSegment

    decoder = MagicMock()
    cfg = DemoFilterConfig()
    fake_segments = [ContentSegment(0.0, 10.0, is_demo=True, text_density=15.0)]
    fake_keyframes = [_make_keyframe(ts=5.0, magnitude="moderate")]
    fake_transcript = [TranscriptSegment(start=4.0, end=6.0, text="hi")]

    with (
        patch("peeklet.core.demo_filter.pytesseract") as mock_pt,
        patch(
            "peeklet.core.demo_filter.classify_content_segments",
            return_value=fake_segments,
        ) as mock_classify,
        patch(
            "peeklet.core.demo_filter.select_demo_keyframes",
            return_value=fake_keyframes,
        ) as mock_select,
    ):
        mock_pt.get_tesseract_version.return_value = "5.3.0"
        result = apply_demo_filter(decoder, fake_keyframes, fake_transcript, cfg)

    mock_classify.assert_called_once_with(decoder, cfg)
    mock_select.assert_called_once_with(fake_keyframes, fake_segments, fake_transcript, cfg)
    assert result == fake_keyframes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_demo_filter.py -v -k "apply_demo_filter"`
Expected: 3 tests FAIL with `ImportError: cannot import name 'apply_demo_filter'`.

- [ ] **Step 3: Implement**

Append to `src/peeklet/core/demo_filter.py`:

```python
_TESSERACT_INSTALL_HINT = (
    "--demo-mode requires the tesseract binary. "
    "Install with: brew install tesseract (macOS) or "
    "apt install tesseract-ocr (Linux)."
)


def _verify_tesseract() -> None:
    """Raise a clear error if Tesseract is unusable."""
    if pytesseract is None:
        raise RuntimeError(_TESSERACT_INSTALL_HINT)
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(_TESSERACT_INSTALL_HINT) from exc


def apply_demo_filter(
    decoder: VideoDecoder,
    keyframes: list[FrameResult],
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
) -> list[FrameResult]:
    """Top-level demo-mode entry point.

    1. Verifies tesseract is installed and usable.
    2. Runs Stage 1 (sparse OCR sweep) to classify content type ranges.
    3. Runs Stages 2 & 3 (drop non-demo + transcript-anchored selection).
    4. Returns the filtered keyframe list.
    """
    _verify_tesseract()
    segments = classify_content_segments(decoder, config)
    logger.info(
        "demo_filter: classified %d content segments (%d demo, %d non-demo)",
        len(segments),
        sum(1 for s in segments if s.is_demo),
        sum(1 for s in segments if not s.is_demo),
    )
    selected = select_demo_keyframes(keyframes, segments, transcript, config)
    logger.info(
        "demo_filter: %d/%d keyframes survived filtering",
        len(selected),
        len(keyframes),
    )
    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_demo_filter.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py
git commit -m "demo_filter: add apply_demo_filter entry point with tesseract guard"
```

---

## Task 9: Wire `apply_demo_filter()` into `process_video()`

**Files:**
- Modify: `src/peeklet/core/video.py:237-399`
- Test: `tests/unit/test_video.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_video.py`:

```python
def test_process_video_calls_demo_filter_when_enabled(tmp_path, monkeypatch):
    """When config.demo_filter.enabled is True, process_video calls apply_demo_filter."""
    from unittest.mock import MagicMock

    from peeklet.config import PeekletConfig
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "synthetic.mp4"
    write_synthetic_video(video_path, duration_sec=3, fps=10, width=64, height=64)

    transcript_path = tmp_path / "transcript.srt"
    transcript_path.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nhello world\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\ngoodbye\n"
    )

    cfg = PeekletConfig()
    cfg.exporter.output_dir = str(tmp_path / "out")
    cfg.video.audio_detection = False
    cfg.video.transcript_path = str(transcript_path)
    cfg.demo_filter.enabled = True

    fake_filtered = []  # demo filter drops everything
    mock_apply = MagicMock(return_value=fake_filtered)
    monkeypatch.setattr(video_module, "apply_demo_filter", mock_apply)

    video_module.process_video(video_path, cfg)

    assert mock_apply.called, "apply_demo_filter should be invoked when demo_filter.enabled"


def test_process_video_skips_demo_filter_when_disabled(tmp_path, monkeypatch):
    """When demo_filter.enabled is False, apply_demo_filter is NOT called."""
    from unittest.mock import MagicMock

    from peeklet.config import PeekletConfig
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "synthetic.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)

    cfg = PeekletConfig()
    cfg.exporter.output_dir = str(tmp_path / "out")
    cfg.video.audio_detection = False
    cfg.demo_filter.enabled = False

    mock_apply = MagicMock()
    monkeypatch.setattr(video_module, "apply_demo_filter", mock_apply)

    video_module.process_video(video_path, cfg)

    assert not mock_apply.called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_video.py::test_process_video_calls_demo_filter_when_enabled -v`
Expected: FAIL — either an `AttributeError` (no `apply_demo_filter` in `video_module`) or the mock is never called.

- [ ] **Step 3: Implement — add the imports**

Edit `src/peeklet/core/video.py`. Add `import logging` to the stdlib import block (after `from pathlib import Path`, around line 6):

```python
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
```

Then after the existing local imports (around line 26, after `from peeklet.utils.types import EventType`), add:

```python
from peeklet.core.demo_filter import apply_demo_filter
```

Then immediately below the imports, before the `if TYPE_CHECKING:` block, add the module-level logger:

```python
logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Implement — call after the keyframe enrichment loop**

In `process_video()` (around line 357, just before the `# --- Audio enrichment ---` comment), insert the demo filter step. The insertion replaces the section between the `for r in results: ... r.total_keyframes = keyframe_count` block and the audio enrichment block. Locate this code:

```python
    # Backfill total_keyframes on all keyframe results
    for r in results:
        if r.is_keyframe:
            r.total_keyframes = keyframe_count

    # --- Audio enrichment ---
```

and replace it with:

```python
    # Backfill total_keyframes on all keyframe results
    for r in results:
        if r.is_keyframe:
            r.total_keyframes = keyframe_count

    # --- Demo-mode filter (opt-in) ---
    # When --demo-mode is set, replace the keyframes in `results` with the
    # filtered subset. Non-keyframe rows in `results` are preserved so the
    # manifest still records the full coarse stream.
    if config.demo_filter.enabled:
        keyframe_results = [r for r in results if r.is_keyframe]
        non_keyframe_results = [r for r in results if not r.is_keyframe]

        transcript_for_filter: list[TranscriptSegment] = []
        if config.video.transcript_path:
            transcript_for_filter = parse_transcript(Path(config.video.transcript_path))

        filtered_keyframes = apply_demo_filter(
            decoder,
            keyframe_results,
            transcript_for_filter,
            config.demo_filter,
        )

        # Demote any keyframe that the filter dropped: clear is_keyframe and
        # event_type so the manifest still has the row but it no longer
        # contributes to the keyframe set.
        kept_ids = {r.frame_id for r in filtered_keyframes}
        for r in keyframe_results:
            if r.frame_id not in kept_ids:
                r.is_keyframe = False
                r.event_type = EventType.SKIPPED

        # Recompute total_keyframes on the survivors.
        for i, kf in enumerate(filtered_keyframes, start=1):
            kf.keyframe_index = i
            kf.total_keyframes = len(filtered_keyframes)

        results = non_keyframe_results + keyframe_results
        results.sort(key=lambda r: (r.video_timestamp or 0.0))

        if not filtered_keyframes:
            logger.warning(
                "Demo filter dropped all keyframes — video may not contain "
                "screen-share content"
            )

    # --- Audio enrichment ---
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_video.py::test_process_video_calls_demo_filter_when_enabled tests/unit/test_video.py::test_process_video_skips_demo_filter_when_disabled -v`
Expected: PASS.

- [ ] **Step 6: Run the full video test file**

Run: `uv run pytest tests/unit/test_video.py -v`
Expected: all tests pass (no regressions in existing video tests).

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/core/video.py tests/unit/test_video.py
git commit -m "video: invoke demo filter from process_video when enabled"
```

---

## Task 10: Add `--demo-mode` CLI flag with validation

**Files:**
- Modify: `src/peeklet/cli.py:46-110`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_cli.py`:

```python
def test_cli_demo_mode_requires_transcript(tmp_path):
    """--demo-mode without --transcript exits with a clear error."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--input", str(video_path), "--output", str(tmp_path / "out"), "--demo-mode"],
    )
    assert result.exit_code == 2
    assert "--demo-mode requires --transcript" in result.output


def test_cli_demo_mode_requires_video(tmp_path):
    """--demo-mode without a video input exits with a clear error."""
    from click.testing import CliRunner

    from peeklet.cli import main

    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    (images_dir / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    transcript = tmp_path / "t.srt"
    transcript.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(images_dir),
            "--output",
            str(tmp_path / "out"),
            "--mode",
            "image",
            "--transcript",
            str(transcript),
            "--demo-mode",
        ],
    )
    assert result.exit_code == 2
    assert "--demo-mode only applies to video inputs" in result.output


def test_cli_demo_mode_sets_config_flag(tmp_path, monkeypatch):
    """--demo-mode happy path sets config.demo_filter.enabled."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)
    transcript = tmp_path / "t.srt"
    transcript.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")

    captured = {}

    real_process_video = video_module.process_video

    def spy_process_video(path, config, **kwargs):
        captured["enabled"] = config.demo_filter.enabled
        return real_process_video(path, config, **kwargs)

    monkeypatch.setattr(video_module, "process_video", spy_process_video)

    # Avoid actual tesseract dependency in this CLI smoke test by stubbing.
    from peeklet.core import demo_filter as df_module

    monkeypatch.setattr(df_module, "apply_demo_filter", lambda *_a, **_kw: [])

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(video_path),
            "--output",
            str(tmp_path / "out"),
            "--no-audio",
            "--transcript",
            str(transcript),
            "--demo-mode",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("enabled") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli.py -v -k "demo_mode"`
Expected: 3 tests FAIL with `Error: No such option: --demo-mode`.

- [ ] **Step 3: Implement — add the click option**

Edit `src/peeklet/cli.py`. After the existing `--transcript` click option (around line 81), add:

```python
@click.option(
    "--demo-mode",
    "demo_mode",
    is_flag=True,
    default=False,
    help="Filter video keyframes for demo content using OCR + transcript anchoring. "
    "Requires --transcript and a video input.",
)
```

- [ ] **Step 4: Implement — wire into `main()` signature**

Update the `main()` function signature (around line 83) to accept the new parameter:

```python
def main(
    input_path: Path,
    output_dir: Path,
    config_path: Path | None,
    no_audio: bool,
    mode: str | None,
    transcript_path: Path | None,
    demo_mode: bool,
) -> None:
```

- [ ] **Step 5: Implement — validate activation rules**

In `main()`, after the existing `config.exporter.output_dir = str(output_dir)` line (around line 101) and BEFORE `image_extensions = ...`, add:

```python
    if demo_mode:
        if not transcript_path:
            raise click.UsageError(
                "--demo-mode requires --transcript. Demo mode needs both a "
                "video and a transcript to filter frames effectively."
            )
        config.demo_filter.enabled = True
```

Then update the mode-detection branch (after `detected_mode = _detect_mode(...)`, around line 105) to validate the video requirement:

```python
    image_extensions = {f".{fmt}" for fmt in config.input.supported_formats}
    detected_mode = _detect_mode(input_path, mode, image_extensions)

    if demo_mode and detected_mode != "video":
        raise click.UsageError("--demo-mode only applies to video inputs.")

    if detected_mode == "video":
        _run_video_mode(input_path, config)
    else:
        _run_image_mode(input_path, config, image_extensions)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli.py -v -k "demo_mode"`
Expected: all 3 demo-mode tests PASS.

- [ ] **Step 7: Run the full CLI test file**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/peeklet/cli.py tests/unit/test_cli.py
git commit -m "cli: add --demo-mode flag with activation rule validation"
```

---

## Task 11: End-to-end integration test

**Files:**
- Create: `tests/integration/test_demo_filter_pipeline.py`

- [ ] **Step 1: Check tesseract availability and add a marker**

Run: `tesseract --version`
Expected: prints a version (e.g. `tesseract 5.3.x`). If it fails, install via `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux) before continuing.

Then add the marker registration to `pyproject.toml`. In `[tool.pytest.ini_options]` `markers` (around line 102), add a new entry:

```toml
markers = [
    "datasets: tests requiring downloaded datasets (deselect with '-m not datasets')",
    "mind2web: tests requiring Mind2Web dataset (deselect with '-m not mind2web')",
    "showui: tests requiring ShowUI dataset (deselect with '-m not showui')",
    "webui: tests requiring WebUI dataset (deselect with '-m not webui')",
    "benchmark: performance benchmark tests",
    "requires_tesseract: tests that need the tesseract binary installed",
]
```

- [ ] **Step 2: Write the integration test**

Create `tests/integration/test_demo_filter_pipeline.py`:

```python
"""End-to-end test for the --demo-mode video pipeline.

Generates a tiny synthetic video that mixes 'gallery view' frames (no text)
with 'demo' frames (rendered text), then runs the full pipeline and asserts
the output matches the demo-filter contract.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.requires_tesseract


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


@pytest.fixture
def synthetic_demo_video(tmp_path: Path) -> tuple[Path, Path]:
    """A 6-second video: 0-3s blank (gallery), 3-6s text (demo)."""
    import imageio.v3 as iio
    from PIL import Image, ImageDraw, ImageFont

    fps = 10
    width, height = 320, 240

    blank_frame = np.full((height, width, 3), 80, dtype=np.uint8)

    text_img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(text_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 30), "Settings Dashboard", fill=(0, 0, 0), font=font)
    draw.text((10, 70), "Save Cancel Login", fill=(0, 0, 0), font=font)
    draw.text((10, 110), "Username Password", fill=(0, 0, 0), font=font)
    text_frame = np.asarray(text_img)

    frames = []
    for _ in range(3 * fps):
        frames.append(blank_frame)
    for _ in range(3 * fps):
        frames.append(text_frame)

    video_path = tmp_path / "demo.mp4"
    iio.imwrite(video_path, np.stack(frames), plugin="pyav", fps=fps, codec="libx264")

    transcript_path = tmp_path / "demo.srt"
    transcript_path.write_text(
        "1\n00:00:00,500 --> 00:00:02,500\nhere is the gallery view\n\n"
        "2\n00:00:03,500 --> 00:00:05,500\nnow look at the settings dashboard\n",
        encoding="utf-8",
    )
    return video_path, transcript_path


@pytest.mark.skipif(not _tesseract_available(), reason="tesseract binary not installed")
def test_demo_mode_filters_gallery_keeps_text(
    synthetic_demo_video: tuple[Path, Path], tmp_path: Path
) -> None:
    from peeklet.config import PeekletConfig
    from peeklet.core.video import process_video

    video_path, transcript_path = synthetic_demo_video

    cfg = PeekletConfig()
    cfg.exporter.output_dir = str(tmp_path / "out")
    cfg.video.audio_detection = False
    cfg.video.transcript_path = str(transcript_path)
    cfg.demo_filter.enabled = True
    cfg.demo_filter.ocr_sample_interval_sec = 1.0  # finer for short test video
    cfg.demo_filter.ocr_min_words = 2

    results = process_video(video_path, cfg)
    keyframes = [r for r in results if r.is_keyframe]

    assert keyframes, "expected at least one keyframe to survive demo filter"
    for kf in keyframes:
        assert kf.content_type == "demo"
        assert kf.selection_reason in {"transcript_anchor", "major_change"}
        # Every surviving keyframe should fall in the second half (text region).
        assert kf.video_timestamp is not None
        assert kf.video_timestamp >= 2.5, (
            f"keyframe at {kf.video_timestamp}s should be in the demo (text) region"
        )
```

- [ ] **Step 3: Run the integration test**

Run: `uv run pytest tests/integration/test_demo_filter_pipeline.py -v`
Expected: PASS (or skipped with a clear "tesseract binary not installed" reason if Tesseract is missing).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_demo_filter_pipeline.py pyproject.toml
git commit -m "test: end-to-end demo-mode pipeline test with synthetic video"
```

---

## Task 12: Lint, format, and full test sweep

**Files:** all changed files

- [ ] **Step 1: Run ruff format check**

Run: `uv run ruff format --check src/ tests/`
Expected: PASS. If anything is unformatted, run `uv run ruff format src/ tests/` and re-run the check.

- [ ] **Step 2: Run ruff lint**

Run: `uv run ruff check src/ tests/`
Expected: no errors. Fix any reported issues inline.

- [ ] **Step 3: Run mypy**

Run: `uv run mypy src/peeklet`
Expected: no errors.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass (the integration test may be skipped if tesseract is unavailable, which is fine).

- [ ] **Step 5: Commit any format/lint fixes**

```bash
git status
# If there are any unstaged changes from formatting, stage and commit:
git add -u
git commit -m "style: ruff format and lint fixes for demo-mode"
```

---

## Task 13: Update README with demo-mode usage

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the existing README to find the right section**

Run: open `README.md` and locate the existing "Usage" or "Video processing" section that documents `--transcript`.

- [ ] **Step 2: Add a demo-mode subsection**

After the existing transcript usage example, add:

````markdown
### Demo mode (smart frame filtering)

For demo videos with narration, use `--demo-mode` to filter the keyframe set
down to ~100-200 high-signal frames using OCR + transcript anchoring:

```bash
peeklet --input demo.mp4 --transcript demo.srt --demo-mode --output ./out
```

`--demo-mode` requires:
- A video input (single file)
- A transcript file via `--transcript` (SRT, VTT, or Fathom-style markdown)
- The `tesseract` binary installed locally (`brew install tesseract` on macOS,
  `apt install tesseract-ocr` on Linux)

The filter drops gallery/talking-head segments by detecting on-screen text
density, then keeps one frame per transcript segment plus any frames with
major visual changes (modal popups, view switches). See
`docs/superpowers/specs/2026-04-08-demo-mode-frame-filtering-design.md` for
the full algorithm.
````

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document --demo-mode usage in README"
```

---

## Task 14: Final verification

- [ ] **Step 1: Run the full test suite one more time**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Verify the CLI help shows --demo-mode**

Run: `uv run peeklet --help`
Expected: `--demo-mode` appears in the options list with the correct help text.

- [ ] **Step 3: Do a manual smoke test (optional but recommended)**

If you have a real demo video + transcript handy, run:

```bash
uv run peeklet --input <real_demo.mp4> --transcript <real_demo.srt> --demo-mode --output /tmp/peeklet-demo
```

Expected:
- Process completes in under 60 seconds for a 60-min video
- The output directory contains `manifest.parquet`, `context.json`, `context.md`
- The keyframe count is roughly 100-200 (vs ~800 without `--demo-mode`)
- Spot-check a few of the kept screenshots — they should be screen-share content, not gallery view

- [ ] **Step 4: Push the branch**

```bash
git push -u origin feat/demo-mode-frame-filtering
```

The feature is complete. Open a PR against `develop` for review.
