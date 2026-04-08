# OCR Content Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Filter out video-call/talking-head keyframes from Peeklet's video output by running Tesseract OCR on each candidate keyframe and demoting frames whose extracted text falls below a configurable character threshold.

**Architecture:** Add a thin OCR wrapper module. After `Pipeline.process_frame` marks a frame as a keyframe in `process_video`, run OCR on the saved image. If text length is below threshold, delete the saved image, flip the result back to non-keyframe with a `demoted_reason`, and rewind pipeline rolling state to the prior keyframe (so the demoted frame doesn't anchor subsequent SSIM comparisons). Forced (transcript-trigger) frames bypass the filter entirely.

**Tech Stack:** Python 3.10+, `pytesseract` (new optional dep, system `tesseract` binary), pydantic config, pyarrow manifest, pytest.

**Spec:** `docs/superpowers/specs/2026-04-08-ocr-content-filter-design.md`

**Branch:** `feat/ocr-content-filter` (cut from `main` after the inline-screenshots PR lands)

---

## File Structure

**Create:**
- `src/peeklet/core/ocr.py` — thin Tesseract wrapper (one public function: `run_ocr`)
- `tests/unit/test_ocr.py` — unit tests for the OCR wrapper (mocked + real)
- `tests/unit/test_video_ocr_filter.py` — unit tests for the demotion logic in `process_video`

**Modify:**
- `src/peeklet/utils/types.py` — add `ocr_text`, `ocr_char_count`, `demoted_reason` fields to `FrameResult`
- `src/peeklet/core/exporter.py` — add three new fields to `MANIFEST_SCHEMA` and `ManifestWriter.append`
- `src/peeklet/config.py` — add `ocr_enabled`, `ocr_min_text_chars`, `ocr_engine` to `VideoConfig`
- `src/peeklet/pipeline.py` — add `snapshot_state()` / `restore_state()` for rollback after demotion
- `src/peeklet/core/video.py` — integrate OCR + demotion into the per-frame loop
- `tests/integration/test_video_pipeline.py` — extend with an end-to-end OCR-filter test
- `pyproject.toml` — add `[ocr]` optional-dependencies extra
- `README.md` — document the new filter, config, and install requirement

---

## Task 1: Create the OCR wrapper module

**Files:**
- Create: `src/peeklet/core/ocr.py`
- Create: `tests/unit/test_ocr.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ocr.py`:

```python
"""Unit tests for the OCR wrapper module."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

pytest.importorskip("pytesseract", reason="requires peeklet[ocr]")

from peeklet.core.ocr import run_ocr  # noqa: E402


def _text_image(text: str, w: int = 600, h: int = 200) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=48)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((20, 60), text, fill=(0, 0, 0), font=font)
    return np.asarray(img, dtype=np.uint8)


def _blank_image(w: int = 600, h: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _noise_image(w: int = 600, h: int = 200) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class TestRunOcr:
    def test_extracts_rendered_text(self) -> None:
        frame = _text_image("Hello World")
        result = run_ocr(frame)
        assert "Hello" in result or "World" in result

    def test_blank_image_returns_near_empty(self) -> None:
        frame = _blank_image()
        result = run_ocr(frame).strip()
        assert len(result) < 5

    def test_noise_image_returns_near_empty(self) -> None:
        frame = _noise_image()
        result = run_ocr(frame).strip()
        assert len(result) < 30  # Tesseract may hallucinate a few chars on noise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peeklet.core.ocr'`

- [ ] **Step 3: Create the OCR wrapper module**

Create `src/peeklet/core/ocr.py`:

```python
"""Thin wrapper around Tesseract OCR for frame text extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _check_ocr_deps() -> None:
    """Raise a clear error if OCR dependencies are not installed."""
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        raise ImportError(
            "OCR support requires additional dependencies. "
            "Install with: pip install peeklet[ocr]\n"
            "Also install the tesseract system binary "
            "(e.g. 'brew install tesseract' or 'apt install tesseract-ocr')."
        ) from None


def run_ocr(frame_rgb: np.ndarray) -> str:
    """Run Tesseract OCR on an RGB frame and return the extracted text.

    Args:
        frame_rgb: HxWx3 uint8 RGB numpy array.

    Returns:
        The raw text extracted by Tesseract. Whitespace is preserved; callers
        that want a character count should call ``len(result.strip())``.

    Raises:
        ImportError: If pytesseract is not installed.
    """
    _check_ocr_deps()
    import pytesseract
    from PIL import Image

    image = Image.fromarray(frame_rgb)
    return pytesseract.image_to_string(image)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ocr.py -v`
Expected: All three tests PASS (assuming `pytesseract` and the `tesseract` binary are installed locally).

- [ ] **Step 5: Lint**

Run: `ruff check src/peeklet/core/ocr.py tests/unit/test_ocr.py && ruff format --check src/peeklet/core/ocr.py tests/unit/test_ocr.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/core/ocr.py tests/unit/test_ocr.py
git commit -m "feat: add Tesseract OCR wrapper module

Thin wrapper exposing run_ocr(frame_rgb) -> str. Lazy-imports pytesseract
and raises a clear ImportError with install hint when missing, mirroring
the existing _check_video_deps pattern."
```

---

## Task 2: Add `[ocr]` optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the extra**

In `pyproject.toml`, find the `[project.optional-dependencies]` section. Add this entry alongside the existing `video`/`dev`/`datasets` extras:

```toml
ocr = [
    "pytesseract>=0.3.10",
]
```

The full section should look like:

```toml
[project.optional-dependencies]
video = [
    "imageio[ffmpeg]>=2.31",
    "av>=14.0",
    "pydub>=0.25",
]
ocr = [
    "pytesseract>=0.3.10",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "pytest-benchmark>=4.0",
    "hypothesis>=6.0",
    "ruff>=0.4",
    "mypy>=1.8",
]
datasets = [
    "datasets>=2.14",
    "huggingface-hub>=0.17",
]
```

- [ ] **Step 2: Install the new extra into the dev environment**

Run: `pip install -e ".[ocr]"`
Expected: pytesseract installed successfully.

- [ ] **Step 3: Verify the system tesseract binary is present**

Run: `tesseract --version`
Expected: prints a version string. If not, install it:
- macOS: `brew install tesseract`
- Linux: `apt install tesseract-ocr`

- [ ] **Step 4: Re-run Task 1 tests against the real engine**

Run: `pytest tests/unit/test_ocr.py -v`
Expected: all three pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add [ocr] optional dependency for pytesseract"
```

---

## Task 3: Add new `FrameResult` fields

**Files:**
- Modify: `src/peeklet/utils/types.py`
- Modify: `tests/unit/test_types.py` (extend existing tests)

- [ ] **Step 1: Write the failing test**

Open `tests/unit/test_types.py`. Add this test class at the end of the file:

```python
class TestFrameResultOcrFields:
    def test_ocr_fields_default_to_none(self) -> None:
        from peeklet.utils.types import EventType, FrameResult

        result = FrameResult(
            frame_id="f1",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abc",
            frame_width=100,
            frame_height=100,
        )
        assert result.ocr_text is None
        assert result.ocr_char_count is None
        assert result.demoted_reason is None

    def test_ocr_fields_can_be_set(self) -> None:
        from peeklet.utils.types import EventType, FrameResult

        result = FrameResult(
            frame_id="f1",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="abc",
            frame_width=100,
            frame_height=100,
        )
        result.ocr_text = "Hello World"
        result.ocr_char_count = 11
        result.demoted_reason = "low_text_content"
        assert result.ocr_text == "Hello World"
        assert result.ocr_char_count == 11
        assert result.demoted_reason == "low_text_content"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_types.py::TestFrameResultOcrFields -v`
Expected: FAIL with `AttributeError: 'FrameResult' object has no attribute 'ocr_text'`

- [ ] **Step 3: Add the fields**

In `src/peeklet/utils/types.py`, append these three fields at the end of the `FrameResult` dataclass (immediately after `change_magnitude: str | None = None`):

```python
    # OCR-driven content filter (video mode only)
    ocr_text: str | None = None
    ocr_char_count: int | None = None
    demoted_reason: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_types.py::TestFrameResultOcrFields -v`
Expected: both tests PASS.

- [ ] **Step 5: Lint**

Run: `ruff check src/peeklet/utils/types.py tests/unit/test_types.py && ruff format --check src/peeklet/utils/types.py tests/unit/test_types.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/utils/types.py tests/unit/test_types.py
git commit -m "feat: add ocr_text, ocr_char_count, demoted_reason to FrameResult"
```

---

## Task 4: Extend the parquet manifest schema

**Files:**
- Modify: `src/peeklet/core/exporter.py`
- Modify: `tests/unit/test_exporter.py` (extend existing tests)

- [ ] **Step 1: Write the failing test**

Open `tests/unit/test_exporter.py`. Add this test at the end of the file (adjust imports at the top if needed):

```python
class TestManifestWriterOcrFields:
    def test_ocr_fields_round_trip_through_parquet(self, tmp_path) -> None:
        import pyarrow.parquet as pq

        from peeklet.core.exporter import ManifestWriter
        from peeklet.utils.types import EventType, FrameResult

        path = tmp_path / "manifest.parquet"
        writer = ManifestWriter(path=path)

        result = FrameResult(
            frame_id="f1",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="abc",
            frame_width=100,
            frame_height=100,
            ocr_text="Hello World",
            ocr_char_count=11,
            demoted_reason="low_text_content",
        )
        writer.append(result)
        writer.flush()

        table = pq.read_table(path)
        assert "ocr_text" in table.column_names
        assert "ocr_char_count" in table.column_names
        assert "demoted_reason" in table.column_names
        row = table.to_pylist()[0]
        assert row["ocr_text"] == "Hello World"
        assert row["ocr_char_count"] == 11
        assert row["demoted_reason"] == "low_text_content"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_exporter.py::TestManifestWriterOcrFields -v`
Expected: FAIL — schema does not contain `ocr_text` and pyarrow rejects the row.

- [ ] **Step 3: Add the fields to `MANIFEST_SCHEMA`**

In `src/peeklet/core/exporter.py`, locate `MANIFEST_SCHEMA` (starts at line 17). Append three fields to the end of the field list, immediately after `pa.field("change_magnitude", pa.string(), nullable=True)`:

```python
        pa.field("ocr_text", pa.string(), nullable=True),
        pa.field("ocr_char_count", pa.int32(), nullable=True),
        pa.field("demoted_reason", pa.string(), nullable=True),
```

- [ ] **Step 4: Add the fields to `ManifestWriter.append`**

In the same file, locate `ManifestWriter.append`. Add three keys to the dict literal being appended to `self._rows`, immediately after `"change_magnitude": result.change_magnitude,`:

```python
                "ocr_text": result.ocr_text,
                "ocr_char_count": result.ocr_char_count,
                "demoted_reason": result.demoted_reason,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_exporter.py::TestManifestWriterOcrFields -v`
Expected: PASS.

- [ ] **Step 6: Run the full exporter test module to confirm no regression**

Run: `pytest tests/unit/test_exporter.py -v`
Expected: all tests PASS.

- [ ] **Step 7: Lint**

Run: `ruff check src/peeklet/core/exporter.py tests/unit/test_exporter.py && ruff format --check src/peeklet/core/exporter.py tests/unit/test_exporter.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/peeklet/core/exporter.py tests/unit/test_exporter.py
git commit -m "feat: extend manifest schema with OCR fields"
```

---

## Task 5: Add OCR config fields to `VideoConfig`

**Files:**
- Modify: `src/peeklet/config.py`
- Modify: `tests/unit/test_config.py` (extend)

- [ ] **Step 1: Write the failing test**

Open `tests/unit/test_config.py`. Add this test at the end of the file:

```python
class TestVideoConfigOcrFields:
    def test_ocr_defaults(self) -> None:
        from peeklet.config import PeekletConfig

        cfg = PeekletConfig()
        assert cfg.video.ocr_enabled is True
        assert cfg.video.ocr_min_text_chars == 50
        assert cfg.video.ocr_engine == "tesseract"

    def test_ocr_can_be_disabled_via_dict(self) -> None:
        from peeklet.config import PeekletConfig

        cfg = PeekletConfig.model_validate(
            {"video": {"ocr_enabled": False, "ocr_min_text_chars": 10}}
        )
        assert cfg.video.ocr_enabled is False
        assert cfg.video.ocr_min_text_chars == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::TestVideoConfigOcrFields -v`
Expected: FAIL with `AttributeError: 'VideoConfig' object has no attribute 'ocr_enabled'`

- [ ] **Step 3: Add the fields**

In `src/peeklet/config.py`, find the `VideoConfig` class (starts at line 71). Append three fields at the end of the class body:

```python
    # OCR-driven content filter
    ocr_enabled: bool = True
    ocr_min_text_chars: int = Field(default=50, ge=0)
    ocr_engine: Literal["tesseract"] = "tesseract"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py::TestVideoConfigOcrFields -v`
Expected: both tests PASS.

- [ ] **Step 5: Lint**

Run: `ruff check src/peeklet/config.py tests/unit/test_config.py && ruff format --check src/peeklet/config.py tests/unit/test_config.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/config.py tests/unit/test_config.py
git commit -m "feat: add OCR config fields to VideoConfig"
```

---

## Task 6: Add `snapshot_state` / `restore_state` to `Pipeline`

Why this exists: when `process_video` decides to demote a frame after OCR, the pipeline has already updated its rolling state (`_last_keyframe`, `_last_hash`, etc.) inside `_make_keyframe`. We need to rewind to the *previous* keyframe so the demoted frame doesn't anchor subsequent SSIM comparisons. Snapshot before `process_frame`, restore after demotion.

**Files:**
- Modify: `src/peeklet/pipeline.py`
- Modify: `tests/unit/test_pipeline.py` (extend)

- [ ] **Step 1: Write the failing test**

Open `tests/unit/test_pipeline.py`. Add this test at the end of the file (adjust imports if needed):

```python
class TestPipelineSnapshotRestore:
    def test_snapshot_then_restore_undoes_keyframe(self, tmp_path) -> None:
        import numpy as np

        from peeklet.config import PeekletConfig
        from peeklet.pipeline import Pipeline

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "out")
        config.masking.enabled = False
        pipe = Pipeline(config)

        # First frame establishes the reference (becomes a keyframe)
        red = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
        r1 = pipe.process_frame(red, frame_id="f1", source_format="image")
        assert r1.is_keyframe

        # Snapshot AFTER the first keyframe is anchored
        snap = pipe.snapshot_state()

        # Process a very different frame (becomes a keyframe and updates state)
        green = np.full((100, 100, 3), (0, 255, 0), dtype=np.uint8)
        r2 = pipe.process_frame(green, frame_id="f2", source_format="image")
        assert r2.is_keyframe

        # Restore — pipeline should now behave as if f2 never happened
        pipe.restore_state(snap)

        # Reprocess green: should again be detected as a new keyframe relative to red
        r3 = pipe.process_frame(green, frame_id="f3", source_format="image")
        assert r3.is_keyframe
        # And the prev_keyframe pointer should still point to f1, not f2
        assert r3.prev_keyframe_id == "f1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline.py::TestPipelineSnapshotRestore -v`
Expected: FAIL with `AttributeError: 'Pipeline' object has no attribute 'snapshot_state'`

- [ ] **Step 3: Add the snapshot dataclass and methods**

In `src/peeklet/pipeline.py`, add this dataclass at module scope, immediately after the existing imports and before the `Pipeline` class:

```python
from dataclasses import dataclass


@dataclass
class _PipelineState:
    """Snapshot of Pipeline rolling state for rollback after demotion."""

    last_keyframe: "np.ndarray | None"
    last_hash: str | None
    last_mean: tuple[float, float, float] | None
    last_keyframe_id: str | None
    last_keyframe_path: str | None
    last_tiled_hashes: list[str] | None
    last_tile_means: list[tuple[float, float, float]] | None
```

Then, inside the `Pipeline` class, add these two methods immediately after `update_reference_state` and before `finalize`:

```python
    def snapshot_state(self) -> _PipelineState:
        """Capture current rolling state so it can be restored later.

        Used by ``process_video`` to roll back after demoting a keyframe via
        the OCR content filter, so the demoted frame does not become the
        anchor for subsequent SSIM comparisons.
        """
        return _PipelineState(
            last_keyframe=self._last_keyframe,
            last_hash=self._last_hash,
            last_mean=self._last_mean,
            last_keyframe_id=self._last_keyframe_id,
            last_keyframe_path=self._last_keyframe_path,
            last_tiled_hashes=self._last_tiled_hashes,
            last_tile_means=self._last_tile_means,
        )

    def restore_state(self, state: _PipelineState) -> None:
        """Restore rolling state from a snapshot taken via ``snapshot_state``."""
        self._last_keyframe = state.last_keyframe
        self._last_hash = state.last_hash
        self._last_mean = state.last_mean
        self._last_keyframe_id = state.last_keyframe_id
        self._last_keyframe_path = state.last_keyframe_path
        self._last_tiled_hashes = state.last_tiled_hashes
        self._last_tile_means = state.last_tile_means
```

Note: the `dataclass` import at module top is new — make sure it's added once at the top of the file alongside the existing imports rather than mid-file. The example above shows it inline only for clarity.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pipeline.py::TestPipelineSnapshotRestore -v`
Expected: PASS.

- [ ] **Step 5: Run the full pipeline test module to confirm no regression**

Run: `pytest tests/unit/test_pipeline.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Lint**

Run: `ruff check src/peeklet/pipeline.py tests/unit/test_pipeline.py && ruff format --check src/peeklet/pipeline.py tests/unit/test_pipeline.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: add Pipeline snapshot_state/restore_state for demotion rollback"
```

---

## Task 7: Integrate OCR filter into `process_video`

This is the heart of the change. After the existing keyframe bookkeeping decides a frame is a keyframe, run OCR. If text is too short and the frame was not transcript-forced, demote: delete the saved image, flip flags on the result, and rewind pipeline state.

**Files:**
- Modify: `src/peeklet/core/video.py`
- Create: `tests/unit/test_video_ocr_filter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_video_ocr_filter.py`:

```python
"""Unit tests for the OCR-based content filter in process_video."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

iio = pytest.importorskip("imageio.v3", reason="requires peeklet[video]")
pytest.importorskip("av", reason="requires peeklet[video]")

from peeklet.config import PeekletConfig  # noqa: E402
from peeklet.core.video import process_video  # noqa: E402


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


def _make_three_scene_video(tmp_path: Path) -> Path:
    """Three visually distinct scenes (~1.5s each) so the pipeline produces
    multiple keyframes regardless of OCR outcome."""
    frames = (
        [_solid_frame((0, 0, 0))] * 45
        + [_solid_frame((255, 255, 255))] * 45
        + [_solid_frame((128, 128, 128))] * 45
    )
    return _make_test_video(tmp_path / "demo.mp4", frames, fps=30)


class TestOcrFilter:
    def test_low_text_keyframes_are_demoted(self, tmp_path: Path) -> None:
        video = _make_three_scene_video(tmp_path)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "out")
        config.video.ocr_enabled = True
        config.video.ocr_min_text_chars = 50

        # Mock OCR to always return empty
        with patch("peeklet.core.video.run_ocr", return_value=""):
            results = process_video(video, config)

        keyframes = [r for r in results if r.is_keyframe]
        demoted = [r for r in results if r.demoted_reason == "low_text_content"]
        assert len(keyframes) == 0
        assert len(demoted) >= 2  # original keyframes all demoted

        # No keyframe images on disk
        out_dir = tmp_path / "out"
        for r in demoted:
            assert r.asset_path is None
        # No leftover keyframe files
        keyframe_files = [
            p for p in out_dir.glob("*") if p.suffix in (".png", ".jpg") and p.is_file()
        ]
        assert keyframe_files == []

    def test_high_text_keyframes_pass_through(self, tmp_path: Path) -> None:
        video = _make_three_scene_video(tmp_path)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "out")
        config.video.ocr_enabled = True
        config.video.ocr_min_text_chars = 50

        # Mock OCR to always return long text
        with patch("peeklet.core.video.run_ocr", return_value="x" * 200):
            results = process_video(video, config)

        keyframes = [r for r in results if r.is_keyframe]
        demoted = [r for r in results if r.demoted_reason == "low_text_content"]
        assert len(keyframes) >= 2
        assert demoted == []
        for r in keyframes:
            assert r.ocr_char_count == 200
            assert r.asset_path is not None
            assert Path(r.asset_path).exists()

    def test_ocr_disabled_skips_filter(self, tmp_path: Path) -> None:
        video = _make_three_scene_video(tmp_path)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "out")
        config.video.ocr_enabled = False

        with patch("peeklet.core.video.run_ocr") as mock_ocr:
            results = process_video(video, config)
            mock_ocr.assert_not_called()

        keyframes = [r for r in results if r.is_keyframe]
        assert len(keyframes) >= 2
        for r in keyframes:
            assert r.ocr_text is None
            assert r.ocr_char_count is None

    def test_demoted_frames_do_not_anchor_subsequent_comparisons(
        self, tmp_path: Path
    ) -> None:
        """A demoted (low-text) frame must not become the SSIM reference,
        otherwise the next genuinely different frame would be compared
        against it instead of the prior surviving keyframe."""
        video = _make_three_scene_video(tmp_path)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "out")
        config.video.ocr_enabled = True
        config.video.ocr_min_text_chars = 50

        # First call returns text (kept), all others return empty (demoted)
        call_count = {"n": 0}

        def side_effect(_frame: np.ndarray) -> str:
            call_count["n"] += 1
            return "x" * 200 if call_count["n"] == 1 else ""

        with patch("peeklet.core.video.run_ocr", side_effect=side_effect):
            results = process_video(video, config)

        keyframes = [r for r in results if r.is_keyframe]
        # Exactly one survivor — the first keyframe
        assert len(keyframes) == 1
        # All others were demoted, not silently dropped
        demoted = [r for r in results if r.demoted_reason == "low_text_content"]
        assert len(demoted) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_video_ocr_filter.py -v`
Expected: FAIL — `run_ocr` is not imported in `peeklet.core.video`, and the demotion logic does not exist.

- [ ] **Step 3: Update the imports in `video.py`**

In `src/peeklet/core/video.py`, add this import alongside the existing `from peeklet.core.exporter import ...` line:

```python
from peeklet.core.ocr import run_ocr
```

- [ ] **Step 4: Add the OCR-deps fail-fast check**

In `src/peeklet/core/video.py`, find `_check_video_deps()` (line 48). Add a new helper immediately after it:

```python
def _check_ocr_deps() -> None:
    """Raise a clear error if OCR is enabled but pytesseract is missing."""
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        raise ImportError(
            "OCR filter is enabled (config.video.ocr_enabled=True) but "
            "pytesseract is not installed. Install with: pip install peeklet[ocr] "
            "and ensure the tesseract system binary is available "
            "(brew install tesseract / apt install tesseract-ocr). "
            "Set config.video.ocr_enabled=False to disable the filter."
        ) from None
```

- [ ] **Step 5: Wire the fail-fast check into `process_video`**

In `src/peeklet/core/video.py`, inside `process_video`, immediately after the line `forced_ts = forced_timestamps or []`, add:

```python
    if config.video.ocr_enabled:
        _check_ocr_deps()
```

- [ ] **Step 6: Implement demotion in the per-frame loop**

In `src/peeklet/core/video.py`, locate the per-frame loop inside `process_video` (starts at the `for frame, ts, frame_num in decoder.iter_coarse_frames_seek(...)` line, around line 297).

Replace the entire loop body — from the `frame_id = ...` line down through but **not** including `results.append(result)` — with this version. The new content snapshots pipeline state before the call, runs OCR after a keyframe is produced, and rolls back on demotion. Compare carefully against the existing code; the only structural change is the new "OCR filter" block and the snapshot/restore pair.

```python
        frame_id = f"frame_{frame_num:06d}"
        snapshot = pipeline.snapshot_state()
        result = pipeline.process_frame(
            frame,
            frame_id=frame_id,
            source_format="video",
        )

        is_forced = _should_force_keyframe(ts, forced_ts)
        was_visual_keyframe = result.is_keyframe

        # Promote forced (transcript-triggered) frames to keyframes
        if is_forced and not result.is_keyframe:
            asset_path = save_keyframe(
                frame,
                output_dir,
                frame_id,
                fmt=config.exporter.keyframe_format,
            )
            result.is_keyframe = True
            result.event_type = EventType.KEYFRAME
            result.asset_path = str(asset_path)
            result.visual_reason = "Transcript trigger"
            pipeline.update_reference_state(frame, frame_id, str(asset_path))

        # Set trigger_type on every keyframe
        if result.is_keyframe:
            result.trigger_type = _merge_trigger_type(
                is_visual=was_visual_keyframe,
                is_transcript=is_forced,
            )

        # --- OCR content filter ---
        # Run OCR on every keyframe. If text is below threshold AND the frame
        # was not transcript-forced, demote: delete the asset, flip is_keyframe
        # back to False, set demoted_reason, and rewind pipeline rolling state
        # so the demoted frame doesn't anchor subsequent SSIM comparisons.
        if result.is_keyframe and config.video.ocr_enabled and not is_forced:
            ocr_text = run_ocr(frame)
            char_count = len(ocr_text.strip())
            result.ocr_text = ocr_text
            result.ocr_char_count = char_count

            if char_count < config.video.ocr_min_text_chars:
                if result.asset_path:
                    Path(result.asset_path).unlink(missing_ok=True)
                result.is_keyframe = False
                result.event_type = EventType.SKIPPED
                result.asset_path = None
                result.demoted_reason = "low_text_content"
                result.trigger_type = None
                pipeline.restore_state(snapshot)

        # Enrich with video metadata
        result.source_video = meta.filename
        result.video_timestamp = ts
        result.video_frame_number = frame_num
        result.video_duration = meta.duration

        if result.is_keyframe:
            keyframe_count += 1
            new_frame_id = timestamp_filename(ts)
            if result.asset_path:
                old_path = Path(result.asset_path)
                new_path = old_path.with_name(new_frame_id + old_path.suffix)
                if old_path.exists():
                    old_path.rename(new_path)
                result.asset_path = str(new_path)
            result.frame_id = new_frame_id
            result.keyframe_index = keyframe_count
            result.change_magnitude = _change_magnitude(result)
            if prev_keyframe_ts is not None:
                result.time_since_prev_keyframe = ts - prev_keyframe_ts
            prev_keyframe_ts = ts

        results.append(result)
```

- [ ] **Step 7: Run unit tests to verify they pass**

Run: `pytest tests/unit/test_video_ocr_filter.py -v`
Expected: all four tests PASS.

- [ ] **Step 8: Run the full unit suite to confirm no regression**

Run: `pytest tests/unit -v`
Expected: all tests PASS.

- [ ] **Step 9: Lint**

Run: `ruff check src/peeklet/core/video.py tests/unit/test_video_ocr_filter.py && ruff format --check src/peeklet/core/video.py tests/unit/test_video_ocr_filter.py`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/peeklet/core/video.py tests/unit/test_video_ocr_filter.py
git commit -m "feat: filter low-text keyframes via OCR in process_video

After Pipeline marks a frame as a keyframe, run Tesseract OCR on it.
If extracted text is below ocr_min_text_chars and the frame was not
transcript-forced, delete the saved asset, flip is_keyframe to False,
set demoted_reason='low_text_content', and rewind pipeline rolling
state via snapshot/restore so the demoted frame does not anchor
subsequent SSIM comparisons."
```

---

## Task 8: Verify forced (transcript-triggered) frames bypass the filter

This is covered implicitly by Task 7's `is_forced` check, but it deserves an explicit dedicated test against a transcript file to ensure the bypass actually works end-to-end.

**Files:**
- Modify: `tests/unit/test_video_ocr_filter.py` (add one test)

- [ ] **Step 1: Add the test**

Append this test to the `TestOcrFilter` class in `tests/unit/test_video_ocr_filter.py`:

```python
    def test_transcript_forced_frame_bypasses_ocr_filter(self, tmp_path: Path) -> None:
        """Transcript-trigger frames must be kept even if OCR returns empty."""
        from unittest.mock import patch as _patch

        video = _make_three_scene_video(tmp_path)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "out")
        config.video.ocr_enabled = True
        config.video.ocr_min_text_chars = 1000  # impossibly high

        # forced_timestamps near scene boundaries
        forced = [0.5, 1.5, 2.5]

        with _patch("peeklet.core.video.run_ocr", return_value=""):
            results = process_video(video, config, forced_timestamps=forced)

        # Forced frames should survive — they bypass the OCR filter
        forced_kept = [
            r
            for r in results
            if r.is_keyframe and r.trigger_type in ("transcript_trigger", "both")
        ]
        assert len(forced_kept) >= 1
        # And they should have no ocr_text set (filter was skipped)
        for r in forced_kept:
            assert r.ocr_text is None
            assert r.demoted_reason is None
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/unit/test_video_ocr_filter.py::TestOcrFilter::test_transcript_forced_frame_bypasses_ocr_filter -v`
Expected: PASS.

- [ ] **Step 3: Lint**

Run: `ruff check tests/unit/test_video_ocr_filter.py && ruff format --check tests/unit/test_video_ocr_filter.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_video_ocr_filter.py
git commit -m "test: verify transcript-forced frames bypass OCR filter"
```

---

## Task 9: End-to-end integration test with real Tesseract

**Files:**
- Modify: `tests/integration/test_video_pipeline.py`

- [ ] **Step 1: Add the integration test**

Open `tests/integration/test_video_pipeline.py`. Add this test class at the end of the file:

```python
class TestVideoOcrFilterIntegration:
    """End-to-end OCR filter test using a real tesseract binary."""

    def test_blank_scenes_are_filtered_out(self, tmp_path: Path) -> None:
        pytest.importorskip("pytesseract", reason="requires peeklet[ocr]")
        from PIL import Image, ImageDraw, ImageFont

        # Build a video where some scenes have rendered text, others are blank.
        # Each scene is 45 frames @ 30 fps = 1.5s.
        def text_frame(text: str, w: int = 640, h: int = 480) -> np.ndarray:
            img = Image.new("RGB", (w, h), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default(size=36)
            except TypeError:
                font = ImageFont.load_default()
            # Render multiple lines so total chars >> threshold
            for i, line in enumerate(text.splitlines()):
                draw.text((20, 30 + i * 50), line, fill=(0, 0, 0), font=font)
            return np.asarray(img, dtype=np.uint8)

        text_a = text_frame(
            "Welcome to Peeklet\nThis is a long line of text\nLine three of text"
        )
        blank = _solid_frame((180, 180, 180), w=640, h=480)
        text_b = text_frame(
            "Settings panel open\nNotifications email on\nTheme dark mode active"
        )

        frames = [text_a] * 45 + [blank] * 45 + [text_b] * 45
        video = _make_test_video(tmp_path / "ocr_demo.mp4", frames, fps=30)

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "out")
        config.video.ocr_enabled = True
        config.video.ocr_min_text_chars = 30

        results = process_video(video, config)

        keyframes = [r for r in results if r.is_keyframe]
        demoted = [r for r in results if r.demoted_reason == "low_text_content"]

        # The two text scenes should each survive at least once.
        # The blank scene's keyframe should be demoted.
        assert len(keyframes) >= 2, f"expected text scenes to survive, got {len(keyframes)}"
        assert len(demoted) >= 1, "expected at least one blank-scene demotion"

        # Surviving keyframes should have non-empty OCR text
        for r in keyframes:
            assert r.ocr_char_count is not None and r.ocr_char_count >= 30
            assert r.asset_path is not None
            assert Path(r.asset_path).exists()
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/integration/test_video_pipeline.py::TestVideoOcrFilterIntegration -v`
Expected: PASS. If it fails because the rendered text is too small for Tesseract on a particular system, increase the font size or rendering resolution rather than weakening the assertions.

- [ ] **Step 3: Run the full integration suite**

Run: `pytest tests/integration -v`
Expected: all PASS.

- [ ] **Step 4: Lint**

Run: `ruff check tests/integration/test_video_pipeline.py && ruff format --check tests/integration/test_video_pipeline.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_video_pipeline.py
git commit -m "test: end-to-end OCR filter test with real tesseract"
```

---

## Task 10: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the install requirement**

Open `README.md` and find the install/quickstart section. Add a note alongside the `peeklet[video]` install instructions:

```markdown
### OCR content filter (recommended for video)

Peeklet runs OCR on each video keyframe and drops frames that contain no shared screen content (e.g. video-call gallery views with no screen-share). This requires:

```bash
pip install peeklet[ocr]
# macOS:
brew install tesseract
# Linux:
sudo apt install tesseract-ocr
```

The filter is enabled by default. Disable it via config:

```yaml
video:
  ocr_enabled: false
  ocr_min_text_chars: 50  # frames with fewer chars are filtered
```
```

- [ ] **Step 2: Add a "Why some keyframes are filtered" note**

In the same README, add a short subsection (or extend an existing FAQ section) explaining what `demoted_reason` means in the manifest:

```markdown
### Why some frames are missing from the output

When the OCR filter is enabled (default), keyframes that contain no recognizable text — typically video-call gallery views, blank desktops, or fullscreen video — are demoted out of the user-facing output. They remain in the parquet manifest with `is_keyframe=False` and `demoted_reason="low_text_content"`, so you can audit what was filtered:

```python
import pyarrow.parquet as pq
table = pq.read_table("output/manifest.parquet")
demoted = table.filter(pq.compute.is_valid(table["demoted_reason"]))
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document OCR content filter, install, and audit"
```

---

## Task 11: Full verification before PR

- [ ] **Step 1: Full lint**

Run: `ruff check . && ruff format --check .`
Expected: clean across the whole repo.

- [ ] **Step 2: Full test suite**

Run: `pytest -v`
Expected: all tests PASS, including the OCR-gated ones (which require pytesseract + tesseract binary).

- [ ] **Step 3: Manual validation against the real call recording**

Run Peeklet against the actual 60-minute call recording the user observed the 800-keyframe inflation on. Inspect:

1. Final keyframe count (should be dramatically lower than 800)
2. Distribution of `ocr_char_count` in the manifest:
   ```python
   import pyarrow.parquet as pq
   t = pq.read_table("output/manifest.parquet")
   counts = [c for c in t["ocr_char_count"].to_pylist() if c is not None]
   print("min", min(counts), "median", sorted(counts)[len(counts)//2], "max", max(counts))
   ```
3. Sample the surviving keyframe images and confirm they all show shared screen content (not gallery views)
4. Sample the demoted rows and confirm they were correctly identified as gallery/blank

If the threshold needs tuning, adjust `ocr_min_text_chars` in the user's config. If the default of 50 is wrong for this corpus, update the default in `src/peeklet/config.py` (under `VideoConfig`) and add a one-liner commit explaining the empirical retune.

- [ ] **Step 4: Open the PR**

```bash
git push -u origin feat/ocr-content-filter
gh pr create --title "feat: OCR-based content filter for video keyframes" --body "$(cat <<'EOF'
## Summary
- Adds Tesseract OCR (via new `peeklet[ocr]` extra) to the video pipeline
- After `Pipeline.process_frame` marks a frame as a keyframe, run OCR; if extracted text is below `ocr_min_text_chars` (default 50) and the frame was not transcript-forced, demote it: delete the saved image, flip `is_keyframe=False`, set `demoted_reason="low_text_content"`, rewind pipeline rolling state
- Cleanly removes video-call gallery / blank-screen noise from user-facing output while preserving full audit trail in the manifest

## Spec
`docs/superpowers/specs/2026-04-08-ocr-content-filter-design.md`

## Test plan
- [ ] `pytest tests/unit/test_ocr.py` — OCR wrapper unit tests pass
- [ ] `pytest tests/unit/test_video_ocr_filter.py` — demotion logic unit tests pass (mocked OCR)
- [ ] `pytest tests/integration/test_video_pipeline.py` — end-to-end test with real tesseract passes
- [ ] `pytest -v` — full suite green
- [ ] `ruff check . && ruff format --check .` — clean
- [ ] Manual run against the 60-min call recording: keyframe count drops from ~800 to a sane number, surviving frames all show shared content, demoted frames are gallery/blank

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Notes

**Spec coverage check:**
- OCR wrapper module → Task 1 ✓
- `pyproject.toml` extra → Task 2 ✓
- `FrameResult` fields → Task 3 ✓
- Manifest schema → Task 4 ✓
- Config fields → Task 5 ✓
- Pipeline reference-state isolation → Task 6 (snapshot/restore) ✓
- `process_video` integration with demotion → Task 7 ✓
- Forced-frame bypass → Task 8 ✓ (also implicit in Task 7's logic)
- Integration test → Task 9 ✓
- README docs → Task 10 ✓
- Lint + manual validation + PR → Task 11 ✓

**Type/name consistency check:**
- `run_ocr` defined in Task 1, imported in Task 7 ✓
- `_PipelineState` / `snapshot_state` / `restore_state` defined in Task 6, used in Task 7 ✓
- `ocr_text` / `ocr_char_count` / `demoted_reason` defined in Task 3, used in Tasks 4, 7, 8, 9, 10, 11 ✓
- `ocr_enabled` / `ocr_min_text_chars` / `ocr_engine` defined in Task 5, used in Tasks 7, 8, 9, 10 ✓
- `_check_ocr_deps` defined in Task 7, mirrors existing `_check_video_deps` pattern ✓

**Placeholder scan:** None found.
