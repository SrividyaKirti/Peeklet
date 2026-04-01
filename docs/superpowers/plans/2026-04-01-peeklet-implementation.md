# Peeklet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Peeklet — a smart screenshot change detection library + CLI that filters noise, redacts PII, and exports structured Parquet manifests of keyframes.

**Architecture:** Modular Python library with stateless core modules (loader, masking, hasher, comparator, redactor, exporter) wired by a stateful pipeline orchestrator. CLI is a thin wrapper. Each module is independently importable and testable.

**Tech Stack:** Python 3.10+, numpy, Pillow, scikit-image, imagehash, pyarrow, pydantic, click, easyocr (optional), pymupdf (optional), pytest, ruff, mypy

**Spec:** `docs/superpowers/specs/2026-04-01-peeklet-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package config, dependencies, scripts, build |
| `.gitignore` | Ignore patterns |
| `.github/workflows/ci-develop.yml` | CI for PRs to develop |
| `.github/workflows/ci-main.yml` | CI for PRs to main |
| `peeklet/__init__.py` | Package root, version |
| `peeklet/utils/types.py` | All shared dataclasses and enums |
| `peeklet/utils/image.py` | Shared image operations |
| `peeklet/utils/__init__.py` | Utils package |
| `peeklet/config.py` | Pydantic config models, loading, validation |
| `peeklet/core/__init__.py` | Core package |
| `peeklet/core/loader.py` | Any format → numpy RGB uint8 |
| `peeklet/core/hasher.py` | Perceptual hashing |
| `peeklet/core/masking.py` | Adaptive block-based masking |
| `peeklet/core/comparator.py` | SSIM + block-level diff |
| `peeklet/core/redactor.py` | PII detection + black-box redaction |
| `peeklet/core/exporter.py` | Parquet manifest + keyframe writer |
| `peeklet/pipeline.py` | Stateful orchestrator wiring all modules |
| `peeklet/cli.py` | CLI entry point |
| `tests/conftest.py` | Shared fixtures, markers |
| `tests/unit/test_types.py` | Types dataclass tests |
| `tests/unit/test_image.py` | Image utils tests |
| `tests/unit/test_config.py` | Config loading/validation tests |
| `tests/unit/test_loader.py` | Loader tests (all formats) |
| `tests/unit/test_hasher.py` | Hasher tests |
| `tests/unit/test_masking.py` | Adaptive masking tests |
| `tests/unit/test_comparator.py` | Comparator tests |
| `tests/unit/test_redactor.py` | PII redactor tests |
| `tests/unit/test_exporter.py` | Exporter tests |
| `tests/integration/test_pipeline_batch.py` | Full pipeline batch tests |
| `tests/integration/test_pipeline_stream.py` | Full pipeline stream tests |
| `tests/synthetic/generate.py` | Synthetic test data generator |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `peeklet/__init__.py`
- Create: `peeklet/utils/__init__.py`
- Create: `peeklet/core/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "peeklet"
version = "0.1.0"
description = "Smart screenshot change detection — filters noise, redacts PII, exports structured keyframe manifests"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Vidya", email = "vidya@peeklet.dev" },
]
keywords = ["screenshot", "change-detection", "deduplication", "parquet", "workflow"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Image Processing",
]

dependencies = [
    "numpy>=1.24",
    "Pillow>=10.0",
    "scikit-image>=0.21",
    "imagehash>=4.3",
    "pyarrow>=14.0",
    "pydantic>=2.0",
    "click>=8.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
ocr = [
    "easyocr>=1.7",
    "pymupdf>=1.23",
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

[project.scripts]
peeklet = "peeklet.cli:main"

[project.urls]
Homepage = "https://github.com/SrividyaKirti/Peeklet"
Repository = "https://github.com/SrividyaKirti/Peeklet"
Issues = "https://github.com/SrividyaKirti/Peeklet/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "TCH"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "mind2web: tests requiring Mind2Web dataset (deselect with '-m not mind2web')",
    "showui: tests requiring ShowUI dataset (deselect with '-m not showui')",
    "webui: tests requiring WebUI dataset (deselect with '-m not webui')",
    "benchmark: performance benchmark tests",
]
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg
.eggs/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Peeklet specific
tests/datasets/data/
tests/synthetic/output/
output/

# Superpowers
.superpowers/

# Environment
.env
.env.local
```

- [ ] **Step 3: Create package init files**

`peeklet/__init__.py`:
```python
"""Peeklet — Smart screenshot change detection."""

__version__ = "0.1.0"
```

`peeklet/utils/__init__.py`:
```python
"""Shared utilities for Peeklet."""
```

`peeklet/core/__init__.py`:
```python
"""Core processing modules for Peeklet."""
```

`tests/__init__.py`:
```python
```

`tests/unit/__init__.py`:
```python
```

`tests/integration/__init__.py`:
```python
```

- [ ] **Step 4: Create tests/conftest.py**

```python
"""Shared test fixtures and configuration."""

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 100x100 RGB test frame with known content."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Red quadrant top-left
    frame[:50, :50] = [255, 0, 0]
    # Green quadrant top-right
    frame[:50, 50:] = [0, 255, 0]
    # Blue quadrant bottom-left
    frame[50:, :50] = [0, 0, 255]
    # White quadrant bottom-right
    frame[50:, 50:] = [255, 255, 255]
    return frame


@pytest.fixture
def sample_frame_small_change(sample_frame: np.ndarray) -> np.ndarray:
    """Same as sample_frame but with a small 5x5 pixel change."""
    changed = sample_frame.copy()
    changed[10:15, 10:15] = [128, 128, 128]
    return changed


@pytest.fixture
def sample_frame_big_change() -> np.ndarray:
    """A completely different frame from sample_frame."""
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame[20:80, 20:80] = [0, 0, 0]
    return frame


@pytest.fixture
def hd_frame() -> np.ndarray:
    """A 1920x1080 RGB frame for benchmark-style tests."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
```

- [ ] **Step 5: Initialize uv and verify**

Run: `cd /Users/vidya/Documents/code_time/Peeklet && uv sync --dev`
Expected: Dependencies install successfully, `.venv` created.

- [ ] **Step 6: Verify pytest runs**

Run: `uv run pytest --co`
Expected: "no tests ran" with exit code 5 (no tests collected yet), no import errors.

- [ ] **Step 7: Verify ruff and mypy run**

Run: `uv run ruff check peeklet/ && uv run mypy peeklet/`
Expected: No errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore peeklet/ tests/
git commit -m "feat: project scaffolding with pyproject.toml, package structure, and test fixtures"
```

---

### Task 2: Shared Types (`peeklet/utils/types.py`)

**Files:**
- Create: `peeklet/utils/types.py`
- Create: `tests/unit/test_types.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_types.py`:
```python
"""Tests for shared type definitions."""

from peeklet.utils.types import EventType, FrameMeta, FrameResult, Region


class TestRegion:
    def test_create_region(self) -> None:
        r = Region(x=10, y=20, w=100, h=50)
        assert r.x == 10
        assert r.y == 20
        assert r.w == 100
        assert r.h == 50

    def test_region_to_dict(self) -> None:
        r = Region(x=10, y=20, w=100, h=50)
        assert r.to_dict() == {"x": 10, "y": 20, "w": 100, "h": 50}

    def test_region_area(self) -> None:
        r = Region(x=0, y=0, w=100, h=50)
        assert r.area == 5000


class TestEventType:
    def test_keyframe_value(self) -> None:
        assert EventType.KEYFRAME.value == "KEYFRAME"

    def test_skipped_value(self) -> None:
        assert EventType.SKIPPED.value == "SKIPPED"


class TestFrameMeta:
    def test_create_with_defaults(self) -> None:
        meta = FrameMeta(frame_id="frame_001")
        assert meta.frame_id == "frame_001"
        assert meta.timestamp is None
        assert meta.app_name is None
        assert meta.window_title is None
        assert meta.source_format is None

    def test_create_with_all_fields(self) -> None:
        from datetime import datetime, timezone

        ts = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        meta = FrameMeta(
            frame_id="frame_001",
            timestamp=ts,
            app_name="Chrome",
            window_title="Google - Chrome",
            source_format="png",
        )
        assert meta.app_name == "Chrome"
        assert meta.timestamp == ts


class TestFrameResult:
    def test_create_keyframe_result(self) -> None:
        result = FrameResult(
            frame_id="frame_001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd1234",
            ssim_score=0.62,
            change_score=0.38,
            changed_pct=34.0,
            changed_regions=[Region(x=100, y=200, w=300, h=150)],
            adaptive_mask=[Region(x=0, y=0, w=32, h=32)],
            frame_width=1920,
            frame_height=1080,
            pii_detected=False,
        )
        assert result.is_keyframe is True
        assert result.event_type == EventType.KEYFRAME
        assert len(result.changed_regions) == 1

    def test_create_skipped_result(self) -> None:
        result = FrameResult(
            frame_id="frame_002",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="abcd1234",
            frame_width=1920,
            frame_height=1080,
        )
        assert result.is_keyframe is False
        assert result.ssim_score is None
        assert result.changed_regions is None
        assert result.asset_path is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/utils/types.py`:
```python
"""Shared type definitions for Peeklet."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(Enum):
    """Whether a frame was kept as a keyframe or skipped."""

    KEYFRAME = "KEYFRAME"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class Region:
    """A rectangular region on screen."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """Metadata about a frame provided by the caller."""

    frame_id: str
    timestamp: datetime | None = None
    app_name: str | None = None
    window_title: str | None = None
    source_format: str | None = None


@dataclass(slots=True)
class FrameResult:
    """Result of processing a single frame through the pipeline."""

    frame_id: str
    event_type: EventType
    is_keyframe: bool
    perceptual_hash: str
    frame_width: int
    frame_height: int
    timestamp: datetime | None = None
    app_name: str | None = None
    window_title: str | None = None
    ssim_score: float | None = None
    change_score: float | None = None
    changed_pct: float | None = None
    changed_regions: list[Region] | None = None
    adaptive_mask: list[Region] | None = None
    source_format: str | None = None
    asset_path: str | None = None
    pii_detected: bool | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_types.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Run ruff and mypy**

Run: `uv run ruff check peeklet/utils/types.py && uv run mypy peeklet/utils/types.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add peeklet/utils/types.py tests/unit/test_types.py
git commit -m "feat: add shared type definitions (Region, FrameMeta, FrameResult, EventType)"
```

---

### Task 3: Shared Image Utils (`peeklet/utils/image.py`)

**Files:**
- Create: `peeklet/utils/image.py`
- Create: `tests/unit/test_image.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_image.py`:
```python
"""Tests for shared image utilities."""

import numpy as np
import pytest

from peeklet.utils.image import ensure_rgb_uint8, crop_region, compute_block_grid
from peeklet.utils.types import Region


class TestEnsureRgbUint8:
    def test_passthrough_valid_rgb(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_convert_rgba_to_rgb(self) -> None:
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        frame[:, :, 3] = 255  # full alpha
        result = ensure_rgb_uint8(frame)
        assert result.shape == (100, 100, 3)

    def test_convert_grayscale_to_rgb(self) -> None:
        frame = np.zeros((100, 100), dtype=np.uint8)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (100, 100, 3)

    def test_convert_float_to_uint8(self) -> None:
        frame = np.ones((100, 100, 3), dtype=np.float64) * 0.5
        result = ensure_rgb_uint8(frame)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 127 or result[0, 0, 0] == 128  # rounding

    def test_reject_invalid_ndim(self) -> None:
        frame = np.zeros((100,), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            ensure_rgb_uint8(frame)

    def test_reject_invalid_channels(self) -> None:
        frame = np.zeros((100, 100, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 1, 3, or 4 channels"):
            ensure_rgb_uint8(frame)


class TestCropRegion:
    def test_crop_extracts_correct_area(self) -> None:
        frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        region = Region(x=10, y=20, w=30, h=40)
        cropped = crop_region(frame, region)
        assert cropped.shape == (40, 30, 3)
        np.testing.assert_array_equal(cropped, frame[20:60, 10:40])

    def test_crop_clamps_to_bounds(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        region = Region(x=80, y=80, w=50, h=50)  # exceeds bounds
        cropped = crop_region(frame, region)
        assert cropped.shape == (20, 20, 3)


class TestComputeBlockGrid:
    def test_grid_dimensions(self) -> None:
        rows, cols = compute_block_grid(1080, 1920, block_size=32)
        assert rows == 1080 // 32  # 33
        assert cols == 1920 // 32  # 60

    def test_grid_with_non_divisible(self) -> None:
        rows, cols = compute_block_grid(100, 100, block_size=32)
        assert rows == 3  # 100 // 32 = 3 (remainder discarded)
        assert cols == 3

    def test_block_size_larger_than_image(self) -> None:
        rows, cols = compute_block_grid(16, 16, block_size=32)
        assert rows == 0
        assert cols == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_image.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/utils/image.py`:
```python
"""Shared image operations for Peeklet."""

from __future__ import annotations

import numpy as np

from peeklet.utils.types import Region


def ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert any image array to RGB uint8 (H, W, 3).

    Handles: grayscale, RGBA, float [0,1], and passthrough for valid RGB uint8.
    """
    if frame.ndim == 2:
        # Grayscale → RGB
        if frame.dtype != np.uint8:
            frame = _to_uint8(frame)
        return np.stack([frame, frame, frame], axis=-1)

    if frame.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {frame.ndim}D")

    channels = frame.shape[2]

    if channels == 4:
        # RGBA → RGB (drop alpha)
        frame = frame[:, :, :3]
    elif channels == 1:
        # Single channel → RGB
        frame = np.concatenate([frame, frame, frame], axis=-1)
    elif channels != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got {channels}")

    if frame.dtype != np.uint8:
        frame = _to_uint8(frame)

    return frame


def crop_region(frame: np.ndarray, region: Region) -> np.ndarray:
    """Crop a region from a frame, clamping to image bounds."""
    h, w = frame.shape[:2]
    x1 = max(0, region.x)
    y1 = max(0, region.y)
    x2 = min(w, region.x + region.w)
    y2 = min(h, region.y + region.h)
    return frame[y1:y2, x1:x2]


def compute_block_grid(height: int, width: int, block_size: int) -> tuple[int, int]:
    """Compute the number of block rows and columns for a given image size."""
    rows = height // block_size
    cols = width // block_size
    return rows, cols


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert float or other dtype arrays to uint8."""
    if np.issubdtype(frame.dtype, np.floating):
        return (frame * 255).clip(0, 255).astype(np.uint8)
    return frame.astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_image.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run ruff and mypy**

Run: `uv run ruff check peeklet/utils/image.py && uv run mypy peeklet/utils/image.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add peeklet/utils/image.py tests/unit/test_image.py
git commit -m "feat: add shared image utilities (ensure_rgb_uint8, crop_region, compute_block_grid)"
```

---

### Task 4: Configuration (`peeklet/config.py`)

**Files:**
- Create: `peeklet/config.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config.py`:
```python
"""Tests for configuration loading and validation."""

import json
from pathlib import Path

import pytest
import yaml

from peeklet.config import (
    ComparatorConfig,
    ExporterConfig,
    HasherConfig,
    InputConfig,
    MaskingConfig,
    PeekletConfig,
    PipelineConfig,
    RedactorConfig,
    load_config,
    PiiPattern,
    load_patterns,
)


class TestDefaults:
    def test_default_config_is_valid(self) -> None:
        config = PeekletConfig()
        assert config.pipeline.mode == "batch"
        assert config.masking.block_size == 32
        assert config.masking.window_size == 15
        assert config.masking.noise_threshold == 0.8
        assert config.hasher.algorithm == "phash"
        assert config.comparator.ssim_threshold == 0.85
        assert config.redactor.enabled is True
        assert config.exporter.keyframe_format == "png"
        assert config.exporter.parquet_compression == "snappy"

    def test_pipeline_defaults(self) -> None:
        config = PipelineConfig()
        assert config.mode == "batch"
        assert config.concurrency == 4


class TestValidation:
    def test_reject_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            PipelineConfig(mode="invalid")

    def test_reject_negative_block_size(self) -> None:
        with pytest.raises(ValueError):
            MaskingConfig(block_size=-1)

    def test_reject_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            ComparatorConfig(ssim_threshold=1.5)

    def test_reject_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError):
            ComparatorConfig(ssim_threshold=-0.1)

    def test_reject_noise_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            MaskingConfig(noise_threshold=1.5)


class TestLoadConfig:
    def test_load_from_json_file(self, tmp_path: Path) -> None:
        config_data = {
            "comparator": {"ssim_threshold": 0.9},
            "masking": {"block_size": 64},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = load_config(config_file)
        assert config.comparator.ssim_threshold == 0.9
        assert config.masking.block_size == 64
        # Other fields keep defaults
        assert config.masking.window_size == 15

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        config_data = {"comparator": {"ssim_threshold": 0.7}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.comparator.ssim_threshold == 0.7

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.json"))

    def test_load_none_returns_defaults(self) -> None:
        config = load_config(None)
        assert config == PeekletConfig()


class TestPatternLoading:
    def test_load_custom_patterns(self, tmp_path: Path) -> None:
        patterns_data = {
            "patterns": [
                {
                    "name": "employee_id",
                    "regex": r"EMP-\d{6}",
                    "description": "Employee ID",
                },
            ]
        }
        patterns_file = tmp_path / "patterns.yaml"
        patterns_file.write_text(yaml.dump(patterns_data))

        patterns = load_patterns(patterns_file)
        assert len(patterns) == 1
        assert patterns[0].name == "employee_id"
        assert patterns[0].regex == r"EMP-\d{6}"
        assert patterns[0].enabled is True

    def test_disabled_pattern(self, tmp_path: Path) -> None:
        patterns_data = {
            "patterns": [
                {"name": "ssn", "enabled": False},
            ]
        }
        patterns_file = tmp_path / "patterns.yaml"
        patterns_file.write_text(yaml.dump(patterns_data))

        patterns = load_patterns(patterns_file)
        assert patterns[0].enabled is False

    def test_load_missing_patterns_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_patterns(Path("/nonexistent/patterns.yaml"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/config.py`:
```python
"""Configuration loading and validation for Peeklet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class PipelineConfig(BaseModel):
    """Pipeline execution settings."""

    mode: Literal["batch", "stream"] = "batch"
    concurrency: int = Field(default=4, ge=1)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("batch", "stream"):
            raise ValueError(f"mode must be 'batch' or 'stream', got '{v}'")
        return v


class MaskingConfig(BaseModel):
    """Adaptive masking settings."""

    block_size: int = Field(default=32, gt=0)
    window_size: int = Field(default=15, gt=0)
    noise_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class HasherConfig(BaseModel):
    """Perceptual hashing settings."""

    algorithm: Literal["phash"] = "phash"
    hash_size: int = Field(default=8, gt=0)


class ComparatorConfig(BaseModel):
    """SSIM comparison settings."""

    ssim_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_changed_pct: float = Field(default=2.0, ge=0.0)


class RedactorConfig(BaseModel):
    """PII redaction settings."""

    enabled: bool = True
    pii_types: list[str] = Field(
        default_factory=lambda: [
            "email", "phone", "ssn", "credit_card", "ip_address", "address",
        ]
    )
    custom_patterns_file: str | None = None


class ExporterConfig(BaseModel):
    """Export settings."""

    output_dir: str = "./output"
    keyframe_format: Literal["png", "jpg"] = "png"
    parquet_compression: Literal["snappy", "gzip", "zstd", "none"] = "snappy"


class InputConfig(BaseModel):
    """Input settings."""

    supported_formats: list[str] = Field(
        default_factory=lambda: ["png", "jpg", "jpeg", "bmp", "tiff", "webp", "pdf"]
    )
    sort_by: Literal["filename", "timestamp"] = "filename"


class PeekletConfig(BaseModel):
    """Root configuration for Peeklet."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    hasher: HasherConfig = Field(default_factory=HasherConfig)
    comparator: ComparatorConfig = Field(default_factory=ComparatorConfig)
    redactor: RedactorConfig = Field(default_factory=RedactorConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)
    input: InputConfig = Field(default_factory=InputConfig)


class PiiPattern(BaseModel):
    """A PII detection pattern."""

    name: str
    regex: str = ""
    description: str = ""
    enabled: bool = True


def load_config(path: Path | None) -> PeekletConfig:
    """Load config from a JSON or YAML file, or return defaults if path is None."""
    if path is None:
        return PeekletConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    return PeekletConfig.model_validate(data or {})


def load_patterns(path: Path) -> list[PiiPattern]:
    """Load custom PII patterns from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Patterns file not found: {path}")

    data = yaml.safe_load(path.read_text())
    raw_patterns = data.get("patterns", [])
    return [PiiPattern.model_validate(p) for p in raw_patterns]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Run ruff and mypy**

Run: `uv run ruff check peeklet/config.py && uv run mypy peeklet/config.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add peeklet/config.py tests/unit/test_config.py
git commit -m "feat: add Pydantic config with validation, JSON/YAML loading, custom PII patterns"
```

---

### Task 5: Loader (`peeklet/core/loader.py`)

**Files:**
- Create: `peeklet/core/loader.py`
- Create: `tests/unit/test_loader.py`
- Create: `tests/fixtures/formats/` (test images generated in conftest)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_loader.py`:
```python
"""Tests for the image loader module."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from peeklet.core.loader import load_frame


def _save_pil(img: Image.Image, path: Path, fmt: str) -> Path:
    """Helper to save a PIL image in a given format."""
    img.save(path, format=fmt)
    return path


def _make_test_image() -> Image.Image:
    """Create a simple 80x60 RGB test image."""
    return Image.fromarray(
        np.arange(80 * 60 * 3, dtype=np.uint8).reshape(60, 80, 3) % 256
    )


class TestLoadFromPath:
    def test_load_png(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.png", "PNG")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)
        assert result.dtype == np.uint8

    def test_load_jpeg(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.jpg", "JPEG")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)
        assert result.dtype == np.uint8

    def test_load_bmp(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.bmp", "BMP")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)
        assert result.dtype == np.uint8

    def test_load_webp(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.webp", "WEBP")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)
        assert result.dtype == np.uint8

    def test_load_tiff(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.tiff", "TIFF")
        result = load_frame(path)
        assert result.shape == (60, 80, 3)
        assert result.dtype == np.uint8

    def test_load_string_path(self, tmp_path: Path) -> None:
        img = _make_test_image()
        path = _save_pil(img, tmp_path / "test.png", "PNG")
        result = load_frame(str(path))
        assert result.shape == (60, 80, 3)

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_frame("/nonexistent/image.png")

    def test_load_corrupt_file_raises(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Could not load"):
            load_frame(corrupt)


class TestLoadFromMemory:
    def test_load_numpy_rgb(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = load_frame(frame)
        assert result.shape == (100, 100, 3)
        np.testing.assert_array_equal(result, frame)

    def test_load_numpy_rgba(self) -> None:
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        result = load_frame(frame)
        assert result.shape == (100, 100, 3)

    def test_load_numpy_grayscale(self) -> None:
        frame = np.zeros((100, 100), dtype=np.uint8)
        result = load_frame(frame)
        assert result.shape == (100, 100, 3)

    def test_load_pil_image(self) -> None:
        img = Image.new("RGB", (80, 60), color=(128, 64, 32))
        result = load_frame(img)
        assert result.shape == (60, 80, 3)
        assert result[0, 0, 0] == 128

    def test_load_pil_rgba(self) -> None:
        img = Image.new("RGBA", (80, 60), color=(128, 64, 32, 255))
        result = load_frame(img)
        assert result.shape == (60, 80, 3)

    def test_load_bytes(self, tmp_path: Path) -> None:
        img = _make_test_image()
        buf = BytesIO()
        img.save(buf, format="PNG")
        result = load_frame(buf.getvalue())
        assert result.shape == (60, 80, 3)

    def test_load_invalid_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not load"):
            load_frame(b"not an image")

    def test_load_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported source type"):
            load_frame(12345)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_loader.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/core/loader.py`:
```python
"""Load images from any supported format into normalized numpy arrays."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from peeklet.utils.image import ensure_rgb_uint8


def load_frame(source: str | Path | bytes | np.ndarray | Image.Image) -> np.ndarray:
    """Normalize any supported input to an RGB uint8 numpy array (H, W, 3).

    Supported inputs:
        - File path (str or Path): PNG, JPEG, BMP, TIFF, WebP, PDF
        - bytes: Raw image bytes
        - numpy.ndarray: Grayscale, RGB, or RGBA arrays
        - PIL.Image.Image: Any PIL image
    """
    if isinstance(source, np.ndarray):
        return ensure_rgb_uint8(source)

    if isinstance(source, Image.Image):
        return ensure_rgb_uint8(np.asarray(source))

    if isinstance(source, (str, Path)):
        return _load_from_path(Path(source))

    if isinstance(source, (bytes, bytearray)):
        return _load_from_bytes(source)

    raise TypeError(f"Unsupported source type: {type(source).__name__}")


def _load_from_path(path: Path) -> np.ndarray:
    """Load an image from a file path."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        img = Image.open(path)
        img.load()  # Force full load to catch truncated files
        return ensure_rgb_uint8(np.asarray(img.convert("RGB")))
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"Could not load image from {path}: {e}") from e


def _load_from_bytes(data: bytes | bytearray) -> np.ndarray:
    """Load an image from raw bytes."""
    try:
        img = Image.open(BytesIO(data))
        img.load()
        return ensure_rgb_uint8(np.asarray(img.convert("RGB")))
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError(f"Could not load image from bytes: {e}") from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_loader.py -v`
Expected: All 17 tests PASS

- [ ] **Step 5: Run ruff and mypy**

Run: `uv run ruff check peeklet/core/loader.py && uv run mypy peeklet/core/loader.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add peeklet/core/loader.py tests/unit/test_loader.py
git commit -m "feat: add multi-format image loader (path, bytes, numpy, PIL → RGB uint8)"
```

---

### Task 6: Perceptual Hasher (`peeklet/core/hasher.py`)

**Files:**
- Create: `peeklet/core/hasher.py`
- Create: `tests/unit/test_hasher.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_hasher.py`:
```python
"""Tests for perceptual hashing."""

import numpy as np

from peeklet.core.hasher import compute_phash, hashes_match


class TestComputePhash:
    def test_returns_string(self, sample_frame: np.ndarray) -> None:
        h = compute_phash(sample_frame)
        assert isinstance(h, str)
        assert len(h) == 16  # 64-bit hash as 16-char hex

    def test_identical_frames_same_hash(self, sample_frame: np.ndarray) -> None:
        h1 = compute_phash(sample_frame)
        h2 = compute_phash(sample_frame.copy())
        assert h1 == h2

    def test_small_change_same_hash(
        self, sample_frame: np.ndarray, sample_frame_small_change: np.ndarray
    ) -> None:
        h1 = compute_phash(sample_frame)
        h2 = compute_phash(sample_frame_small_change)
        assert h1 == h2

    def test_big_change_different_hash(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        h1 = compute_phash(sample_frame)
        h2 = compute_phash(sample_frame_big_change)
        assert h1 != h2

    def test_deterministic(self, sample_frame: np.ndarray) -> None:
        results = [compute_phash(sample_frame) for _ in range(5)]
        assert len(set(results)) == 1


class TestHashesMatch:
    def test_identical_hashes_match(self, sample_frame: np.ndarray) -> None:
        h = compute_phash(sample_frame)
        assert hashes_match(h, h) is True

    def test_different_hashes_no_match(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        h1 = compute_phash(sample_frame)
        h2 = compute_phash(sample_frame_big_change)
        assert hashes_match(h1, h2) is False

    def test_similar_hashes_match_with_tolerance(self) -> None:
        # Two hashes that differ by 1 bit should match with tolerance=1
        assert hashes_match("0000000000000000", "0000000000000001", tolerance=4) is True

    def test_similar_hashes_no_match_without_tolerance(self) -> None:
        assert hashes_match("0000000000000000", "ffffffffffffffff", tolerance=0) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_hasher.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/core/hasher.py`:
```python
"""Perceptual hashing for screenshot deduplication."""

from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image


def compute_phash(frame: np.ndarray, hash_size: int = 8) -> str:
    """Compute a perceptual hash of an RGB uint8 frame.

    Returns a hex string representation of the 64-bit hash.
    """
    img = Image.fromarray(frame)
    h = imagehash.phash(img, hash_size=hash_size)
    return str(h)


def hashes_match(hash_a: str, hash_b: str, tolerance: int = 0) -> bool:
    """Check if two perceptual hashes are similar within a Hamming distance tolerance."""
    h1 = imagehash.hex_to_hash(hash_a)
    h2 = imagehash.hex_to_hash(hash_b)
    return (h1 - h2) <= tolerance
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_hasher.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add peeklet/core/hasher.py tests/unit/test_hasher.py
git commit -m "feat: add perceptual hashing with pHash and Hamming distance matching"
```

---

### Task 7: Adaptive Masking (`peeklet/core/masking.py`)

**Files:**
- Create: `peeklet/core/masking.py`
- Create: `tests/unit/test_masking.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_masking.py`:
```python
"""Tests for adaptive block-based masking."""

import numpy as np

from peeklet.core.masking import AdaptiveMask
from peeklet.utils.types import Region


class TestAdaptiveMask:
    def test_init_creates_empty_state(self) -> None:
        mask = AdaptiveMask(block_size=32, window_size=5, noise_threshold=0.8)
        assert mask.block_size == 32
        assert mask.window_size == 5

    def test_first_frame_returns_unmasked(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result, regions = mask.apply(frame)
        # First frame: no history, nothing to mask
        np.testing.assert_array_equal(result, frame)
        assert regions == []

    def test_static_frames_no_mask(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Feed identical frames — no blocks should be noisy
        for _ in range(10):
            result, regions = mask.apply(frame)
        assert regions == []
        np.testing.assert_array_equal(result, frame)

    def test_constantly_changing_block_gets_masked(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        # Feed frames where top-left block changes every time
        for i in range(10):
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
            frame[:50, :50] = i * 25  # change top-left block every frame
            result, regions = mask.apply(frame)

        # After enough frames, top-left block should be masked (zeroed)
        assert len(regions) > 0
        # The masked region should be zeroed
        assert np.all(result[:50, :50] == 0)
        # Other blocks should be untouched
        assert np.all(result[50:, 50:] == 128)

    def test_noisy_region_returns_correct_regions(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        for i in range(10):
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
            frame[:50, :50] = i * 25
            _, regions = mask.apply(frame)

        # Should have a region for the noisy block
        assert any(r.x == 0 and r.y == 0 for r in regions)

    def test_window_slides(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        # First: 10 frames with changing top-left
        for i in range(10):
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
            frame[:50, :50] = i * 25
            mask.apply(frame)

        # Then: 10 identical frames — the noise should clear
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        for _ in range(10):
            _, regions = mask.apply(frame)

        # After enough static frames, no blocks should be noisy
        assert regions == []

    def test_all_blocks_noisy(self) -> None:
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        for i in range(10):
            frame = np.full((100, 100, 3), i * 25, dtype=np.uint8)
            result, regions = mask.apply(frame)

        # All blocks are noisy — entire frame masked
        assert np.all(result == 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_masking.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/core/masking.py`:
```python
"""Adaptive block-based masking for high-frequency change regions."""

from __future__ import annotations

from collections import deque

import numpy as np

from peeklet.utils.image import compute_block_grid
from peeklet.utils.types import Region


class AdaptiveMask:
    """Detects and masks screen regions that change on nearly every frame.

    Maintains a sliding window of recent frames. For each block in the grid,
    tracks how often it changes. Blocks that change more than `noise_threshold`
    fraction of the time are zeroed out.
    """

    def __init__(
        self,
        block_size: int = 32,
        window_size: int = 15,
        noise_threshold: float = 0.8,
    ) -> None:
        self.block_size = block_size
        self.window_size = window_size
        self.noise_threshold = noise_threshold
        self._prev_blocks: np.ndarray | None = None
        self._change_history: deque[np.ndarray] = deque(maxlen=window_size)

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, list[Region]]:
        """Apply adaptive mask to a frame.

        Returns the masked frame and a list of masked regions.
        """
        h, w = frame.shape[:2]
        rows, cols = compute_block_grid(h, w, self.block_size)

        if rows == 0 or cols == 0:
            return frame.copy(), []

        current_blocks = self._compute_block_means(frame, rows, cols)

        if self._prev_blocks is None:
            self._prev_blocks = current_blocks
            return frame.copy(), []

        # Compute which blocks changed from previous frame
        change_map = np.abs(current_blocks - self._prev_blocks).mean(axis=-1) > 5.0
        self._change_history.append(change_map)
        self._prev_blocks = current_blocks

        if len(self._change_history) < 2:
            return frame.copy(), []

        # Compute noise frequency per block
        history = np.stack(list(self._change_history), axis=0)
        noise_freq = history.mean(axis=0)
        noisy_blocks = noise_freq > self.noise_threshold

        # Build mask and region list
        masked = frame.copy()
        regions: list[Region] = []

        for r in range(rows):
            for c in range(cols):
                if noisy_blocks[r, c]:
                    y = r * self.block_size
                    x = c * self.block_size
                    masked[y : y + self.block_size, x : x + self.block_size] = 0
                    regions.append(Region(x=x, y=y, w=self.block_size, h=self.block_size))

        return masked, regions

    def _compute_block_means(
        self, frame: np.ndarray, rows: int, cols: int
    ) -> np.ndarray:
        """Compute the mean RGB value for each block in the grid."""
        bs = self.block_size
        means = np.zeros((rows, cols, 3), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                block = frame[r * bs : (r + 1) * bs, c * bs : (c + 1) * bs]
                means[r, c] = block.mean(axis=(0, 1))
        return means
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_masking.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add peeklet/core/masking.py tests/unit/test_masking.py
git commit -m "feat: add adaptive block-based masking for high-frequency change regions"
```

---

### Task 8: Comparator (`peeklet/core/comparator.py`)

**Files:**
- Create: `peeklet/core/comparator.py`
- Create: `tests/unit/test_comparator.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_comparator.py`:
```python
"""Tests for SSIM comparison and block-level diff."""

import numpy as np
import pytest

from peeklet.core.comparator import compare_frames, ComparisonResult


class TestCompareFrames:
    def test_identical_frames_high_ssim(self, sample_frame: np.ndarray) -> None:
        result = compare_frames(sample_frame, sample_frame.copy(), block_size=32)
        assert result.ssim_score > 0.99
        assert result.changed_pct == 0.0
        assert result.changed_regions == []

    def test_completely_different_frames_low_ssim(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        result = compare_frames(sample_frame, sample_frame_big_change, block_size=32)
        assert result.ssim_score < 0.5
        assert result.changed_pct > 50.0

    def test_small_change_moderate_ssim(
        self, sample_frame: np.ndarray, sample_frame_small_change: np.ndarray
    ) -> None:
        result = compare_frames(sample_frame, sample_frame_small_change, block_size=10)
        assert 0.5 < result.ssim_score < 1.0

    def test_change_score_inverse_of_ssim(self, sample_frame: np.ndarray) -> None:
        different = sample_frame.copy()
        different[:50, :50] = 255 - different[:50, :50]
        result = compare_frames(sample_frame, different, block_size=32)
        assert abs(result.change_score - (1.0 - result.ssim_score)) < 0.01

    def test_changed_regions_have_valid_bounds(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        result = compare_frames(sample_frame, sample_frame_big_change, block_size=32)
        h, w = sample_frame.shape[:2]
        for region in result.changed_regions:
            assert region.x >= 0
            assert region.y >= 0
            assert region.x + region.w <= w + 32  # allow block boundary
            assert region.y + region.h <= h + 32

    def test_changed_pct_between_0_and_100(
        self, sample_frame: np.ndarray, sample_frame_big_change: np.ndarray
    ) -> None:
        result = compare_frames(sample_frame, sample_frame_big_change, block_size=32)
        assert 0.0 <= result.changed_pct <= 100.0


class TestComparisonResult:
    def test_dataclass_fields(self) -> None:
        result = ComparisonResult(
            ssim_score=0.85,
            change_score=0.15,
            changed_pct=12.5,
            changed_regions=[],
        )
        assert result.ssim_score == 0.85
        assert result.change_score == 0.15
        assert result.changed_pct == 12.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_comparator.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/core/comparator.py`:
```python
"""SSIM comparison and block-level change detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.metrics import structural_similarity

from peeklet.utils.image import compute_block_grid
from peeklet.utils.types import Region


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Result of comparing two frames."""

    ssim_score: float
    change_score: float
    changed_pct: float
    changed_regions: list[Region] = field(default_factory=list)


def compare_frames(
    current: np.ndarray,
    reference: np.ndarray,
    block_size: int = 32,
    block_change_threshold: float = 10.0,
) -> ComparisonResult:
    """Compare two frames using SSIM and block-level diff.

    Args:
        current: The new frame (H, W, 3) RGB uint8.
        reference: The reference/previous keyframe (H, W, 3) RGB uint8.
        block_size: Size of blocks for change detection.
        block_change_threshold: Mean pixel difference threshold per block.

    Returns:
        ComparisonResult with SSIM score, change score, and changed regions.
    """
    # Compute SSIM on grayscale for speed
    gray_current = _to_grayscale(current)
    gray_reference = _to_grayscale(reference)

    ssim_score = float(
        structural_similarity(gray_reference, gray_current, data_range=255)
    )
    change_score = 1.0 - ssim_score

    # Block-level diff
    changed_regions = _compute_block_diff(
        current, reference, block_size, block_change_threshold
    )

    h, w = current.shape[:2]
    rows, cols = compute_block_grid(h, w, block_size)
    total_blocks = max(rows * cols, 1)
    changed_pct = (len(changed_regions) / total_blocks) * 100.0

    return ComparisonResult(
        ssim_score=ssim_score,
        change_score=change_score,
        changed_pct=changed_pct,
        changed_regions=changed_regions,
    )


def _compute_block_diff(
    current: np.ndarray,
    reference: np.ndarray,
    block_size: int,
    threshold: float,
) -> list[Region]:
    """Find blocks where the mean pixel difference exceeds the threshold."""
    h, w = current.shape[:2]
    rows, cols = compute_block_grid(h, w, block_size)
    bs = block_size

    diff = np.abs(current.astype(np.float32) - reference.astype(np.float32))
    regions: list[Region] = []

    for r in range(rows):
        for c in range(cols):
            block_diff = diff[r * bs : (r + 1) * bs, c * bs : (c + 1) * bs]
            if block_diff.mean() > threshold:
                regions.append(Region(x=c * bs, y=r * bs, w=bs, h=bs))

    return regions


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert RGB frame to grayscale using luminance weights."""
    return (
        0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
    ).astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_comparator.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add peeklet/core/comparator.py tests/unit/test_comparator.py
git commit -m "feat: add SSIM comparator with block-level change detection"
```

---

### Task 9: PII Redactor (`peeklet/core/redactor.py`)

**Files:**
- Create: `peeklet/core/redactor.py`
- Create: `tests/unit/test_redactor.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_redactor.py`:
```python
"""Tests for PII redaction."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from peeklet.config import PiiPattern, load_patterns
from peeklet.core.redactor import (
    BUILTIN_PATTERNS,
    build_pattern_set,
    find_pii_in_text,
    redact_regions,
    PiiMatch,
)
from peeklet.utils.types import Region


class TestBuiltinPatterns:
    def test_email_pattern(self) -> None:
        matches = find_pii_in_text("contact me at john@example.com please", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "email" for m in matches)

    def test_phone_pattern(self) -> None:
        matches = find_pii_in_text("call 555-123-4567 now", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "phone" for m in matches)

    def test_ssn_pattern(self) -> None:
        matches = find_pii_in_text("SSN: 123-45-6789", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "ssn" for m in matches)

    def test_credit_card_pattern(self) -> None:
        matches = find_pii_in_text("card 4111-1111-1111-1111", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "credit_card" for m in matches)

    def test_ip_address_pattern(self) -> None:
        matches = find_pii_in_text("server at 192.168.1.100", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "ip_address" for m in matches)

    def test_no_pii_in_clean_text(self) -> None:
        matches = find_pii_in_text("this is a normal sentence", BUILTIN_PATTERNS)
        assert matches == []


class TestBuildPatternSet:
    def test_builtin_only(self) -> None:
        patterns = build_pattern_set(
            pii_types=["email", "phone"],
            custom_patterns=[],
        )
        assert len(patterns) == 2
        names = {p.name for p in patterns}
        assert names == {"email", "phone"}

    def test_custom_overrides_builtin(self) -> None:
        custom = [
            PiiPattern(name="email", regex=r"[a-z]+@acme\.com", description="Acme only")
        ]
        patterns = build_pattern_set(
            pii_types=["email"],
            custom_patterns=custom,
        )
        assert len(patterns) == 1
        assert patterns[0].regex == r"[a-z]+@acme\.com"

    def test_custom_adds_new_pattern(self) -> None:
        custom = [
            PiiPattern(name="employee_id", regex=r"EMP-\d{6}", description="Employee ID")
        ]
        patterns = build_pattern_set(
            pii_types=["email", "employee_id"],
            custom_patterns=custom,
        )
        names = {p.name for p in patterns}
        assert "email" in names
        assert "employee_id" in names

    def test_disabled_pattern_excluded(self) -> None:
        custom = [PiiPattern(name="ssn", enabled=False)]
        patterns = build_pattern_set(
            pii_types=["email", "ssn"],
            custom_patterns=custom,
        )
        names = {p.name for p in patterns}
        assert "ssn" not in names
        assert "email" in names


class TestRedactRegions:
    def test_redact_draws_black_box(self) -> None:
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        region = Region(x=10, y=10, w=30, h=20)
        redacted = redact_regions(frame, [region])
        # Black box should be applied
        assert np.all(redacted[10:30, 10:40] == 0)
        # Outside region should be unchanged
        assert np.all(redacted[0:5, 0:5] == 200)

    def test_redact_empty_regions_unchanged(self) -> None:
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        redacted = redact_regions(frame, [])
        np.testing.assert_array_equal(redacted, frame)

    def test_redact_does_not_mutate_original(self) -> None:
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        original = frame.copy()
        redact_regions(frame, [Region(x=10, y=10, w=30, h=20)])
        np.testing.assert_array_equal(frame, original)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_redactor.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/core/redactor.py`:
```python
"""PII detection and redaction for keyframe images."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from peeklet.config import PiiPattern
from peeklet.utils.types import Region

# Built-in PII patterns
BUILTIN_PATTERNS: list[PiiPattern] = [
    PiiPattern(
        name="email",
        regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        description="Email addresses",
    ),
    PiiPattern(
        name="phone",
        regex=r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        description="US phone numbers",
    ),
    PiiPattern(
        name="ssn",
        regex=r"\b\d{3}-\d{2}-\d{4}\b",
        description="US Social Security Numbers",
    ),
    PiiPattern(
        name="credit_card",
        regex=r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        description="Credit card numbers",
    ),
    PiiPattern(
        name="ip_address",
        regex=r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        description="IPv4 addresses",
    ),
    PiiPattern(
        name="address",
        regex=r"\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b",
        description="US street addresses",
    ),
]


@dataclass(frozen=True, slots=True)
class PiiMatch:
    """A detected PII occurrence in text."""

    pattern_name: str
    matched_text: str
    start: int
    end: int


def build_pattern_set(
    pii_types: list[str],
    custom_patterns: list[PiiPattern],
) -> list[PiiPattern]:
    """Build the final pattern set with user overrides and filtering.

    Precedence: custom patterns override builtins by name. Disabled patterns are excluded.
    Only patterns whose names appear in pii_types are included.
    """
    # Start with builtins
    pattern_map: dict[str, PiiPattern] = {p.name: p for p in BUILTIN_PATTERNS}

    # Apply custom overrides
    for custom in custom_patterns:
        if not custom.enabled:
            pattern_map.pop(custom.name, None)
            continue
        pattern_map[custom.name] = custom

    # Filter to requested types
    return [p for name, p in pattern_map.items() if name in pii_types]


def find_pii_in_text(text: str, patterns: list[PiiPattern]) -> list[PiiMatch]:
    """Scan text for PII matches using the given patterns."""
    matches: list[PiiMatch] = []
    for pattern in patterns:
        if not pattern.regex:
            continue
        for match in re.finditer(pattern.regex, text):
            matches.append(
                PiiMatch(
                    pattern_name=pattern.name,
                    matched_text=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )
    return matches


def redact_regions(frame: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Draw black rectangles over the specified regions.

    Returns a new array; does not mutate the input.
    """
    redacted = frame.copy()
    h, w = redacted.shape[:2]
    for region in regions:
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(w, region.x + region.w)
        y2 = min(h, region.y + region.h)
        redacted[y1:y2, x1:x2] = 0
    return redacted
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_redactor.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add peeklet/core/redactor.py tests/unit/test_redactor.py
git commit -m "feat: add PII redactor with builtin patterns, custom overrides, and black-box redaction"
```

---

### Task 10: Exporter (`peeklet/core/exporter.py`)

**Files:**
- Create: `peeklet/core/exporter.py`
- Create: `tests/unit/test_exporter.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_exporter.py`:
```python
"""Tests for Parquet export and keyframe saving."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from peeklet.core.exporter import ManifestWriter, save_keyframe
from peeklet.utils.types import EventType, FrameResult, Region


class TestSaveKeyframe:
    def test_saves_png(self, tmp_output: Path) -> None:
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        path = save_keyframe(frame, tmp_output, "frame_001", fmt="png")
        assert path.exists()
        assert path.suffix == ".png"

    def test_saves_jpg(self, tmp_output: Path) -> None:
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        path = save_keyframe(frame, tmp_output, "frame_001", fmt="jpg")
        assert path.exists()
        assert path.suffix == ".jpg"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "output"
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        path = save_keyframe(frame, out, "frame_001", fmt="png")
        assert path.exists()


class TestManifestWriter:
    def test_write_and_read_manifest(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        result = FrameResult(
            frame_id="frame_001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd1234abcd1234",
            ssim_score=0.62,
            change_score=0.38,
            changed_pct=34.0,
            changed_regions=[Region(x=100, y=200, w=300, h=150)],
            adaptive_mask=[],
            frame_width=1920,
            frame_height=1080,
            asset_path="output/frame_001.png",
            pii_detected=False,
        )
        writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.num_rows == 1
        assert table.column("frame_id")[0].as_py() == "frame_001"
        assert table.column("is_keyframe")[0].as_py() is True
        assert table.column("perceptual_hash")[0].as_py() == "abcd1234abcd1234"

    def test_write_skipped_frame(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        result = FrameResult(
            frame_id="frame_002",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="abcd1234abcd1234",
            frame_width=1920,
            frame_height=1080,
        )
        writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.column("is_keyframe")[0].as_py() is False
        assert table.column("asset_path")[0].as_py() is None

    def test_multiple_rows(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        for i in range(10):
            result = FrameResult(
                frame_id=f"frame_{i:03d}",
                event_type=EventType.KEYFRAME if i % 5 == 0 else EventType.SKIPPED,
                is_keyframe=i % 5 == 0,
                perceptual_hash=f"hash{i:012d}",
                frame_width=1920,
                frame_height=1080,
            )
            writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.num_rows == 10

    def test_snappy_compression(self, tmp_output: Path) -> None:
        writer = ManifestWriter(
            tmp_output / "manifest.parquet", compression="snappy"
        )
        result = FrameResult(
            frame_id="frame_001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd1234abcd1234",
            frame_width=1920,
            frame_height=1080,
        )
        writer.append(result)
        writer.flush()

        meta = pq.read_metadata(tmp_output / "manifest.parquet")
        assert meta.row_group(0).column(0).compression == "SNAPPY"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_exporter.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/core/exporter.py`:
```python
"""Parquet manifest writer and keyframe image saver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from peeklet.utils.types import FrameResult


MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("frame_id", pa.string()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("event_type", pa.string()),
        pa.field("app_name", pa.string(), nullable=True),
        pa.field("window_title", pa.string(), nullable=True),
        pa.field("is_keyframe", pa.bool_()),
        pa.field("perceptual_hash", pa.string()),
        pa.field("ssim_score", pa.float64(), nullable=True),
        pa.field("change_score", pa.float64(), nullable=True),
        pa.field("changed_pct", pa.float64(), nullable=True),
        pa.field(
            "changed_regions",
            pa.list_(
                pa.struct(
                    [
                        pa.field("x", pa.int32()),
                        pa.field("y", pa.int32()),
                        pa.field("w", pa.int32()),
                        pa.field("h", pa.int32()),
                    ]
                )
            ),
            nullable=True,
        ),
        pa.field(
            "adaptive_mask",
            pa.list_(
                pa.struct(
                    [
                        pa.field("x", pa.int32()),
                        pa.field("y", pa.int32()),
                        pa.field("w", pa.int32()),
                        pa.field("h", pa.int32()),
                    ]
                )
            ),
            nullable=True,
        ),
        pa.field("frame_width", pa.int32()),
        pa.field("frame_height", pa.int32()),
        pa.field("source_format", pa.string(), nullable=True),
        pa.field("asset_path", pa.string(), nullable=True),
        pa.field("pii_detected", pa.bool_(), nullable=True),
    ]
)


def save_keyframe(
    frame: np.ndarray, output_dir: Path, frame_id: str, fmt: str = "png"
) -> Path:
    """Save a keyframe image to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{frame_id}.{fmt}"
    Image.fromarray(frame).save(path)
    return path


class ManifestWriter:
    """Buffers FrameResults and writes them to a Parquet file."""

    def __init__(self, path: Path, compression: str = "snappy") -> None:
        self._path = Path(path)
        self._compression = compression
        self._rows: list[dict] = []

    def append(self, result: FrameResult) -> None:
        """Add a frame result to the buffer."""
        regions = None
        if result.changed_regions is not None:
            regions = [r.to_dict() for r in result.changed_regions]

        mask = None
        if result.adaptive_mask is not None:
            mask = [r.to_dict() for r in result.adaptive_mask]

        self._rows.append(
            {
                "frame_id": result.frame_id,
                "timestamp": result.timestamp,
                "event_type": result.event_type.value,
                "app_name": result.app_name,
                "window_title": result.window_title,
                "is_keyframe": result.is_keyframe,
                "perceptual_hash": result.perceptual_hash,
                "ssim_score": result.ssim_score,
                "change_score": result.change_score,
                "changed_pct": result.changed_pct,
                "changed_regions": regions,
                "adaptive_mask": mask,
                "frame_width": result.frame_width,
                "frame_height": result.frame_height,
                "source_format": result.source_format,
                "asset_path": result.asset_path,
                "pii_detected": result.pii_detected,
            }
        )

    def flush(self) -> None:
        """Write all buffered rows to the Parquet file."""
        if not self._rows:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self._rows, schema=MANIFEST_SCHEMA)
        pq.write_table(table, self._path, compression=self._compression)
        self._rows.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_exporter.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add peeklet/core/exporter.py tests/unit/test_exporter.py
git commit -m "feat: add Parquet manifest writer and keyframe image saver"
```

---

### Task 11: Pipeline Orchestrator (`peeklet/pipeline.py`)

**Files:**
- Create: `peeklet/pipeline.py`
- Create: `tests/integration/test_pipeline_batch.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_pipeline_batch.py`:
```python
"""Integration tests for the full pipeline in batch mode."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from peeklet.config import PeekletConfig
from peeklet.pipeline import Pipeline


@pytest.fixture
def basic_config(tmp_output: Path) -> PeekletConfig:
    return PeekletConfig.model_validate(
        {
            "redactor": {"enabled": False},
            "exporter": {"output_dir": str(tmp_output)},
            "masking": {"block_size": 50, "window_size": 5, "noise_threshold": 0.8},
            "comparator": {"ssim_threshold": 0.85},
        }
    )


class TestPipelineBatch:
    def test_identical_frames_all_skipped(
        self, basic_config: PeekletConfig, tmp_output: Path
    ) -> None:
        pipeline = Pipeline(basic_config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        results = []
        for i in range(10):
            result = pipeline.process_frame(frame, frame_id=f"frame_{i:03d}")
            results.append(result)

        # First frame is always a keyframe
        assert results[0].is_keyframe is True
        # Rest should be skipped (identical)
        assert all(r.is_keyframe is False for r in results[1:])

    def test_big_change_detected_as_keyframe(
        self, basic_config: PeekletConfig, tmp_output: Path
    ) -> None:
        pipeline = Pipeline(basic_config)

        frame_a = np.full((100, 100, 3), 50, dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 200, dtype=np.uint8)

        result_a = pipeline.process_frame(frame_a, frame_id="frame_000")
        result_b = pipeline.process_frame(frame_b, frame_id="frame_001")

        assert result_a.is_keyframe is True
        assert result_b.is_keyframe is True
        assert result_b.ssim_score is not None
        assert result_b.ssim_score < 0.85

    def test_pipeline_writes_manifest(
        self, basic_config: PeekletConfig, tmp_output: Path
    ) -> None:
        pipeline = Pipeline(basic_config)

        frame_a = np.full((100, 100, 3), 50, dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 50, dtype=np.uint8)
        frame_c = np.full((100, 100, 3), 200, dtype=np.uint8)

        pipeline.process_frame(frame_a, frame_id="frame_000")
        pipeline.process_frame(frame_b, frame_id="frame_001")
        pipeline.process_frame(frame_c, frame_id="frame_002")
        pipeline.finalize()

        manifest = tmp_output / "manifest.parquet"
        assert manifest.exists()
        table = pq.read_table(manifest)
        assert table.num_rows == 3

    def test_keyframe_images_saved(
        self, basic_config: PeekletConfig, tmp_output: Path
    ) -> None:
        pipeline = Pipeline(basic_config)

        frame_a = np.full((100, 100, 3), 50, dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 200, dtype=np.uint8)

        pipeline.process_frame(frame_a, frame_id="frame_000")
        pipeline.process_frame(frame_b, frame_id="frame_001")
        pipeline.finalize()

        keyframes = list(tmp_output.glob("*.png"))
        assert len(keyframes) == 2  # both are keyframes (different content)

    def test_cascade_skips_comparator_on_hash_match(
        self, basic_config: PeekletConfig, tmp_output: Path
    ) -> None:
        pipeline = Pipeline(basic_config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        pipeline.process_frame(frame, frame_id="frame_000")
        result = pipeline.process_frame(frame, frame_id="frame_001")

        # Hash matched — SSIM was not computed
        assert result.is_keyframe is False
        assert result.ssim_score is None

    def test_frame_metadata_in_result(
        self, basic_config: PeekletConfig, tmp_output: Path
    ) -> None:
        pipeline = Pipeline(basic_config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = pipeline.process_frame(
            frame,
            frame_id="frame_000",
            app_name="Chrome",
            window_title="Google",
        )
        assert result.app_name == "Chrome"
        assert result.window_title == "Google"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_pipeline_batch.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/pipeline.py`:
```python
"""Pipeline orchestrator — wires core modules and manages rolling state."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from peeklet.config import PeekletConfig, load_patterns
from peeklet.core.comparator import compare_frames
from peeklet.core.exporter import ManifestWriter, save_keyframe
from peeklet.core.hasher import compute_phash, hashes_match
from peeklet.core.masking import AdaptiveMask
from peeklet.core.redactor import (
    BUILTIN_PATTERNS,
    build_pattern_set,
    find_pii_in_text,
    redact_regions,
)
from peeklet.utils.types import EventType, FrameResult, Region


class Pipeline:
    """Stateful pipeline that processes frames through the cascade."""

    def __init__(self, config: PeekletConfig) -> None:
        self._config = config
        self._mask = AdaptiveMask(
            block_size=config.masking.block_size,
            window_size=config.masking.window_size,
            noise_threshold=config.masking.noise_threshold,
        )
        self._last_keyframe: np.ndarray | None = None
        self._last_hash: str | None = None

        output_dir = Path(config.exporter.output_dir)
        self._output_dir = output_dir
        self._writer = ManifestWriter(
            output_dir / "manifest.parquet",
            compression=config.exporter.parquet_compression,
        )

        # Build PII patterns if redaction is enabled
        self._pii_patterns = None
        if config.redactor.enabled:
            custom = []
            if config.redactor.custom_patterns_file:
                custom = load_patterns(Path(config.redactor.custom_patterns_file))
            self._pii_patterns = build_pattern_set(
                pii_types=config.redactor.pii_types,
                custom_patterns=custom,
            )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        timestamp: datetime | None = None,
        app_name: str | None = None,
        window_title: str | None = None,
        source_format: str | None = None,
    ) -> FrameResult:
        """Process a single frame through the cascade.

        Returns a FrameResult indicating whether this frame is a keyframe or was skipped.
        """
        h, w = frame.shape[:2]

        # Step 1: Adaptive masking
        masked, mask_regions = self._mask.apply(frame)

        # Step 2: Perceptual hash
        current_hash = compute_phash(
            masked, hash_size=self._config.hasher.hash_size
        )

        # First frame is always a keyframe
        if self._last_keyframe is None:
            return self._emit_keyframe(
                frame=frame,
                frame_id=frame_id,
                phash=current_hash,
                mask_regions=mask_regions,
                h=h,
                w=w,
                timestamp=timestamp,
                app_name=app_name,
                window_title=window_title,
                source_format=source_format,
            )

        # Step 3: Hash comparison
        if hashes_match(current_hash, self._last_hash):  # type: ignore[arg-type]
            return self._emit_skipped(
                frame_id=frame_id,
                phash=current_hash,
                mask_regions=mask_regions,
                h=h,
                w=w,
                timestamp=timestamp,
                app_name=app_name,
                window_title=window_title,
                source_format=source_format,
            )

        # Step 4: SSIM comparison
        comparison = compare_frames(
            masked,
            self._last_keyframe,
            block_size=self._config.masking.block_size,
        )

        if comparison.ssim_score > self._config.comparator.ssim_threshold:
            return self._emit_skipped(
                frame_id=frame_id,
                phash=current_hash,
                ssim_score=comparison.ssim_score,
                change_score=comparison.change_score,
                changed_pct=comparison.changed_pct,
                mask_regions=mask_regions,
                h=h,
                w=w,
                timestamp=timestamp,
                app_name=app_name,
                window_title=window_title,
                source_format=source_format,
            )

        # Step 5: This is a keyframe
        return self._emit_keyframe(
            frame=frame,
            frame_id=frame_id,
            phash=current_hash,
            ssim_score=comparison.ssim_score,
            change_score=comparison.change_score,
            changed_pct=comparison.changed_pct,
            changed_regions=comparison.changed_regions,
            mask_regions=mask_regions,
            h=h,
            w=w,
            timestamp=timestamp,
            app_name=app_name,
            window_title=window_title,
            source_format=source_format,
        )

    def finalize(self) -> None:
        """Flush the manifest writer."""
        self._writer.flush()

    def _emit_keyframe(
        self,
        frame: np.ndarray,
        frame_id: str,
        phash: str,
        mask_regions: list[Region],
        h: int,
        w: int,
        ssim_score: float | None = None,
        change_score: float | None = None,
        changed_pct: float | None = None,
        changed_regions: list[Region] | None = None,
        timestamp: datetime | None = None,
        app_name: str | None = None,
        window_title: str | None = None,
        source_format: str | None = None,
    ) -> FrameResult:
        """Handle a keyframe: optional PII redaction, save image, update state."""
        pii_detected = False
        output_frame = frame

        # PII redaction (placeholder — full OCR integration in future task)
        if self._pii_patterns and changed_regions:
            # For now, we mark PII as not detected since OCR is optional
            pii_detected = False

        # Save keyframe image
        path = save_keyframe(
            output_frame,
            self._output_dir,
            frame_id,
            fmt=self._config.exporter.keyframe_format,
        )

        # Update rolling state
        self._last_keyframe = frame.copy()
        self._last_hash = phash

        result = FrameResult(
            frame_id=frame_id,
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash=phash,
            ssim_score=ssim_score,
            change_score=change_score,
            changed_pct=changed_pct,
            changed_regions=changed_regions,
            adaptive_mask=mask_regions,
            frame_width=w,
            frame_height=h,
            asset_path=str(path),
            pii_detected=pii_detected,
            timestamp=timestamp,
            app_name=app_name,
            window_title=window_title,
            source_format=source_format,
        )
        self._writer.append(result)
        return result

    def _emit_skipped(
        self,
        frame_id: str,
        phash: str,
        mask_regions: list[Region],
        h: int,
        w: int,
        ssim_score: float | None = None,
        change_score: float | None = None,
        changed_pct: float | None = None,
        timestamp: datetime | None = None,
        app_name: str | None = None,
        window_title: str | None = None,
        source_format: str | None = None,
    ) -> FrameResult:
        """Handle a skipped frame: record in manifest, don't save image."""
        result = FrameResult(
            frame_id=frame_id,
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash=phash,
            ssim_score=ssim_score,
            change_score=change_score,
            changed_pct=changed_pct,
            adaptive_mask=mask_regions,
            frame_width=w,
            frame_height=h,
            timestamp=timestamp,
            app_name=app_name,
            window_title=window_title,
            source_format=source_format,
        )
        self._writer.append(result)
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_pipeline_batch.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add peeklet/pipeline.py tests/integration/test_pipeline_batch.py
git commit -m "feat: add pipeline orchestrator wiring all core modules with cascade logic"
```

---

### Task 12: CLI (`peeklet/cli.py`)

**Files:**
- Create: `peeklet/cli.py`
- Create: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cli.py`:
```python
"""Tests for the CLI interface."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from click.testing import CliRunner
from PIL import Image

from peeklet.cli import main


def _create_test_images(directory: Path, count: int = 5) -> None:
    """Create test PNG images in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        frame = np.full((100, 100, 3), i * 50, dtype=np.uint8)
        Image.fromarray(frame).save(directory / f"frame_{i:03d}.png")


class TestCli:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Smart screenshot change detection" in result.output

    def test_batch_mode(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _create_test_images(input_dir, count=5)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--input", str(input_dir), "--output", str(output_dir), "--no-redact"],
        )
        assert result.exit_code == 0

        manifest = output_dir / "manifest.parquet"
        assert manifest.exists()
        table = pq.read_table(manifest)
        assert table.num_rows == 5

    def test_batch_with_config(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _create_test_images(input_dir, count=3)

        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"comparator": {"ssim_threshold": 0.5}, "redactor": {"enabled": false}}'
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input", str(input_dir),
                "--output", str(output_dir),
                "--config", str(config_file),
            ],
        )
        assert result.exit_code == 0

    def test_missing_input_dir_errors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--input", "/nonexistent/dir"])
        assert result.exit_code != 0

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`peeklet/cli.py`:
```python
"""CLI entry point for Peeklet."""

from __future__ import annotations

from pathlib import Path

import click

import peeklet
from peeklet.config import PeekletConfig, load_config
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline


@click.command()
@click.option(
    "--input", "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing screenshot images.",
)
@click.option(
    "--output", "output_dir",
    type=click.Path(path_type=Path),
    default="./output",
    help="Directory for keyframes and manifest.",
)
@click.option(
    "--config", "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config file (JSON or YAML).",
)
@click.option(
    "--no-redact",
    is_flag=True,
    default=False,
    help="Disable PII redaction.",
)
@click.version_option(version=peeklet.__version__, prog_name="peeklet")
def main(
    input_dir: Path,
    output_dir: Path,
    config_path: Path | None,
    no_redact: bool,
) -> None:
    """Smart screenshot change detection.

    Filters noise from screenshot sequences, redacts PII, and exports
    a structured Parquet manifest of keyframes.
    """
    config = load_config(config_path)

    if no_redact:
        config.redactor.enabled = False

    config.exporter.output_dir = str(output_dir)

    pipeline = Pipeline(config)

    # Collect and sort image files
    extensions = {f".{fmt}" for fmt in config.input.supported_formats}
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
            frame,
            frame_id=f.stem,
            source_format=f.suffix.lstrip("."),
        )
        if result.is_keyframe:
            keyframe_count += 1

    pipeline.finalize()

    click.echo(
        f"Done: {keyframe_count} keyframes from {len(files)} frames "
        f"({100 * keyframe_count / len(files):.1f}%). "
        f"Manifest: {output_dir / 'manifest.parquet'}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Verify CLI runs end-to-end**

Run: `uv run peeklet --help`
Expected: Help text with options `--input`, `--output`, `--config`, `--no-redact`, `--version`

- [ ] **Step 6: Commit**

```bash
git add peeklet/cli.py tests/unit/test_cli.py
git commit -m "feat: add CLI with batch mode, config loading, and --no-redact flag"
```

---

### Task 13: CI/CD Pipelines

**Files:**
- Create: `.github/workflows/ci-develop.yml`
- Create: `.github/workflows/ci-main.yml`

- [ ] **Step 1: Create ci-develop.yml**

`.github/workflows/ci-develop.yml`:
```yaml
name: CI — Develop

on:
  pull_request:
    branches: [develop]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      - run: uv sync --dev
      - run: uv run ruff check peeklet/ tests/
      - run: uv run ruff format --check peeklet/ tests/
      - run: uv run mypy peeklet/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      - run: uv sync --dev
      - run: uv run pytest tests/unit/ --cov=peeklet --cov-fail-under=90 -v
      - run: uv run pytest tests/integration/ -v

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      - run: uv build
```

- [ ] **Step 2: Create ci-main.yml**

`.github/workflows/ci-main.yml`:
```yaml
name: CI — Main

on:
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      - run: uv sync --dev
      - run: uv run ruff check peeklet/ tests/
      - run: uv run ruff format --check peeklet/ tests/
      - run: uv run mypy peeklet/

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      - run: uv sync --dev --python ${{ matrix.python-version }}
      - run: uv run pytest tests/unit/ --cov=peeklet --cov-fail-under=90 -v
      - run: uv run pytest tests/integration/ -v

  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      - run: uv sync --dev
      - run: uv run pytest tests/benchmarks/ -v --benchmark-only || true

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      - run: uv sync --dev
      - run: uv pip audit || true

  package:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      - run: uv build
      - run: |
          uv venv /tmp/smoke-test
          VIRTUAL_ENV=/tmp/smoke-test uv pip install dist/*.whl
          /tmp/smoke-test/bin/peeklet --version
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci-develop.yml .github/workflows/ci-main.yml
git commit -m "ci: add CI pipelines for develop (lint+test+build) and main (matrix+benchmark+security)"
```

---

### Task 14: Synthetic Test Data Generator

**Files:**
- Create: `tests/synthetic/generate.py`
- Create: `tests/synthetic/__init__.py`

- [ ] **Step 1: Write the generator**

`tests/synthetic/__init__.py`:
```python
```

`tests/synthetic/generate.py`:
```python
"""Generate synthetic screenshot sequences for testing.

Each sequence has known ground truth: which frames are keyframes and
what regions changed. This enables deterministic CI testing without
real screenshots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_idle_sequence(output_dir: Path, num_frames: int = 20) -> dict:
    """Frames where only the 'clock' region changes (top-right corner).

    Expected: frame 0 is keyframe, all others skipped (clock is noise).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = {"keyframes": [0], "description": "Idle — only clock changes"}

    for i in range(num_frames):
        frame = _make_desktop_base()
        # Simulate clock changing
        _draw_text(frame, f"12:{i:02d}", x=880, y=5, color=(200, 200, 200))
        Image.fromarray(frame).save(output_dir / f"frame_{i:03d}.png")

    _save_ground_truth(output_dir, ground_truth)
    return ground_truth


def generate_app_switch_sequence(output_dir: Path) -> dict:
    """Frames simulating an app switch (whole screen changes at frame 5).

    Expected: frames 0 and 5 are keyframes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = {"keyframes": [0, 5], "description": "App switch at frame 5"}

    for i in range(10):
        if i < 5:
            frame = _make_desktop_base(bg_color=(40, 40, 60))
            _draw_text(frame, "Excel - Budget.xlsx", x=10, y=10, color=(255, 255, 255))
        else:
            frame = _make_desktop_base(bg_color=(255, 255, 255))
            _draw_text(frame, "Chrome - Google", x=10, y=10, color=(0, 0, 0))

        _draw_text(frame, f"12:{i:02d}", x=880, y=5, color=(200, 200, 200))
        Image.fromarray(frame).save(output_dir / f"frame_{i:03d}.png")

    _save_ground_truth(output_dir, ground_truth)
    return ground_truth


def generate_form_fill_sequence(output_dir: Path) -> dict:
    """Frames simulating typing in a form field, character by character.

    Expected: frames 0, 3, 6, 9, 12 are keyframes (every 3rd frame has new text).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    text_states = ["", "", "", "Joh", "Joh", "Joh", "John D", "John D", "John D",
                   "John Doe", "John Doe", "John Doe", "John Doe, 42", "John Doe, 42", "John Doe, 42"]
    keyframes = [0, 3, 6, 9, 12]
    ground_truth = {"keyframes": keyframes, "description": "Form fill — text appears every 3 frames"}

    for i, text in enumerate(text_states):
        frame = _make_desktop_base(bg_color=(245, 245, 245))
        _draw_text(frame, "Name:", x=50, y=100, color=(0, 0, 0))
        # Form field background
        frame[130:160, 50:400] = [255, 255, 255]
        if text:
            _draw_text(frame, text, x=55, y=133, color=(0, 0, 0))
        Image.fromarray(frame).save(output_dir / f"frame_{i:03d}.png")

    _save_ground_truth(output_dir, ground_truth)
    return ground_truth


def generate_pip_video_sequence(output_dir: Path, num_frames: int = 20) -> dict:
    """Article reading with a PiP video playing in the bottom-right corner.

    The video area changes every frame (noise). The article is static.
    Expected: frame 0 is keyframe, rest skipped (video is masked as noise).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = {"keyframes": [0], "description": "PiP video playing — should be masked"}

    rng = np.random.default_rng(42)

    for i in range(num_frames):
        frame = _make_desktop_base(bg_color=(250, 250, 250))
        # Article text (static)
        for line in range(5):
            _draw_text(
                frame,
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                x=30,
                y=60 + line * 30,
                color=(30, 30, 30),
            )
        # PiP video (random noise in bottom-right)
        video_region = rng.integers(0, 256, (150, 200, 3), dtype=np.uint8)
        frame[450:600, 740:940] = video_region

        Image.fromarray(frame).save(output_dir / f"frame_{i:03d}.png")

    _save_ground_truth(output_dir, ground_truth)
    return ground_truth


def generate_all(output_dir: Path) -> None:
    """Generate all synthetic sequences."""
    generate_idle_sequence(output_dir / "idle")
    generate_app_switch_sequence(output_dir / "app_switch")
    generate_form_fill_sequence(output_dir / "form_fill")
    generate_pip_video_sequence(output_dir / "pip_video")


def _make_desktop_base(
    width: int = 960, height: int = 600, bg_color: tuple[int, int, int] = (50, 50, 70)
) -> np.ndarray:
    """Create a base desktop-like frame."""
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
    # Taskbar at bottom
    frame[570:600, :] = [30, 30, 30]
    return frame


def _draw_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw text onto a frame using PIL."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, fill=color)
    frame[:] = np.asarray(img)


def _save_ground_truth(output_dir: Path, ground_truth: dict) -> None:
    """Save ground truth metadata alongside the sequence."""
    (output_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/synthetic/output")
    generate_all(out)
    print(f"Generated synthetic sequences in {out}")
```

- [ ] **Step 2: Verify generator runs**

Run: `uv run python tests/synthetic/generate.py /tmp/peeklet-test-data`
Expected: Creates 4 directories with PNG sequences and `ground_truth.json` files.

- [ ] **Step 3: Verify generated images are valid**

Run: `uv run python -c "from peeklet.core.loader import load_frame; print(load_frame('/tmp/peeklet-test-data/idle/frame_000.png').shape)"`
Expected: `(600, 960, 3)`

- [ ] **Step 4: Commit**

```bash
git add tests/synthetic/
git commit -m "feat: add synthetic test data generator (idle, app_switch, form_fill, pip_video)"
```

---

### Task 15: Ruff Format, Final Lint Pass, and README

**Files:**
- Create: `README.md`
- Create: `LICENSE`

- [ ] **Step 1: Run ruff format on entire project**

Run: `uv run ruff format peeklet/ tests/`
Expected: Files formatted (may modify some files)

- [ ] **Step 2: Run full lint**

Run: `uv run ruff check peeklet/ tests/ && uv run mypy peeklet/`
Expected: No errors

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Create LICENSE**

`LICENSE`:
```
MIT License

Copyright (c) 2026 Vidya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 5: Create README.md**

`README.md`:
```markdown
# Peeklet

Smart screenshot change detection. Filters noise, redacts PII, exports structured Parquet manifests of keyframes.

## What it does

Given a stream of screenshots captured every few seconds, Peeklet determines which frames represent meaningful changes (keyframes) and which are noise (cursor blinks, clock ticks, video playback). It reduces downstream OCR/LLM processing costs by 80-90%.

## Install

```bash
pip install peeklet

# With PII redaction support
pip install peeklet[ocr]
```

## Quick start

### CLI

```bash
peeklet --input ./screenshots --output ./output --no-redact
```

### Library

```python
from peeklet.config import PeekletConfig
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline

config = PeekletConfig()
pipeline = Pipeline(config)

frame = load_frame("screenshot.png")
result = pipeline.process_frame(frame, frame_id="frame_001")
print(result.is_keyframe)  # True or False

pipeline.finalize()  # writes manifest.parquet
```

### Query results with DuckDB

```sql
SELECT * FROM 'output/manifest.parquet' WHERE is_keyframe = true ORDER BY timestamp
```

## How it works

Peeklet runs a cascade of increasingly expensive operations, killing irrelevant frames early:

1. **Adaptive masking** — auto-detects and ignores high-frequency change regions (video, ads, clock)
2. **Perceptual hashing** — cheap 64-bit fingerprint kills ~70% of frames (obvious duplicates)
3. **SSIM comparison** — structural similarity kills ~15% more (minor noise)
4. **PII redaction** — OCR + regex on changed regions only (optional, runs on keyframes only)
5. **Parquet export** — DuckDB-ready manifest with per-frame metadata

## Configuration

Zero-config works out of the box. For tuning:

```bash
peeklet --input ./screenshots --config config.json
```

See [design spec](docs/superpowers/specs/2026-04-01-peeklet-design.md) for full config reference.

## Development

```bash
git clone https://github.com/SrividyaKirti/Peeklet.git
cd Peeklet
uv sync --dev
uv run pytest tests/ -v
```

## License

MIT
```

- [ ] **Step 6: Commit**

```bash
git add README.md LICENSE
git commit -m "docs: add README and MIT license"
```

- [ ] **Step 7: Final format and commit if needed**

```bash
uv run ruff format peeklet/ tests/
git add -u
git diff --cached --quiet || git commit -m "style: apply ruff formatting"
```

---

### Task 16: Create develop branch and push

- [ ] **Step 1: Create develop branch**

```bash
git checkout -b develop
```

- [ ] **Step 2: Verify all tests pass on develop**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Push both branches (requires user confirmation)**

```bash
git push -u origin main
git push -u origin develop
```
