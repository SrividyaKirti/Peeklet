# Peeklet Design Spec

**Date:** 2026-04-01
**Status:** Approved
**Author:** Vidya + Claude

## Overview

Peeklet is a smart screenshot gate that determines whether a meaningful UI change occurred between consecutive screenshots. It filters noise (cursor blinks, clock ticks, video playback), redacts PII, and outputs a structured Parquet manifest of keyframes — the frames that actually matter.

It sits between screen capture and expensive OCR/LLM processing in workflow analysis pipelines, cutting downstream compute costs by 80-90%.

### Problem

When capturing screenshots every few seconds for workflow analysis, 80-90% of consecutive frames are near-identical. Processing every frame with OCR and LLMs is wasteful and expensive. No open-source tool exists that does semantic screenshot change detection — existing tools either do pixel-level diffing or process every frame indiscriminately.

### Solution

A Python library + CLI that runs a cascade of increasingly expensive operations to kill irrelevant frames early:

1. Perceptual hash (cheapest) — kills obvious duplicates
2. SSIM + block-level diff — kills minor noise
3. PII redaction (most expensive) — only runs on confirmed keyframes

## Architecture

### Project Structure

```
peeklet/
├── peeklet/
│   ├── __init__.py
│   ├── config.py              # Config loading, validation, defaults
│   ├── pipeline.py            # Orchestrator — wires modules, runs cascade
│   ├── cli.py                 # Thin CLI wrapper
│   ├── core/
│   │   ├── __init__.py
│   │   ├── loader.py          # Any format → numpy RGB uint8
│   │   ├── masking.py         # Adaptive block-based mask engine
│   │   ├── hasher.py          # Perceptual hashing (pHash)
│   │   ├── comparator.py      # SSIM + block-level change detection
│   │   ├── redactor.py        # PII detection via OCR + regex/NER, black-box
│   │   └── exporter.py        # Parquet manifest + keyframe writer
│   └── utils/
│       ├── __init__.py
│       ├── image.py           # Shared image ops (resize, colorspace, crop)
│       └── types.py           # Shared dataclasses (FrameResult, Region, Config)
├── tests/
│   ├── conftest.py            # Shared fixtures, test dataset loading
│   ├── data/                  # Test dataset (synthetic + real examples)
│   ├── unit/                  # One test file per module
│   ├── integration/           # Full pipeline tests
│   └── benchmarks/            # Performance regression tests
├── docs/
├── pyproject.toml
├── .github/
│   └── workflows/
│       ├── ci-develop.yml     # Lint + test on PR to develop
│       └── ci-main.yml        # Full suite on PR to main
└── README.md
```

### Design Principles

- **Stateless modules**: Every module in `core/` takes input, returns output, holds no global state. Independently testable and reusable.
- **Single stateful component**: `pipeline.py` maintains rolling state (last keyframe, adaptive mask buffer, frame window).
- **Shared types**: `utils/types.py` defines all data structures. Single source of truth.
- **Shared image ops**: `utils/image.py` holds colorspace conversion, resizing, cropping. No module reimplements these.
- **Format-agnostic pipeline**: Only `loader.py` knows about image formats. Everything downstream operates on numpy arrays.

## Modules

### 1. Loader (`core/loader.py`)

Single entry point that normalizes any supported input to a numpy array.

```python
def load_frame(source: str | Path | bytes | np.ndarray | Image.Image) -> np.ndarray:
    """Normalize any supported input to RGB uint8 numpy array (H, W, 3)."""
```

**Internal format:** `numpy.ndarray`, shape `(H, W, 3)`, dtype `uint8`, RGB color space.

**Loader registry:**

| Input Type | Strategy |
|------------|----------|
| PNG/JPEG/BMP/TIFF/WebP | Pillow → numpy |
| PDF | pymupdf renders page to pixmap → numpy |
| numpy array | Validate shape/dtype, convert colorspace if needed |
| PIL Image | `np.asarray()` |
| bytes/buffer | Pillow from buffer → numpy |
| File path (str/Path) | Detect format from header bytes, dispatch to above |

Adding a new format = writing one loader function and registering it. No downstream changes.

### 2. Adaptive Masking (`core/masking.py`)

Automatically detects and ignores high-frequency change regions (playing video, animated ads, blinking cursor, clock) without user configuration.

**Algorithm:**
1. Divide screen into a grid of blocks (e.g., 32x32 pixels per block)
2. Maintain a sliding window of the last N frames (configurable, default 15)
3. For each block, track change frequency across the window
4. Blocks that change on >80% of frames (configurable threshold) are "noisy"
5. Zero out noisy blocks before passing to the hash/comparison stages

**Why blocks, not pixels:** A 1920x1080 screen at 32px blocks = 60x34 = 2,040 blocks. Cheap to compute, precise enough to isolate a PiP video window.

**No user-defined masks.** The system adapts to each user's screen behavior automatically. This scales to hundreds of users without per-user configuration.

### 3. Perceptual Hasher (`core/hasher.py`)

Computes a perceptual hash (pHash) of the masked frame. Produces a 64-bit fingerprint.

- Identical screenshots → identical hash
- Minor noise that survived masking → likely identical hash
- Meaningful change → different hash

This is the cheapest check in the cascade. ~70% of frames die here.

### 4. Comparator (`core/comparator.py`)

Runs only when hashes differ. Two operations:

1. **SSIM (Structural Similarity Index):** Compares the frame against the last keyframe. Returns a 0-1 score. Frames above the threshold (default 0.85) are dropped as minor noise.
2. **Block-level diff:** For frames that pass SSIM, identifies which blocks changed. Returns bounding boxes of changed regions.

### 5. PII Redactor (`core/redactor.py`)

Runs only on confirmed keyframes. Three-step process:

1. **OCR the changed regions only** (not the full frame — we already know where changes are from the comparator)
2. **Pattern matching:** Run regex/NER against extracted text to find PII
3. **Black-box:** Draw black rectangles over detected PII regions in the keyframe image

**Built-in PII types:** email, phone, SSN, credit card, IP address, physical address.

**Custom patterns** via external YAML file:

```yaml
# patterns.yaml
patterns:
  - name: employee_id
    regex: "EMP-\\d{6}"
    description: "Internal employee ID"
```

**Pattern precedence:**
1. Load built-in patterns
2. Load user patterns from `custom_patterns_file`
3. User pattern with matching name **replaces** the built-in
4. User pattern with new name **adds** to the set
5. User can disable a built-in: `enabled: false`
6. `pii_types` in config acts as master filter — only listed types run

**Opt-in:** When `redactor.enabled` is `false`, this module is never called and OCR dependencies are not required.

### 6. Exporter (`core/exporter.py`)

Writes two outputs:

1. **Keyframe images** to disk (PNG by default)
2. **Parquet manifest** with per-frame metadata

**Parquet schema:**

| Column | Type | Description |
|--------|------|-------------|
| `frame_id` | string | Filename or sequence index |
| `timestamp` | timestamp | Microsecond capture time |
| `event_type` | enum | `KEYFRAME` or `SKIPPED` |
| `app_name` | string | Active window application |
| `window_title` | string | Specific tab or document name |
| `is_keyframe` | bool | Whether a meaningful change occurred |
| `perceptual_hash` | string | pHash fingerprint used for deduping |
| `ssim_score` | float | Similarity vs previous keyframe (null if first) |
| `change_score` | float | How different from last frame (0.0 to 1.0) |
| `changed_pct` | float | Percentage of screen blocks that changed |
| `changed_regions` | list[struct] | Bounding boxes [{x, y, w, h}] |
| `adaptive_mask` | list[struct] | Auto-masked blocks this frame [{x, y, w, h}] |
| `frame_width` | int | Source image width |
| `frame_height` | int | Source image height |
| `source_format` | string | Original input format |
| `asset_path` | string | URI to saved keyframe (null if skipped) |
| `pii_detected` | bool | Whether PII was found and redacted |

Output is DuckDB-ready. Queryable immediately:

```sql
SELECT * FROM 'output/manifest.parquet' WHERE is_keyframe = true
```

## Pipeline Flow

```
INPUT (any format)
  │
  ▼
┌─────────┐
│ LOADER  │  load_frame() → numpy (H,W,3) RGB uint8
└────┬────┘
     │
     ▼
┌──────────┐
│ MASKING  │  Apply adaptive mask (zero out noisy blocks)
└────┬─────┘
     │
     ▼
┌──────────┐
│ HASHER   │  Compute pHash → uint64
└────┬─────┘
     │
     ├── hash matches rolling state? → SKIP (emit SKIPPED row)
     │
     ▼
┌────────────┐
│ COMPARATOR │  SSIM against last keyframe + block-level diff
└────┬───────┘
     │
     ├── SSIM > threshold? → SKIP (emit SKIPPED row)
     │
     ▼
┌───────────┐
│ REDACTOR  │  OCR changed regions → detect PII → black-box
└────┬──────┘
     │
     ▼
┌───────────┐
│ EXPORTER  │  Write keyframe image + append Parquet manifest row
└───────────┘
```

**Cascade efficiency target:** ~70% die at hash, ~15% at SSIM, ~15% are real keyframes.

**Rolling state (managed by `pipeline.py`):**
- `last_keyframe`: numpy array of most recent keyframe
- `mask_buffer`: circular buffer of last N frames for adaptive mask calculation
- `block_change_counts`: per-block change frequency array

**Module contracts:**

| Module | Input | Output |
|--------|-------|--------|
| `loader` | str/Path/bytes/ndarray/PIL | ndarray (H,W,3) uint8 |
| `masking` | ndarray + mask_buffer | ndarray (masked) + mask regions |
| `hasher` | ndarray | uint64 |
| `comparator` | ndarray + last_keyframe | ssim_score, changed_pct, changed_regions |
| `redactor` | ndarray + changed_regions | ndarray (redacted) + pii_detected bool |
| `exporter` | FrameResult dataclass | Parquet row + keyframe file |

## Configuration

### Main Config (JSON)

```json
{
  "pipeline": {
    "mode": "batch",
    "concurrency": 4
  },
  "masking": {
    "block_size": 32,
    "window_size": 15,
    "noise_threshold": 0.8
  },
  "hasher": {
    "algorithm": "phash",
    "hash_size": 8
  },
  "comparator": {
    "ssim_threshold": 0.85,
    "min_changed_pct": 2.0
  },
  "redactor": {
    "enabled": true,
    "pii_types": ["email", "phone", "ssn", "credit_card", "ip_address", "address"],
    "custom_patterns_file": "./patterns.yaml"
  },
  "exporter": {
    "output_dir": "./output",
    "keyframe_format": "png",
    "parquet_compression": "snappy"
  },
  "input": {
    "supported_formats": ["png", "jpg", "jpeg", "bmp", "tiff", "webp", "pdf"],
    "sort_by": "filename"
  }
}
```

All thresholds have sensible defaults. Zero-config usage works: `peeklet --input ./screenshots`.

Config is validated at startup via Pydantic. Bad values fail fast with clear error messages.

### Custom PII Patterns (YAML)

Separate file referenced by `redactor.custom_patterns_file`:

```yaml
patterns:
  - name: employee_id
    regex: "EMP-\\d{6}"
    description: "Internal employee ID"

  - name: email              # overrides built-in email pattern
    regex: "[a-z]+\\.[a-z]+@acme\\.com"
    description: "Only match Acme corp emails"

  - name: ssn
    enabled: false           # disable built-in SSN detection
```

YAML is used for patterns because regex in JSON requires painful double-escaping.

## Input & Output

### Input Modes

- **Batch:** Directory of image files, processed in timestamp/filename order
- **Stream:** Frames from stdin or named pipe, processed as they arrive
- **Library API:** `from peeklet.core import loader; frame = loader.load_frame(any_input)`

All modes feed into the same pipeline. Input adapters are thin wrappers around `loader.load_frame()`.

**Input metadata:** Fields like `app_name`, `window_title`, and `timestamp` are not extracted from the image — they come from the caller. In batch mode, a sidecar JSON file per image (e.g., `frame_001.json` alongside `frame_001.png`) provides metadata. In stream mode, each frame is a JSON message with a base64-encoded image and metadata fields. In library mode, the caller passes a `FrameMeta` dataclass alongside the image. If metadata is absent, those fields are null in the manifest.

### Output

For a sequence of 44 frames where 4 are keyframes:

**Keyframe images:** 4 PII-redacted PNG files saved to `output_dir`.

**Parquet manifest:** 44 rows, one per input frame. Queryable via DuckDB:

```sql
SELECT * FROM 'output/manifest.parquet' WHERE is_keyframe = true ORDER BY timestamp
```

## Installation

```bash
# Standard install
pip install peeklet
# or
uv add peeklet

# With PII redaction support (adds OCR dependencies)
pip install peeklet[ocr]

# Development
uv sync
```

### Dependency Groups (pyproject.toml)

- **Core:** numpy, Pillow, scikit-image, imagehash, pyarrow
- **Optional `[ocr]`:** pymupdf, easyocr
- **Dev:** pytest, pytest-cov, pytest-asyncio, pytest-benchmark, hypothesis, ruff, mypy

## Testing Strategy

### Framework

`pytest` with plugins:
- `pytest-cov` — coverage enforcement (>=90%)
- `pytest-asyncio` — async pipeline tests
- `pytest-benchmark` — performance regression tests
- `hypothesis` — property-based testing for edge cases

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `test_loader.py` | Every format → numpy. Corrupt files. Wrong dimensions. PDF multi-page. Bytes/PIL/numpy input. Colorspace conversion (RGBA, grayscale → RGB). |
| `test_masking.py` | Block change frequency. Noisy block detection. Mask zeros correct regions. Window buffer rotation. Edge cases: all/no blocks noisy. |
| `test_hasher.py` | Identical → same hash. Slightly different → same hash. Significantly different → different hash. Masked regions ignored. |
| `test_comparator.py` | SSIM scores for known pairs. Block diff bounding boxes. Threshold boundaries. Changed percentage calculation. |
| `test_redactor.py` | Each built-in PII type. Custom patterns. User override precedence. Disabled patterns. Black box placement. Re-OCR confirms redaction. |
| `test_exporter.py` | Parquet schema. Keyframe on disk. Skipped frames null path. DuckDB readable. Compression applied. |
| `test_config.py` | Defaults. Invalid values rejected. Pattern file loading. Override precedence. Missing file error. |

### Integration Tests

| Test | Coverage |
|------|----------|
| `test_pipeline_batch.py` | 50 synthetic frames through full pipeline. Correct keyframes. Manifest matches expected. |
| `test_pipeline_stream.py` | Stdin frames. Same results as batch. |
| `test_cascade_efficiency.py` | ~70% die at hash, ~15% at SSIM. Redactor not called for skipped frames. |
| `test_pii_e2e.py` | Frames with PII → keyframes have black boxes → re-OCR confirms PII gone. |

### Performance Benchmarks

| Benchmark | Target |
|-----------|--------|
| `bench_loader.py` | <10ms per 1920x1080 PNG |
| `bench_hasher.py` | <1ms per hash |
| `bench_comparator.py` | <5ms per SSIM on 1080p |
| `bench_pipeline.py` | >100 frames/sec on 1080p batch |

### Test Dataset — Four Tiers

| Tier | Dataset | What it tests | Size | In repo? |
|------|---------|---------------|------|----------|
| 1. Fixtures | Hand-crafted tiny images | Unit tests — fast, deterministic | ~1 MB | Yes (committed) |
| 2. Synthetic | Generated fake UIs | CI pipeline — reproducible, known ground truth | ~50 MB | Generated on demand |
| 3. Mind2Web | 2,350 real task sequences (17k screenshots) | Keyframe detection accuracy, sequential change detection | Large | Downloaded on demand |
| 4. ShowUI | 7.5k desktop screenshots with bounding boxes | Block-level change region accuracy, UI element grounding | 328 MB | Downloaded on demand |

**WebUI (400k pages)** is available as an optional tier for stress-testing DuckDB/Parquet export at scale and benchmarking perceptual hash dedup against screen similarity labels.

**Directory structure:**

```
tests/
├── fixtures/                 # Tier 1 — committed, tiny, fast unit tests
│   ├── formats/              # One sample per supported format
│   │   ├── sample.png
│   │   ├── sample.jpg
│   │   ├── sample.pdf
│   │   ├── sample.bmp
│   │   ├── sample.webp
│   │   └── corrupt.png       # Truncated file for error handling
│   └── pii/
│       ├── email_visible.png
│       ├── phone_visible.png
│       ├── ssn_visible.png
│       └── clean.png
├── synthetic/                # Tier 2 — generated, not committed
│   └── generate.py           # Renders fake UIs with known mutations
│                             # Sequences: idle, app_switch, form_fill,
│                             # pip_video, mixed (50 frames realistic)
├── datasets/                 # Tiers 3-4 — downloaded on demand, gitignored
│   ├── __init__.py
│   ├── download.py           # Unified downloader for all external datasets
│   ├── mind2web.py           # Fetch + prepare Mind2Web subset from HuggingFace
│   ├── showui.py             # Fetch + prepare ShowUI (328 MB)
│   ├── webui.py              # Fetch + prepare WebUI subset (optional, large)
│   └── README.md             # Setup instructions for contributors
├── unit/                     # One test file per module
├── integration/              # Full pipeline tests
├── benchmarks/               # Performance regression tests
└── conftest.py               # pytest markers: @pytest.mark.mind2web,
                              # @pytest.mark.showui, @pytest.mark.webui
```

**CI runs tiers 1-2 only** (fast, no downloads). Tiers 3-4 run locally or in a nightly CI job:

```bash
pytest -m mind2web    # Run Mind2Web validation tests
pytest -m showui      # Run ShowUI validation tests
pytest -m webui       # Run WebUI scale tests (optional)
```

**What each external dataset validates:**

- **Mind2Web:** Sequential task steps = known keyframes. Measures whether Peeklet correctly identifies which frames are keyframes vs noise. Tests the full pipeline cascade.
- **ShowUI:** Bounding box annotations on interactive elements. Synthetically mutate those regions and verify block-level diff finds them. Tests comparator and masking accuracy.
- **WebUI:** Screen similarity labels for dedup benchmarking at scale. Tests Parquet export performance with DuckDB on large volumes.

Each tier includes expected output (which frames are keyframes, what regions changed) for assertion. Synthetic sequences encode ground truth in the generation script. External datasets use annotation-derived ground truth.

## CI/CD

### Pipeline 1: PR to `develop`

Triggers on pull requests targeting `develop`.

1. **Lint & Format:** ruff check, ruff format --check, mypy
2. **Unit Tests:** pytest tests/unit/ --cov=peeklet --cov-fail-under=90
3. **Integration Tests:** pytest tests/integration/
4. **Build Check:** uv build

### Pipeline 2: PR from `develop` to `main`

Triggers on pull requests targeting `main` (from `develop` only).

1. Everything from Pipeline 1
2. **Performance Benchmarks:** pytest tests/benchmarks/ --benchmark-compare (fail on >10% regression)
3. **Matrix Test:** Python 3.10, 3.11, 3.12, 3.13 on Ubuntu + macOS
4. **Security:** dependency audit, secrets scan
5. **Package Validation:** build wheel, install in clean env, run smoke test

### Branch Rules

| Branch | Direct Push | Merge From | Required Checks |
|--------|-------------|------------|-----------------|
| `main` | Blocked | `develop` only | Pipeline 2 passes, 1 approval |
| `develop` | Blocked | `feature/*`, `bugfix/*` | Pipeline 1 passes |
| `feature/*` | Allowed | — | — |
| `bugfix/*` | Allowed | — | — |

### Workflow

```
feature/add-loader → PR → develop → PR → main
bugfix/fix-ssim    → PR → develop → PR → main
```
