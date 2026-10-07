# Transcript Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Peeklet into a single-purpose tool: screen recording + timestamped transcript → `transcript.json` (one entry per screen block, with screenshot filename and LLM-written `visual_context`) plus `frame_<sec>.jpg` files, and evaluate it on LPM and GUIDE.

**Architecture:** Parse and split the transcript; collect checkpoints (Fathom action items, speech onsets, verbal cues); stream 1 fps samples through change detection; OCR + fingerprint candidates into global screens (bounded memory, OCR cached by masked pHash + mean colour); align each line to at most one screen; heuristic score → shortlist → LLM judge+describe (cached, optional) → select ≤ N; render blocks.

**Tech Stack:** Python 3.10+, numpy, Pillow, scikit-image, scipy, imagehash, pydantic v2, click, PyYAML, imageio + PyAV, pydub, pytesseract (+ tesseract binary), anthropic, openai SDKs; pytest, ruff, mypy (strict), uv.

**Spec:** `docs/superpowers/specs/2026-10-06-transcript-enrichment-design.md`

## Global Constraints

- Branch: `feat/transcript-enrichment` (from `develop`); PR targets `develop`.
- `requires-python = ">=3.10"`; ruff line-length 100, rules `E,F,I,N,W,UP,B,SIM,TCH`; `mypy --strict` on `src/peeklet/` must pass.
- Unit-test coverage gate stays `--cov-fail-under=90` (CI runs `pytest tests/unit/ --cov=src/peeklet`).
- No network or API keys in unit/integration tests: LLM is stubbed; OCR is stubbed except the integration test (skipped when `tesseract` binary is absent).
- Defaults (verbatim from spec): `sample_fps=1.0`, `checkpoint_window_seconds=5`, `min_pause_seconds=1.5`, cue threshold `1.0`, `anchor_bonus=1.0`, `max_line_seconds=8`, `lead_seconds=1.5`, `max_images=20`, `shortlist_factor=2`, `llm_concurrency=8`, `llm_max_lines=8`, `max_image_edge=1568`, rejector `min_text_lines=10`, `min_grid_cells=12`, `min_edge_ratio=0.02`, `ocr_max_dim=1920`, fingerprint `phash_threshold=6`, `ocr_field_min_chars=2`, default LLM `anthropic` / `claude-haiku-4-5`.
- Score weights: references 0.25, text_overlap 0.20, onsets 0.20, verbal_cues 0.15, visual_change 0.10, time_on_screen 0.10.
- Output JSON keys exactly: `timestamp` (`HH:MM:SS`), `transcript`, `image` (filename or `null`), optional `visual_context`, optional `action_item`.
- Image filenames: `frame_<int seconds>.jpg`.
- Dataset files are never committed; evals write under git-ignored `eval/data/` and `eval/results/`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure (end state)

```
src/peeklet/
  __init__.py      exports annotate, Entries
  annotate.py      orchestrator (steps 1–9 of the spec)
  cli.py           single click command
  config.py        PeekletConfig + ScoreWeights + load_config
  types.py         Region, Line, Checkpoint, Screen, ScreenJudgment, Entry, format_hms
  transcript.py    SRT/VTT/Fathom parsers, TranscriptError, split_long_lines, action_item_checkpoints
  speech.py        detect_speech_segments, speech_onset_checkpoints
  cues.py          lexicon, cue_score, verbal_cue_checkpoints
  video.py         VideoDecoder (metadata + iter_coarse_frames_seek)
  change.py        AdaptiveMask, compute_phash, hashes_match, compare_frames, ChangeDetector
  screen.py        WordBox, ocr_word_boxes, rejector, Fingerprint, FingerprintIndex, require_tesseract
  screens.py       Sample, ScreenPass, build_screens (streaming)
  align.py         screen_at, align_lines
  score.py         content_words, ScreenSignals, compute_signals, score_screens, select_screens
  llm/
    __init__.py
    base.py        prompt, LLMClient protocol, parse_judgment, JudgmentCache, judge_screens, build_llm_client
    anthropic_client.py
    openai_client.py
    openrouter_client.py
  render.py        image_filenames, build_entries, Entries, RunStats, write_outputs, write_debug
  image_utils.py   ensure_rgb_uint8, crop_region, compute_block_grid, downscale_to_max_dim,
                   encode_jpeg, decode_jpeg, dhash_64, hamming_distance
eval/
  common.py        token lookup, downloads, VTT helpers, matching metrics
  lpm_eval.py
  guide_eval.py
tests/unit/        one test file per module
tests/integration/test_annotate.py
tests/helpers/synth.py   synthetic demo video + VTT writer
```

---

### Task 1: Remove dead modes and their tests

Removes everything not on the single path. Surviving modules keep working; tests stay green.

**Files:**
- Delete: `src/peeklet/pipeline.py`, `src/peeklet/core/exporter.py`, `src/peeklet/core/loader.py`, `src/peeklet/core/context_exporter.py`
- Delete: `scripts/` (all), `tests/datasets/` (all), `tests/synthetic/` (all)
- Delete tests: `tests/unit/test_pipeline.py`, `test_exporter.py`, `test_loader.py`, `test_context_exporter.py`, `test_quality_fallback.py`, `test_video_trigger_integration.py`, `test_cli.py`, `tests/integration/test_demo_dedup_pipeline.py`, `test_demo_mode_pipeline.py`, `test_pipeline_batch.py`, `test_video_pipeline.py`
- Delete docs: `docs/superpowers/specs/2026-04-*.md`, `docs/superpowers/plans/2026-04-*.md`
- Modify: `src/peeklet/core/video.py`, `src/peeklet/core/demo_filter.py`, `src/peeklet/core/audio.py`, `src/peeklet/core/hasher.py`, `src/peeklet/utils/types.py`, `src/peeklet/cli.py`, `src/peeklet/config.py`
- Modify tests: `tests/unit/test_video.py`, `test_demo_filter.py`, `test_audio.py`, `test_hasher.py`, `test_types.py`, `test_edge_cases.py`, `test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a green suite containing only code that later tasks move/reuse.

- [ ] **Step 1: Delete whole files and directories**

```bash
git rm -q src/peeklet/pipeline.py src/peeklet/core/exporter.py src/peeklet/core/loader.py \
  src/peeklet/core/context_exporter.py
git rm -rq scripts tests/datasets tests/synthetic
git rm -q tests/unit/test_pipeline.py tests/unit/test_exporter.py tests/unit/test_loader.py \
  tests/unit/test_context_exporter.py tests/unit/test_quality_fallback.py \
  tests/unit/test_video_trigger_integration.py tests/unit/test_cli.py \
  tests/integration/test_demo_dedup_pipeline.py tests/integration/test_demo_mode_pipeline.py \
  tests/integration/test_pipeline_batch.py tests/integration/test_video_pipeline.py
git rm -q docs/superpowers/specs/2026-04-*.md docs/superpowers/plans/2026-04-*.md
```

- [ ] **Step 2: Trim `core/video.py` to the decoder**

Keep only: module docstring, imports needed by what remains (`logging`, `dataclass`, `Path`, `TYPE_CHECKING`, `numpy`, `ensure_rgb_uint8`), `VideoMeta`, `_check_video_deps`, and `VideoDecoder` with exactly `__init__`, `get_metadata`, `iter_coarse_frames_seek`. Delete `extract_coarse_frames`, `extract_frame_range`, `extract_frame_at`, `_change_magnitude`, `_should_force_keyframe`, `_merge_trigger_type`, `process_video`, and all imports of `audio`, `context_exporter`, `demo_filter`, `exporter`, `pipeline`, `EventType`, `FrameResult`. Change the `_check_video_deps` message to `"Video support requires imageio and av. Reinstall peeklet."`.

- [ ] **Step 3: Trim `core/demo_filter.py` to OCR + rejector helpers**

Keep: `pytesseract` lazy import block, `_MIN_WORD_LENGTH`, `_MIN_WORD_CONFIDENCE`, `WordBox`, `_downscale_for_ocr`, `_ocr_word_boxes`, `_count_text_lines`, `_GRID_DIM`, `_count_occupied_grid_cells`, `_EDGE_MAGNITUDE_THRESHOLD`, `_edge_pixel_ratio`, `_is_low_info_frame`. Delete `_count_words_in_frame`, `_quality_capture`, `_ocr_joined_text`, `apply_demo_filter`, and imports of `build_llm_client`, `Moment`, `MomentEntry`, `Screen`, `VideoDecoder`, `TranscriptSegment`.

- [ ] **Step 4: Trim `core/audio.py`**

Delete `align_transcript` and `get_audio_activity`. Keep parsers, `parse_fathom_anchors`, `_check_audio_deps`, `detect_speech_segments`.

- [ ] **Step 5: Trim `core/hasher.py`, `utils/types.py`, `config.py`, `cli.py`**

- `hasher.py`: delete `compute_phash_tiled` and `tiled_hashes_match` (and the `math` import).
- `utils/types.py`: keep `Region` and `Moment` (still used by `core/llm*.py` until Task 12). Delete `EventType`, `FrameMeta`, `FrameResult`, `Screen`, `MomentEntry` and the `Fingerprint` import.
- `config.py`: delete `PipelineConfig`, `ExporterConfig`, `InputConfig`, `QUALITY_PRESETS`, `apply_quality_preset`, `SENSITIVITY_PRESETS`, `apply_sensitivity_preset`, and the matching fields on `PeekletConfig`. (Full rewrite comes in Task 3.)
- `cli.py`: replace the whole file with a placeholder until Task 15:

```python
"""Peeklet CLI (being rebuilt; see Task 15 of the implementation plan)."""

from __future__ import annotations

import click


@click.command()
def main() -> None:
    """Placeholder entry point."""
    raise click.ClickException("peeklet CLI is being rebuilt on this branch")
```

- [ ] **Step 6: Prune tests that target deleted code**

- `tests/unit/test_video.py`: keep `TestVideoMeta` and any test that only uses `VideoDecoder.get_metadata` / `iter_coarse_frames_seek`; delete `TestSmartSampling`, `TestAudioEnrichment`, `test_extract_frame_at_*`, `test_process_video_*`, and tests of `extract_coarse_frames`.
- `tests/unit/test_demo_filter.py`: delete `test_count_words_*` tests that call `_count_words_in_frame` (rewrite each as the same assertion on `len(_ocr_word_boxes(...))`), `test_demo_filter_config_*`, `test_apply_demo_filter_*`, `TestApplyDemoFilterLinearPass`.
- `tests/unit/test_audio.py`: delete `TestAlignTranscript` and `test_get_audio_activity_at_timestamp`.
- `tests/unit/test_hasher.py`: delete tiled-hash tests.
- `tests/unit/test_types.py`: keep `TestRegion` and the `test_moment_*` tests; delete the rest.
- `tests/unit/test_edge_cases.py`: delete `TestLoaderEdgeCases`, `TestConfigEdgeCases`, `TestExporterEdgeCases`, `TestPipelineEdgeCases`, `TestPipelineTiledEdgeCases`, and any `TestTypesEdgeCases` test touching deleted types.
- `tests/unit/test_config.py`: delete tests for removed config sections and presets.
- `pyproject.toml`: remove the pytest markers `datasets`, `mind2web`, `showui`, `webui`, `rejector_live`.

- [ ] **Step 7: Verify nothing references deleted code**

Run: `grep -rnE "pipeline|exporter|loader|context_exporter|FrameResult|MomentEntry|apply_demo_filter|process_video|align_transcript" src tests`
Expected: no matches (except the word "pipeline" in prose comments, which you should reword).

- [ ] **Step 8: Run the suite and linters**

Run: `uv run pytest -q && uv run ruff check src tests && uv run mypy src/peeklet/`
Expected: all pass. (Coverage gate is not enforced until Task 15.)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove image mode, Parquet export, demo mode, presets, and dataset tooling

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Flatten the package layout (moves only, no behavior change)

**Files:**
- Move: `core/video.py` → `video.py`; `utils/image.py` → `image_utils.py`; `utils/types.py` → `types.py`
- Create by concatenation: `change.py` (from `core/masking.py` + `core/hasher.py` + `core/comparator.py`), `screen.py` (from `core/demo_filter.py` + `core/fingerprint.py`)
- Split: `core/audio.py` → `transcript.py` (parsers + Fathom anchors) and `speech.py` (`_check_audio_deps`, `detect_speech_segments`)
- Move LLM modules: `core/llm.py` → `llm/base.py`, `core/llm_anthropic.py` → `llm/anthropic_client.py`, `core/llm_openai.py` → `llm/openai_client.py`, `core/llm_openrouter.py` → `llm/openrouter_client.py`; create empty `llm/__init__.py`
- Delete: `core/`, `utils/` packages
- Tests: `test_audio.py` → `test_transcript.py` + `test_speech.py`; `test_masking.py` + `test_hasher.py` + `test_comparator.py` → `test_change.py`; `test_demo_filter.py` → `test_screen_ocr.py`; `test_fingerprint.py` → `test_screen_fingerprint.py`; `test_image.py` → `test_image_utils.py`; update imports in all remaining tests.

**Interfaces:**
- Produces module paths used by every later task: `peeklet.video`, `peeklet.change`, `peeklet.screen`, `peeklet.transcript`, `peeklet.speech`, `peeklet.image_utils`, `peeklet.types`, `peeklet.llm.base`.
- `TranscriptSegment` is renamed to `Line` and moved into `peeklet/types.py`:

```python
@dataclass(frozen=True, slots=True)
class Line:
    """One timestamped transcript line."""

    start: float  # seconds
    end: float  # seconds
    text: str
    speaker: str | None = None
```

- [ ] **Step 1: Do the moves with git**

```bash
mkdir -p src/peeklet/llm
git mv src/peeklet/core/video.py src/peeklet/video.py
git mv src/peeklet/utils/image.py src/peeklet/image_utils.py
git mv src/peeklet/utils/types.py src/peeklet/types.py
git mv src/peeklet/core/llm.py src/peeklet/llm/base.py
git mv src/peeklet/core/llm_anthropic.py src/peeklet/llm/anthropic_client.py
git mv src/peeklet/core/llm_openai.py src/peeklet/llm/openai_client.py
git mv src/peeklet/core/llm_openrouter.py src/peeklet/llm/openrouter_client.py
touch src/peeklet/llm/__init__.py
git mv src/peeklet/core/audio.py src/peeklet/transcript.py
git mv src/peeklet/core/demo_filter.py src/peeklet/screen.py
git mv src/peeklet/core/masking.py src/peeklet/change.py
git mv tests/unit/test_audio.py tests/unit/test_transcript.py
git mv tests/unit/test_demo_filter.py tests/unit/test_screen_ocr.py
git mv tests/unit/test_fingerprint.py tests/unit/test_screen_fingerprint.py
git mv tests/unit/test_image.py tests/unit/test_image_utils.py
git mv tests/unit/test_masking.py tests/unit/test_change.py
```

- [ ] **Step 2: Merge the remaining sources**

- Append the body (everything after imports) of `core/hasher.py` and `core/comparator.py` to `change.py`; merge their imports at the top; `git rm` both files.
- Append the body of `core/fingerprint.py` to `screen.py`; merge imports; `git rm core/fingerprint.py`.
- Move `_check_audio_deps` and `detect_speech_segments` from `transcript.py` into a new `src/peeklet/speech.py` (module docstring `"""Speech/silence detection from a media file's audio track."""`).
- In `types.py` add `Line` (above); in `transcript.py` delete the `TranscriptSegment` class and replace every use with `Line` (`from peeklet.types import Line`). `detect_speech_segments` keeps returning `list[Line]` for now (changed in Task 7).
- Append the bodies of `tests/unit/test_hasher.py` and `test_comparator.py` to `test_change.py`; `git rm` them. Move `TestSpeechSilenceDetection` from `test_transcript.py` into new `tests/unit/test_speech.py`.
- `git rm -r src/peeklet/core src/peeklet/utils`.

- [ ] **Step 3: Rewrite imports everywhere**

Apply these replacements across `src/` and `tests/`:

| Old | New |
|---|---|
| `peeklet.core.video` | `peeklet.video` |
| `peeklet.utils.image` | `peeklet.image_utils` |
| `peeklet.utils.types` | `peeklet.types` |
| `peeklet.core.masking`, `peeklet.core.hasher`, `peeklet.core.comparator` | `peeklet.change` |
| `peeklet.core.demo_filter`, `peeklet.core.fingerprint` | `peeklet.screen` |
| `peeklet.core.audio` (parsers/anchors) | `peeklet.transcript` |
| `peeklet.core.audio` (speech) | `peeklet.speech` |
| `peeklet.core.llm` | `peeklet.llm.base` |
| `peeklet.core.llm_openai` | `peeklet.llm.openai_client` |
| `TranscriptSegment` | `Line` |

Also update `mock.patch("peeklet.core....")` strings in tests the same way.

- [ ] **Step 4: Verify**

Run: `grep -rn "peeklet.core\|peeklet.utils\|TranscriptSegment" src tests`
Expected: no matches.
Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format src tests && uv run mypy src/peeklet/`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: flatten package layout around the single annotation path

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: New configuration model

**Files:**
- Rewrite: `src/peeklet/config.py`
- Rewrite: `tests/unit/test_config.py`
- Modify: `src/peeklet/screen.py` (rejector reads the new config), `tests/unit/test_screen_ocr.py` (build configs with the new model)

**Interfaces:**
- Produces:

```python
class ScoreWeights(BaseModel): references, text_overlap, onsets, verbal_cues, visual_change, time_on_screen
class PeekletConfig(BaseModel)  # fields below
def load_config(path: Path | None) -> PeekletConfig
```

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_config.py`

```python
"""Tests for PeekletConfig and load_config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from peeklet.config import PeekletConfig, ScoreWeights, load_config

if TYPE_CHECKING:
    from pathlib import Path


def test_defaults_match_spec() -> None:
    cfg = PeekletConfig()
    assert cfg.sample_fps == 1.0
    assert cfg.checkpoint_window_seconds == 5.0
    assert cfg.min_pause_seconds == 1.5
    assert cfg.cue_threshold == 1.0
    assert cfg.anchor_bonus == 1.0
    assert cfg.max_line_seconds == 8.0
    assert cfg.lead_seconds == 1.5
    assert cfg.max_images == 20
    assert cfg.shortlist_factor == 2.0
    assert cfg.llm_concurrency == 8
    assert cfg.llm_max_lines == 8
    assert cfg.max_image_edge == 1568
    assert (cfg.min_text_lines, cfg.min_grid_cells, cfg.min_edge_ratio) == (10, 12, 0.02)
    assert cfg.ocr_max_dim == 1920
    assert (cfg.phash_threshold, cfg.ocr_field_min_chars) == (6, 2)
    assert (cfg.llm_provider, cfg.llm_model) == ("anthropic", "claude-haiku-4-5")
    assert cfg.use_llm is True


def test_score_weight_defaults() -> None:
    w = ScoreWeights()
    assert (w.references, w.text_overlap, w.onsets) == (0.25, 0.20, 0.20)
    assert (w.verbal_cues, w.visual_change, w.time_on_screen) == (0.15, 0.10, 0.10)


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        PeekletConfig.model_validate({"not_a_field": 1})


def test_negative_max_images_rejected() -> None:
    with pytest.raises(ValidationError):
        PeekletConfig(max_images=-1)


def test_load_none_returns_defaults() -> None:
    assert load_config(None) == PeekletConfig()


def test_load_yaml(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("max_images: 5\nscore_weights:\n  references: 0.5\n")
    cfg = load_config(p)
    assert cfg.max_images == 5
    assert cfg.score_weights.references == 0.5


def test_load_json(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text('{"llm_model": "claude-sonnet-5-5"}')
    assert load_config(p).llm_model == "claude-sonnet-5-5"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_config.py -q`
Expected: FAIL (`ImportError: cannot import name 'ScoreWeights'`).

- [ ] **Step 3: Implement** — replace `src/peeklet/config.py`

```python
"""Configuration for Peeklet's single annotation path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ScoreWeights(BaseModel):
    """Weights for the heuristic screen score (each signal is scaled to [0, 1])."""

    model_config = {"extra": "forbid"}

    references: float = Field(default=0.25, ge=0.0)
    text_overlap: float = Field(default=0.20, ge=0.0)
    onsets: float = Field(default=0.20, ge=0.0)
    verbal_cues: float = Field(default=0.15, ge=0.0)
    visual_change: float = Field(default=0.10, ge=0.0)
    time_on_screen: float = Field(default=0.10, ge=0.0)


class PeekletConfig(BaseModel):
    """All tunables for annotate(). Unknown keys are rejected."""

    model_config = {"extra": "forbid"}

    # Sampling and change detection
    sample_fps: float = Field(default=1.0, gt=0.0)
    change_max_dim: int = Field(default=720, ge=64)
    mask_block_size: int = Field(default=32, gt=0)
    mask_window_size: int = Field(default=15, gt=0)
    mask_noise_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    ssim_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_changed_blocks: int = Field(default=3, ge=0)

    # Checkpoints
    checkpoint_window_seconds: float = Field(default=5.0, ge=0.0)
    min_pause_seconds: float = Field(default=1.5, ge=0.0)
    silence_threshold_dbfs: float = -40.0
    cue_threshold: float = Field(default=1.0, gt=0.0)
    anchor_bonus: float = Field(default=1.0, ge=0.0)

    # Screens (OCR, low-info rejector, fingerprint)
    ocr_max_dim: int = Field(default=1920, gt=0)
    min_text_lines: int = Field(default=10, ge=0)
    min_grid_cells: int = Field(default=12, ge=0)
    min_edge_ratio: float = Field(default=0.020, ge=0.0, le=1.0)
    phash_threshold: int = Field(default=6, ge=0, le=64)
    ocr_field_min_chars: int = Field(default=2, ge=0)

    # Transcript and alignment
    max_line_seconds: float = Field(default=8.0, gt=0.0)
    lead_seconds: float = Field(default=1.5, ge=0.0)

    # Scoring
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)

    # LLM judge + describe
    use_llm: bool = True
    llm_provider: Literal["anthropic", "openai", "openrouter"] = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    shortlist_factor: float = Field(default=2.0, ge=1.0)
    llm_concurrency: int = Field(default=8, ge=1)
    llm_max_lines: int = Field(default=8, ge=1)
    llm_cache_dir: str = "~/.cache/peeklet/judgments"

    # Output
    max_images: int = Field(default=20, ge=0)
    max_image_edge: int = Field(default=1568, ge=64)
    jpeg_quality: int = Field(default=90, ge=1, le=100)


def load_config(path: Path | None) -> PeekletConfig:
    """Load config from YAML/JSON, or return defaults when path is None."""
    if path is None:
        return PeekletConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    data = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
    return PeekletConfig.model_validate(data or {})
```

- [ ] **Step 4: Point the rejector at the new config**

In `screen.py`, change `_is_low_info_frame(frame, config: DemoFilterConfig)` to take `config: PeekletConfig` and read `config.ocr_max_dim` instead of `config.gallery_ocr_min_dim` (other field names are unchanged). Update the `TYPE_CHECKING` import to `from peeklet.config import PeekletConfig`. In `tests/unit/test_screen_ocr.py` replace `DemoFilterConfig(...)` with `PeekletConfig(...)` and `gallery_ocr_min_dim=` with `ocr_max_dim=`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q && uv run mypy src/peeklet/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: flat PeekletConfig for the annotation path

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Core types and image helpers

**Files:**
- Modify: `src/peeklet/types.py`, `src/peeklet/image_utils.py`, `src/peeklet/screen.py` (use `downscale_to_max_dim`)
- Test: `tests/unit/test_types.py`, `tests/unit/test_image_utils.py`

**Interfaces:**
- Produces (in `peeklet.types`):

```python
def format_hms(seconds: float) -> str                       # 134.9 -> "00:02:14"
CheckpointKind = Literal["action_item", "speech_onset", "verbal_cue"]
@dataclass(frozen=True, slots=True) class Checkpoint: t: float; kind: CheckpointKind; label: str = ""
@dataclass(slots=True) class Screen: id, image_jpeg: bytes, frame_t: float, ocr_text: str,
    word_count: int, first_change: float, occurrences: list[tuple[float, float]]
@dataclass(frozen=True, slots=True) class ScreenJudgment: include: bool; reason: str; visual_context: str
@dataclass(frozen=True, slots=True) class Entry: timestamp, lines: tuple[Line, ...], image, visual_context,
    action_items: tuple[str, ...], screen_id;  def to_dict(self) -> dict[str, Any]
```

- Produces (in `peeklet.image_utils`): `downscale_to_max_dim(frame, max_dim) -> np.ndarray`, `encode_jpeg(frame, quality=90, max_edge=None) -> bytes`, `decode_jpeg(data: bytes) -> np.ndarray`.

- [ ] **Step 1: Write failing tests** — append to `tests/unit/test_types.py`

```python
from peeklet.types import Checkpoint, Entry, Line, Screen, ScreenJudgment, format_hms


def test_format_hms_truncates_to_whole_seconds() -> None:
    assert format_hms(0) == "00:00:00"
    assert format_hms(134.9) == "00:02:14"
    assert format_hms(3725.0) == "01:02:05"


def test_checkpoint_defaults() -> None:
    cp = Checkpoint(t=3.0, kind="speech_onset")
    assert cp.label == ""


def test_screen_occurrences_default_empty() -> None:
    s = Screen(id="S1", image_jpeg=b"x", frame_t=1.0, ocr_text="", word_count=0, first_change=1.0)
    assert s.occurrences == []


def test_entry_to_dict_full() -> None:
    e = Entry(
        timestamp=134.2,
        lines=(Line(134.2, 137.0, "Look left.", "Alice"), Line(137.0, 140.0, "See it?", None)),
        image="frame_134.jpg",
        visual_context="Policy page.",
        action_items=("Fix tag check", "Add test"),
        screen_id="S1",
    )
    assert e.to_dict() == {
        "timestamp": "00:02:14",
        "transcript": "Alice: Look left.\nSpeaker: See it?",
        "image": "frame_134.jpg",
        "visual_context": "Policy page.",
        "action_item": "Fix tag check; Add test",
    }


def test_entry_to_dict_without_image_omits_visual_context() -> None:
    e = Entry(timestamp=5.0, lines=(Line(5.0, 6.0, "Hi", "Bob"),), visual_context="ignored")
    assert e.to_dict() == {"timestamp": "00:00:05", "transcript": "Bob: Hi", "image": None}


def test_screen_judgment_is_frozen() -> None:
    j = ScreenJudgment(include=True, reason="r", visual_context="v")
    assert j.include
```

Append to `tests/unit/test_image_utils.py`:

```python
import numpy as np

from peeklet.image_utils import decode_jpeg, downscale_to_max_dim, encode_jpeg


def test_downscale_keeps_small_frames() -> None:
    f = np.zeros((100, 200, 3), dtype=np.uint8)
    assert downscale_to_max_dim(f, 300) is f


def test_downscale_preserves_aspect() -> None:
    f = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert downscale_to_max_dim(f, 960).shape == (540, 960, 3)


def test_jpeg_round_trip_and_max_edge() -> None:
    f = np.full((400, 800, 3), 128, dtype=np.uint8)
    data = encode_jpeg(f, quality=90, max_edge=200)
    assert data[:2] == b"\xff\xd8"
    back = decode_jpeg(data)
    assert back.shape == (100, 200, 3)
    assert abs(int(back.mean()) - 128) <= 2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_types.py tests/unit/test_image_utils.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement** — add to `src/peeklet/types.py` (keep `Region`, `Line`, and `Moment` until Task 12)

```python
from typing import Any, Literal

from dataclasses import dataclass, field


def format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS (truncating fractions)."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


CheckpointKind = Literal["action_item", "speech_onset", "verbal_cue"]


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A moment where Peeklet guarantees a capture and boosts the screen on display."""

    t: float
    kind: CheckpointKind
    label: str = ""


@dataclass(slots=True)
class Screen:
    """A distinct screen state, global across the whole video."""

    id: str
    image_jpeg: bytes  # chosen frame (most OCR words; ties -> latest)
    frame_t: float  # timestamp of the chosen frame
    ocr_text: str
    word_count: int
    first_change: float  # 1 - SSIM when the screen first appeared (0..1)
    occurrences: list[tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ScreenJudgment:
    """LLM verdict for one screen."""

    include: bool
    reason: str
    visual_context: str


@dataclass(frozen=True, slots=True)
class Entry:
    """One output block: consecutive lines spoken over the same screen (or none)."""

    timestamp: float
    lines: tuple[Line, ...]
    image: str | None = None
    visual_context: str | None = None
    action_items: tuple[str, ...] = ()
    screen_id: str | None = None  # internal; not serialized

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "timestamp": format_hms(self.timestamp),
            "transcript": "\n".join(f"{ln.speaker or 'Speaker'}: {ln.text}" for ln in self.lines),
            "image": self.image,
        }
        if self.image and self.visual_context:
            out["visual_context"] = self.visual_context
        if self.action_items:
            out["action_item"] = "; ".join(self.action_items)
        return out
```

Add to `src/peeklet/image_utils.py`:

```python
import io


def downscale_to_max_dim(frame: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize so the longest edge is at most max_dim; returns the input if already small."""
    from PIL import Image

    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR))


def encode_jpeg(frame: np.ndarray, quality: int = 90, max_edge: int | None = None) -> bytes:
    """Encode an RGB uint8 frame as JPEG, optionally shrinking the long edge first."""
    from PIL import Image

    if max_edge is not None:
        frame = downscale_to_max_dim(frame, max_edge)
    buf = io.BytesIO()
    Image.fromarray(ensure_rgb_uint8(frame)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to an RGB uint8 array."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))
```

In `screen.py`, delete `_downscale_for_ocr` and call `downscale_to_max_dim` instead; move its tests in `test_screen_ocr.py` to target `downscale_to_max_dim` (or delete them if Step 1's tests cover the same cases).

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q && uv run mypy src/peeklet/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: core types (Checkpoint, Screen, Entry) and JPEG helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Transcript parsing, errors, long-line splitting, action items

**Files:**
- Modify: `src/peeklet/transcript.py`
- Test: `tests/unit/test_transcript.py`

**Interfaces:**
- Consumes: `Line`, `Checkpoint` from `peeklet.types`.
- Produces:

```python
class TranscriptError(ValueError)
def parse_transcript(path: Path) -> list[Line]            # raises TranscriptError
def split_long_lines(lines: list[Line], max_seconds: float) -> list[Line]
def parse_fathom_action_items(text: str) -> list[Checkpoint]   # kind="action_item"
def action_item_checkpoints(path: Path) -> list[Checkpoint]    # [] unless .md
```

- [ ] **Step 1: Write failing tests** — add to `tests/unit/test_transcript.py`; rewrite `TestParseFathomAnchors` to call `parse_fathom_action_items` and assert on `Checkpoint` fields (`t`, `kind == "action_item"`, `label`) instead of `Moment` fields.

```python
import pytest

from peeklet.transcript import (
    TranscriptError,
    action_item_checkpoints,
    parse_fathom_action_items,
    parse_transcript,
    split_long_lines,
)
from peeklet.types import Line


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_transcript.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement** in `src/peeklet/transcript.py`

Replace `parse_transcript` and `parse_fathom_anchors`, and add the new functions (parsers `_parse_srt`, `_parse_vtt`, `_parse_fathom_md`, regexes unchanged):

```python
class TranscriptError(ValueError):
    """The transcript is missing, unreadable, or contains no timestamped lines."""


def parse_transcript(path: Path) -> list[Line]:
    """Parse .srt, .vtt or Fathom .md into lines. Raises TranscriptError on bad input."""
    path = Path(path)
    if not path.is_file():
        raise TranscriptError(f"Transcript not found: {path}")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise TranscriptError(f"Could not read transcript {path}: {exc}") from exc
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        lines = _parse_vtt(text)
    elif suffix == ".md":
        lines = _parse_fathom_md(text)
    else:
        lines = _parse_srt(text)
    if not lines:
        raise TranscriptError(f"Transcript {path} has no timestamped lines")
    return lines


_SENTENCE_BREAK_RE = re.compile(r"(?<=[.?!])\s+")


def split_long_lines(lines: list[Line], max_seconds: float) -> list[Line]:
    """Split lines longer than max_seconds at sentence boundaries, timing pieces by length."""
    out: list[Line] = []
    for ln in lines:
        duration = ln.end - ln.start
        parts = [p.strip() for p in _SENTENCE_BREAK_RE.split(ln.text) if p.strip()]
        if duration <= max_seconds or len(parts) < 2:
            out.append(ln)
            continue
        total = sum(len(p) for p in parts)
        t = ln.start
        for i, part in enumerate(parts):
            end = ln.end if i == len(parts) - 1 else round(t + duration * len(part) / total, 3)
            out.append(Line(start=t, end=end, text=part, speaker=ln.speaker))
            t = end
    return out


def parse_fathom_action_items(text: str) -> list[Checkpoint]:
    """Extract Fathom ACTION ITEM ... [WATCH](...?timestamp=N) markers, deduplicated."""
    seen: set[tuple[float, str]] = set()
    found: list[Checkpoint] = []
    for match in _FATHOM_ACTION_RE.finditer(text):
        label = match.group(1).strip()
        t = float(match.group(2))
        if (t, label) in seen:
            continue
        seen.add((t, label))
        found.append(Checkpoint(t=t, kind="action_item", label=label))
    found.sort(key=lambda c: c.t)
    return found


def action_item_checkpoints(path: Path) -> list[Checkpoint]:
    """Action-item checkpoints for Fathom markdown transcripts; [] for other formats."""
    path = Path(path)
    if path.suffix.lower() != ".md":
        return []
    return parse_fathom_action_items(path.read_text(encoding="utf-8"))
```

Import `Checkpoint` alongside `Line` from `peeklet.types`. Delete `parse_fathom_anchors`. In `src/peeklet/llm/base.py` and adapters, remove their dependency on `parse_fathom_anchors` if any (they only use `Moment`; leave as is).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_transcript.py -q && uv run mypy src/peeklet/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: transcript errors, long-line splitting, action-item checkpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Verbal-cue checkpoints (restored lexicon)

**Files:**
- Create: `src/peeklet/cues.py`
- Test: `tests/unit/test_cues.py`

**Interfaces:**
- Produces: `cue_score(text: str) -> float`; `verbal_cue_checkpoints(lines: list[Line], threshold: float = 1.0) -> list[Checkpoint]` (`t = line.start`, `label = line.text[:80]`).

- [ ] **Step 1: Write failing tests** — `tests/unit/test_cues.py` (ported from the deleted `test_transcript_trigger.py` at commit `50a3252^`)

```python
"""Tests for verbal-cue checkpoints."""

from __future__ import annotations

import pytest

from peeklet.cues import cue_score, verbal_cue_checkpoints
from peeklet.types import Line


def test_high_plus_medium_triggers() -> None:
    [cp] = verbal_cue_checkpoints([Line(2.0, 5.0, "Now look at this dashboard")])
    assert (cp.t, cp.kind) == (2.0, "verbal_cue")
    assert cp.label == "Now look at this dashboard"


def test_single_high_signal_triggers() -> None:
    assert len(verbal_cue_checkpoints([Line(0.0, 1.0, "the sidebar")])) == 1


def test_single_medium_does_not_trigger() -> None:
    assert verbal_cue_checkpoints([Line(0.0, 3.0, "I see what you mean")]) == []


def test_no_keywords() -> None:
    assert verbal_cue_checkpoints([Line(0.0, 3.0, "Welcome to the presentation")]) == []


def test_case_insensitive_and_punctuation() -> None:
    assert cue_score("CLICK, THIS... BUTTON!") == pytest.approx(1.0 + 0.5 + 0.3)


def test_repeated_word_counts_once() -> None:
    assert cue_score("button button button") == pytest.approx(1.0)


def test_threshold_is_configurable() -> None:
    lines = [Line(0.0, 1.0, "the sidebar")]
    assert verbal_cue_checkpoints(lines, threshold=1.5) == []


def test_label_truncated_to_80_chars() -> None:
    [cp] = verbal_cue_checkpoints([Line(0.0, 1.0, "dashboard " + "x" * 200)])
    assert len(cp.label) == 80
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_cues.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** — `src/peeklet/cues.py`

```python
"""Verbal-cue checkpoints: transcript lines that reference something on screen.

Lexicon restored from the former ``transcript_trigger.py`` (develop, before #30).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from peeklet.types import Checkpoint

if TYPE_CHECKING:
    from peeklet.types import Line

HIGH_SIGNAL_WORDS: frozenset[str] = frozenset(
    {
        "chart", "graph", "dashboard", "button", "menu", "sidebar", "diagram", "screen",
        "page", "modal", "dropdown", "tab", "panel", "table", "form", "icon", "window",
        "popup", "field", "toolbar", "flowchart", "heatmap", "rollout", "revamp", "rewrite",
        "view", "row", "column", "filter", "toggle", "status", "label", "beta", "empty",
        "loading", "cost", "count", "banner", "badge", "card", "list", "header", "footer",
        "input", "search", "link",
    }
)
MEDIUM_SIGNAL_WORDS: frozenset[str] = frozenset(
    {
        "notice", "look", "see", "click", "shown", "display", "hover", "scroll", "select",
        "drag", "zoom", "highlight", "observe", "watch",
    }
)
DEICTIC_WORDS: frozenset[str] = frozenset({"this", "that", "here", "there"})

_HIGH_WEIGHT = 1.0
_MEDIUM_WEIGHT = 0.5
_DEICTIC_WEIGHT = 0.3
_PUNCT_RE = re.compile(r"[^\w\s]")


def cue_score(text: str) -> float:
    """Weighted count of distinct cue words (high 1.0, medium 0.5, deictic 0.3)."""
    words = set(_PUNCT_RE.sub(" ", text.lower()).split())
    return (
        _HIGH_WEIGHT * len(words & HIGH_SIGNAL_WORDS)
        + _MEDIUM_WEIGHT * len(words & MEDIUM_SIGNAL_WORDS)
        + _DEICTIC_WEIGHT * len(words & DEICTIC_WORDS)
    )


def verbal_cue_checkpoints(lines: list[Line], threshold: float = 1.0) -> list[Checkpoint]:
    """One checkpoint at the start of each line whose cue score reaches the threshold."""
    return [
        Checkpoint(t=ln.start, kind="verbal_cue", label=ln.text[:80])
        for ln in lines
        if cue_score(ln.text) >= threshold
    ]
```

(Run `uv run ruff format src/peeklet/cues.py`; it will reflow the word sets one-per-line, which is fine.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_cues.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/cues.py tests/unit/test_cues.py
git commit -m "feat: verbal-cue checkpoints from restored trigger lexicon

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Speech-onset checkpoints

**Files:**
- Modify: `src/peeklet/speech.py`
- Test: `tests/unit/test_speech.py`

**Interfaces:**
- Produces: `detect_speech_segments(media_path: Path, chunk_ms: int = 500, silence_threshold_dbfs: float = -40.0) -> list[tuple[float, float]]`; `speech_onset_checkpoints(ranges: list[tuple[float, float]], min_pause_seconds: float) -> list[Checkpoint]`.

- [ ] **Step 1: Write failing tests** — in `tests/unit/test_speech.py`, update the existing `TestSpeechSilenceDetection` assertions to expect `(start, end)` tuples instead of `Line` objects, then add:

```python
from peeklet.speech import speech_onset_checkpoints


def test_first_range_is_an_onset() -> None:
    [cp] = speech_onset_checkpoints([(0.5, 3.0)], min_pause_seconds=1.5)
    assert (cp.t, cp.kind) == (0.5, "speech_onset")


def test_onset_requires_minimum_pause() -> None:
    ranges = [(0.0, 2.0), (2.5, 4.0), (6.0, 8.0)]
    cps = speech_onset_checkpoints(ranges, min_pause_seconds=1.5)
    assert [c.t for c in cps] == [0.0, 6.0]


def test_pause_exactly_at_threshold_counts() -> None:
    cps = speech_onset_checkpoints([(0.0, 1.0), (2.5, 3.0)], min_pause_seconds=1.5)
    assert [c.t for c in cps] == [0.0, 2.5]


def test_no_ranges() -> None:
    assert speech_onset_checkpoints([], min_pause_seconds=1.5) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_speech.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement** in `src/peeklet/speech.py`

Change `detect_speech_segments` to return `list[tuple[float, float]]` (replace the final `return [TranscriptSegment(...)...]` with `return speech_ranges`; drop the `Line` import), and add:

```python
def speech_onset_checkpoints(
    ranges: list[tuple[float, float]], min_pause_seconds: float
) -> list[Checkpoint]:
    """Onsets of speech that follow at least min_pause_seconds of silence.

    The first speech range always counts (silence before the recording starts).
    """
    out: list[Checkpoint] = []
    prev_end: float | None = None
    for start, end in ranges:
        if prev_end is None or start - prev_end >= min_pause_seconds:
            out.append(Checkpoint(t=start, kind="speech_onset"))
        prev_end = end
    return out
```

Import: `from peeklet.types import Checkpoint`. Also replace the `_check_audio_deps` error text with `"Speech detection requires pydub. Reinstall peeklet."`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_speech.py -q && uv run mypy src/peeklet/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: speech-onset checkpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Change detector and OCR coordinate fix

**Files:**
- Modify: `src/peeklet/change.py`, `src/peeklet/screen.py`
- Test: `tests/unit/test_change.py`, `tests/unit/test_screen_ocr.py`

**Interfaces:**
- Produces (in `peeklet.change`):

```python
@dataclass(frozen=True, slots=True)
class ChangeResult: changed: bool; change: float; key: str
class ChangeDetector:
    def __init__(self, cfg: PeekletConfig) -> None
    def update(self, frame: np.ndarray) -> ChangeResult
```

- Produces (in `peeklet.screen`, renamed public): `ocr_word_boxes(frame, max_dim) -> list[WordBox]` (box coordinates in the **original** frame's pixel space), `is_low_info_frame(frame, boxes, cfg) -> bool` (takes precomputed boxes), `count_text_lines`, `count_occupied_grid_cells`, `edge_pixel_ratio`, `class MissingDependencyError(RuntimeError)`, `require_tesseract() -> None`.

- [ ] **Step 1: Write failing tests** — append to `tests/unit/test_change.py`

```python
import numpy as np

from peeklet.change import ChangeDetector
from peeklet.config import PeekletConfig


def _solid(rgb: tuple[int, int, int], size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def test_first_frame_is_a_change() -> None:
    res = ChangeDetector(PeekletConfig()).update(_solid((10, 20, 30)))
    assert res.changed and res.change == 1.0


def test_identical_frame_is_not_a_change() -> None:
    det = ChangeDetector(PeekletConfig())
    det.update(_solid((10, 20, 30)))
    res = det.update(_solid((10, 20, 30)))
    assert not res.changed and res.change == 0.0


def test_large_change_detected() -> None:
    det = ChangeDetector(PeekletConfig())
    det.update(_solid((10, 10, 10)))
    rng = np.random.default_rng(0)
    res = det.update(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    assert res.changed and 0.0 < res.change <= 1.0


def test_key_stable_for_identical_frames_and_differs_by_colour() -> None:
    det = ChangeDetector(PeekletConfig())
    k1 = det.update(_solid((50, 50, 50))).key
    k2 = det.update(_solid((50, 50, 50))).key
    k3 = det.update(_solid((90, 50, 50))).key
    assert k1 == k2 != k3


def test_frames_are_downscaled_before_comparison() -> None:
    det = ChangeDetector(PeekletConfig(change_max_dim=64))
    res = det.update(_solid((1, 2, 3), size=1000))
    assert res.changed
```

Append to `tests/unit/test_screen_ocr.py`:

```python
from unittest import mock

import numpy as np
import pytest

from peeklet import screen
from peeklet.screen import MissingDependencyError, ocr_word_boxes, require_tesseract


def test_ocr_boxes_scaled_back_to_original_coordinates() -> None:
    frame = np.zeros((2000, 4000, 3), dtype=np.uint8)  # downscaled by 0.5 for max_dim=2000
    data = {"text": ["Hello"], "conf": ["90"], "left": [10], "top": [20], "width": [30],
            "height": [40]}
    with mock.patch.object(screen, "pytesseract") as pt:
        pt.image_to_data.return_value = data
        pt.Output.DICT = "dict"
        [box] = ocr_word_boxes(frame, 2000)
    assert (box.x, box.y, box.w, box.h) == (20, 40, 60, 80)


def test_require_tesseract_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(screen.shutil, "which", lambda _: None)
    with pytest.raises(MissingDependencyError, match="brew install tesseract"):
        require_tesseract()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_change.py tests/unit/test_screen_ocr.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `ChangeDetector`** — append to `src/peeklet/change.py`

```python
@dataclass(frozen=True, slots=True)
class ChangeResult:
    """Outcome of comparing one sample against the current reference frame."""

    changed: bool
    change: float  # 1 - SSIM vs reference; 1.0 for the first frame; 0.0 on hash match
    key: str  # OCR cache key: masked pHash + rounded mean colour


class ChangeDetector:
    """Masking -> pHash -> SSIM cascade over a stream of samples.

    The reference is the last sample that counted as a change, so slow drift
    eventually registers. Frames are downscaled to cfg.change_max_dim first.
    """

    def __init__(self, cfg: PeekletConfig) -> None:
        self._cfg = cfg
        self._mask = AdaptiveMask(
            block_size=cfg.mask_block_size,
            window_size=cfg.mask_window_size,
            noise_threshold=cfg.mask_noise_threshold,
        )
        self._ref: np.ndarray | None = None
        self._ref_hash = ""
        self._ref_mean = (0.0, 0.0, 0.0)

    def update(self, frame: np.ndarray) -> ChangeResult:
        small = downscale_to_max_dim(frame, self._cfg.change_max_dim)
        masked, _ = self._mask.apply(small)
        phash = compute_phash(masked)
        mean = (
            float(masked[:, :, 0].mean()),
            float(masked[:, :, 1].mean()),
            float(masked[:, :, 2].mean()),
        )
        key = f"{phash}:{round(mean[0])}:{round(mean[1])}:{round(mean[2])}"
        if self._ref is None or self._ref.shape != masked.shape:
            self._set_ref(masked, phash, mean)
            return ChangeResult(changed=True, change=1.0, key=key)
        mean_diff = max(abs(a - b) for a, b in zip(mean, self._ref_mean, strict=True))
        if hashes_match(phash, self._ref_hash) and mean_diff < 5.0:
            return ChangeResult(changed=False, change=0.0, key=key)
        cmp = compare_frames(masked, self._ref, block_size=self._cfg.mask_block_size)
        changed = (
            cmp.ssim_score <= self._cfg.ssim_threshold
            or len(cmp.changed_regions) > self._cfg.min_changed_blocks
        )
        if changed:
            self._set_ref(masked, phash, mean)
        return ChangeResult(changed=changed, change=min(1.0, max(0.0, cmp.change_score)), key=key)

    def _set_ref(self, masked: np.ndarray, phash: str, mean: tuple[float, float, float]) -> None:
        self._ref = masked
        self._ref_hash = phash
        self._ref_mean = mean
```

Imports to add at top of `change.py`: `from typing import TYPE_CHECKING`, `from peeklet.image_utils import downscale_to_max_dim`, and under `TYPE_CHECKING`: `from peeklet.config import PeekletConfig`.

- [ ] **Step 4: Implement OCR changes in `screen.py`**

- Rename `_ocr_word_boxes` → `ocr_word_boxes`, `_count_text_lines` → `count_text_lines`, `_count_occupied_grid_cells` → `count_occupied_grid_cells`, `_edge_pixel_ratio` → `edge_pixel_ratio`, `_is_low_info_frame` → `is_low_info_frame` (update tests).
- In `ocr_word_boxes`, after computing `downscaled`, compute `scale = frame.shape[1] / downscaled.shape[1]` and build each box with `x=int(round(int(x) * scale))`, same for `y`, `w`, `h`.
- Change `is_low_info_frame` to take precomputed boxes:

```python
def is_low_info_frame(frame: np.ndarray, boxes: list[WordBox], config: PeekletConfig) -> bool:
    """Triple-AND rejector: low on text lines, occupied grid cells AND edge density."""
    if edge_pixel_ratio(frame) >= config.min_edge_ratio:
        return False
    if count_text_lines(boxes) >= config.min_text_lines:
        return False
    return count_occupied_grid_cells(boxes, frame.shape) < config.min_grid_cells
```

- Add:

```python
import shutil


class MissingDependencyError(RuntimeError):
    """A required system dependency (tesseract) is not installed."""


def require_tesseract() -> None:
    """Fail fast with an install hint if pytesseract or the tesseract binary is missing."""
    if pytesseract is None or shutil.which("tesseract") is None:
        raise MissingDependencyError(
            "Peeklet needs the tesseract OCR binary. Install it with "
            "`brew install tesseract` (macOS) or `apt install tesseract-ocr` (Debian/Ubuntu)."
        )
```

Update existing `TestIsLowInfoFrame` tests to compute boxes first (`boxes = ocr_word_boxes(frame, cfg.ocr_max_dim)`) and pass them.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit -q && uv run mypy src/peeklet/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: streaming ChangeDetector; OCR boxes in original coordinates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Streaming screen pass

**Files:**
- Create: `src/peeklet/screens.py`
- Test: `tests/unit/test_screens.py`

**Interfaces:**
- Consumes: `ChangeDetector`, `ChangeResult` (Task 8); `WordBox`, `is_low_info_frame`, `compute_fingerprint`, `FingerprintIndex`, `Fingerprint` (`peeklet.screen`); `encode_jpeg`; `Checkpoint`, `Screen`, `format_hms`.
- Produces:

```python
OcrFn = Callable[[np.ndarray], list[WordBox]]
@dataclass(frozen=True, slots=True) class Sample: t: float; frame: np.ndarray
@dataclass(slots=True) class ScreenPass: screens: list[Screen]; events: list[tuple[float, str | None]]; warnings: list[str]
def build_screens(samples: Iterable[Sample], checkpoints: Sequence[Checkpoint],
                  cfg: PeekletConfig, ocr: OcrFn, video_end: float) -> ScreenPass
```

- [ ] **Step 1: Write failing tests** — `tests/unit/test_screens.py`

```python
"""Tests for the streaming screen pass (OCR stubbed by colour)."""

from __future__ import annotations

import numpy as np

from peeklet.config import PeekletConfig
from peeklet.screen import WordBox
from peeklet.screens import Sample, build_screens
from peeklet.types import Checkpoint

A = (200, 20, 20)
A_RICH = (202, 20, 20)  # same screen as A, more OCR words (same pHash, different mean)
B = (20, 200, 20)
GALLERY = (0, 0, 0)


def frame(rgb: tuple[int, int, int]) -> np.ndarray:
    return np.full((64, 64, 3), rgb, dtype=np.uint8)


def boxes_for(heading: str, n_lines: int) -> list[WordBox]:
    out = [WordBox(heading, 95.0, 20, 2, 30, 6), WordBox("Home", 95.0, 0, 30, 8, 4)]
    out += [WordBox(f"word{i}", 95.0, 12, 10 + i * 5, 20, 4) for i in range(n_lines)]
    return out


def fake_ocr(f: np.ndarray) -> list[WordBox]:
    rgb = tuple(int(v) for v in f[0, 0])
    return {
        A: boxes_for("ScreenA", 12),
        A_RICH: boxes_for("ScreenA", 20),
        B: boxes_for("ScreenB", 12),
    }.get(rgb, [])  # type: ignore[arg-type]


def run(seq: list[tuple[int, int, int]], checkpoints: list[Checkpoint] | None = None,
        **cfg_kw: float):
    samples = [Sample(t=float(i), frame=frame(rgb)) for i, rgb in enumerate(seq)]
    return build_screens(samples, checkpoints or [], PeekletConfig(**cfg_kw), fake_ocr,
                         video_end=float(len(seq)))


def test_screen_returning_later_is_one_screen_with_two_occurrences() -> None:
    res = run([A] * 3 + [B] * 3 + [A] * 3)
    assert len(res.screens) == 2
    a = next(s for s in res.screens if "ScreenA" in s.ocr_text)
    assert a.occurrences == [(0.0, 3.0), (6.0, 9.0)]


def test_different_screens_not_merged() -> None:
    res = run([A] * 2 + [B] * 2)
    assert {s.ocr_text.split()[0] for s in res.screens} == {"ScreenA", "ScreenB"}


def test_low_info_frames_create_no_screen_and_end_occurrence() -> None:
    res = run([A] * 3 + [GALLERY] * 3 + [A] * 2)
    [a] = res.screens
    assert a.occurrences == [(0.0, 3.0), (6.0, 8.0)]
    assert (3.0, None) in res.events


def test_empty_ocr_frames_never_merge() -> None:
    def ocr_body_only(f: np.ndarray) -> list[WordBox]:
        # 12 text lines (passes the rejector) but all below the midline and right of the
        # sidebar band, so Part A (url, heading, sidebar) is empty.
        return [WordBox(f"w{i}", 95.0, 60, 110 + i * 7, 20, 4) for i in range(12)]

    big = [np.full((200, 200, 3), rgb, dtype=np.uint8) for rgb in (A, B)]
    samples = [Sample(0.0, big[0]), Sample(1.0, big[1])]
    res = build_screens(samples, [], PeekletConfig(), ocr_body_only, video_end=2.0)
    assert len(res.screens) == 2


def test_checkpoint_window_picks_richest_frame() -> None:
    seq = [A] * 12 + [A_RICH] + [A] * 8  # A_RICH at t=12 is not a change (hash match)
    res = run(seq, [Checkpoint(t=10.0, kind="speech_onset")])
    [a] = res.screens
    assert a.frame_t == 12.0
    assert a.word_count == len(boxes_for("ScreenA", 20))


def test_checkpoint_window_ties_prefer_closest_sample() -> None:
    res = run([A] * 20, [Checkpoint(t=10.0, kind="verbal_cue")])
    assert (10.0, res.screens[0].id) in res.events


def test_all_low_info_window_warns() -> None:
    res = run([GALLERY] * 20, [Checkpoint(t=10.0, kind="speech_onset")])
    assert res.screens == []
    assert any("no usable frame" in w for w in res.warnings)


def test_ocr_runs_once_per_visual_state() -> None:
    calls: list[int] = []

    def counting_ocr(f: np.ndarray) -> list[WordBox]:
        calls.append(1)
        return fake_ocr(f)

    samples = [Sample(float(i), frame(A)) for i in range(30)]
    cps = [Checkpoint(t=float(t), kind="verbal_cue") for t in (5, 10, 15, 20, 25)]
    build_screens(samples, cps, PeekletConfig(), counting_ocr, video_end=30.0)
    assert len(calls) == 1


def test_chosen_frame_is_jpeg() -> None:
    [a] = run([A] * 2).screens
    assert a.image_jpeg[:2] == b"\xff\xd8"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_screens.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** — `src/peeklet/screens.py`

```python
"""Streaming screen pass: candidates -> OCR + fingerprint -> global screens.

Memory stays bounded: per candidate we keep (t, screen id); full frames are kept
only as each screen's current best frame (JPEG bytes). OCR results are cached by
ChangeResult.key so a static screen is OCR'd once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from peeklet.change import ChangeDetector
from peeklet.image_utils import encode_jpeg
from peeklet.screen import FingerprintIndex, compute_fingerprint, is_low_info_frame
from peeklet.types import Screen, format_hms

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy as np

    from peeklet.config import PeekletConfig
    from peeklet.screen import Fingerprint, WordBox
    from peeklet.types import Checkpoint

    OcrFn = Callable[[np.ndarray], list[WordBox]]


@dataclass(frozen=True, slots=True)
class Sample:
    """One decoded video sample."""

    t: float
    frame: np.ndarray


@dataclass(slots=True)
class ScreenPass:
    """Result of the screen pass."""

    screens: list[Screen]
    events: list[tuple[float, str | None]]  # time-ordered; None = low-info frame
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class _Analysis:
    low_info: bool
    fingerprint: Fingerprint
    word_count: int
    ocr_text: str


@dataclass(slots=True)
class _Window:
    cp: Checkpoint
    best_key: tuple[int, float] | None = None
    best: tuple[float, _Analysis, np.ndarray, float] | None = None


class _Tracker:
    def __init__(self, cfg: PeekletConfig, ocr: OcrFn) -> None:
        self.cfg = cfg
        self.ocr = ocr
        self.cache: dict[str, _Analysis] = {}
        self.index = FingerprintIndex(
            phash_threshold=cfg.phash_threshold, ocr_field_min_chars=cfg.ocr_field_min_chars
        )
        self.screens: dict[str, Screen] = {}
        self.events: dict[float, str | None] = {}
        self.warnings: list[str] = []

    def analyze(self, frame: np.ndarray, key: str) -> _Analysis:
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        boxes = self.ocr(frame)
        result = _Analysis(
            low_info=is_low_info_frame(frame, boxes, self.cfg),
            fingerprint=compute_fingerprint(frame, boxes),  # type: ignore[arg-type]
            word_count=len(boxes),
            ocr_text=" ".join(b.text for b in boxes),
        )
        self.cache[key] = result
        return result

    def assign(self, t: float, a: _Analysis, frame: np.ndarray, change: float) -> None:
        if a.low_info:
            self.events[t] = None
            return
        sid = self.index.lookup(a.fingerprint)
        if sid is None:
            sid = f"S{len(self.screens) + 1}"
            self.index.register(a.fingerprint, screen_id=sid)
            self.screens[sid] = Screen(
                id=sid,
                image_jpeg=encode_jpeg(frame, self.cfg.jpeg_quality),
                frame_t=t,
                ocr_text=a.ocr_text,
                word_count=a.word_count,
                first_change=change,
            )
        else:
            s = self.screens[sid]
            if a.word_count > s.word_count or (a.word_count == s.word_count and t > s.frame_t):
                s.image_jpeg = encode_jpeg(frame, self.cfg.jpeg_quality)
                s.frame_t = t
                s.ocr_text = a.ocr_text
                s.word_count = a.word_count
        self.events[t] = sid

    def close(self, win: _Window) -> None:
        if win.best is None:
            self.warnings.append(
                f"no usable frame within ±{self.cfg.checkpoint_window_seconds:g}s of "
                f"{win.cp.kind} checkpoint at {format_hms(win.cp.t)}"
            )
            return
        t, a, frame, change = win.best
        self.assign(t, a, frame, change)


def build_screens(
    samples: Iterable[Sample],
    checkpoints: Sequence[Checkpoint],
    cfg: PeekletConfig,
    ocr: OcrFn,
    video_end: float,
) -> ScreenPass:
    """Run the streaming screen pass over time-ordered samples."""
    tracker = _Tracker(cfg, ocr)
    detector = ChangeDetector(cfg)
    half = cfg.checkpoint_window_seconds
    pending = sorted(checkpoints, key=lambda c: c.t)
    nxt = 0
    open_windows: list[_Window] = []

    for sample in samples:
        res = detector.update(sample.frame)
        while nxt < len(pending) and pending[nxt].t - half <= sample.t:
            open_windows.append(_Window(pending[nxt]))
            nxt += 1
        still_open: list[_Window] = []
        for win in open_windows:
            if sample.t > win.cp.t + half:
                tracker.close(win)
            else:
                still_open.append(win)
        open_windows = still_open
        active = [w for w in open_windows if abs(sample.t - w.cp.t) <= half]
        if not res.changed and not active:
            continue
        analysis = tracker.analyze(sample.frame, res.key)
        if res.changed:
            tracker.assign(sample.t, analysis, sample.frame, res.change)
        if analysis.low_info:
            continue
        for win in active:
            rank = (analysis.word_count, -abs(sample.t - win.cp.t))
            if win.best_key is None or rank > win.best_key:
                win.best_key = rank
                win.best = (sample.t, analysis, sample.frame, res.change)

    for win in open_windows:
        tracker.close(win)
    for cp in pending[nxt:]:
        tracker.close(_Window(cp))

    ordered = sorted(tracker.events.items())
    _fill_occurrences(tracker.screens, ordered, video_end)
    return ScreenPass(
        screens=list(tracker.screens.values()), events=ordered, warnings=tracker.warnings
    )


def _fill_occurrences(
    screens: dict[str, Screen], ordered: list[tuple[float, str | None]], video_end: float
) -> None:
    for i, (t, sid) in enumerate(ordered):
        if sid is None:
            continue
        end = ordered[i + 1][0] if i + 1 < len(ordered) else max(video_end, t)
        occ = screens[sid].occurrences
        if occ and abs(occ[-1][1] - t) < 1e-9:
            occ[-1] = (occ[-1][0], end)
        else:
            occ.append((t, end))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_screens.py -q && uv run mypy src/peeklet/`
Expected: PASS. If `test_checkpoint_window_picks_richest_frame` fails because `A_RICH` registers as a change, confirm `hashes_match` on two flat frames returns True and the mean diff (2) is < 5; adjust `A_RICH` to `(201, 20, 20)` only if needed.

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/screens.py tests/unit/test_screens.py
git commit -m "feat: streaming screen pass with checkpoint windows and OCR cache

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Alignment

**Files:**
- Create: `src/peeklet/align.py`
- Test: `tests/unit/test_align.py`

**Interfaces:**
- Produces: `screen_at(screens: Sequence[Screen], t: float) -> str | None`; `align_lines(lines: Sequence[Line], screens: Sequence[Screen], lead_seconds: float) -> list[str | None]`.

- [ ] **Step 1: Write failing tests** — `tests/unit/test_align.py`

```python
"""Tests for line-to-screen alignment."""

from __future__ import annotations

from peeklet.align import align_lines, screen_at
from peeklet.types import Line, Screen


def scr(sid: str, occ: list[tuple[float, float]]) -> Screen:
    return Screen(id=sid, image_jpeg=b"", frame_t=occ[0][0], ocr_text="", word_count=0,
                  first_change=1.0, occurrences=occ)


SCREENS = [scr("S1", [(0.0, 10.0), (30.0, 40.0)]), scr("S2", [(10.0, 20.0)])]


def test_screen_at() -> None:
    assert screen_at(SCREENS, 5.0) == "S1"
    assert screen_at(SCREENS, 10.0) == "S2"
    assert screen_at(SCREENS, 25.0) is None


def test_line_gets_largest_overlap() -> None:
    assert align_lines([Line(7.0, 15.0, "x")], SCREENS, lead_seconds=0.0) == ["S2"]


def test_lead_window_extends_line_end() -> None:
    assert align_lines([Line(8.0, 9.5, "now I'll click")], SCREENS, 1.5) == ["S1"]
    assert align_lines([Line(20.5, 29.0, "x")], SCREENS, 1.5) == ["S1"]


def test_no_overlap_gives_none() -> None:
    assert align_lines([Line(21.0, 25.0, "x")], SCREENS, 1.5) == [None]


def test_tie_prefers_screen_that_appeared_first() -> None:
    screens = [scr("S2", [(10.0, 20.0)]), scr("S1", [(0.0, 10.0)])]
    assert align_lines([Line(8.0, 12.0, "x")], screens, 0.0) == ["S1"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_align.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement** — `src/peeklet/align.py`

```python
"""Align transcript lines to the screen on display while they were spoken."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peeklet.types import Line, Screen

_EPS = 1e-9


def screen_at(screens: Sequence[Screen], t: float) -> str | None:
    """Id of the screen whose occurrence covers t (half-open intervals), else None."""
    for s in screens:
        for start, end in s.occurrences:
            if start <= t < end:
                return s.id
    return None


def align_lines(
    lines: Sequence[Line], screens: Sequence[Screen], lead_seconds: float
) -> list[str | None]:
    """At most one screen per line: largest overlap with [start, end + lead]."""
    first_seen = {
        s.id: min((a for a, _ in s.occurrences), default=math.inf) for s in screens
    }
    out: list[str | None] = []
    for ln in lines:
        lo, hi = ln.start, ln.end + lead_seconds
        best: str | None = None
        best_overlap = 0.0
        for s in screens:
            overlap = sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in s.occurrences)
            if overlap <= _EPS:
                continue
            better = overlap > best_overlap + _EPS
            tie = best is not None and abs(overlap - best_overlap) <= _EPS
            if better or (tie and first_seen[s.id] < first_seen[best]):  # type: ignore[index]
                best, best_overlap = s.id, overlap
        out.append(best)
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_align.py -q && uv run mypy src/peeklet/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/align.py tests/unit/test_align.py
git commit -m "feat: align each transcript line to at most one screen

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Heuristic scoring and selection

**Files:**
- Create: `src/peeklet/score.py`
- Test: `tests/unit/test_score.py`

**Interfaces:**
- Consumes: `screen_at` (Task 10), `ScoreWeights`, `Screen`, `Line`, `Checkpoint`, `ScreenJudgment`.
- Produces:

```python
def content_words(text: str) -> set[str]
@dataclass(frozen=True, slots=True) class ScreenSignals: references, text_overlap, onsets,
    verbal_cues, visual_change, time_on_screen: float; anchored: bool; def as_dict(self) -> dict[str, float | bool]
def compute_signals(screens, lines, line_screens, checkpoints) -> dict[str, ScreenSignals]
def score_screens(signals: Mapping[str, ScreenSignals], weights: ScoreWeights, anchor_bonus: float) -> dict[str, float]
def first_seen(screen: Screen) -> float
def rank_screens(screens, scores) -> list[Screen]          # score desc, then first appearance
def select_screens(screens, scores, signals, judgments: Mapping[str, ScreenJudgment | None] | None,
                   max_images: int) -> list[Screen]       # result ordered by first appearance
```

- [ ] **Step 1: Write failing tests** — `tests/unit/test_score.py`

```python
"""Tests for heuristic scoring and selection."""

from __future__ import annotations

import math

import pytest

from peeklet.config import ScoreWeights
from peeklet.score import (
    ScreenSignals,
    compute_signals,
    content_words,
    rank_screens,
    score_screens,
    select_screens,
)
from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment


def scr(sid: str, occ: list[tuple[float, float]], ocr: str = "", change: float = 0.5) -> Screen:
    return Screen(id=sid, image_jpeg=b"", frame_t=occ[0][0], ocr_text=ocr, word_count=1,
                  first_change=change, occurrences=occ)


def test_content_words_drops_stopwords_and_short_tokens() -> None:
    assert content_words("The Billing settings, on a page!") == {"billing", "settings", "page"}


def test_signals_each_scaled_to_unit_range() -> None:
    s1 = scr("S1", [(0.0, 30.0)], ocr="Billing Settings Invoices", change=0.8)
    s2 = scr("S2", [(30.0, 40.0)], ocr="Welcome")
    lines = [Line(1.0, 2.0, "open billing settings"), Line(5.0, 6.0, "and invoices"),
             Line(31.0, 32.0, "hello")]
    cps = [Checkpoint(1.0, "speech_onset"), Checkpoint(5.0, "verbal_cue"),
           Checkpoint(31.0, "speech_onset"), Checkpoint(32.0, "action_item", "x")]
    sig = compute_signals([s1, s2], lines, ["S1", "S1", "S2"], cps)
    assert sig["S1"].references == pytest.approx(1.0)
    assert sig["S2"].references == pytest.approx(math.log1p(1) / math.log1p(2))
    assert sig["S1"].text_overlap > sig["S2"].text_overlap == 0.0
    assert sig["S1"].onsets == sig["S2"].onsets == pytest.approx(1.0)
    assert sig["S1"].verbal_cues == pytest.approx(1.0) and sig["S2"].verbal_cues == 0.0
    assert sig["S1"].visual_change == pytest.approx(0.8)
    assert sig["S1"].time_on_screen == pytest.approx(1.0)
    assert sig["S2"].anchored and not sig["S1"].anchored


def test_score_is_weighted_sum_plus_anchor_bonus() -> None:
    sig = {"S1": ScreenSignals(1, 1, 1, 1, 1, 1, anchored=True),
           "S2": ScreenSignals(1, 0, 0, 0, 0, 0, anchored=False)}
    scores = score_screens(sig, ScoreWeights(), anchor_bonus=1.0)
    assert scores["S1"] == pytest.approx(2.0)
    assert scores["S2"] == pytest.approx(0.25)


def test_rank_ties_prefer_first_appearance() -> None:
    a, b = scr("S1", [(5.0, 6.0)]), scr("S2", [(1.0, 2.0)])
    assert [s.id for s in rank_screens([a, b], {"S1": 0.5, "S2": 0.5})] == ["S2", "S1"]


def _sig(anchored: bool = False) -> ScreenSignals:
    return ScreenSignals(0, 0, 0, 0, 0, 0, anchored=anchored)


def test_select_without_llm_takes_top_n_in_time_order() -> None:
    screens = [scr("S1", [(0.0, 1.0)]), scr("S2", [(1.0, 2.0)]), scr("S3", [(2.0, 3.0)])]
    scores = {"S1": 0.1, "S2": 0.9, "S3": 0.5}
    sig = {s.id: _sig() for s in screens}
    kept = select_screens(screens, scores, sig, None, max_images=2)
    assert [s.id for s in kept] == ["S2", "S3"]


def test_select_with_llm_uses_verdicts_and_anchor_override() -> None:
    screens = [scr("S1", [(0.0, 1.0)]), scr("S2", [(1.0, 2.0)]), scr("S3", [(2.0, 3.0)]),
               scr("S4", [(3.0, 4.0)])]
    scores = {"S1": 0.9, "S2": 0.8, "S3": 0.7, "S4": 0.6}
    sig = {"S1": _sig(), "S2": _sig(anchored=True), "S3": _sig(), "S4": _sig()}
    judgments = {
        "S1": ScreenJudgment(False, "not needed", ""),
        "S2": ScreenJudgment(False, "not needed", ""),  # anchored -> kept anyway
        "S3": None,  # failed call -> include
        # S4 not shortlisted -> not eligible
    }
    kept = select_screens(screens, scores, sig, judgments, max_images=5)
    assert [s.id for s in kept] == ["S2", "S3"]


def test_select_zero_images() -> None:
    screens = [scr("S1", [(0.0, 1.0)])]
    assert select_screens(screens, {"S1": 1.0}, {"S1": _sig()}, None, max_images=0) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_score.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement** — `src/peeklet/score.py`

```python
"""Heuristic screen scoring and final selection."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from peeklet.align import screen_at

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from peeklet.config import ScoreWeights
    from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the and for are but not you all any can had her was one our out has have this that with "
    "they from what when your will there their about would which into them then than some "
    "just like here okay yeah so now going get got see look let lets it's its also".split()
)


def content_words(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of length >= 3 that are not stopwords."""
    return {w for w in _TOKEN_RE.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


@dataclass(frozen=True, slots=True)
class ScreenSignals:
    """Per-screen ranking signals, each in [0, 1]."""

    references: float
    text_overlap: float
    onsets: float
    verbal_cues: float
    visual_change: float
    time_on_screen: float
    anchored: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _log_scaled(counts: Mapping[str, float], ids: Sequence[str]) -> dict[str, float]:
    top = max((math.log1p(counts.get(i, 0.0)) for i in ids), default=0.0)
    return {i: (math.log1p(counts.get(i, 0.0)) / top if top > 0 else 0.0) for i in ids}


def compute_signals(
    screens: Sequence[Screen],
    lines: Sequence[Line],
    line_screens: Sequence[str | None],
    checkpoints: Sequence[Checkpoint],
) -> dict[str, ScreenSignals]:
    """Compute all ranking signals for every screen."""
    ids = [s.id for s in screens]
    refs = Counter(sid for sid in line_screens if sid is not None)
    words: dict[str, set[str]] = {i: set() for i in ids}
    for ln, sid in zip(lines, line_screens, strict=True):
        if sid is not None:
            words[sid] |= content_words(ln.text)
    kind_counts: dict[str, Counter[str]] = {
        "speech_onset": Counter(), "verbal_cue": Counter(), "action_item": Counter()
    }
    for cp in checkpoints:
        sid = screen_at(screens, cp.t)
        if sid is not None:
            kind_counts[cp.kind][sid] += 1
    dwell = {s.id: float(sum(b - a for a, b in s.occurrences)) for s in screens}
    references = _log_scaled(refs, ids)
    onsets = _log_scaled(kind_counts["speech_onset"], ids)
    cues = _log_scaled(kind_counts["verbal_cue"], ids)
    time_on = _log_scaled(dwell, ids)
    out: dict[str, ScreenSignals] = {}
    for s in screens:
        ocr_words = content_words(s.ocr_text)
        union = words[s.id] | ocr_words
        overlap = len(words[s.id] & ocr_words) / len(union) if union else 0.0
        out[s.id] = ScreenSignals(
            references=references[s.id],
            text_overlap=overlap,
            onsets=onsets[s.id],
            verbal_cues=cues[s.id],
            visual_change=min(1.0, max(0.0, s.first_change)),
            time_on_screen=time_on[s.id],
            anchored=kind_counts["action_item"][s.id] > 0,
        )
    return out


def score_screens(
    signals: Mapping[str, ScreenSignals], weights: ScoreWeights, anchor_bonus: float
) -> dict[str, float]:
    """Weighted sum of signals, plus anchor_bonus for screens shown at an action item."""
    return {
        sid: weights.references * g.references
        + weights.text_overlap * g.text_overlap
        + weights.onsets * g.onsets
        + weights.verbal_cues * g.verbal_cues
        + weights.visual_change * g.visual_change
        + weights.time_on_screen * g.time_on_screen
        + (anchor_bonus if g.anchored else 0.0)
        for sid, g in signals.items()
    }


def first_seen(screen: Screen) -> float:
    return min((a for a, _ in screen.occurrences), default=screen.frame_t)


def rank_screens(screens: Sequence[Screen], scores: Mapping[str, float]) -> list[Screen]:
    """Highest score first; ties go to the screen that appeared first."""
    return sorted(screens, key=lambda s: (-scores[s.id], first_seen(s)))


def select_screens(
    screens: Sequence[Screen],
    scores: Mapping[str, float],
    signals: Mapping[str, ScreenSignals],
    judgments: Mapping[str, ScreenJudgment | None] | None,
    max_images: int,
) -> list[Screen]:
    """Pick at most max_images screens; result is ordered by first appearance.

    Without judgments: top by score. With judgments (keys = shortlisted ids): eligible
    if the verdict is include, the call failed (None), or the screen is anchored.
    """
    ranked = rank_screens(screens, scores)
    if judgments is not None:
        ranked = [
            s
            for s in ranked
            if s.id in judgments
            and (
                judgments[s.id] is None
                or judgments[s.id].include  # type: ignore[union-attr]
                or signals[s.id].anchored
            )
        ]
    return sorted(ranked[:max_images], key=first_seen)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_score.py -q && uv run mypy src/peeklet/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/score.py tests/unit/test_score.py
git commit -m "feat: heuristic screen scoring and budgeted selection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: LLM judge + describe (multi-provider, cached)

**Files:**
- Rewrite: `src/peeklet/llm/base.py`, `src/peeklet/llm/anthropic_client.py`, `src/peeklet/llm/openai_client.py`, `src/peeklet/llm/openrouter_client.py`
- Modify: `src/peeklet/types.py` (delete `Moment`)
- Rewrite: `tests/unit/test_llm.py`; delete `test_moment_*` from `tests/unit/test_types.py`

**Interfaces:**
- Consumes: `ScreenJudgment`.
- Produces (in `peeklet.llm.base`):

```python
PROMPT_VERSION: str
SYSTEM_PROMPT: str
class LLMResponseError(RuntimeError)
class LLMClient(Protocol):
    provider: str; model: str
    def judge_screen(self, image_jpeg: bytes, ocr_text: str, lines: list[str],
                     action_items: list[str]) -> ScreenJudgment: ...
def build_user_text(ocr_text: str, lines: list[str], action_items: list[str]) -> str
def parse_judgment(raw: str) -> ScreenJudgment
def has_credentials(provider: str) -> bool
def build_llm_client(provider: str, model: str) -> LLMClient
@dataclass(frozen=True) class JudgeRequest: screen_id, image_jpeg, ocr_text, lines: tuple[str, ...], action_items: tuple[str, ...]
class JudgmentCache: __init__(directory: Path, provider: str, model: str); get(req) -> ScreenJudgment | None; put(req, j)
def judge_screens(client, requests, cache: JudgmentCache | None, concurrency: int)
    -> tuple[dict[str, ScreenJudgment | None], list[str]]
```

- [ ] **Step 1: Write failing tests** — replace `tests/unit/test_llm.py`

```python
"""Tests for the LLM judge layer (all SDKs mocked)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from peeklet.llm import base
from peeklet.llm.base import (
    JudgeRequest,
    JudgmentCache,
    LLMResponseError,
    build_user_text,
    has_credentials,
    judge_screens,
    parse_judgment,
)
from peeklet.types import ScreenJudgment

if TYPE_CHECKING:
    from pathlib import Path

REQ = JudgeRequest("S1", b"\xff\xd8jpeg", "Billing Invoices", ("[00:00:01] A: open billing",),
                   ("Fix invoice total",))


class StubClient:
    provider = "stub"
    model = "m"

    def __init__(self, result: ScreenJudgment | Exception) -> None:
        self.result = result
        self.calls = 0

    def judge_screen(self, image_jpeg: bytes, ocr_text: str, lines: list[str],
                     action_items: list[str]) -> ScreenJudgment:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_user_text_contains_all_inputs() -> None:
    text = build_user_text("OCR WORDS", ["[00:00:01] A: hi"], ["Do X"])
    assert "OCR WORDS" in text and "[00:00:01] A: hi" in text and "Do X" in text


def test_parse_judgment_plain_and_fenced() -> None:
    raw = '{"include": true, "reason": "r", "visual_context": "v"}'
    assert parse_judgment(raw) == ScreenJudgment(True, "r", "v")
    assert parse_judgment(f"```json\n{raw}\n```") == ScreenJudgment(True, "r", "v")


@pytest.mark.parametrize("raw", ["not json", '{"include": "yes"}', '{"reason": "r"}', "[]"])
def test_parse_judgment_rejects_bad_output(raw: str) -> None:
    with pytest.raises(LLMResponseError):
        parse_judgment(raw)


def test_has_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert not has_credentials("anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    assert has_credentials("anthropic")
    assert not has_credentials("openai")


def test_cache_round_trip_and_key_sensitivity(tmp_path: Path) -> None:
    cache = JudgmentCache(tmp_path, "anthropic", "claude-haiku-4-5")
    j = ScreenJudgment(True, "r", "v")
    cache.put(REQ, j)
    assert cache.get(REQ) == j
    changed = JudgeRequest("S1", REQ.image_jpeg, "different ocr", REQ.lines, REQ.action_items)
    assert cache.get(changed) is None
    other_model = JudgmentCache(tmp_path, "anthropic", "claude-sonnet-5-5")
    assert other_model.get(REQ) is None


def test_judge_screens_uses_cache(tmp_path: Path) -> None:
    client = StubClient(ScreenJudgment(True, "r", "v"))
    cache = JudgmentCache(tmp_path, client.provider, client.model)
    judge_screens(client, [REQ], cache, concurrency=2)
    out, warnings = judge_screens(client, [REQ], cache, concurrency=2)
    assert client.calls == 1
    assert out == {"S1": ScreenJudgment(True, "r", "v")} and warnings == []


def test_judge_screens_failure_gives_none_and_warning() -> None:
    out, warnings = judge_screens(StubClient(RuntimeError("boom")), [REQ], None, concurrency=1)
    assert out == {"S1": None}
    assert len(warnings) == 1 and "S1" in warnings[0]


def test_anthropic_adapter_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from peeklet.llm import anthropic_client

    fake_sdk = mock.MagicMock()
    reply = mock.MagicMock()
    reply.content = [mock.MagicMock(text='{"include": false, "reason": "r", "visual_context": ""}')]
    fake_sdk.Anthropic.return_value.messages.create.return_value = reply
    monkeypatch.setattr(anthropic_client, "anthropic", fake_sdk)
    client = anthropic_client.AnthropicClient(model="claude-haiku-4-5")
    j = client.judge_screen(b"img", "ocr", ["l1"], [])
    assert j == ScreenJudgment(False, "r", "")
    kwargs = fake_sdk.Anthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["system"] == base.SYSTEM_PROMPT
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1]["type"] == "text"


def test_openai_adapter_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from peeklet.llm import openai_client

    fake_sdk = mock.MagicMock()
    choice = mock.MagicMock()
    choice.message.content = json.dumps({"include": True, "reason": "r", "visual_context": "v"})
    fake_sdk.OpenAI.return_value.chat.completions.create.return_value.choices = [choice]
    monkeypatch.setattr(openai_client, "openai", fake_sdk)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    client = openai_client.OpenAIClient(model="gpt-x")
    assert client.judge_screen(b"img", "ocr", ["l1"], []).include
    msgs = fake_sdk.OpenAI.return_value.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert user[0]["type"] == "image_url"
    assert user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_adapter_retries_once_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from peeklet.llm import anthropic_client

    fake_sdk = mock.MagicMock()
    bad, good = mock.MagicMock(), mock.MagicMock()
    bad.content = [mock.MagicMock(text="nope")]
    good.content = [mock.MagicMock(text='{"include": true, "reason": "r", "visual_context": "v"}')]
    fake_sdk.Anthropic.return_value.messages.create.side_effect = [bad, good]
    monkeypatch.setattr(anthropic_client, "anthropic", fake_sdk)
    assert anthropic_client.AnthropicClient("m").judge_screen(b"i", "o", [], []).include
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_llm.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `src/peeklet/llm/base.py`** (replace the file)

```python
"""LLM judge + describe: decides whether a screenshot is needed and describes it."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from peeklet.types import ScreenJudgment

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

PROMPT_VERSION = "judge-v1"

SYSTEM_PROMPT = (
    "You review screenshots from a screen recording for a downstream AI that will read "
    "the transcript. For one screenshot you get the transcript lines spoken while it was "
    "on screen, the screen's OCR text, and any action items flagged at that moment.\n\n"
    "Decide whether understanding these transcript lines requires seeing this screenshot "
    "(the speaker refers to things that are only visible on screen), and describe the "
    "screen.\n\n"
    "Rules for visual_context:\n"
    "- Describe only what is visible on the screen: the application, the page or view, "
    "and the specific elements, values, messages or errors shown.\n"
    "- Do not paraphrase or summarise the transcript.\n"
    "- Copy on-screen text exactly only where the OCR text confirms it.\n"
    "- One to three sentences.\n\n"
    'Respond with only a JSON object: {"include": true|false, "reason": "<one sentence>", '
    '"visual_context": "<description>"}'
)


class LLMResponseError(RuntimeError):
    """The model's output could not be parsed into a ScreenJudgment."""


class LLMClient(Protocol):
    provider: str
    model: str

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment: ...


def build_user_text(ocr_text: str, lines: list[str], action_items: list[str]) -> str:
    """The text part of the user message (the image is sent alongside it)."""
    parts = ["## Transcript lines spoken while this screen was shown", *lines]
    if action_items:
        parts += ["", "## Action items flagged at this moment", *action_items]
    parts += ["", "## OCR text of the screen", ocr_text or "(none)"]
    return "\n".join(parts)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_judgment(raw: str) -> ScreenJudgment:
    """Parse the model's JSON reply (optionally fenced) into a ScreenJudgment."""
    text = _FENCE_RE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"not JSON: {raw[:200]!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("include"), bool):
        raise LLMResponseError(f"missing boolean 'include': {raw[:200]!r}")
    return ScreenJudgment(
        include=data["include"],
        reason=str(data.get("reason", "")),
        visual_context=str(data.get("visual_context", "")),
    )


_PROVIDER_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def has_credentials(provider: str) -> bool:
    """True if any of the provider's credential env vars is set."""
    return any(os.environ.get(v) for v in _PROVIDER_ENV.get(provider, ()))


def build_llm_client(provider: str, model: str) -> LLMClient:
    """Instantiate the adapter for provider (SDKs are imported lazily)."""
    if provider == "anthropic":
        from peeklet.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=model)
    if provider == "openai":
        from peeklet.llm.openai_client import OpenAIClient

        return OpenAIClient(model=model)
    if provider == "openrouter":
        from peeklet.llm.openrouter_client import OpenRouterClient

        return OpenRouterClient(model=model)
    raise ValueError(f"Unknown LLM provider {provider!r}")


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    screen_id: str
    image_jpeg: bytes
    ocr_text: str
    lines: tuple[str, ...]
    action_items: tuple[str, ...]


class JudgmentCache:
    """On-disk cache of judgments keyed by every input that affects the answer."""

    def __init__(self, directory: Path, provider: str, model: str) -> None:
        self._dir = Path(directory).expanduser()
        self._provider = provider
        self._model = model

    def _path(self, req: JudgeRequest) -> Path:
        payload = json.dumps(
            [
                PROMPT_VERSION,
                self._provider,
                self._model,
                hashlib.sha256(req.image_jpeg).hexdigest(),
                req.ocr_text,
                list(req.lines),
                list(req.action_items),
            ]
        )
        return self._dir / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"

    def get(self, req: JudgeRequest) -> ScreenJudgment | None:
        path = self._path(req)
        if not path.is_file():
            return None
        try:
            return parse_judgment(path.read_text())
        except (OSError, LLMResponseError):
            return None

    def put(self, req: JudgeRequest, judgment: ScreenJudgment) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(req).write_text(
            json.dumps(
                {
                    "include": judgment.include,
                    "reason": judgment.reason,
                    "visual_context": judgment.visual_context,
                }
            )
        )


def judge_screens(
    client: LLMClient,
    requests: Sequence[JudgeRequest],
    cache: JudgmentCache | None,
    concurrency: int,
) -> tuple[dict[str, ScreenJudgment | None], list[str]]:
    """Judge every request (cache first, then the API in parallel).

    Returns verdicts keyed by screen id (None where the call failed) and warnings.
    """
    results: dict[str, ScreenJudgment | None] = {}
    warnings: list[str] = []
    todo: list[JudgeRequest] = []
    for req in requests:
        hit = cache.get(req) if cache else None
        if hit is not None:
            results[req.screen_id] = hit
        else:
            todo.append(req)

    def call(req: JudgeRequest) -> tuple[JudgeRequest, ScreenJudgment | None, str | None]:
        try:
            j = client.judge_screen(req.image_jpeg, req.ocr_text, list(req.lines),
                                    list(req.action_items))
        except Exception as exc:  # noqa: BLE001 - any provider failure falls back
            return req, None, f"LLM judge failed for screen {req.screen_id}: {exc}"
        return req, j, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for req, judgment, warning in pool.map(call, todo):
            results[req.screen_id] = judgment
            if warning:
                logger.warning(warning)
                warnings.append(warning)
            elif judgment is not None and cache:
                cache.put(req, judgment)
    return results, warnings
```

Note: ruff's `BLE` rules aren't enabled, so drop the `noqa` comment if ruff reports it as unused (`RUF100` is also not enabled; leave it out entirely to be safe).

- [ ] **Step 4: Implement the adapters**

`src/peeklet/llm/anthropic_client.py`:

```python
"""Anthropic Messages API adapter for the LLM judge."""

from __future__ import annotations

import base64
import logging

from peeklet.llm.base import SYSTEM_PROMPT, LLMResponseError, build_user_text, parse_judgment
from peeklet.types import ScreenJudgment

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment, unused-ignore]

_MAX_TOKENS = 1024


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str) -> None:
        if anthropic is None:  # pragma: no cover
            raise RuntimeError("The anthropic package is required for provider 'anthropic'.")
        self.model = model
        self._client = anthropic.Anthropic()

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(image_jpeg).decode("ascii"),
                },
            },
            {"type": "text", "text": build_user_text(ocr_text, lines, action_items)},
        ]
        for attempt in (1, 2):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            raw = "".join(getattr(b, "text", "") for b in response.content)
            try:
                return parse_judgment(raw)
            except LLMResponseError:
                if attempt == 2:
                    raise
                logger.warning("Anthropic returned unparseable JSON, retrying once")
        raise AssertionError("unreachable")
```

`src/peeklet/llm/openai_client.py`:

```python
"""OpenAI chat-completions adapter (honours OPENAI_BASE_URL) for the LLM judge."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from peeklet.llm.base import SYSTEM_PROMPT, LLMResponseError, build_user_text, parse_judgment
from peeklet.types import ScreenJudgment

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment, unused-ignore]


def chat_judge(
    client: Any, model: str, image_jpeg: bytes, ocr_text: str, lines: list[str],
    action_items: list[str], label: str,
) -> ScreenJudgment:
    """Shared by OpenAI and OpenRouter: one vision chat call with one JSON retry."""
    data_url = "data:image/jpeg;base64," + base64.standard_b64encode(image_jpeg).decode("ascii")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": build_user_text(ocr_text, lines, action_items)},
            ],
        },
    ]
    for attempt in (1, 2):
        response = client.chat.completions.create(model=model, messages=messages)
        raw = response.choices[0].message.content or ""
        try:
            return parse_judgment(raw)
        except LLMResponseError:
            if attempt == 2:
                raise
            logger.warning("%s returned unparseable JSON, retrying once", label)
    raise AssertionError("unreachable")


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str) -> None:
        if openai is None:  # pragma: no cover
            raise RuntimeError("The openai package is required for provider 'openai'.")
        self.model = model
        base_url = os.environ.get("OPENAI_BASE_URL")
        self._client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment:
        return chat_judge(self._client, self.model, image_jpeg, ocr_text, lines, action_items,
                          "OpenAI")
```

`src/peeklet/llm/openrouter_client.py`:

```python
"""OpenRouter adapter (OpenAI-compatible API) for the LLM judge."""

from __future__ import annotations

import os

from peeklet.llm.openai_client import chat_judge
from peeklet.types import ScreenJudgment

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment, unused-ignore]

_BASE_URL = "https://openrouter.ai/api/v1"
_HEADERS = {"HTTP-Referer": "https://github.com/SrividyaKirti/Peeklet", "X-Title": "Peeklet"}


class OpenRouterClient:
    provider = "openrouter"

    def __init__(self, model: str) -> None:
        if openai is None:  # pragma: no cover
            raise RuntimeError("The openai package is required for provider 'openrouter'.")
        self.model = model
        self._client = openai.OpenAI(
            base_url=_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers=_HEADERS,
        )

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment:
        return chat_judge(self._client, self.model, image_jpeg, ocr_text, lines, action_items,
                          "OpenRouter")
```

Delete `Moment` from `src/peeklet/types.py` and the `test_moment_*` tests.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit -q && uv run ruff check src tests && uv run mypy src/peeklet/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: multi-provider LLM judge+describe with on-disk cache

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Blocks, output files, Claude content, debug report

**Files:**
- Create: `src/peeklet/render.py`
- Test: `tests/unit/test_render.py`

**Interfaces:**
- Consumes: `Entry`, `Line`, `Screen`, `ScreenJudgment`, `Checkpoint`, `format_hms`, `decode_jpeg`, `encode_jpeg`.
- Produces:

```python
def image_filenames(screens: Sequence[Screen]) -> dict[str, str]     # frame_<int>.jpg, de-duplicated
def build_entries(lines, line_screens, files: Mapping[str, str],
                  judgments: Mapping[str, ScreenJudgment | None], action_items: Sequence[Checkpoint]) -> list[Entry]
@dataclass class RunStats: lines: int; screens_found: int; shortlisted: int; kept: int; warnings: list[str]
@dataclass class Entries: items: list[Entry]; out_dir: Path; stats: RunStats; max_image_edge: int = 1568
    def __iter__ / __len__ / __getitem__; def to_claude_content(self) -> list[dict[str, Any]]
def write_outputs(entries: list[Entry], kept: Sequence[Screen], files: Mapping[str, str], out_dir: Path) -> None
def write_debug(path: Path, payload: dict[str, Any]) -> None
```

- [ ] **Step 1: Write failing tests** — `tests/unit/test_render.py`

```python
"""Tests for blocks, output files and Claude content rendering."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from peeklet.image_utils import decode_jpeg, encode_jpeg
from peeklet.render import Entries, RunStats, build_entries, image_filenames, write_outputs
from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment

if TYPE_CHECKING:
    from pathlib import Path

JPEG = encode_jpeg(np.full((2000, 3000, 3), 100, dtype=np.uint8))


def scr(sid: str, t: float) -> Screen:
    return Screen(id=sid, image_jpeg=JPEG, frame_t=t, ocr_text="", word_count=1,
                  first_change=1.0, occurrences=[(t, t + 1)])


LINES = [
    Line(134.2, 137.0, "Look at the left.", "Alice"),
    Line(137.0, 140.0, "This blocks the bucket.", "Alice"),
    Line(147.0, 150.0, "Here is the remediation.", "Alice"),
    Line(302.0, 304.0, "Can you hear me?", "Bob"),
    Line(500.0, 503.0, "Back to the policy.", "Alice"),
]
SCREENS = ["S1", "S1", "S2", None, "S1"]


def test_image_filenames_use_whole_seconds_and_dedupe() -> None:
    files = image_filenames([scr("S1", 134.9), scr("S2", 134.2), scr("S3", 147.0)])
    # Sorted by frame time: S2 (134.2) claims frame_134.jpg first.
    assert files == {"S2": "frame_134.jpg", "S1": "frame_134_2.jpg", "S3": "frame_147.jpg"}


def test_blocks_break_on_screen_change_and_repeat_filename() -> None:
    files = {"S1": "frame_134.jpg", "S2": "frame_147.jpg"}
    judg = {"S1": ScreenJudgment(True, "r", "Policy page."), "S2": None}
    cps = [Checkpoint(148.0, "action_item", "Add team tag")]
    entries = build_entries(LINES, SCREENS, files, judg, cps)
    assert [e.to_dict() for e in entries] == [
        {"timestamp": "00:02:14",
         "transcript": "Alice: Look at the left.\nAlice: This blocks the bucket.",
         "image": "frame_134.jpg", "visual_context": "Policy page."},
        {"timestamp": "00:02:27", "transcript": "Alice: Here is the remediation.",
         "image": "frame_147.jpg", "action_item": "Add team tag"},
        {"timestamp": "00:05:02", "transcript": "Bob: Can you hear me?", "image": None},
        {"timestamp": "00:08:20", "transcript": "Alice: Back to the policy.",
         "image": "frame_134.jpg", "visual_context": "Policy page."},
    ]


def test_lines_on_dropped_screens_get_null_image() -> None:
    entries = build_entries(LINES[:2], ["S9", "S9"], {}, {}, [])
    assert [e.image for e in entries] == [None]


def test_write_outputs(tmp_path: Path) -> None:
    files = {"S1": "frame_134.jpg"}
    entries = build_entries(LINES[:1], ["S1"], files, {}, [])
    write_outputs(entries, [scr("S1", 134.0)], files, tmp_path)
    data = json.loads((tmp_path / "transcript.json").read_text())
    assert data[0]["image"] == "frame_134.jpg"
    assert (tmp_path / "frame_134.jpg").read_bytes() == JPEG


def test_claude_content_layout(tmp_path: Path) -> None:
    files = {"S1": "frame_134.jpg", "S2": "frame_147.jpg"}
    judg = {"S1": ScreenJudgment(True, "r", "Policy page."), "S2": None}
    entries = build_entries(LINES, SCREENS, files, judg,
                            [Checkpoint(148.0, "action_item", "Add team tag")])
    write_outputs(entries, [scr("S1", 134.0), scr("S2", 147.0)], files, tmp_path)
    blocks = Entries(entries, tmp_path, RunStats(5, 2, 2, 2, [])).to_claude_content()
    kinds = [b["type"] for b in blocks]
    assert kinds == ["text", "image", "text", "image", "text"]
    assert blocks[0]["text"].startswith("[00:02:14]\nAlice: Look at the left.")
    assert blocks[0]["text"].endswith("[SCREENSHOT @ 00:02:14]")
    assert blocks[2]["text"].startswith("Visual context: Policy page.")
    assert "[ACTION ITEM @ 00:02:27] Add team tag" in blocks[2]["text"]
    assert "[SCREENSHOT @ 00:08:20 — same screen as 00:02:14]" in blocks[4]["text"]
    img = decode_jpeg(__import__("base64").b64decode(blocks[1]["source"]["data"]))
    assert max(img.shape[:2]) == 1568
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_render.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement** — `src/peeklet/render.py`

```python
"""Blocks, output files, Claude content blocks and the debug report."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from peeklet.image_utils import decode_jpeg, encode_jpeg
from peeklet.types import Entry, format_hms

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment


def image_filenames(screens: Sequence[Screen]) -> dict[str, str]:
    """frame_<whole seconds>.jpg per screen; suffix _2, _3... on collisions."""
    files: dict[str, str] = {}
    taken: set[str] = set()
    for s in sorted(screens, key=lambda s: (s.frame_t, s.id)):
        stem = f"frame_{int(s.frame_t)}"
        name, n = f"{stem}.jpg", 2
        while name in taken:
            name, n = f"{stem}_{n}.jpg", n + 1
        taken.add(name)
        files[s.id] = name
    return files


def build_entries(
    lines: Sequence[Line],
    line_screens: Sequence[str | None],
    files: Mapping[str, str],
    judgments: Mapping[str, ScreenJudgment | None],
    action_items: Sequence[Checkpoint],
) -> list[Entry]:
    """Group consecutive lines on the same kept screen (or none) into entries."""
    labels: dict[int, list[str]] = defaultdict(list)
    for cp in action_items:
        idx = max((i for i, ln in enumerate(lines) if ln.start <= cp.t), default=0)
        labels[idx].append(cp.label)

    entries: list[Entry] = []
    block: list[int] = []
    block_key: str | None = None

    def flush() -> None:
        if not block:
            return
        judgment = judgments.get(block_key) if block_key else None
        entries.append(
            Entry(
                timestamp=lines[block[0]].start,
                lines=tuple(lines[i] for i in block),
                image=files.get(block_key) if block_key else None,
                visual_context=judgment.visual_context if judgment else None,
                action_items=tuple(lab for i in block for lab in labels.get(i, [])),
                screen_id=block_key,
            )
        )

    for i, sid in enumerate(line_screens):
        key = sid if sid is not None and sid in files else None
        if block and key != block_key:
            flush()
            block = []
        block_key = key
        block.append(i)
    flush()
    return entries


@dataclass(slots=True)
class RunStats:
    lines: int
    screens_found: int
    shortlisted: int
    kept: int
    warnings: list[str]


@dataclass(slots=True)
class Entries:
    """The annotated transcript plus where its images live."""

    items: list[Entry]
    out_dir: Path
    stats: RunStats
    max_image_edge: int = 1568

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Entry:
        return self.items[i]

    def to_claude_content(self) -> list[dict[str, Any]]:
        """Anthropic content blocks: text, then each new screenshot once."""
        blocks: list[dict[str, Any]] = []
        buf: list[str] = []
        seen: dict[str, str] = {}

        def flush_text() -> None:
            if buf:
                blocks.append({"type": "text", "text": "\n\n".join(buf)})
                buf.clear()

        for e in self.items:
            hms = format_hms(e.timestamp)
            buf.append(f"[{hms}]\n" + e.to_dict()["transcript"])
            buf.extend(f"[ACTION ITEM @ {hms}] {label}" for label in e.action_items)
            if not e.image or e.screen_id is None:
                continue
            if e.screen_id in seen:
                buf.append(f"[SCREENSHOT @ {hms} — same screen as {seen[e.screen_id]}]")
                continue
            buf.append(f"[SCREENSHOT @ {hms}]")
            flush_text()
            raw = (self.out_dir / e.image).read_bytes()
            small = encode_jpeg(decode_jpeg(raw), quality=90, max_edge=self.max_image_edge)
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(small).decode("ascii"),
                    },
                }
            )
            if e.visual_context:
                buf.append(f"Visual context: {e.visual_context}")
            seen[e.screen_id] = hms
        flush_text()
        return blocks


def write_outputs(
    entries: list[Entry], kept: Sequence[Screen], files: Mapping[str, str], out_dir: Path
) -> None:
    """Write transcript.json and the chosen frame of every kept screen."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.json").write_text(
        json.dumps([e.to_dict() for e in entries], indent=2, ensure_ascii=False) + "\n"
    )
    for s in kept:
        (out_dir / files[s.id]).write_bytes(s.image_jpeg)


def write_debug(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_render.py -q && uv run mypy src/peeklet/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/render.py tests/unit/test_render.py
git commit -m "feat: render blocks, transcript.json, Claude content and debug output

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: `annotate()` orchestrator + integration test

**Files:**
- Create: `src/peeklet/annotate.py`, `tests/helpers/__init__.py`, `tests/helpers/synth.py`, `tests/unit/test_annotate.py`, `tests/integration/test_annotate.py`
- Modify: `src/peeklet/__init__.py`

**Interfaces:**
- Consumes everything above.
- Produces:

```python
def annotate(video: str | Path, transcript: str | Path, out_dir: str | Path, *,
             max_images: int | None = None, config: PeekletConfig | None = None,
             use_llm: bool | None = None, debug: bool = False,
             llm_client: LLMClient | None = None, ocr: OcrFn | None = None) -> Entries
```
- `peeklet/__init__.py` exports `annotate`, `Entries`.

- [ ] **Step 1: Write the synthetic video helper** — `tests/helpers/synth.py`

```python
"""Synthetic screen-share demo: A (10s) -> B (10s) -> gallery (5s) -> A (15s)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from pathlib import Path

W, H, FPS = 1280, 720, 2


def _screen(heading: str, body_prefix: str, background: tuple[int, int, int]) -> np.ndarray:
    img = Image.new("RGB", (W, H), background)
    d = ImageDraw.Draw(img)
    big = ImageFont.load_default(size=48)
    small = ImageFont.load_default(size=22)
    d.text((220, 90), heading, fill="black", font=big)
    for i, item in enumerate(["Home", "Reports", "Settings", "Billing"]):
        d.text((20, 160 + 40 * i), item, fill="black", font=small)
    for i in range(12):
        d.text((220, 180 + 34 * i), f"{body_prefix} row {i + 1}: value {i * 17}",
               fill="black", font=small)
    return np.asarray(img)


def _gallery() -> np.ndarray:
    img = Image.new("RGB", (W, H), (40, 40, 40))
    d = ImageDraw.Draw(img)
    for i, colour in enumerate([(90, 60, 60), (60, 90, 60), (60, 60, 90), (90, 90, 60)]):
        x, y = (i % 2) * (W // 2), (i // 2) * (H // 2)
        d.rectangle([x + 10, y + 10, x + W // 2 - 10, y + H // 2 - 10], fill=colour)
    return np.asarray(img)


def _with_webcam(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = frame.copy()
    out[H - 82 : H - 10, W - 106 : W - 10] = rng.integers(0, 255, (72, 96, 3), dtype=np.uint8)
    return out


def write_demo(video_path: Path, vtt_path: Path) -> None:
    import imageio.v3 as iio

    rng = np.random.default_rng(7)
    a = _screen("Revenue Dashboard", "Revenue", (255, 255, 255))
    b = _screen("Billing Settings", "Invoice", (236, 236, 236))  # tint lets stub OCR tell A/B apart
    g = _gallery()
    plan = [(a, 10), (b, 10), (g, 5), (a, 15)]
    frames = [_with_webcam(f, rng) for f, secs in plan for _ in range(secs * FPS)]
    iio.imwrite(video_path, np.stack(frames), plugin="pyav", fps=FPS, codec="libx264")
    vtt_path.write_text(
        "WEBVTT\n\n"
        "00:00:00.500 --> 00:00:04.000\nWelcome, this is the revenue dashboard.\n\n"
        "00:00:11.000 --> 00:00:15.000\nNow look at the billing settings page.\n\n"
        "00:00:21.000 --> 00:00:24.000\nCan everyone hear me okay?\n\n"
        "00:00:27.000 --> 00:00:31.000\nBack on the dashboard, notice the totals.\n"
    )
```

- [ ] **Step 2: Write failing unit tests** — `tests/unit/test_annotate.py` (OCR and LLM stubbed; uses the real decoder on the synthetic video)

```python
"""Orchestrator tests with stubbed OCR and LLM."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from peeklet import annotate
from peeklet.screen import WordBox
from peeklet.transcript import TranscriptError
from peeklet.types import ScreenJudgment
from tests.helpers.synth import write_demo

if TYPE_CHECKING:
    from pathlib import Path


def colour_ocr(frame: np.ndarray) -> list[WordBox]:
    """Identify synthetic screens by background tint in an empty area (right of the body)."""
    h, w = frame.shape[:2]
    tint = float(frame[int(h * 0.4), int(w * 0.85)].mean())
    if tint > 248:
        heading = "Revenue Dashboard"
    elif tint > 225:
        heading = "Billing Settings"
    else:
        return []  # gallery view
    boxes = [WordBox(heading, 95, 220, 90, 400, 48), WordBox("Home", 95, 20, 160, 60, 22)]
    boxes += [WordBox(f"row{i}", 95, 220, 180 + 34 * i, 300, 22) for i in range(12)]
    return boxes


class StubLLM:
    provider, model = "stub", "stub"

    def judge_screen(self, image_jpeg: bytes, ocr_text: str, lines: list[str],
                     action_items: list[str]) -> ScreenJudgment:
        return ScreenJudgment(True, "needed", f"Screen showing {ocr_text.split()[0]}")


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("demo")
    write_demo(d / "demo.mp4", d / "demo.vtt")
    return d / "demo.mp4", d / "demo.vtt"


@pytest.fixture(autouse=True)
def no_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("peeklet.annotate.detect_speech_segments",
                        lambda *a, **k: [(0.5, 4.0), (11.0, 15.0), (27.0, 31.0)])


def test_no_llm_end_to_end(demo: tuple[Path, Path], tmp_path: Path) -> None:
    entries = annotate(*demo, tmp_path, use_llm=False, ocr=colour_ocr)
    data = json.loads((tmp_path / "transcript.json").read_text())
    images = [e["image"] for e in data]
    assert images[0] is not None and images[0] == images[3]
    assert images[1] is not None and images[1] != images[0]
    assert images[2] is None
    assert all("visual_context" not in e for e in data)
    assert entries.stats.screens_found == 2
    for name in {i for i in images if i}:
        assert (tmp_path / name).is_file()


def test_llm_descriptions_and_cache(demo: tuple[Path, Path], tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from peeklet.config import PeekletConfig

    cfg = PeekletConfig(llm_cache_dir=str(tmp_path / "cache"))
    annotate(*demo, tmp_path / "out", config=cfg, llm_client=StubLLM(), ocr=colour_ocr)
    data = json.loads((tmp_path / "out" / "transcript.json").read_text())
    assert data[0]["visual_context"].startswith("Screen showing")


def test_missing_key_falls_back_to_no_llm(demo: tuple[Path, Path], tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    entries = annotate(*demo, tmp_path, ocr=colour_ocr)
    assert any("credentials" in w for w in entries.stats.warnings)


def test_max_images_zero_is_text_only(demo: tuple[Path, Path], tmp_path: Path) -> None:
    annotate(*demo, tmp_path, use_llm=False, max_images=0, ocr=colour_ocr)
    data = json.loads((tmp_path / "transcript.json").read_text())
    assert all(e["image"] is None for e in data)


def test_debug_report(demo: tuple[Path, Path], tmp_path: Path) -> None:
    annotate(*demo, tmp_path, use_llm=False, debug=True, ocr=colour_ocr)
    dbg = json.loads((tmp_path / "debug.json").read_text())
    assert {"screens", "lines", "checkpoints", "warnings", "timing"} <= dbg.keys()
    assert all("occurrences" in s and "signals" in s for s in dbg["screens"])


def test_bad_transcript_raises(demo: tuple[Path, Path], tmp_path: Path) -> None:
    bad = tmp_path / "empty.vtt"
    bad.write_text("WEBVTT\n")
    with pytest.raises(TranscriptError):
        annotate(demo[0], bad, tmp_path / "o", use_llm=False, ocr=colour_ocr)
```

Add `tests/helpers/__init__.py` (empty) and make sure `tests/__init__.py` exists so `tests.helpers.synth` imports.

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/test_annotate.py -q`
Expected: FAIL (`ImportError: cannot import name 'annotate'`).

- [ ] **Step 4: Implement** — `src/peeklet/annotate.py`

```python
"""annotate(): video + transcript -> annotated transcript with screenshots."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from peeklet.align import align_lines
from peeklet.config import PeekletConfig
from peeklet.cues import verbal_cue_checkpoints
from peeklet.llm.base import (
    JudgeRequest,
    JudgmentCache,
    build_llm_client,
    has_credentials,
    judge_screens,
)
from peeklet.render import (
    Entries,
    RunStats,
    build_entries,
    image_filenames,
    write_debug,
    write_outputs,
)
from peeklet.score import compute_signals, rank_screens, score_screens, select_screens
from peeklet.screen import ocr_word_boxes, require_tesseract
from peeklet.screens import Sample, build_screens
from peeklet.speech import detect_speech_segments, speech_onset_checkpoints
from peeklet.transcript import action_item_checkpoints, parse_transcript, split_long_lines
from peeklet.types import format_hms
from peeklet.video import VideoDecoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peeklet.llm.base import LLMClient
    from peeklet.screens import OcrFn
    from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment

logger = logging.getLogger(__name__)


def annotate(
    video: str | Path,
    transcript: str | Path,
    out_dir: str | Path,
    *,
    max_images: int | None = None,
    config: PeekletConfig | None = None,
    use_llm: bool | None = None,
    debug: bool = False,
    llm_client: LLMClient | None = None,
    ocr: OcrFn | None = None,
) -> Entries:
    """Run the full pipeline and write transcript.json + frame images to out_dir."""
    t0 = time.monotonic()
    cfg = (config or PeekletConfig()).model_copy(deep=True)
    if max_images is not None:
        cfg.max_images = max_images
    if use_llm is not None:
        cfg.use_llm = use_llm
    video, transcript, out_dir = Path(video), Path(transcript), Path(out_dir)
    warnings: list[str] = []

    def warn(msg: str) -> None:
        logger.warning(msg)
        warnings.append(msg)

    # 1. Transcript
    lines = split_long_lines(parse_transcript(transcript), cfg.max_line_seconds)

    # 2. Checkpoints
    checkpoints: list[Checkpoint] = list(action_item_checkpoints(transcript))
    checkpoints += verbal_cue_checkpoints(lines, cfg.cue_threshold)
    try:
        ranges = detect_speech_segments(video, silence_threshold_dbfs=cfg.silence_threshold_dbfs)
        checkpoints += speech_onset_checkpoints(ranges, cfg.min_pause_seconds)
    except Exception as exc:  # no audio track, undecodable audio, missing ffmpeg
        warn(f"speech detection skipped ({exc}); no speech-onset checkpoints")

    # 3–4. Screens
    if ocr is None:
        require_tesseract()

        def ocr_fn(frame: Any) -> Any:
            return ocr_word_boxes(frame, cfg.ocr_max_dim)

        ocr = ocr_fn
    decoder = VideoDecoder(video)
    meta = decoder.get_metadata()
    if lines and lines[-1].end > meta.duration + 1.0:
        warn(f"transcript runs past the end of the video ({format_hms(meta.duration)})")
    samples = (
        Sample(t=ts, frame=frame)
        for frame, ts, _ in decoder.iter_coarse_frames_seek(cfg.sample_fps)
    )
    t_screens = time.monotonic()
    sp = build_screens(samples, checkpoints, cfg, ocr, video_end=meta.duration)
    screen_seconds = time.monotonic() - t_screens
    for w in sp.warnings:
        warn(w)
    screens = sp.screens
    if not screens:
        warn("no usable screens found (only low-information frames)")

    # 5–6. Align and score
    line_screens = align_lines(lines, screens, cfg.lead_seconds)
    signals = compute_signals(screens, lines, line_screens, checkpoints)
    scores = score_screens(signals, cfg.score_weights, cfg.anchor_bonus)

    # 7. LLM judge
    judgments: dict[str, ScreenJudgment | None] | None = None
    shortlist: list[Screen] = []
    if cfg.use_llm and cfg.max_images > 0 and screens:
        client = llm_client
        if client is None:
            if has_credentials(cfg.llm_provider):
                client = build_llm_client(cfg.llm_provider, cfg.llm_model)
            else:
                warn(f"no credentials for LLM provider {cfg.llm_provider!r}; running as --no-llm")
        if client is not None:
            k = math.ceil(cfg.shortlist_factor * cfg.max_images)
            shortlist = rank_screens(screens, scores)[:k]
            requests = _build_requests(shortlist, lines, line_screens, checkpoints, cfg)
            cache = JudgmentCache(Path(cfg.llm_cache_dir), client.provider, client.model)
            judgments, llm_warnings = judge_screens(client, requests, cache, cfg.llm_concurrency)
            for w in llm_warnings:
                warnings.append(w)
            if judgments and all(v is None for v in judgments.values()):
                warn("every LLM call failed; running as --no-llm")
                judgments = None

    # 8–9. Select and render
    kept = select_screens(screens, scores, signals, judgments, cfg.max_images)
    files = image_filenames(kept)
    action_items = [c for c in checkpoints if c.kind == "action_item"]
    entries = build_entries(lines, line_screens, files, judgments or {}, action_items)
    write_outputs(entries, kept, files, out_dir)
    stats = RunStats(
        lines=len(lines),
        screens_found=len(screens),
        shortlisted=len(shortlist),
        kept=len(kept),
        warnings=warnings,
    )
    if debug:
        write_debug(
            out_dir / "debug.json",
            _debug_payload(screens, signals, scores, shortlist, judgments, kept, lines,
                           line_screens, checkpoints, warnings,
                           {"seconds_total": time.monotonic() - t0,
                            "seconds_screen_pass": screen_seconds}),
        )
    return Entries(items=entries, out_dir=out_dir, stats=stats, max_image_edge=cfg.max_image_edge)


def _build_requests(
    shortlist: Sequence[Screen],
    lines: Sequence[Line],
    line_screens: Sequence[str | None],
    checkpoints: Sequence[Checkpoint],
    cfg: PeekletConfig,
) -> list[JudgeRequest]:
    from peeklet.align import screen_at
    from peeklet.image_utils import decode_jpeg, encode_jpeg

    requests: list[JudgeRequest] = []
    for s in shortlist:
        idx = [i for i, sid in enumerate(line_screens) if sid == s.id]
        has_cp = {
            i for i in idx
            if any(lines[i].start <= c.t <= lines[i].end + cfg.lead_seconds for c in checkpoints)
        }
        chosen = sorted(sorted(idx, key=lambda i: (i not in has_cp, i))[: cfg.llm_max_lines])
        rendered = tuple(
            f"[{format_hms(lines[i].start)}] {lines[i].speaker or 'Speaker'}: {lines[i].text}"
            for i in chosen
        )
        items = tuple(
            c.label for c in checkpoints if c.kind == "action_item" and screen_at([s], c.t)
        )
        image = encode_jpeg(decode_jpeg(s.image_jpeg), quality=90, max_edge=cfg.max_image_edge)
        requests.append(JudgeRequest(s.id, image, s.ocr_text, rendered, items))
    return requests


def _debug_payload(
    screens: Sequence[Screen],
    signals: dict[str, Any],
    scores: dict[str, float],
    shortlist: Sequence[Screen],
    judgments: dict[str, ScreenJudgment | None] | None,
    kept: Sequence[Screen],
    lines: Sequence[Line],
    line_screens: Sequence[str | None],
    checkpoints: Sequence[Checkpoint],
    warnings: list[str],
    timing: dict[str, float],
) -> dict[str, Any]:
    short_ids = {s.id for s in shortlist}
    kept_ids = {s.id for s in kept}
    return {
        "screens": [
            {
                "id": s.id,
                "frame_t": s.frame_t,
                "occurrences": s.occurrences,
                "word_count": s.word_count,
                "signals": signals[s.id].as_dict(),
                "score": scores[s.id],
                "shortlisted": s.id in short_ids,
                "judgment": (
                    None
                    if judgments is None or judgments.get(s.id) is None
                    else {"include": judgments[s.id].include,  # type: ignore[union-attr]
                          "reason": judgments[s.id].reason}  # type: ignore[union-attr]
                ),
                "kept": s.id in kept_ids,
            }
            for s in screens
        ],
        "lines": [
            {"start": ln.start, "end": ln.end, "screen_id": sid}
            for ln, sid in zip(lines, line_screens, strict=True)
        ],
        "checkpoints": [{"t": c.t, "kind": c.kind, "label": c.label} for c in checkpoints],
        "warnings": warnings,
        "timing": timing,
    }
```

Replace `src/peeklet/__init__.py`:

```python
"""Peeklet — annotate a screen-recording transcript with the screenshots it refers to."""

from peeklet.annotate import annotate
from peeklet.render import Entries

__all__ = ["Entries", "annotate"]
__version__ = "0.2.0"
```

- [ ] **Step 5: Write the integration test** — `tests/integration/test_annotate.py` (real tesseract)

```python
"""End-to-end with real OCR (skipped when tesseract is not installed)."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from peeklet import annotate
from tests.helpers.synth import write_demo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract")


def test_real_ocr_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("peeklet.annotate.detect_speech_segments",
                        lambda *a, **k: [(0.5, 4.0), (11.0, 15.0), (27.0, 31.0)])
    write_demo(tmp_path / "demo.mp4", tmp_path / "demo.vtt")
    entries = annotate(tmp_path / "demo.mp4", tmp_path / "demo.vtt", tmp_path / "out",
                       use_llm=False, debug=True)
    data = json.loads((tmp_path / "out" / "transcript.json").read_text())
    images = [e["image"] for e in data]
    assert entries.stats.screens_found == 2, json.loads((tmp_path / "out/debug.json").read_text())
    assert images[0] == images[3] is not None
    assert images[1] not in (None, images[0])
    assert images[2] is None
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/test_annotate.py tests/integration -q && uv run mypy src/peeklet/`
Expected: unit PASS; integration PASS where tesseract is installed (SKIPPED otherwise). If the integration test finds 3 screens, inspect `debug.json`: the usual cause is a webcam-tile frame registering as a separate fingerprint before the mask has learned it; fix by making the synthetic webcam tile smaller or starting with 2s of static A.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: annotate() orchestrator with synthetic end-to-end tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: CLI, dependencies, CI, README

**Files:**
- Rewrite: `src/peeklet/cli.py`, `README.md`
- Create: `tests/unit/test_cli.py`
- Modify: `pyproject.toml`, `.github/workflows/ci-develop.yml`, `.github/workflows/ci-main.yml`, `.gitignore`

**Interfaces:**
- Consumes: `annotate`, `load_config`, `TranscriptError`, `MissingDependencyError`.
- Produces: console script `peeklet = "peeklet.cli:main"`.

- [ ] **Step 1: Write failing tests** — `tests/unit/test_cli.py`

```python
"""CLI smoke tests (annotate is mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from peeklet.cli import main
from peeklet.render import Entries, RunStats
from peeklet.transcript import TranscriptError


def _files(tmp: Path) -> tuple[Path, Path]:
    v, t = tmp / "v.mp4", tmp / "t.vtt"
    v.write_bytes(b"x")
    t.write_text("WEBVTT\n")
    return v, t


def test_cli_passes_options_and_prints_summary(tmp_path: Path) -> None:
    v, t = _files(tmp_path)
    fake = Entries([], tmp_path / "out", RunStats(12, 5, 4, 3, ["w"]))
    with mock.patch("peeklet.cli.annotate", return_value=fake) as ann:
        res = CliRunner().invoke(main, [str(v), "--transcript", str(t), "--out",
                                        str(tmp_path / "out"), "--max-images", "7", "--no-llm",
                                        "--llm-model", "claude-sonnet-5-5", "--debug"])
    assert res.exit_code == 0, res.output
    kwargs = ann.call_args.kwargs
    assert kwargs["max_images"] == 7 and kwargs["use_llm"] is False and kwargs["debug"] is True
    assert kwargs["config"].llm_model == "claude-sonnet-5-5"
    assert "12 lines, 5 screens found, 4 shortlisted, 3 kept, 1 warning" in res.output


def test_cli_reports_transcript_error(tmp_path: Path) -> None:
    v, t = _files(tmp_path)
    with mock.patch("peeklet.cli.annotate", side_effect=TranscriptError("bad transcript")):
        res = CliRunner().invoke(main, [str(v), "--transcript", str(t), "--out", str(tmp_path)])
    assert res.exit_code != 0
    assert "bad transcript" in res.output


def test_cli_requires_transcript(tmp_path: Path) -> None:
    v, _ = _files(tmp_path)
    res = CliRunner().invoke(main, [str(v), "--out", str(tmp_path)])
    assert res.exit_code != 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_cli.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement** — replace `src/peeklet/cli.py`

```python
"""peeklet: annotate a screen-recording transcript with screenshots."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from peeklet.annotate import annotate
from peeklet.config import load_config
from peeklet.screen import MissingDependencyError
from peeklet.transcript import TranscriptError


@click.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--transcript", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="SRT, VTT or Fathom markdown transcript.")
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Output directory for transcript.json and frame images.")
@click.option("--max-images", type=click.IntRange(min=0), default=None,
              help="Maximum screenshots (default 20).")
@click.option("--no-llm", is_flag=True, help="Heuristic selection only; no descriptions.")
@click.option("--llm-provider", type=click.Choice(["anthropic", "openai", "openrouter"]),
              default=None)
@click.option("--llm-model", default=None, help="Model id (default claude-haiku-4-5).")
@click.option("--config", "config_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--debug", is_flag=True, help="Also write debug.json with scoring details.")
def main(
    video: Path,
    transcript: Path,
    out_dir: Path,
    max_images: int | None,
    no_llm: bool,
    llm_provider: str | None,
    llm_model: str | None,
    config_path: Path | None,
    debug: bool,
) -> None:
    """Annotate VIDEO's transcript with the screenshots it refers to."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    cfg = load_config(config_path)
    if llm_provider:
        cfg.llm_provider = llm_provider  # type: ignore[assignment]
    if llm_model:
        cfg.llm_model = llm_model
    try:
        entries = annotate(video, transcript, out_dir, max_images=max_images, config=cfg,
                           use_llm=not no_llm, debug=debug)
    except (TranscriptError, MissingDependencyError) as exc:
        raise click.ClickException(str(exc)) from exc
    s = entries.stats
    n_warn = len(s.warnings)
    click.echo(
        f"{s.lines} lines, {s.screens_found} screens found, {s.shortlisted} shortlisted, "
        f"{s.kept} kept, {n_warn} warning{'' if n_warn == 1 else 's'} "
        f"-> {out_dir / 'transcript.json'}"
    )
```

- [ ] **Step 4: Dependencies and tooling** — in `pyproject.toml`:
  - `version = "0.2.0"`; `description = "Annotate screen-recording transcripts with the screenshots they refer to, for LLMs."`; `keywords = ["screen-recording", "transcript", "screenshots", "llm", "multimodal"]`.
  - `dependencies = ["numpy>=1.24", "Pillow>=10.1", "scikit-image>=0.21", "scipy>=1.10", "imagehash>=4.3", "pydantic>=2.0", "click>=8.0", "pyyaml>=6.0", "imageio[ffmpeg]>=2.31", "av>=14.0", "pydub>=0.25", "pytesseract>=0.3.10", "anthropic>=0.40", "openai>=1.0"]` (pyarrow removed; Pillow ≥10.1 for `load_default(size=)`).
  - `[project.optional-dependencies]`: keep only `dev`.
  - mypy overrides: drop `pyarrow.*`, add `scipy.*`.
  - Run `uv lock` to refresh `uv.lock`.
- `.gitignore`: add `eval/data/` and `eval/results/`.
- CI (`ci-develop.yml` and `ci-main.yml`): keep `apt-get install ... ffmpeg ... tesseract-ocr`; `uv sync --dev` (no extras); unit tests `uv run pytest tests/unit/ --cov=src/peeklet --cov-fail-under=90 -v`; integration `uv run pytest tests/integration/ -v`.

- [ ] **Step 5: Rewrite `README.md`**

Sections, in order:
1. Title + one-line description: "Turn a screen recording + its transcript into an annotated transcript with the screenshots (and short descriptions) a downstream LLM needs."
2. **Output** — the `transcript.json` example from the spec (4 entries including `null` image and repeated `frame_134.jpg`) and the `to_claude_content()` layout.
3. **Install** — `pip install peeklet`; `brew install tesseract` / `apt install tesseract-ocr`.
4. **Usage** — the CLI line from the spec with all flags; the Python `annotate(...)` + `to_claude_content()` snippet; supported transcripts (SRT, VTT, Fathom markdown) and that long lines are split at sentence boundaries.
5. **How it works** — the spec's pipeline diagram plus one paragraph each for checkpoints, screens, alignment, scoring, LLM judge (cost estimate ~$0.15/video on Haiku 4.5; `--no-llm`), budget.
6. **Configuration** — YAML example overriding `max_images`, `score_weights.references`, `llm_model`; link to `src/peeklet/config.py` for all fields.
7. **LLM providers** — env vars per provider (`ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`, `OPENAI_API_KEY` + optional `OPENAI_BASE_URL`, `OPENROUTER_API_KEY`); cache location.
8. **Evaluation** — how to run `eval/lpm_eval.py` and `eval/guide_eval.py` (Task 16/17), dataset licenses (LPM CC BY-NC-SA 4.0; GUIDE CC BY 4.0, gated), and that data is never committed.
9. **Development** — `uv sync --dev`, `uv run pytest`, `ruff`, `mypy`.

- [ ] **Step 6: Full verification**

Run: `uv sync --dev && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src/peeklet/ && uv run pytest tests/unit --cov=src/peeklet --cov-fail-under=90 -q && uv run pytest tests/integration -q`
Expected: all pass (integration skipped without tesseract). If coverage < 90, add tests for the uncovered branches reported by `--cov-report=term-missing` (most likely `annotate.py` fallback paths) before committing.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: single peeklet CLI, core dependencies, CI and README

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: LPM evaluation script

**Files:**
- Create: `eval/__init__.py` (empty), `eval/common.py`, `eval/lpm_eval.py`, `tests/unit/test_eval_common.py`

**Interfaces:**
- Produces (in `eval/common.py`): `clean_youtube_vtt(text: str) -> str`, `match_events(pred: list[float], truth: list[float], tolerance: float) -> tuple[int, int, int]` (tp, fp, fn), `prf(tp, fp, fn) -> tuple[float, float, float]`, `segment_index(boundaries: list[float], t: float) -> int`, `change_times_from_debug(debug: dict) -> list[float]`, `ffmpeg_exe() -> str`, `hf_token() -> str | None`, `download(url, dest, headers=None) -> Path`, `whisperx_to_vtt(items: list[dict]) -> str`.
- CLI: `uv run --with yt-dlp python -m eval.lpm_eval --limit 10 --max-minutes 30 [--tolerance 2.0]`.

Ground truth: `raw_video_links.csv` from `https://raw.githubusercontent.com/dondongwon/LPMDataset/main/dataset/raw_video_links.csv`. Column `Answer.startTimeList` = `"Start Time|t1|t2|..."` = annotated slide-transition times (seconds); `seconds` = duration; `speaker`; `youtube_url`.

Metrics per video: change-detection precision/recall/F1 (predicted change times = occurrence starts in `debug.json`, excluding t < 1s, deduped within 0.5s; greedy one-to-one matching within ±tolerance); line coverage (lines with a screen / all lines); line-screen accuracy (for lines with a screen: GT slide index of the line midpoint == GT slide index of the screen's `frame_t`; note revisited slides count as misses, so this is a lower bound).

- [ ] **Step 1: Write failing tests** — `tests/unit/test_eval_common.py`

```python
"""Tests for evaluation helpers (no network)."""

from __future__ import annotations

import pytest

from eval.common import (
    change_times_from_debug,
    clean_youtube_vtt,
    match_events,
    prf,
    segment_index,
    whisperx_to_vtt,
)
from peeklet.transcript import _parse_vtt

YT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000 align:start position:0%
hello<00:00:01.500><c> world</c>

00:00:03.000 --> 00:00:03.010 align:start position:0%
hello world

00:00:03.010 --> 00:00:05.000 align:start position:0%
hello world
next<00:00:03.500><c> line</c>
"""


def test_clean_youtube_vtt_strips_tags_and_rolling_duplicates() -> None:
    lines = _parse_vtt(clean_youtube_vtt(YT))
    assert [(ln.start, ln.text) for ln in lines] == [(1.0, "hello world"), (3.01, "next line")]


def test_match_events_one_to_one_within_tolerance() -> None:
    assert match_events([10.0, 10.5, 30.0], [10.2, 20.0], tolerance=2.0) == (1, 2, 1)


def test_prf_handles_zero() -> None:
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(1, 1, 0) == pytest.approx((0.5, 1.0, 2 / 3))


def test_segment_index() -> None:
    assert segment_index([10.0, 20.0], 5.0) == 0
    assert segment_index([10.0, 20.0], 10.0) == 1
    assert segment_index([10.0, 20.0], 25.0) == 2


def test_change_times_from_debug() -> None:
    debug = {"screens": [{"occurrences": [[0.0, 10.0], [30.0, 40.0]]},
                         {"occurrences": [[10.0, 30.0], [30.2, 31.0]]}]}
    assert change_times_from_debug(debug) == [10.0, 30.0]


def test_whisperx_to_vtt() -> None:
    vtt = whisperx_to_vtt([{"start": 1.617, "end": 30.305, "sentence": "Alright. We start."}])
    [ln] = _parse_vtt(vtt)
    assert (ln.start, ln.end, ln.text) == (1.617, 30.305, "Alright. We start.")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_eval_common.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** — `eval/common.py`

```python
"""Shared helpers for Peeklet evaluation scripts (not part of the package)."""

from __future__ import annotations

import os
import re
import urllib.request
from bisect import bisect_right
from pathlib import Path
from typing import Any

_CUE_RE = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")
_TAG_RE = re.compile(r"<[^>]+>")


def _secs(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _vtt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d}.{ms % 1000:03d}"


def clean_youtube_vtt(text: str) -> str:
    """Strip inline timing tags and YouTube's rolling duplicate lines; keep one line per cue."""
    out = ["WEBVTT", ""]
    last = ""
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        match = next((_CUE_RE.search(ln) for ln in lines if _CUE_RE.search(ln)), None)
        if match is None:
            continue
        start, end = _secs(match.group(1)), _secs(match.group(2))
        if end - start < 0.05:
            continue
        body = [_TAG_RE.sub("", ln).strip() for ln in lines[lines.index(
            next(ln for ln in lines if _CUE_RE.search(ln))) + 1:]]
        body = [b for b in body if b]
        new = body[-1] if body else ""
        if not new or new == last:
            continue
        last = new
        out += [f"{_vtt_ts(start)} --> {_vtt_ts(end)}", new, ""]
    return "\n".join(out)


def whisperx_to_vtt(items: list[dict[str, Any]]) -> str:
    """GUIDE transcript JSON ([{start, end, sentence}]) -> WebVTT."""
    out = ["WEBVTT", ""]
    for it in items:
        text = str(it.get("sentence", it.get("text", ""))).strip()
        if text:
            out += [f"{_vtt_ts(float(it['start']))} --> {_vtt_ts(float(it['end']))}", text, ""]
    return "\n".join(out)


def match_events(pred: list[float], truth: list[float], tolerance: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching; returns (true positives, false positives, false negatives)."""
    unused = sorted(pred)
    tp = 0
    for t in sorted(truth):
        best = min(unused, key=lambda p: abs(p - t), default=None)
        if best is not None and abs(best - t) <= tolerance:
            unused.remove(best)
            tp += 1
    return tp, len(pred) - tp, len(truth) - tp


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def segment_index(boundaries: list[float], t: float) -> int:
    """Index of the ground-truth segment containing t (boundaries are segment starts)."""
    return bisect_right(boundaries, t)


def change_times_from_debug(debug: dict[str, Any], min_t: float = 1.0) -> list[float]:
    """Predicted screen-change times: occurrence starts (excluding the video start), deduped."""
    starts = sorted(float(o[0]) for s in debug["screens"] for o in s["occurrences"])
    out: list[float] = []
    for t in starts:
        if t >= min_t and (not out or t - out[-1] > 0.5):
            out.append(t)
    return out


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def hf_token() -> str | None:
    """HF_TOKEN env var, else ~/.cache/huggingface/token."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    path = Path.home() / ".cache" / "huggingface" / "token"
    return path.read_text().strip() if path.is_file() else None


def download(url: str, dest: Path, headers: dict[str, str] | None = None) -> Path:
    """Download url to dest unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers=headers or {})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    return dest
```

`eval/lpm_eval.py`:

```python
"""LPM evaluation: change detection and alignment vs hand-labelled slide transitions.

Usage: uv run --with yt-dlp python -m eval.lpm_eval --limit 10 --max-minutes 30
Data (CC BY-NC-SA 4.0; videos under YouTube terms) goes to eval/data/lpm/, never committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from eval.common import (
    change_times_from_debug,
    clean_youtube_vtt,
    download,
    ffmpeg_exe,
    match_events,
    prf,
    segment_index,
)
from peeklet import annotate

CSV_URL = "https://raw.githubusercontent.com/dondongwon/LPMDataset/main/dataset/raw_video_links.csv"
DATA = Path("eval/data/lpm")
RESULTS = Path("eval/results")


def pick_videos(rows: list[dict[str, str]], limit: int, max_minutes: float) -> list[dict[str, str]]:
    """Shortest eligible video per speaker first (diversity), then the next shortest."""
    eligible = [r for r in rows if float(r["seconds"]) <= max_minutes * 60 and r["youtube_url"]]
    eligible.sort(key=lambda r: float(r["seconds"]))
    picked, speakers = [], set()
    for r in eligible:
        if r["speaker"] not in speakers:
            picked.append(r)
            speakers.add(r["speaker"])
    for r in eligible:
        if r not in picked:
            picked.append(r)
    return picked[:limit]


def fetch(row: dict[str, str]) -> tuple[Path, Path] | None:
    vid = row["video_id"].removesuffix(".mp4")
    out = DATA / vid
    video = out / f"{vid}.mp4"
    common = ["yt-dlp", "--ffmpeg-location", ffmpeg_exe(), "-o", str(out / f"{vid}.%(ext)s")]
    if not video.exists():
        subprocess.run([*common, "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
                        "--merge-output-format", "mp4", row["youtube_url"]], check=False)
    subs = sorted(out.glob("*.vtt"))
    if not subs:
        subprocess.run([*common, "--skip-download", "--write-subs", "--sub-langs", "en.*",
                        "--sub-format", "vtt", row["youtube_url"]], check=False)
        subs = sorted(out.glob("*.vtt"))
    if not subs:
        subprocess.run([*common, "--skip-download", "--write-auto-subs", "--sub-langs", "en",
                        "--sub-format", "vtt", row["youtube_url"]], check=False)
        subs = sorted(out.glob("*.vtt"))
    if not video.exists() or not subs:
        return None
    clean = out / "clean.vtt"
    clean.write_text(clean_youtube_vtt(subs[0].read_text(encoding="utf-8")))
    return video, clean


def evaluate(row: dict[str, str], video: Path, vtt: Path, tolerance: float) -> dict[str, Any]:
    duration = float(row["seconds"])
    raw = [float(x) for x in row["Answer.startTimeList"].split("|")[1:] if x.strip()]
    truth = sorted({round(t, 2) for t in raw if 0.0 < t < duration})
    out = RESULTS / "lpm" / video.stem
    t0 = time.monotonic()
    annotate(video, vtt, out, max_images=10_000, use_llm=False, debug=True)
    runtime = time.monotonic() - t0
    debug = json.loads((out / "debug.json").read_text())
    tp, fp, fn = match_events(change_times_from_debug(debug), truth, tolerance)
    p, r, f = prf(tp, fp, fn)
    frame_t = {s["id"]: s["frame_t"] for s in debug["screens"]}
    lines = debug["lines"]
    with_screen = [ln for ln in lines if ln["screen_id"]]
    correct = sum(
        segment_index(truth, (ln["start"] + ln["end"]) / 2)
        == segment_index(truth, frame_t[ln["screen_id"]])
        for ln in with_screen
    )
    return {
        "video_id": video.stem, "speaker": row["speaker"], "duration_s": duration,
        "runtime_s": round(runtime, 1), "truth_changes": len(truth),
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
        "line_coverage": round(len(with_screen) / len(lines), 3) if lines else 0.0,
        "line_screen_accuracy": round(correct / len(with_screen), 3) if with_screen else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--tolerance", type=float, default=2.0)
    args = ap.parse_args()
    csv_path = download(CSV_URL, DATA / "raw_video_links.csv")
    rows = list(csv.DictReader(csv_path.open()))
    results = []
    for row in pick_videos(rows, args.limit, args.max_minutes):
        got = fetch(row)
        if got is None:
            print(f"skip {row['video_id']}: download or captions unavailable", file=sys.stderr)
            continue
        res = evaluate(row, *got, tolerance=args.tolerance)
        print(json.dumps(res))
        results.append(res)
    if not results:
        print("no videos evaluated", file=sys.stderr)
        return 1
    mean = {k: round(sum(r[k] for r in results) / len(results), 3)
            for k in ("precision", "recall", "f1", "line_coverage", "line_screen_accuracy")}
    report = {"date": str(date.today()), "tolerance_s": args.tolerance, "videos": results,
              "mean": mean}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"lpm_{date.today()}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nMEAN over {len(results)} videos: {mean}\nreport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `eval/` is imported as a package in tests (`from eval.common import ...`); confirm `pythonpath` in `[tool.pytest.ini_options]` includes the repo root by adding `"."` → `pythonpath = ["src", "."]`. Exclude `eval/` from mypy's `src/peeklet` run (it already is) but run `uv run ruff check eval`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_eval_common.py -q && uv run ruff check eval tests`
Expected: PASS.

- [ ] **Step 5: Smoke-run on one video (manual, needs network + tesseract)**

Run: `uv run --with yt-dlp python -m eval.lpm_eval --limit 1 --max-minutes 10`
Expected: one JSON result line and `eval/results/lpm_<date>.json`. If YouTube blocks the download, the script prints `skip ...` and exits 1 — report that rather than retrying in a loop.

- [ ] **Step 6: Commit**

```bash
git add eval/__init__.py eval/common.py eval/lpm_eval.py tests/unit/test_eval_common.py pyproject.toml
git commit -m "feat(eval): LPM change-detection and alignment evaluation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 17: GUIDE evaluation script

**Files:**
- Create: `eval/guide_eval.py`
- Modify: `tests/unit/test_eval_common.py` (add `help_switch_recall` tests), `eval/common.py`

**Interfaces:**
- Consumes: `eval.common` helpers; `peeklet.annotate`; `peeklet.image_utils.dhash_64`, `hamming_distance`, `decode_jpeg`.
- Produces (in `eval/common.py`): `help_switch_recall(change_times: list[float], segments: list[tuple[float, float]], slack: float = 2.0) -> tuple[int, int]` (hits, total); `near_duplicate_count(images: list[bytes], max_distance: int = 4) -> int`.
- CLI: `uv run python -m eval.guide_eval --limit 5 [--llm] [--video-id ID ...]`.

Dataset facts (verified 2026-10-07): repo `kixlab/GuideBench` (gated "auto", CC BY 4.0). `videos/video_urls.csv` has 50 rows `video_id,file_name` (public S3 `.mp4` URLs, ~270 MB each). Transcripts at `transcriptions/transcripts_{video_id}.json`, a list of `{"start","end","sentence"}` (no speakers; median segment 24.5s). `annotations/3_help_annotations.json` is a list of `{video_id,start_time,end_time,options,answer,type}`; `type == "1_explicit"` marks switches to Google/YouTube/ChatGPT (91 in the 50 full videos).

- [ ] **Step 1: Write failing tests** — append to `tests/unit/test_eval_common.py`

```python
import numpy as np

from eval.common import help_switch_recall, near_duplicate_count
from peeklet.image_utils import encode_jpeg


def test_help_switch_recall_with_slack() -> None:
    assert help_switch_recall([10.0, 50.0], [(11.0, 20.0), (30.0, 40.0), (51.0, 60.0)]) == (2, 3)


def test_near_duplicate_count() -> None:
    a = encode_jpeg(np.tile(np.arange(64, dtype=np.uint8), (64, 1))[..., None].repeat(3, 2))
    b = encode_jpeg(np.full((64, 64, 3), 128, dtype=np.uint8))
    assert near_duplicate_count([a, a, b]) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_eval_common.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement** — append to `eval/common.py`

```python
def help_switch_recall(
    change_times: list[float], segments: list[tuple[float, float]], slack: float = 2.0
) -> tuple[int, int]:
    """How many app-switch segments contain a predicted screen change (± slack)."""
    hits = sum(any(s - slack <= t <= e + slack for t in change_times) for s, e in segments)
    return hits, len(segments)


def near_duplicate_count(images: list[bytes], max_distance: int = 4) -> int:
    """Images whose dHash is within max_distance of an earlier image."""
    from peeklet.image_utils import decode_jpeg, dhash_64, hamming_distance

    hashes: list[int] = []
    dupes = 0
    for data in images:
        h = dhash_64(decode_jpeg(data))
        if any(hamming_distance(h, prev) <= max_distance for prev in hashes):
            dupes += 1
        hashes.append(h)
    return dupes
```

`eval/guide_eval.py`:

```python
"""GUIDE evaluation: real software recordings with think-aloud narration.

Usage: uv run python -m eval.guide_eval --limit 5 [--llm]
Requires a Hugging Face token with access to kixlab/GuideBench (HF_TOKEN or
~/.cache/huggingface/token). Data (CC BY 4.0) goes to eval/data/guide/, never committed.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from eval.common import (
    change_times_from_debug,
    download,
    help_switch_recall,
    hf_token,
    near_duplicate_count,
    whisperx_to_vtt,
)
from peeklet import annotate

HF = "https://huggingface.co/datasets/kixlab/GuideBench/resolve/main"
DATA = Path("eval/data/guide")
RESULTS = Path("eval/results")


def gallery(out: Path, video_id: str) -> None:
    entries = json.loads((out / "transcript.json").read_text())
    rows = []
    for e in entries:
        img = f'<img src="{html.escape(e["image"])}" width="640">' if e["image"] else "<em>no image</em>"
        vc = f"<p><b>Visual context:</b> {html.escape(e.get('visual_context', ''))}</p>"
        text = html.escape(e["transcript"]).replace("\n", "<br>")
        rows.append(f"<section><h3>{e['timestamp']}</h3><p>{text}</p>{img}{vc}</section>")
    (out / "index.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>{video_id}</title>"
        "<style>body{font-family:sans-serif;max-width:900px;margin:auto}"
        "section{border-bottom:1px solid #ccc;padding:12px 0}</style>" + "".join(rows)
    )


def evaluate(video_id: str, url: str, headers: dict[str, str], help_items: list[dict[str, Any]],
             use_llm: bool) -> dict[str, Any]:
    vdir = DATA / video_id
    video = download(url, vdir / f"{video_id}.mp4")
    tx = download(f"{HF}/transcriptions/transcripts_{video_id}.json", vdir / "transcript.json",
                  headers)
    vtt = vdir / "transcript.vtt"
    vtt.write_text(whisperx_to_vtt(json.loads(tx.read_text())))
    out = RESULTS / "guide" / video_id
    t0 = time.monotonic()
    entries = annotate(video, vtt, out, use_llm=use_llm, debug=True)
    runtime = time.monotonic() - t0
    debug = json.loads((out / "debug.json").read_text())
    duration_min = max(ln["end"] for ln in debug["lines"]) / 60 if debug["lines"] else 0.0
    kept_imgs = [(out / e.image).read_bytes() for e in entries if e.image]
    unique_imgs = list({e.image: None for e in entries if e.image})
    segments = [(float(h["start_time"]), float(h["end_time"])) for h in help_items
                if h["video_id"] == video_id and h["type"] == "1_explicit"]
    hits, total = help_switch_recall(change_times_from_debug(debug), segments)
    lines = debug["lines"]
    gallery(out, video_id)
    return {
        "video_id": video_id,
        "duration_min": round(duration_min, 1),
        "runtime_s": round(runtime, 1),
        "runtime_per_video_hour_s": round(runtime / (duration_min / 60), 1) if duration_min else None,
        "screen_pass_s": round(debug["timing"]["seconds_screen_pass"], 1),
        "lines": len(lines),
        "line_image_coverage": round(sum(1 for e in entries for _ in e.lines if e.image) / len(lines), 3)
        if lines else 0.0,
        "screens_found": entries.stats.screens_found,
        "screens_per_min": round(entries.stats.screens_found / duration_min, 2) if duration_min else None,
        "kept": entries.stats.kept,
        "near_duplicate_kept": near_duplicate_count(
            [(out / name).read_bytes() for name in unique_imgs]) if kept_imgs else 0,
        "llm_include_rate": _include_rate(debug) if use_llm else None,
        "help_switch_hits": hits,
        "help_switch_total": total,
        "warnings": len(entries.stats.warnings),
        "gallery": str(out / "index.html"),
    }


def _include_rate(debug: dict[str, Any]) -> float | None:
    judged = [s["judgment"] for s in debug["screens"] if s["shortlisted"] and s["judgment"]]
    return round(sum(j["include"] for j in judged) / len(judged), 3) if judged else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--video-id", action="append", default=[])
    ap.add_argument("--llm", action="store_true", help="Run the LLM judge (costs money).")
    args = ap.parse_args()
    token = hf_token()
    if not token:
        print("No Hugging Face token: set HF_TOKEN or run `hf auth login`, and accept the "
              "terms at https://huggingface.co/datasets/kixlab/GuideBench", file=sys.stderr)
        return 2
    headers = {"Authorization": f"Bearer {token}"}
    urls_csv = download(f"{HF}/videos/video_urls.csv", DATA / "video_urls.csv", headers)
    help_json = download(f"{HF}/annotations/3_help_annotations.json",
                         DATA / "3_help_annotations.json", headers)
    rows = list(csv.DictReader(urls_csv.open()))
    help_items = json.loads(help_json.read_text())
    if args.video_id:
        rows = [r for r in rows if r["video_id"] in set(args.video_id)]
    else:
        rows = sorted(rows, key=lambda r: r["video_id"])[: args.limit]
    results = []
    for r in rows:
        res = evaluate(r["video_id"], r["file_name"], headers, help_items, args.llm)
        print(json.dumps(res))
        results.append(res)
    if not results:
        print("no videos evaluated", file=sys.stderr)
        return 1
    hits = sum(r["help_switch_hits"] for r in results)
    total = sum(r["help_switch_total"] for r in results)
    summary = {
        "videos": len(results),
        "mean_runtime_per_video_hour_s": round(
            sum(r["runtime_per_video_hour_s"] or 0 for r in results) / len(results), 1),
        "mean_line_image_coverage": round(
            sum(r["line_image_coverage"] for r in results) / len(results), 3),
        "mean_screens_per_min": round(sum(r["screens_per_min"] or 0 for r in results) / len(results), 2),
        "help_switch_recall": round(hits / total, 3) if total else None,
    }
    report = {"date": str(date.today()), "llm": args.llm, "videos": results, "summary": summary}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"guide_{date.today()}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSUMMARY: {summary}\nreport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_eval_common.py -q && uv run ruff check eval && uv run ruff format eval`
Expected: PASS.

- [ ] **Step 5: Smoke-run on one video (manual, needs network, tesseract, HF token)**

Run: `uv run python -m eval.guide_eval --limit 1`
Expected: one JSON result line, `eval/results/guide_<date>.json`, and a browsable `eval/results/guide/<video_id>/index.html`. Record `runtime_per_video_hour_s` — it answers the spec's open question about OCR speed.

- [ ] **Step 6: Commit**

```bash
git add eval/guide_eval.py eval/common.py tests/unit/test_eval_common.py
git commit -m "feat(eval): GUIDE real-recording evaluation with app-switch recall

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage check

| Spec section | Task |
|---|---|
| Single path, removals, dependencies, README | 1, 2, 15 |
| Interface (CLI flags, `annotate`, `to_claude_content`) | 14, 15, 13 |
| Output `transcript.json` rules, filenames, `debug.json` | 4, 13, 14 |
| 1. Parse + long-line splitting + `TranscriptError` | 5 |
| 2. Checkpoints: action items / speech onsets / verbal cues | 5, 7, 6 |
| 3. Candidates, streaming, OCR cache, checkpoint windows | 8, 9 |
| 4. Screens: rejector, fingerprint, grouping, occurrences, chosen frame | 8, 9 |
| 5. Align (≤ 1 screen per line, lead) | 10 |
| 6. Heuristic score + anchor bonus | 11 |
| 7. LLM judge: shortlist, prompt, providers, cache, failures, missing key | 12, 14 |
| 8. Select (LLM / no-LLM, action-item override, `max_images=0`) | 11, 14 |
| 9. Blocks and render | 13 |
| Configuration | 3 |
| Error handling table | 5, 7, 8, 9, 12, 14, 15 |
| Testing (unit, integration, CLI, manual) | every task; 14; 15; 16–17 |
| Evaluation (LPM, GUIDE) | 16, 17 |
