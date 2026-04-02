# Peeklet Dataset Testing & Visual Results Explorer

**Date:** 2026-04-01
**Status:** Approved

## Overview

Thorough testing of Peeklet against real-world data (Mind2Web dataset) before PR to main, plus a self-contained HTML visual explorer to inspect pipeline decisions frame-by-frame.

## Part 1: Dataset Test Infrastructure

### Structure

```
tests/datasets/
├── conftest.py          # Fixtures, skip-if-no-data logic
├── test_web_tasks.py    # Test logic for web task sequences
├── data/                # .gitignore'd
│   └── web_tasks/       # Downloaded Mind2Web trace files
scripts/
├── setup_test_data.py   # Downloads & prepares data
```

### Behavior

- `setup_test_data.py` downloads Mind2Web trace files (screenshots + action metadata) into `tests/datasets/data/web_tasks/`.
- `conftest.py` provides a `web_tasks_available` fixture and applies `pytest.mark.datasets` to auto-skip when data is missing.
- Test cases process each task's screenshot sequence through Peeklet's pipeline and validate:
  - Keyframes align with action steps (click/type/scroll should trigger a keyframe)
  - Idle sequences between actions are correctly skipped
  - SSIM scores and hash decisions are reasonable
  - Adaptive mask doesn't suppress real UI changes
  - Export produces valid manifest with correct metadata
- Results (pipeline output + per-frame metrics) are saved to a `results/` directory for the explorer.

### Test marker

Tests use `pytest.mark.datasets` and auto-skip when data directory is absent.

## Part 2: Visual Results Explorer

### Output

A self-contained HTML file: `results/report.html`

### Generator

`scripts/generate_report.py` — reads pipeline results directory (keyframe images + manifest parquet), produces the HTML report.

### Layout

- **Top bar:** Summary stats — total frames, keyframes, skip rate, processing time.
- **Left sidebar:** List of task sequences, each showing name + keyframe count + skip rate. Click to select.
- **Main area:** For the selected sequence, a vertical timeline of all frames:
  - Each frame shows: thumbnail, frame index, decision badge (KEYFRAME/SKIPPED).
  - Keyframes expanded by default: full-size image, hash value, SSIM score, changed regions highlighted, mask overlay toggle.
  - Skipped frames collapsed by default: one-line summary with hash match / SSIM score. Click to expand.
  - Side-by-side comparison: current frame vs previous keyframe with diff overlay.

### Tech

- Pure HTML + CSS + vanilla JS (no dependencies).
- Images base64-encoded inline or referenced from results directory.
- CSS grid layout, dark/light mode toggle.
- Collapsible sections for skipped frames.

## Part 3: Integration Flow

### Developer workflow

```bash
# One-time setup
python scripts/setup_test_data.py

# Run all tests (dataset tests auto-skip if data missing)
pytest

# Run dataset tests specifically
pytest tests/datasets/ -m datasets

# Generate visual report after tests
python scripts/generate_report.py results/ -o results/report.html

# Open in browser
open results/report.html
```

### What gets committed

- `tests/datasets/conftest.py`, `test_web_tasks.py` — test logic
- `scripts/setup_test_data.py`, `scripts/generate_report.py` — tooling
- Updated `.gitignore` — excludes `tests/datasets/data/` and `results/`

### What stays local

- `tests/datasets/data/web_tasks/` — downloaded dataset
- `results/` — pipeline output + HTML report

## Design Decisions

- **Mind2Web first (approach B):** Best match for Peeklet's screenshot-sequence-to-keyframe pipeline. WebUI and ShowUI can be added later under `tests/datasets/data/`.
- **Static HTML explorer:** Zero dependencies, just open in browser. No server needed.
- **Generic directory naming (`datasets`, `web_tasks`):** Future-proof for adding more datasets without renaming.
- **JUnit XML not used for explorer:** Standard pytest output for CI, but the visual explorer reads directly from pipeline output (manifest parquet + keyframe images). No intermediate format.
