# Dataset Testing & Visual Results Explorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test Peeklet against real-world Mind2Web screenshot sequences and build a self-contained HTML explorer to visually inspect pipeline decisions.

**Architecture:** A download script fetches Mind2Web data from Hugging Face into a gitignored local directory. Pytest tests run each task's screenshot sequence through Peeklet's Pipeline and save results. A report generator reads the manifest parquet + keyframe images and produces a standalone HTML file with per-frame cascade visualization.

**Tech Stack:** Python 3.10+, `datasets` (Hugging Face), `pyarrow`, `pytest`, vanilla HTML/CSS/JS

---

### Task 1: Update .gitignore and pytest markers

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml:85-93`

- [ ] **Step 1: Add results/ to .gitignore**

In `.gitignore`, add under the `# Peeklet specific` section:

```
results/
```

- [ ] **Step 2: Add datasets marker to pyproject.toml**

Replace the existing markers list in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "datasets: tests requiring downloaded datasets (deselect with '-m not datasets')",
    "mind2web: tests requiring Mind2Web dataset (deselect with '-m not mind2web')",
    "showui: tests requiring ShowUI dataset (deselect with '-m not showui')",
    "webui: tests requiring WebUI dataset (deselect with '-m not webui')",
    "benchmark: performance benchmark tests",
]
```

- [ ] **Step 3: Add datasets optional dependency**

Add a new optional dependency group in `pyproject.toml` after the `dev` group:

```toml
datasets = [
    "datasets>=2.14",
    "huggingface-hub>=0.17",
]
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore pyproject.toml
git commit -m "chore: add datasets marker, results/ to gitignore, HF dependency"
```

---

### Task 2: Create the data download script

**Files:**
- Create: `scripts/setup_test_data.py`

- [ ] **Step 1: Write the download script**

```python
"""Download and prepare Mind2Web dataset for Peeklet testing.

Usage:
    python scripts/setup_test_data.py [--num-tasks 10] [--split test_task]

Downloads screenshot sequences from Hugging Face's Multimodal-Mind2Web
dataset and organizes them into tests/datasets/data/web_tasks/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "datasets" / "data" / "web_tasks"
HF_DATASET = "osunlp/Multimodal-Mind2Web"


def download_tasks(num_tasks: int = 10, split: str = "test_task") -> None:
    """Download Mind2Web tasks and save as screenshot sequences."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. Install with:")
        print("  pip install 'peeklet[datasets]'")
        sys.exit(1)

    print(f"Loading {split} split from {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split=split)

    # Group rows by annotation_id (each row is one action step)
    tasks: dict[str, list] = {}
    for row in ds:
        task_id = row["annotation_id"]
        if task_id not in tasks:
            tasks[task_id] = []
        tasks[task_id].append(row)

    # Sort each task's steps by action index
    for task_id in tasks:
        tasks[task_id].sort(key=lambda r: int(r["target_action_index"]))

    # Take first N tasks
    selected_ids = list(tasks.keys())[:num_tasks]
    print(f"Found {len(tasks)} tasks, selecting {len(selected_ids)}...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for task_id in selected_ids:
        steps = tasks[task_id]
        task_dir = DATA_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "annotation_id": task_id,
            "website": steps[0].get("website", ""),
            "domain": steps[0].get("domain", ""),
            "subdomain": steps[0].get("subdomain", ""),
            "confirmed_task": steps[0].get("confirmed_task", ""),
            "num_steps": len(steps),
            "actions": [],
        }

        for i, step in enumerate(steps):
            # Save screenshot
            screenshot = step["screenshot"]  # PIL Image
            img_path = task_dir / f"step_{i:03d}.png"
            screenshot.save(img_path)

            # Collect action metadata
            metadata["actions"].append({
                "step_index": i,
                "action_uid": step.get("action_uid", ""),
                "operation": step.get("operation", {}),
                "target_action_reprs": step.get("target_action_reprs", ""),
            })

        # Save task metadata
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"  Saved {task_id}: {len(steps)} steps")

    # Write index file
    index = {
        "source": HF_DATASET,
        "split": split,
        "num_tasks": len(selected_ids),
        "task_ids": selected_ids,
    }
    (DATA_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nDone! {len(selected_ids)} tasks saved to {DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Mind2Web test data")
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks to download")
    parser.add_argument("--split", default="test_task", help="Dataset split to use")
    args = parser.parse_args()
    download_tasks(num_tasks=args.num_tasks, split=args.split)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script runs (dry check)**

```bash
python scripts/setup_test_data.py --help
```

Expected: help text with `--num-tasks` and `--split` options.

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_test_data.py
git commit -m "feat: add Mind2Web data download script"
```

---

### Task 3: Create dataset test fixtures

**Files:**
- Create: `tests/datasets/__init__.py`
- Create: `tests/datasets/conftest.py`

- [ ] **Step 1: Create empty __init__.py**

```python
```

(Empty file — makes `tests/datasets/` a package.)

- [ ] **Step 2: Write conftest.py with fixtures and skip logic**

```python
"""Fixtures and auto-skip logic for dataset-based tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
WEB_TASKS_DIR = DATA_DIR / "web_tasks"


def web_tasks_available() -> bool:
    """Check if Mind2Web data has been downloaded."""
    index_file = WEB_TASKS_DIR / "index.json"
    return index_file.exists()


def load_task_ids() -> list[str]:
    """Load list of available task IDs."""
    index_file = WEB_TASKS_DIR / "index.json"
    if not index_file.exists():
        return []
    index = json.loads(index_file.read_text())
    return index.get("task_ids", [])


def load_task_metadata(task_id: str) -> dict:
    """Load metadata for a specific task."""
    meta_path = WEB_TASKS_DIR / task_id / "metadata.json"
    return json.loads(meta_path.read_text())


def get_task_screenshots(task_id: str) -> list[Path]:
    """Get sorted list of screenshot paths for a task."""
    task_dir = WEB_TASKS_DIR / task_id
    return sorted(task_dir.glob("step_*.png"))


skip_no_data = pytest.mark.skipif(
    not web_tasks_available(),
    reason="Mind2Web data not downloaded. Run: python scripts/setup_test_data.py",
)


@pytest.fixture
def web_tasks_dir() -> Path:
    """Path to the web_tasks data directory."""
    return WEB_TASKS_DIR


@pytest.fixture
def task_ids() -> list[str]:
    """List of available task IDs."""
    return load_task_ids()


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Temporary results directory for test pipeline output."""
    out = tmp_path / "results"
    out.mkdir()
    return out
```

- [ ] **Step 3: Commit**

```bash
git add tests/datasets/__init__.py tests/datasets/conftest.py
git commit -m "feat: add dataset test fixtures with auto-skip logic"
```

---

### Task 4: Write the web tasks test suite

**Files:**
- Create: `tests/datasets/test_web_tasks.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests that run Mind2Web screenshot sequences through Peeklet's pipeline.

These tests validate that Peeklet correctly identifies keyframes at action
boundaries (click, type, select) and skips idle frames between actions.

Requires data download: python scripts/setup_test_data.py
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from peeklet.config import PeekletConfig
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline
from peeklet.utils.types import EventType

from .conftest import (
    get_task_screenshots,
    load_task_ids,
    load_task_metadata,
    skip_no_data,
)


def _make_config(output_dir: Path) -> PeekletConfig:
    """Create pipeline config for dataset testing."""
    return PeekletConfig.model_validate({
        "redactor": {"enabled": False},
        "exporter": {"output_dir": str(output_dir)},
    })


def _run_task(task_id: str, output_dir: Path) -> list[dict]:
    """Run a single task's screenshots through the pipeline, return results."""
    task_output = output_dir / task_id
    task_output.mkdir(parents=True, exist_ok=True)

    config = _make_config(task_output)
    pipeline = Pipeline(config)
    metadata = load_task_metadata(task_id)
    screenshots = get_task_screenshots(task_id)

    results = []
    for i, img_path in enumerate(screenshots):
        frame = load_frame(img_path)
        result = pipeline.process_frame(
            frame,
            frame_id=f"step_{i:03d}",
            app_name=metadata.get("website", ""),
            window_title=metadata.get("confirmed_task", ""),
        )
        results.append({
            "frame_id": result.frame_id,
            "event_type": result.event_type.value,
            "is_keyframe": result.is_keyframe,
            "perceptual_hash": result.perceptual_hash,
            "ssim_score": result.ssim_score,
            "change_score": result.change_score,
            "changed_pct": result.changed_pct,
            "asset_path": result.asset_path,
            "action": metadata["actions"][i] if i < len(metadata["actions"]) else None,
        })

    pipeline.finalize()
    return results


@pytest.mark.datasets
class TestWebTasks:
    """Tests against real Mind2Web screenshot sequences."""

    @skip_no_data
    def test_first_frame_is_always_keyframe(self, results_dir: Path) -> None:
        """The first frame of every task must be a keyframe."""
        task_ids = load_task_ids()
        assert len(task_ids) > 0, "No tasks available"

        for task_id in task_ids:
            results = _run_task(task_id, results_dir)
            assert results[0]["is_keyframe"] is True, (
                f"Task {task_id}: first frame should be keyframe"
            )

    @skip_no_data
    def test_not_all_frames_skipped(self, results_dir: Path) -> None:
        """Each task should have at least one keyframe beyond the first frame.
        Mind2Web tasks have meaningful UI changes (clicks, typing, navigation).
        """
        task_ids = load_task_ids()
        for task_id in task_ids:
            results = _run_task(task_id, results_dir)
            keyframes = [r for r in results if r["is_keyframe"]]
            assert len(keyframes) >= 1, (
                f"Task {task_id}: expected at least 1 keyframe, got {len(keyframes)}"
            )

    @skip_no_data
    def test_keyframe_reduction_rate(self, results_dir: Path) -> None:
        """Pipeline should reduce frames — not every frame is a keyframe.
        Expect at least 20% reduction across all tasks.
        """
        task_ids = load_task_ids()
        total_frames = 0
        total_keyframes = 0

        for task_id in task_ids:
            results = _run_task(task_id, results_dir)
            total_frames += len(results)
            total_keyframes += sum(1 for r in results if r["is_keyframe"])

        skip_rate = 1 - (total_keyframes / total_frames)
        assert skip_rate >= 0.2, (
            f"Expected at least 20% frame reduction, got {skip_rate:.1%}"
        )

    @skip_no_data
    def test_ssim_scores_within_range(self, results_dir: Path) -> None:
        """All SSIM scores should be between 0 and 1."""
        task_ids = load_task_ids()
        for task_id in task_ids[:3]:  # Spot-check first 3 tasks
            results = _run_task(task_id, results_dir)
            for r in results:
                if r["ssim_score"] is not None:
                    assert 0.0 <= r["ssim_score"] <= 1.0, (
                        f"Task {task_id}, {r['frame_id']}: "
                        f"SSIM {r['ssim_score']} out of range"
                    )

    @skip_no_data
    def test_manifest_parquet_written(self, results_dir: Path) -> None:
        """Each task should produce a valid manifest.parquet."""
        task_ids = load_task_ids()
        task_id = task_ids[0]
        _run_task(task_id, results_dir)

        manifest = results_dir / task_id / "manifest.parquet"
        assert manifest.exists(), f"Manifest not written for {task_id}"

        table = pq.read_table(manifest)
        assert table.num_rows > 0
        assert "frame_id" in table.column_names
        assert "event_type" in table.column_names
        assert "ssim_score" in table.column_names

    @skip_no_data
    def test_keyframe_images_saved(self, results_dir: Path) -> None:
        """Keyframe images should be saved to disk."""
        task_ids = load_task_ids()
        task_id = task_ids[0]
        results = _run_task(task_id, results_dir)

        keyframe_count = sum(1 for r in results if r["is_keyframe"])
        saved_images = list((results_dir / task_id).glob("*.png"))
        assert len(saved_images) == keyframe_count, (
            f"Expected {keyframe_count} saved keyframes, found {len(saved_images)}"
        )

    @skip_no_data
    def test_hash_match_skips_ssim(self, results_dir: Path) -> None:
        """When hash matches, SSIM should be None (cascade optimization)."""
        task_ids = load_task_ids()
        task_id = task_ids[0]
        results = _run_task(task_id, results_dir)

        skipped_no_ssim = [
            r for r in results
            if not r["is_keyframe"] and r["ssim_score"] is None
        ]
        # If there are any hash-matched skips, verify they have no SSIM
        for r in skipped_no_ssim:
            assert r["ssim_score"] is None
            assert r["change_score"] is None
```

- [ ] **Step 2: Run tests to verify they skip when data is missing**

```bash
pytest tests/datasets/ -v
```

Expected: all tests SKIPPED with reason "Mind2Web data not downloaded."

- [ ] **Step 3: Commit**

```bash
git add tests/datasets/test_web_tasks.py
git commit -m "feat: add Mind2Web dataset test suite with 7 validation tests"
```

---

### Task 5: Create the results saver (bridge between tests and explorer)

**Files:**
- Create: `scripts/run_dataset_tests.py`

The pytest tests use `tmp_path` (cleaned up after). We need a separate script that processes tasks and saves results persistently for the HTML explorer.

- [ ] **Step 1: Write the results runner script**

```python
"""Run Peeklet pipeline on downloaded dataset and save results for the HTML explorer.

Usage:
    python scripts/run_dataset_tests.py [--output results/]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from peeklet.config import PeekletConfig
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "tests" / "datasets" / "data" / "web_tasks"


def process_all_tasks(output_dir: Path) -> dict:
    """Process all downloaded tasks and save results."""
    index_file = DATA_DIR / "index.json"
    if not index_file.exists():
        print("Error: No dataset found. Run: python scripts/setup_test_data.py")
        return {}

    index = json.loads(index_file.read_text())
    task_ids = index["task_ids"]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": index["source"],
        "split": index["split"],
        "tasks": [],
        "total_frames": 0,
        "total_keyframes": 0,
        "total_time_ms": 0,
    }

    for task_id in task_ids:
        task_dir = DATA_DIR / task_id
        meta = json.loads((task_dir / "metadata.json").read_text())
        screenshots = sorted(task_dir.glob("step_*.png"))

        task_output = output_dir / task_id
        task_output.mkdir(parents=True, exist_ok=True)

        config = PeekletConfig.model_validate({
            "redactor": {"enabled": False},
            "exporter": {"output_dir": str(task_output)},
        })
        pipeline = Pipeline(config)

        task_results = []
        start = time.perf_counter()

        for i, img_path in enumerate(screenshots):
            frame = load_frame(img_path)
            result = pipeline.process_frame(
                frame,
                frame_id=f"step_{i:03d}",
                app_name=meta.get("website", ""),
                window_title=meta.get("confirmed_task", ""),
            )
            action = meta["actions"][i] if i < len(meta["actions"]) else None

            task_results.append({
                "frame_id": result.frame_id,
                "event_type": result.event_type.value,
                "is_keyframe": result.is_keyframe,
                "perceptual_hash": result.perceptual_hash,
                "ssim_score": result.ssim_score,
                "change_score": result.change_score,
                "changed_pct": result.changed_pct,
                "changed_regions": (
                    [r.to_dict() for r in result.changed_regions]
                    if result.changed_regions else None
                ),
                "adaptive_mask": (
                    [r.to_dict() for r in result.adaptive_mask]
                    if result.adaptive_mask else None
                ),
                "asset_path": result.asset_path,
                "action": action,
                "source_image": str(img_path.name),
            })

        pipeline.finalize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        keyframe_count = sum(1 for r in task_results if r["is_keyframe"])
        task_summary = {
            "task_id": task_id,
            "website": meta.get("website", ""),
            "domain": meta.get("domain", ""),
            "confirmed_task": meta.get("confirmed_task", ""),
            "num_frames": len(task_results),
            "num_keyframes": keyframe_count,
            "skip_rate": 1 - (keyframe_count / len(task_results)) if task_results else 0,
            "processing_time_ms": round(elapsed_ms, 1),
            "frames": task_results,
        }

        # Save per-task results JSON
        (task_output / "results.json").write_text(json.dumps(task_summary, indent=2))

        summary["tasks"].append({
            "task_id": task_id,
            "website": task_summary["website"],
            "domain": task_summary["domain"],
            "confirmed_task": task_summary["confirmed_task"],
            "num_frames": task_summary["num_frames"],
            "num_keyframes": task_summary["num_keyframes"],
            "skip_rate": task_summary["skip_rate"],
            "processing_time_ms": task_summary["processing_time_ms"],
        })
        summary["total_frames"] += len(task_results)
        summary["total_keyframes"] += keyframe_count
        summary["total_time_ms"] += round(elapsed_ms, 1)

        print(
            f"  {task_id}: {len(task_results)} frames, "
            f"{keyframe_count} keyframes, "
            f"{task_summary['skip_rate']:.0%} skip rate, "
            f"{elapsed_ms:.0f}ms"
        )

    # Save global summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    total_skip = 1 - (summary["total_keyframes"] / summary["total_frames"]) if summary["total_frames"] else 0
    print(f"\nTotal: {summary['total_frames']} frames, {summary['total_keyframes']} keyframes, {total_skip:.0%} skip rate")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Peeklet on dataset and save results")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()
    process_all_tasks(Path(args.output))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script shows help**

```bash
python scripts/run_dataset_tests.py --help
```

Expected: help text with `--output` option.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_dataset_tests.py
git commit -m "feat: add dataset results runner for HTML explorer"
```

---

### Task 6: Build the HTML report generator

**Files:**
- Create: `scripts/generate_report.py`

This is the largest task. The script reads `results/summary.json` and per-task `results.json` files, encodes images as base64, and produces a single self-contained HTML file.

- [ ] **Step 1: Write the report generator**

```python
"""Generate a self-contained HTML report from Peeklet pipeline results.

Usage:
    python scripts/generate_report.py [results_dir] [-o report.html]

Reads summary.json and per-task results.json from the results directory.
Produces a standalone HTML file with embedded images and interactive UI.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def encode_image_base64(path: Path) -> str:
    """Encode an image file as a base64 data URI."""
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def load_results(results_dir: Path) -> dict:
    """Load summary and all per-task results."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run scripts/run_dataset_tests.py first.")
        sys.exit(1)

    summary = json.loads(summary_path.read_text())

    # Load per-task details and encode images
    for task in summary["tasks"]:
        task_dir = results_dir / task["task_id"]
        task_results_path = task_dir / "results.json"
        if task_results_path.exists():
            task_detail = json.loads(task_results_path.read_text())
            task["frames"] = task_detail["frames"]

            # Encode source images from dataset
            data_task_dir = (
                Path(__file__).resolve().parent.parent
                / "tests" / "datasets" / "data" / "web_tasks" / task["task_id"]
            )
            for frame in task["frames"]:
                src_img = data_task_dir / frame["source_image"]
                if src_img.exists():
                    frame["source_image_b64"] = encode_image_base64(src_img)
                # Encode keyframe image if it exists
                if frame["asset_path"]:
                    asset = Path(frame["asset_path"])
                    if asset.exists():
                        frame["keyframe_image_b64"] = encode_image_base64(asset)

    return summary


def generate_html(summary: dict) -> str:
    """Generate the complete HTML report string."""
    tasks_json = json.dumps(summary["tasks"])
    total_skip_rate = (
        1 - (summary["total_keyframes"] / summary["total_frames"])
        if summary["total_frames"] else 0
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Peeklet Results Explorer</title>
<style>
:root {{
    --bg: #1a1a2e;
    --bg-card: #16213e;
    --bg-sidebar: #0f3460;
    --text: #e6e6e6;
    --text-muted: #8888aa;
    --accent: #e94560;
    --accent-green: #4ecca3;
    --accent-yellow: #ffd700;
    --border: #2a2a4a;
}}
[data-theme="light"] {{
    --bg: #f5f5f5;
    --bg-card: #ffffff;
    --bg-sidebar: #e8e8e8;
    --text: #222222;
    --text-muted: #666666;
    --accent: #d63031;
    --accent-green: #00b894;
    --accent-yellow: #f39c12;
    --border: #dddddd;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    background: var(--bg);
    color: var(--text);
    display: grid;
    grid-template-rows: auto 1fr;
    grid-template-columns: 300px 1fr;
    height: 100vh;
}}
/* Top bar */
.topbar {{
    grid-column: 1 / -1;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 24px;
}}
.topbar h1 {{ font-size: 16px; font-weight: 600; }}
.stat {{
    display: flex;
    flex-direction: column;
    align-items: center;
}}
.stat-value {{ font-size: 20px; font-weight: 700; }}
.stat-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; }}
.stat-value.green {{ color: var(--accent-green); }}
.stat-value.accent {{ color: var(--accent); }}
.theme-toggle {{
    margin-left: auto;
    background: var(--border);
    border: none;
    color: var(--text);
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
}}
/* Sidebar */
.sidebar {{
    background: var(--bg-sidebar);
    overflow-y: auto;
    border-right: 1px solid var(--border);
    padding: 8px;
}}
.task-item {{
    padding: 10px;
    border-radius: 6px;
    cursor: pointer;
    margin-bottom: 4px;
    border: 1px solid transparent;
}}
.task-item:hover {{ border-color: var(--border); }}
.task-item.active {{ background: var(--bg-card); border-color: var(--accent); }}
.task-name {{ font-size: 13px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.task-meta {{ font-size: 11px; color: var(--text-muted); margin-top: 2px; }}
/* Main area */
.main {{
    overflow-y: auto;
    padding: 16px;
}}
.frame-timeline {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}
.frame-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}}
.frame-header {{
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    cursor: pointer;
    user-select: none;
}}
.frame-header:hover {{ background: var(--bg-sidebar); }}
.badge {{
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 3px;
    text-transform: uppercase;
}}
.badge.keyframe {{ background: var(--accent); color: white; }}
.badge.skipped {{ background: var(--border); color: var(--text-muted); }}
.frame-meta-inline {{ font-size: 12px; color: var(--text-muted); }}
.frame-detail {{
    padding: 14px;
    border-top: 1px solid var(--border);
    display: none;
}}
.frame-detail.open {{ display: block; }}
.frame-detail img {{
    max-width: 100%;
    border-radius: 4px;
    margin-bottom: 8px;
}}
.frame-props {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px;
    font-size: 12px;
}}
.prop-label {{ color: var(--text-muted); }}
.prop-value {{ font-weight: 500; }}
.action-repr {{
    margin-top: 8px;
    padding: 6px 10px;
    background: var(--bg-sidebar);
    border-radius: 4px;
    font-size: 12px;
}}
.image-compare {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 8px;
}}
.image-compare img {{ width: 100%; border-radius: 4px; }}
.image-label {{ font-size: 11px; color: var(--text-muted); text-align: center; margin-top: 4px; }}
.no-tasks {{ padding: 40px; text-align: center; color: var(--text-muted); }}
</style>
</head>
<body>

<div class="topbar">
    <h1>Peeklet Results Explorer</h1>
    <div class="stat">
        <span class="stat-value">{summary['total_frames']}</span>
        <span class="stat-label">Total Frames</span>
    </div>
    <div class="stat">
        <span class="stat-value accent">{summary['total_keyframes']}</span>
        <span class="stat-label">Keyframes</span>
    </div>
    <div class="stat">
        <span class="stat-value green">{total_skip_rate:.0%}</span>
        <span class="stat-label">Skip Rate</span>
    </div>
    <div class="stat">
        <span class="stat-value">{summary['total_time_ms']:.0f}ms</span>
        <span class="stat-label">Processing</span>
    </div>
    <button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
</div>

<div class="sidebar" id="sidebar"></div>
<div class="main" id="main">
    <div class="no-tasks">Select a task from the sidebar</div>
</div>

<script>
const TASKS = {tasks_json};
let activeTask = null;

function toggleTheme() {{
    document.documentElement.toggleAttribute('data-theme');
    if (document.documentElement.hasAttribute('data-theme')) {{
        document.documentElement.setAttribute('data-theme', 'light');
    }} else {{
        document.documentElement.removeAttribute('data-theme');
    }}
}}

function renderSidebar() {{
    const sidebar = document.getElementById('sidebar');
    sidebar.innerHTML = TASKS.map((t, i) => `
        <div class="task-item ${{activeTask === i ? 'active' : ''}}" onclick="selectTask(${{i}})">
            <div class="task-name" title="${{t.confirmed_task}}">${{t.confirmed_task || t.task_id}}</div>
            <div class="task-meta">
                ${{t.website}} &middot; ${{t.num_keyframes}}/${{t.num_frames}} keyframes &middot; ${{Math.round(t.skip_rate * 100)}}% skip
            </div>
        </div>
    `).join('');
}}

function selectTask(index) {{
    activeTask = index;
    renderSidebar();
    renderMain(TASKS[index]);
}}

function renderMain(task) {{
    const main = document.getElementById('main');
    if (!task.frames) {{
        main.innerHTML = '<div class="no-tasks">No frame data available</div>';
        return;
    }}
    main.innerHTML = `
        <h2 style="margin-bottom:12px;font-size:15px;">${{task.confirmed_task || task.task_id}}</h2>
        <div class="frame-timeline">
            ${{task.frames.map((f, i) => renderFrame(f, i, task)).join('')}}
        </div>
    `;
}}

function renderFrame(frame, index, task) {{
    const isKF = frame.is_keyframe;
    const openByDefault = isKF;
    const ssimStr = frame.ssim_score !== null ? frame.ssim_score.toFixed(4) : 'N/A (hash match)';
    const actionStr = frame.action ? frame.action.target_action_reprs || '' : '';
    const opStr = frame.action && frame.action.operation ? frame.action.operation.op || '' : '';

    let imageSection = '';
    if (frame.source_image_b64) {{
        if (isKF && frame.keyframe_image_b64) {{
            imageSection = `
                <div class="image-compare">
                    <div><img src="${{frame.source_image_b64}}"><div class="image-label">Source Screenshot</div></div>
                    <div><img src="${{frame.keyframe_image_b64}}"><div class="image-label">Saved Keyframe</div></div>
                </div>
            `;
        }} else {{
            imageSection = `<img src="${{frame.source_image_b64}}">`;
        }}
    }}

    return `
        <div class="frame-card">
            <div class="frame-header" onclick="this.nextElementSibling.classList.toggle('open')">
                <span class="badge ${{isKF ? 'keyframe' : 'skipped'}}">${{isKF ? 'Keyframe' : 'Skipped'}}</span>
                <strong>${{frame.frame_id}}</strong>
                <span class="frame-meta-inline">
                    SSIM: ${{ssimStr}}
                    ${{opStr ? '&middot; ' + opStr : ''}}
                </span>
            </div>
            <div class="frame-detail ${{openByDefault ? 'open' : ''}}">
                ${{imageSection}}
                <div class="frame-props">
                    <div><span class="prop-label">Hash:</span> <span class="prop-value">${{frame.perceptual_hash}}</span></div>
                    <div><span class="prop-label">SSIM:</span> <span class="prop-value">${{ssimStr}}</span></div>
                    <div><span class="prop-label">Change %:</span> <span class="prop-value">${{frame.changed_pct !== null ? frame.changed_pct.toFixed(1) + '%' : 'N/A'}}</span></div>
                    <div><span class="prop-label">Mask Regions:</span> <span class="prop-value">${{frame.adaptive_mask ? frame.adaptive_mask.length : 0}}</span></div>
                </div>
                ${{actionStr ? `<div class="action-repr"><strong>Action:</strong> ${{actionStr}}</div>` : ''}}
            </div>
        </div>
    `;
}}

renderSidebar();
if (TASKS.length > 0) selectTask(0);
</script>

</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Peeklet HTML results report")
    parser.add_argument("results_dir", nargs="?", default="results", help="Results directory")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "report.html"

    print(f"Loading results from {results_dir}...")
    summary = load_results(results_dir)
    print(f"Generating report for {len(summary['tasks'])} tasks...")
    html = generate_html(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script shows help**

```bash
python scripts/generate_report.py --help
```

Expected: help text with `results_dir` and `-o` options.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_report.py
git commit -m "feat: add self-contained HTML results explorer generator"
```

---

### Task 7: End-to-end integration test

**Files:**
- None new — uses existing scripts

This task validates the full workflow works end-to-end. It should be done manually after all code is committed.

- [ ] **Step 1: Install datasets dependency**

```bash
pip install 'peeklet[datasets]'
```

Or if using editable install:

```bash
pip install -e '.[datasets]'
```

- [ ] **Step 2: Download test data**

```bash
python scripts/setup_test_data.py --num-tasks 5
```

Expected: 5 tasks downloaded with screenshots to `tests/datasets/data/web_tasks/`.

- [ ] **Step 3: Run existing unit tests (sanity check)**

```bash
pytest tests/unit/ tests/integration/ -v
```

Expected: all existing tests pass.

- [ ] **Step 4: Run dataset tests**

```bash
pytest tests/datasets/ -v -m datasets
```

Expected: 7 tests pass (not skipped, since data is now present).

- [ ] **Step 5: Run the results pipeline**

```bash
python scripts/run_dataset_tests.py --output results
```

Expected: per-task output in `results/`, summary printed to terminal.

- [ ] **Step 6: Generate the HTML report**

```bash
python scripts/generate_report.py results -o results/report.html
```

Expected: `results/report.html` created.

- [ ] **Step 7: Open and verify the report**

```bash
open results/report.html
```

Verify:
- Top bar shows correct summary stats
- Sidebar lists all tasks with skip rates
- Clicking a task shows the frame timeline
- Keyframes are expanded with side-by-side images
- Skipped frames are collapsed, expandable on click
- SSIM scores and hash values are displayed
- Action labels (CLICK/TYPE/SELECT) shown per frame
- Dark/light theme toggle works

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "chore: verify end-to-end dataset testing and HTML explorer workflow"
```

Note: this commit should only contain any small fixes found during integration. If nothing changed, skip this step.
