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
