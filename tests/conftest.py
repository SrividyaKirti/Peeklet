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
