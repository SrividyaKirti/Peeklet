"""Tests for Pipeline state management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from peeklet.config import PeekletConfig
from peeklet.pipeline import Pipeline

if TYPE_CHECKING:
    from pathlib import Path


def test_update_reference_state_updates_rolling_state(tmp_path: Path) -> None:
    """After update_reference_state, subsequent comparisons use the new anchor."""
    config = PeekletConfig.model_validate(
        {
            "exporter": {"output_dir": str(tmp_path / "out")},
            "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
            "comparator": {"ssim_threshold": 0.85},
        }
    )
    pipeline = Pipeline(config)

    # First frame: black
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline.process_frame(frame_a, frame_id="a")

    # Now manually set a different reference frame
    frame_b = np.full((100, 100, 3), 200, dtype=np.uint8)
    pipeline.update_reference_state(frame_b, frame_id="b_forced", asset_path="dummy.png")

    # Process frame_b again — should now be SKIPPED because it matches the new anchor
    result = pipeline.process_frame(frame_b, frame_id="b_again")
    assert result.is_keyframe is False
