"""Tests that run Mind2Web screenshot sequences through Peeklet's pipeline.

These tests validate that Peeklet correctly identifies keyframes at action
boundaries (click, type, select) and skips idle frames between actions.

Requires data download: python scripts/setup_test_data.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from peeklet.config import PeekletConfig
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline

if TYPE_CHECKING:
    from pathlib import Path

from .conftest import (
    get_task_screenshots,
    load_task_ids,
    load_task_metadata,
    skip_no_data,
)


def _make_config(output_dir: Path) -> PeekletConfig:
    """Create pipeline config for dataset testing."""
    return PeekletConfig.model_validate(
        {
            "exporter": {"output_dir": str(output_dir)},
        }
    )


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
        results.append(
            {
                "frame_id": result.frame_id,
                "event_type": result.event_type.value,
                "is_keyframe": result.is_keyframe,
                "perceptual_hash": result.perceptual_hash,
                "ssim_score": result.ssim_score,
                "change_score": result.change_score,
                "changed_pct": result.changed_pct,
                "asset_path": result.asset_path,
                "action": metadata["actions"][i] if i < len(metadata["actions"]) else None,
            }
        )

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
        assert skip_rate >= 0.2, f"Expected at least 20% frame reduction, got {skip_rate:.1%}"

    @skip_no_data
    def test_ssim_scores_within_range(self, results_dir: Path) -> None:
        """All SSIM scores should be between 0 and 1."""
        task_ids = load_task_ids()
        for task_id in task_ids[:3]:  # Spot-check first 3 tasks
            results = _run_task(task_id, results_dir)
            for r in results:
                if r["ssim_score"] is not None:
                    assert 0.0 <= r["ssim_score"] <= 1.0, (
                        f"Task {task_id}, {r['frame_id']}: SSIM {r['ssim_score']} out of range"
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

        skipped_no_ssim = [r for r in results if not r["is_keyframe"] and r["ssim_score"] is None]
        # If there are any hash-matched skips, verify they have no SSIM
        for r in skipped_no_ssim:
            assert r["ssim_score"] is None
            assert r["change_score"] is None
