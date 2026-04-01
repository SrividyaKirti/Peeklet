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

        assert results[0].is_keyframe is True
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

    def test_pipeline_writes_manifest(self, basic_config: PeekletConfig, tmp_output: Path) -> None:
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

    def test_keyframe_images_saved(self, basic_config: PeekletConfig, tmp_output: Path) -> None:
        pipeline = Pipeline(basic_config)

        frame_a = np.full((100, 100, 3), 50, dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 200, dtype=np.uint8)

        pipeline.process_frame(frame_a, frame_id="frame_000")
        pipeline.process_frame(frame_b, frame_id="frame_001")
        pipeline.finalize()

        keyframes = list(tmp_output.glob("*.png"))
        assert len(keyframes) == 2

    def test_cascade_skips_comparator_on_hash_match(
        self, basic_config: PeekletConfig, tmp_output: Path
    ) -> None:
        pipeline = Pipeline(basic_config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        pipeline.process_frame(frame, frame_id="frame_000")
        result = pipeline.process_frame(frame, frame_id="frame_001")

        assert result.is_keyframe is False
        assert result.ssim_score is None

    def test_frame_metadata_in_result(self, basic_config: PeekletConfig, tmp_output: Path) -> None:
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
