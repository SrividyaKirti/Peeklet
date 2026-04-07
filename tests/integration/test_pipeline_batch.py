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


class TestTiledDetection:
    def test_small_change_on_tall_image_detected(self, tmp_output: Path) -> None:
        """A small localized change on a tall full-page screenshot must be detected."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85, "min_changed_blocks": 2},
                "hasher": {"tile_aspect_ratio": 1.5},
            }
        )
        pipeline = Pipeline(config)

        # Simulate a full-page screenshot (200 wide, 600 tall -> aspect 3.0 > 1.5)
        frame_a = np.full((600, 200, 3), 220, dtype=np.uint8)
        frame_b = frame_a.copy()
        # Small change in the middle: simulate typed text (black on light background)
        frame_b[280:310, 80:140] = [10, 10, 10]

        result_a = pipeline.process_frame(frame_a, frame_id="frame_000")
        result_b = pipeline.process_frame(frame_b, frame_id="frame_001")

        assert result_a.is_keyframe is True
        assert result_b.is_keyframe is True
        reason = (result_b.visual_reason or "").lower()
        assert "block" in reason or "localized" in reason

    def test_small_change_on_tall_image_skipped_without_tiling(self, tmp_output: Path) -> None:
        """Without tiled pHash, a small change on a tall image gets skipped at the hash gate."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85, "min_changed_blocks": 2},
                # Very high aspect ratio threshold -> tiling never activates
                "hasher": {"tile_aspect_ratio": 100.0},
            }
        )
        pipeline = Pipeline(config)

        frame_a = np.full((600, 200, 3), 220, dtype=np.uint8)
        frame_b = frame_a.copy()
        frame_b[280:310, 80:140] = [10, 10, 10]

        pipeline.process_frame(frame_a, frame_id="frame_000")
        result_b = pipeline.process_frame(frame_b, frame_id="frame_001")

        # Without tiling, pHash matches -> skipped
        assert result_b.is_keyframe is False

    def test_identical_tall_frames_still_skipped(self, tmp_output: Path) -> None:
        """Tiling should not cause false positives on identical tall frames."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85, "min_changed_blocks": 2},
                "hasher": {"tile_aspect_ratio": 1.5},
            }
        )
        pipeline = Pipeline(config)

        frame = np.full((600, 200, 3), 220, dtype=np.uint8)

        pipeline.process_frame(frame, frame_id="frame_000")
        result = pipeline.process_frame(frame, frame_id="frame_001")

        assert result.is_keyframe is False

    def test_block_count_triggers_keyframe_even_with_high_ssim(self, tmp_output: Path) -> None:
        """Block count threshold should trigger keyframe even when SSIM is above threshold."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                # Very high SSIM threshold -- normally would skip
                "comparator": {"ssim_threshold": 0.99, "min_changed_blocks": 1},
                "hasher": {"tile_aspect_ratio": 1.5},
            }
        )
        pipeline = Pipeline(config)

        frame_a = np.full((600, 200, 3), 220, dtype=np.uint8)
        frame_b = frame_a.copy()
        # Change enough blocks to exceed min_changed_blocks=1
        frame_b[280:320, 80:150] = [10, 10, 10]

        pipeline.process_frame(frame_a, frame_id="frame_000")
        result_b = pipeline.process_frame(frame_b, frame_id="frame_001")

        assert result_b.is_keyframe is True

    def test_non_tall_image_uses_single_hash(self, tmp_output: Path) -> None:
        """Normal aspect ratio images should use single pHash (existing behavior)."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 50, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85},
                "hasher": {"tile_aspect_ratio": 1.5},
            }
        )
        pipeline = Pipeline(config)

        # Square image -- not tall
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        pipeline.process_frame(frame, frame_id="frame_000")
        result = pipeline.process_frame(frame, frame_id="frame_001")

        # Identical frames still skipped -- same behavior as before
        assert result.is_keyframe is False
        assert result.ssim_score is None  # skipped at hash gate
