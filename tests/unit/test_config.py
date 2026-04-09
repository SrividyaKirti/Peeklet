"""Tests for configuration loading and validation."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from peeklet.config import (
    QUALITY_PRESETS,
    SENSITIVITY_PRESETS,
    ComparatorConfig,
    HasherConfig,
    MaskingConfig,
    PeekletConfig,
    PipelineConfig,
    VideoConfig,
    apply_quality_preset,
    apply_sensitivity_preset,
    load_config,
)


class TestDefaults:
    def test_default_config_is_valid(self) -> None:
        config = PeekletConfig()
        assert config.pipeline.mode == "batch"
        assert config.masking.block_size == 32
        assert config.masking.window_size == 15
        assert config.masking.noise_threshold == 0.8
        assert config.hasher.algorithm == "phash"
        assert config.comparator.ssim_threshold == 0.85
        assert config.exporter.keyframe_format == "jpg"
        assert config.exporter.parquet_compression == "snappy"

    def test_pipeline_defaults(self) -> None:
        config = PipelineConfig()
        assert config.mode == "batch"
        assert config.concurrency == 4

    def test_default_tile_aspect_ratio(self) -> None:
        config = PeekletConfig()
        assert config.hasher.tile_aspect_ratio == 1.5

    def test_default_min_changed_blocks(self) -> None:
        config = PeekletConfig()
        assert config.comparator.min_changed_blocks == 3


class TestValidation:
    def test_reject_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            PipelineConfig(mode="invalid")

    def test_reject_negative_block_size(self) -> None:
        with pytest.raises(ValueError):
            MaskingConfig(block_size=-1)

    def test_reject_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            ComparatorConfig(ssim_threshold=1.5)

    def test_reject_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError):
            ComparatorConfig(ssim_threshold=-0.1)

    def test_reject_noise_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            MaskingConfig(noise_threshold=1.5)

    def test_reject_negative_tile_aspect_ratio(self) -> None:
        with pytest.raises(ValueError):
            HasherConfig(tile_aspect_ratio=-1.0)

    def test_reject_negative_min_changed_blocks(self) -> None:
        with pytest.raises(ValueError):
            ComparatorConfig(min_changed_blocks=-1)

    def test_processing_max_dim_rejects_tiny_values(self) -> None:
        """processing_max_dim below 64 is silently broken — must error."""
        with pytest.raises(ValidationError):
            VideoConfig(processing_max_dim=10)

    def test_processing_max_dim_accepts_64(self) -> None:
        """64 is the minimum; anything below it errors."""
        config = VideoConfig(processing_max_dim=64)
        assert config.processing_max_dim == 64

    def test_processing_max_dim_accepts_none(self) -> None:
        """None still means 'no downscaling' — must remain valid."""
        config = VideoConfig(processing_max_dim=None)
        assert config.processing_max_dim is None


class TestLoadConfig:
    def test_load_from_json_file(self, tmp_path: Path) -> None:
        config_data = {
            "comparator": {"ssim_threshold": 0.9},
            "masking": {"block_size": 64},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = load_config(config_file)
        assert config.comparator.ssim_threshold == 0.9
        assert config.masking.block_size == 64
        assert config.masking.window_size == 15

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        config_data = {"comparator": {"ssim_threshold": 0.7}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.comparator.ssim_threshold == 0.7

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.json"))

    def test_load_none_returns_defaults(self) -> None:
        config = load_config(None)
        assert config == PeekletConfig()


class TestVideoConfig:
    def test_defaults(self) -> None:
        config = PeekletConfig()
        assert config.video.sample_fps == 1.0
        assert config.video.formats == ["mp4", "mov", "webm"]
        assert config.video.audio_detection is True
        assert config.video.transcript_path is None

    def test_custom_video_config_from_dict(self) -> None:
        config = PeekletConfig.model_validate(
            {
                "video": {
                    "sample_fps": 2.0,
                    "formats": ["mp4"],
                    "audio_detection": False,
                    "transcript_path": "/path/to/transcript.srt",
                }
            }
        )
        assert config.video.sample_fps == 2.0
        assert config.video.formats == ["mp4"]
        assert config.video.audio_detection is False
        assert config.video.transcript_path == "/path/to/transcript.srt"

    def test_sample_fps_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PeekletConfig.model_validate({"video": {"sample_fps": 0}})

    def test_load_video_config_from_yaml(self, tmp_path: Path) -> None:
        config_data = {
            "video": {
                "sample_fps": 5.0,
                "formats": ["mp4", "webm"],
                "audio_detection": False,
                "transcript_path": "/tmp/captions.srt",
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.video.sample_fps == 5.0
        assert config.video.formats == ["mp4", "webm"]
        assert config.video.audio_detection is False
        assert config.video.transcript_path == "/tmp/captions.srt"


def test_demo_filter_config_defaults():
    cfg = PeekletConfig()
    assert cfg.demo_filter.enabled is False
    assert cfg.demo_filter.llm_provider == "anthropic"
    assert cfg.demo_filter.llm_model == "claude-haiku-4-5"
    assert cfg.demo_filter.frame_search_resolution == 360
    assert cfg.demo_filter.ssim_stability_threshold == 0.92
    assert cfg.demo_filter.forward_search_step_sec == 0.5
    assert cfg.demo_filter.forward_search_window_max_sec == 5.0
    assert cfg.demo_filter.gallery_min_words == 5


def test_demo_filter_config_rejects_unknown_provider():
    import pytest
    from pydantic import ValidationError

    from peeklet.config import DemoFilterConfig

    with pytest.raises(ValidationError):
        DemoFilterConfig(llm_provider="cohere")  # type: ignore[arg-type]


class TestQualityPresets:
    def test_quality_presets_exist(self) -> None:
        assert set(QUALITY_PRESETS.keys()) == {"fast", "balanced", "precise"}

    def test_apply_quality_preset_balanced_matches_current_defaults(self) -> None:
        """The 'balanced' preset must match today's default values exactly,
        so users who don't pass --quality see no behavior change."""
        config = PeekletConfig()
        baseline_max_dim = config.video.processing_max_dim
        baseline_sample_fps = config.video.sample_fps
        baseline_search_res = config.demo_filter.frame_search_resolution

        apply_quality_preset(config, "balanced")

        assert config.video.processing_max_dim == baseline_max_dim
        assert config.video.sample_fps == baseline_sample_fps
        assert config.demo_filter.frame_search_resolution == baseline_search_res

    def test_apply_quality_preset_fast(self) -> None:
        config = PeekletConfig()
        apply_quality_preset(config, "fast")

        assert config.video.processing_max_dim == 480
        assert config.video.sample_fps == 0.5
        assert config.demo_filter.frame_search_resolution == 240

    def test_apply_quality_preset_precise(self) -> None:
        config = PeekletConfig()
        apply_quality_preset(config, "precise")

        assert config.video.processing_max_dim == 1080
        assert config.video.sample_fps == 2.0
        assert config.demo_filter.frame_search_resolution == 540

    def test_apply_quality_preset_invalid_raises(self) -> None:
        config = PeekletConfig()
        with pytest.raises(ValueError, match="unknown quality preset"):
            apply_quality_preset(config, "ludicrous")


class TestSensitivityPresets:
    def test_sensitivity_presets_exist(self) -> None:
        assert set(SENSITIVITY_PRESETS.keys()) == {"low", "medium", "high"}

    def test_apply_sensitivity_medium_matches_current_defaults(self) -> None:
        """'medium' must match today's defaults so unflagged users see no change."""
        config = PeekletConfig()
        baseline_ssim = config.comparator.ssim_threshold
        baseline_pct = config.comparator.min_changed_pct
        baseline_blocks = config.comparator.min_changed_blocks

        apply_sensitivity_preset(config, "medium")

        assert config.comparator.ssim_threshold == baseline_ssim
        assert config.comparator.min_changed_pct == baseline_pct
        assert config.comparator.min_changed_blocks == baseline_blocks

    def test_apply_sensitivity_low(self) -> None:
        """'low' = fewer keyframes (stricter thresholds)."""
        config = PeekletConfig()
        apply_sensitivity_preset(config, "low")

        assert config.comparator.ssim_threshold == 0.92
        assert config.comparator.min_changed_pct == 5.0
        assert config.comparator.min_changed_blocks == 5

    def test_apply_sensitivity_high(self) -> None:
        """'high' = more keyframes (looser thresholds)."""
        config = PeekletConfig()
        apply_sensitivity_preset(config, "high")

        assert config.comparator.ssim_threshold == 0.75
        assert config.comparator.min_changed_pct == 1.0
        assert config.comparator.min_changed_blocks == 2

    def test_apply_sensitivity_invalid_raises(self) -> None:
        config = PeekletConfig()
        with pytest.raises(ValueError, match="unknown sensitivity preset"):
            apply_sensitivity_preset(config, "extreme")
