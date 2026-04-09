"""Tests for configuration loading and validation."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from peeklet.config import (
    ComparatorConfig,
    HasherConfig,
    MaskingConfig,
    PeekletConfig,
    PipelineConfig,
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
    from peeklet.config import PeekletConfig

    cfg = PeekletConfig()
    assert cfg.demo_filter.enabled is False
    assert cfg.demo_filter.ocr_sample_interval_sec == 30.0
    assert cfg.demo_filter.ocr_downscale_dim == 360
    assert cfg.demo_filter.ocr_min_words == 5
    assert cfg.demo_filter.major_change_ssim == 0.70
    assert cfg.demo_filter.major_change_blocks == 15
    assert cfg.demo_filter.visual_change_weight == 0.7
    assert cfg.demo_filter.text_density_weight == 0.3
