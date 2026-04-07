"""Configuration loading and validation for Peeklet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class PipelineConfig(BaseModel):
    """Pipeline execution settings."""

    mode: Literal["batch", "stream"] = "batch"
    concurrency: int = Field(default=4, ge=1)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("batch", "stream"):
            raise ValueError(f"mode must be 'batch' or 'stream', got '{v}'")
        return v


class MaskingConfig(BaseModel):
    """Adaptive masking settings."""

    block_size: int = Field(default=32, gt=0)
    window_size: int = Field(default=15, gt=0)
    noise_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class HasherConfig(BaseModel):
    """Perceptual hashing settings."""

    algorithm: Literal["phash"] = "phash"
    hash_size: int = Field(default=8, gt=0)
    tile_aspect_ratio: float = Field(default=1.5, gt=0.0)


class ComparatorConfig(BaseModel):
    """SSIM comparison settings."""

    ssim_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_changed_pct: float = Field(default=2.0, ge=0.0)
    min_changed_blocks: int = Field(default=3, ge=0)


class ExporterConfig(BaseModel):
    """Export settings."""

    output_dir: str = "./output"
    keyframe_format: Literal["png", "jpg"] = "png"
    parquet_compression: Literal["snappy", "gzip", "zstd", "none"] = "snappy"


class InputConfig(BaseModel):
    """Input settings."""

    supported_formats: list[str] = Field(
        default_factory=lambda: ["png", "jpg", "jpeg", "bmp", "tiff", "webp", "pdf"]
    )
    sort_by: Literal["filename", "timestamp"] = "filename"


class VideoConfig(BaseModel):
    """Video input settings."""

    sample_fps: float = Field(default=1.0, gt=0.0)
    formats: list[str] = Field(default_factory=lambda: ["mp4", "mov", "webm"])
    audio_detection: bool = True
    transcript_path: str | None = None


class PeekletConfig(BaseModel):
    """Root configuration for Peeklet."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    hasher: HasherConfig = Field(default_factory=HasherConfig)
    comparator: ComparatorConfig = Field(default_factory=ComparatorConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)


def load_config(path: Path | None) -> PeekletConfig:
    """Load config from a JSON or YAML file, or return defaults if path is None."""
    if path is None:
        return PeekletConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text()
    data = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)

    return PeekletConfig.model_validate(data or {})
