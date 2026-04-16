"""Configuration loading and validation for Peeklet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


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

    enabled: bool = True
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
    # JPEG default — ~5x faster to encode than PNG and ~10x smaller on disk,
    # which matters at scale and is fine for downstream LLM consumption.
    keyframe_format: Literal["png", "jpg"] = "jpg"
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
    # Performance: downscale frames before pipeline processing. None = no downscale.
    # Saved keyframe images are at the downscaled resolution.
    # Minimum of 64 prevents misconfiguration that silently degrades quality.
    # Anything smaller produces useless frames for the cascade.
    processing_max_dim: int | None = Field(default=720, ge=64)


class DemoFilterConfig(BaseModel):
    """Demo-mode frame filtering settings.

    Activated via the CLI ``--demo-mode`` flag. The LLM picks screenshot-worthy
    moments from the transcript and Peeklet picks the exact frame for each
    moment using forward-search bidirectional SSIM stability plus an OCR
    gallery check. See the design spec for full details.
    """

    enabled: bool = False
    llm_provider: Literal["anthropic", "openai", "openrouter"] = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    # Frame search and selection
    frame_search_resolution: int = Field(default=360, gt=0)
    ssim_stability_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    forward_search_step_sec: float = Field(default=0.5, gt=0.0)
    forward_search_window_max_sec: float = Field(default=10.0, gt=0.0)
    # Minimum longest-edge resolution for OCR. Frames are downscaled only if
    # they exceed this value, so OCR runs at near-source resolution. Reusing
    # ``frame_search_resolution`` (which can be 240–540 under quality presets)
    # destroys text before Tesseract sees it on 720p+ screen-share recordings,
    # silently flagging every demo frame as gallery view. Default 1920 leaves
    # 1080p and smaller frames untouched and downscales 4K to 1920 wide.
    gallery_ocr_min_dim: int = Field(default=1920, gt=0)
    # Near-duplicate suppression: drop a moment whose candidate frame SSIMs
    # above this threshold against the most recently saved keyframe. 1.0
    # disables dedup; 0.0 drops every moment after the first. Only compared
    # against the immediately previous keyframe, so a later moment that
    # legitimately revisits an earlier page is still allowed through.
    dedup_ssim_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    # Skip LLM moments whose timestamp falls in the final fraction of the
    # video. Meeting recordings often continue past the end of the demo with
    # social chatter, and the LLM sometimes grasps at those transcript
    # tokens and picks a wrap-up frame with no meaningful content. 0.0
    # disables the tail skip entirely.
    tail_skip_ratio: float = Field(default=0.02, ge=0.0, le=0.5)
    # --- Layout-based low-info rejector (PR A) ---
    # Replaces the raw word-count gate. A frame is rejected only if ALL three
    # signals below fall under their thresholds (triple-AND), so a legit
    # minimalist-but-valid UI passes on at least one axis while gallery views
    # and blank scratch pages get caught.
    min_text_lines: int = Field(default=10, ge=0)
    min_grid_cells: int = Field(default=12, ge=0)
    min_edge_ratio: float = Field(default=0.015, ge=0.0, le=1.0)
    # Post-selection dHash dedup. A newly picked frame is dropped if its
    # 64-bit dHash Hamming distance to the most recently saved keyframe is <=
    # this value. 0 means exact-match only; 64 disables dedup. Runs after
    # dedup_ssim_threshold as a backstop for near-duplicates SSIM missed.
    dhash_hamming_threshold: int = Field(default=5, ge=0, le=64)

    @model_validator(mode="before")
    @classmethod
    def _migrate_phash_alias(cls, data):
        if (
            isinstance(data, dict)
            and "phash_hamming_threshold" in data
            and "dhash_hamming_threshold" not in data
        ):
            data["dhash_hamming_threshold"] = data.pop("phash_hamming_threshold")
        return data

    # Primary dedup signal — Jaccard of normalized OCR token sets.
    # Screen-share recordings have stable text but pixel-level noise, so
    # token overlap is a more faithful "same screen" signal than SSIM or
    # dHash on raw pixels. Only applied when both frames have enough
    # tokens (see min_ocr_tokens_for_jaccard); otherwise dHash is the
    # fallback. 1.0 disables Jaccard dedup.
    dedup_jaccard_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    # Minimum tokens in both frames for Jaccard to be trustworthy. Below
    # this, fall back to dHash. Prevents spurious "identical" matches on
    # photo/video frames where OCR returns nothing.
    min_ocr_tokens_for_jaccard: int = Field(default=5, ge=0)
    # --- Coverage gap closer (PR C) ---
    # Maximum allowed gap between consecutive kept keyframes, in seconds.
    # When a longer gap is detected after frame selection, a synthetic
    # ``gap_fill`` moment is injected at the midpoint and re-run through
    # the normal selection + low-info gates. 0.0 disables gap filling.
    max_seconds_between_keyframes: float = Field(default=120.0, ge=0.0)
    # How far before each moment's timestamp the search window starts.
    # Product-review speakers typically reference a UI element *before*
    # naming the action, so biasing the window backward captures the
    # frame the speaker was actually pointing at.
    search_window_lookback_sec: float = Field(default=3.0, ge=0.0)


class PeekletConfig(BaseModel):
    """Root configuration for Peeklet."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    hasher: HasherConfig = Field(default_factory=HasherConfig)
    comparator: ComparatorConfig = Field(default_factory=ComparatorConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    demo_filter: DemoFilterConfig = Field(default_factory=DemoFilterConfig)


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


# --- CLI presets ---
# Presets bundle multiple raw config knobs into one user-facing concept
# so the CLI surface stays small while still letting users tune the
# things they actually care about. See the repo restructure spec
# (D1) for the full rationale. Power users can still override
# individual fields via --config <yaml>.

QUALITY_PRESETS: dict[str, dict[str, float | int]] = {
    "fast": {
        "processing_max_dim": 480,
        "sample_fps": 0.5,
        "frame_search_resolution": 240,
    },
    "balanced": {
        "processing_max_dim": 720,
        "sample_fps": 1.0,
        "frame_search_resolution": 360,
    },
    "precise": {
        "processing_max_dim": 1080,
        "sample_fps": 2.0,
        "frame_search_resolution": 540,
    },
}


def apply_quality_preset(config: PeekletConfig, preset: str) -> None:
    """Apply a quality preset in place. Overrides any existing values."""
    if preset not in QUALITY_PRESETS:
        raise ValueError(
            f"unknown quality preset '{preset}'. Valid: {sorted(QUALITY_PRESETS.keys())}"
        )
    values = QUALITY_PRESETS[preset]
    config.video.processing_max_dim = int(values["processing_max_dim"])
    config.video.sample_fps = float(values["sample_fps"])
    config.demo_filter.frame_search_resolution = int(values["frame_search_resolution"])


SENSITIVITY_PRESETS: dict[str, dict[str, float | int]] = {
    "low": {
        "ssim_threshold": 0.92,
        "min_changed_pct": 5.0,
        "min_changed_blocks": 5,
    },
    "medium": {
        "ssim_threshold": 0.85,
        "min_changed_pct": 2.0,
        "min_changed_blocks": 3,
    },
    "high": {
        "ssim_threshold": 0.75,
        "min_changed_pct": 1.0,
        "min_changed_blocks": 2,
    },
}


def apply_sensitivity_preset(config: PeekletConfig, preset: str) -> None:
    """Apply a sensitivity preset in place. Overrides any existing values."""
    if preset not in SENSITIVITY_PRESETS:
        raise ValueError(
            f"unknown sensitivity preset '{preset}'. Valid: {sorted(SENSITIVITY_PRESETS.keys())}"
        )
    values = SENSITIVITY_PRESETS[preset]
    config.comparator.ssim_threshold = float(values["ssim_threshold"])
    config.comparator.min_changed_pct = float(values["min_changed_pct"])
    config.comparator.min_changed_blocks = int(values["min_changed_blocks"])
