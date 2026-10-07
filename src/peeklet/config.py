"""Configuration loading and validation for Peeklet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


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


class VideoConfig(BaseModel):
    """Video input settings."""

    sample_fps: float = Field(default=1.0, gt=0.0)
    formats: list[str] = Field(default_factory=lambda: ["mp4", "mov", "webm"])
    audio_detection: bool = True
    transcript_path: str | None = None
    # Performance: downscale frames before processing. None = no downscale.
    # Saved keyframe images are at the downscaled resolution.
    # Minimum of 64 prevents misconfiguration that silently degrades quality.
    # Anything smaller produces useless frames for the cascade.
    processing_max_dim: int | None = Field(default=720, ge=64)


class DemoFilterConfig(BaseModel):
    """Demo-mode frame filtering settings.

    Activated via the CLI ``--demo-mode`` flag. The LLM picks
    screenshot-worthy moments from the transcript; for each moment
    Peeklet captures a frame, runs the universal layout/info-density
    quality gate (with bounded ±N-second fallback), then computes a
    content-addressable fingerprint to deduplicate against screens
    already saved this run.

    See ``docs/superpowers/specs/2026-04-30-content-addressable-screen-dedup-design.md``.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    llm_provider: Literal["anthropic", "openai", "openrouter"] = "anthropic"
    llm_model: str = "claude-haiku-4-5"

    # Frame OCR resolution. Frames are downscaled only if they exceed
    # this; OCR runs at near-source resolution. 1920 leaves 1080p
    # untouched and downscales 4K to 1920 wide.
    gallery_ocr_min_dim: int = Field(default=1920, gt=0)

    # Layout/info-density rejector — triple-AND. A frame is rejected
    # only if all three signals fall under their thresholds.
    min_text_lines: int = Field(default=10, ge=0)
    min_grid_cells: int = Field(default=12, ge=0)
    min_edge_ratio: float = Field(default=0.020, ge=0.0, le=1.0)

    # Tail-skip: drop LLM picks whose timestamp lands in the final
    # fraction of the video. Anchors are guaranteed by Fathom and not
    # subject to tail-skip; only LLM picks (which sometimes grasp at
    # meeting-end chatter) are filtered. 0.0 disables.
    tail_skip_ratio: float = Field(default=0.02, ge=0.0, le=0.5)

    # --- Bounded quality fallback ---
    # When the layout rejector fails the frame at the moment's
    # timestamp, try nearby frames at ±step, ±2*step, ... up to
    # ±half_window seconds. First passing frame wins; original
    # timestamp is preserved in the moment metadata. If all attempts
    # fail, emit caption with image_unavailable=True.
    quality_fallback_max_attempts: int = Field(default=8, ge=0)
    quality_fallback_half_window_seconds: float = Field(default=4.0, gt=0.0)
    quality_fallback_step_seconds: float = Field(default=1.0, gt=0.0)

    # --- Fingerprint dedup ---
    # Header-strip pHash Hamming threshold for "same screen" within a
    # Part-A bucket. Lenient (high) → filter-chip splits collapse;
    # strict (low) → scroll cases split.
    phash_threshold: int = Field(default=6, ge=0, le=64)
    # Minimum chars per Part-A field for the field to count as
    # populated. If all three fields are below this, the dedup safety
    # fallback bypasses collapse so unreadable frames don't all
    # cluster under the same empty key.
    ocr_field_min_chars: int = Field(default=2, ge=0)


class PeekletConfig(BaseModel):
    """Root configuration for Peeklet."""

    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    hasher: HasherConfig = Field(default_factory=HasherConfig)
    comparator: ComparatorConfig = Field(default_factory=ComparatorConfig)
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
