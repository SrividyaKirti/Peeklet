"""Configuration for Peeklet's single annotation path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ScoreWeights(BaseModel):
    """Weights for the heuristic screen score (each signal is scaled to [0, 1])."""

    model_config = {"extra": "forbid"}

    references: float = Field(default=0.25, ge=0.0)
    text_overlap: float = Field(default=0.20, ge=0.0)
    onsets: float = Field(default=0.20, ge=0.0)
    verbal_cues: float = Field(default=0.15, ge=0.0)
    visual_change: float = Field(default=0.10, ge=0.0)
    time_on_screen: float = Field(default=0.10, ge=0.0)


class PeekletConfig(BaseModel):
    """All tunables for annotate(). Unknown keys are rejected."""

    model_config = {"extra": "forbid"}

    # Sampling and change detection
    sample_fps: float = Field(default=1.0, gt=0.0)
    change_max_dim: int = Field(default=720, ge=64)
    mask_block_size: int = Field(default=32, gt=0)
    mask_window_size: int = Field(default=15, gt=0)
    mask_noise_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    ssim_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_changed_blocks: int = Field(default=3, ge=0)

    # Checkpoints
    checkpoint_window_seconds: float = Field(default=5.0, ge=0.0)
    min_pause_seconds: float = Field(default=1.5, ge=0.0)
    silence_threshold_dbfs: float = -40.0
    cue_threshold: float = Field(default=1.0, gt=0.0)
    anchor_bonus: float = Field(default=1.0, ge=0.0)

    # Screens (OCR, low-info rejector, fingerprint)
    ocr_max_dim: int = Field(default=1920, gt=0)
    min_text_lines: int = Field(default=10, ge=0)
    min_grid_cells: int = Field(default=12, ge=0)
    min_edge_ratio: float = Field(default=0.020, ge=0.0, le=1.0)
    phash_threshold: int = Field(default=6, ge=0, le=64)
    ocr_field_min_chars: int = Field(default=2, ge=0)

    # Transcript and alignment
    max_line_seconds: float = Field(default=8.0, gt=0.0)
    lead_seconds: float = Field(default=1.5, ge=0.0)

    # Scoring
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)

    # LLM judge + describe
    use_llm: bool = True
    llm_provider: Literal["anthropic", "openai", "openrouter"] = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    shortlist_factor: float = Field(default=2.0, ge=1.0)
    llm_concurrency: int = Field(default=8, ge=1)
    llm_max_lines: int = Field(default=8, ge=1)
    llm_cache_dir: str = "~/.cache/peeklet/judgments"

    # Output
    max_images: int = Field(default=20, ge=0)
    max_image_edge: int = Field(default=1568, ge=64)
    jpeg_quality: int = Field(default=90, ge=1, le=100)


def load_config(path: Path | None) -> PeekletConfig:
    """Load config from YAML/JSON, or return defaults when path is None."""
    if path is None:
        return PeekletConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
    return PeekletConfig.model_validate(data or {})
