"""Tests for PeekletConfig and load_config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from peeklet.config import PeekletConfig, ScoreWeights, load_config

if TYPE_CHECKING:
    from pathlib import Path


def test_defaults_match_spec() -> None:
    cfg = PeekletConfig()
    assert cfg.sample_fps == 1.0
    assert cfg.checkpoint_window_seconds == 5.0
    assert cfg.min_pause_seconds == 1.5
    assert cfg.cue_threshold == 1.0
    assert cfg.anchor_bonus == 1.0
    assert cfg.max_line_seconds == 8.0
    assert cfg.lead_seconds == 1.5
    assert cfg.max_images == 20
    assert cfg.shortlist_factor == 2.0
    assert cfg.llm_concurrency == 8
    assert cfg.llm_max_lines == 8
    assert cfg.max_image_edge == 1568
    assert (cfg.min_text_lines, cfg.min_grid_cells, cfg.min_edge_ratio) == (10, 12, 0.02)
    assert cfg.ocr_max_dim == 1920
    assert (cfg.phash_threshold, cfg.ocr_field_min_chars) == (6, 2)
    assert (cfg.llm_provider, cfg.llm_model) == ("anthropic", "claude-haiku-4-5")
    assert cfg.use_llm is True


def test_score_weight_defaults() -> None:
    w = ScoreWeights()
    assert (w.references, w.text_overlap, w.onsets) == (0.25, 0.20, 0.20)
    assert (w.verbal_cues, w.visual_change, w.time_on_screen) == (0.15, 0.10, 0.10)


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        PeekletConfig.model_validate({"not_a_field": 1})


def test_negative_max_images_rejected() -> None:
    with pytest.raises(ValidationError):
        PeekletConfig(max_images=-1)


def test_load_none_returns_defaults() -> None:
    assert load_config(None) == PeekletConfig()


def test_load_yaml(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("max_images: 5\nscore_weights:\n  references: 0.5\n")
    cfg = load_config(p)
    assert cfg.max_images == 5
    assert cfg.score_weights.references == 0.5


def test_load_json(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text('{"llm_model": "claude-sonnet-5-5"}')
    assert load_config(p).llm_model == "claude-sonnet-5-5"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
