"""Tests for verbal-cue checkpoints."""

from __future__ import annotations

import pytest

from peeklet.cues import cue_score, verbal_cue_checkpoints
from peeklet.types import Line


def test_high_plus_medium_triggers() -> None:
    [cp] = verbal_cue_checkpoints([Line(2.0, 5.0, "Now look at this dashboard")])
    assert (cp.t, cp.kind) == (2.0, "verbal_cue")
    assert cp.label == "Now look at this dashboard"


def test_single_high_signal_triggers() -> None:
    assert len(verbal_cue_checkpoints([Line(0.0, 1.0, "the sidebar")])) == 1


def test_single_medium_does_not_trigger() -> None:
    assert verbal_cue_checkpoints([Line(0.0, 3.0, "I see what you mean")]) == []


def test_no_keywords() -> None:
    assert verbal_cue_checkpoints([Line(0.0, 3.0, "Welcome to the presentation")]) == []


def test_case_insensitive_and_punctuation() -> None:
    assert cue_score("CLICK, THIS... BUTTON!") == pytest.approx(1.0 + 0.5 + 0.3)


def test_repeated_word_counts_once() -> None:
    assert cue_score("button button button") == pytest.approx(1.0)


def test_threshold_is_configurable() -> None:
    lines = [Line(0.0, 1.0, "the sidebar")]
    assert verbal_cue_checkpoints(lines, threshold=1.5) == []


def test_label_truncated_to_80_chars() -> None:
    [cp] = verbal_cue_checkpoints([Line(0.0, 1.0, "dashboard " + "x" * 200)])
    assert len(cp.label) == 80
