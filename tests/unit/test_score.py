"""Tests for heuristic scoring and selection."""

from __future__ import annotations

import math

import pytest

from peeklet.config import ScoreWeights
from peeklet.score import (
    ScreenSignals,
    compute_signals,
    content_words,
    rank_screens,
    score_screens,
    select_screens,
)
from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment


def scr(sid: str, occ: list[tuple[float, float]], ocr: str = "", change: float = 0.5) -> Screen:
    return Screen(
        id=sid,
        image_jpeg=b"",
        frame_t=occ[0][0],
        ocr_text=ocr,
        word_count=1,
        first_change=change,
        occurrences=occ,
    )


def test_content_words_drops_stopwords_and_short_tokens() -> None:
    assert content_words("The Billing settings, on a page!") == {"billing", "settings", "page"}


def test_signals_each_scaled_to_unit_range() -> None:
    s1 = scr("S1", [(0.0, 30.0)], ocr="Billing Settings Invoices", change=0.8)
    s2 = scr("S2", [(30.0, 40.0)], ocr="Welcome")
    lines = [
        Line(1.0, 2.0, "open billing settings"),
        Line(5.0, 6.0, "and invoices"),
        Line(31.0, 32.0, "hello"),
    ]
    cps = [
        Checkpoint(1.0, "speech_onset"),
        Checkpoint(5.0, "verbal_cue"),
        Checkpoint(31.0, "speech_onset"),
        Checkpoint(32.0, "action_item", "x"),
    ]
    sig = compute_signals([s1, s2], lines, ["S1", "S1", "S2"], cps)
    assert sig["S1"].references == pytest.approx(1.0)
    assert sig["S2"].references == pytest.approx(math.log1p(1) / math.log1p(2))
    assert sig["S1"].text_overlap > sig["S2"].text_overlap == 0.0
    assert sig["S1"].onsets == sig["S2"].onsets == pytest.approx(1.0)
    assert sig["S1"].verbal_cues == pytest.approx(1.0) and sig["S2"].verbal_cues == 0.0
    assert sig["S1"].visual_change == pytest.approx(0.8)
    assert sig["S1"].time_on_screen == pytest.approx(1.0)
    assert sig["S2"].anchored and not sig["S1"].anchored


def test_score_is_weighted_sum_plus_anchor_bonus() -> None:
    sig = {
        "S1": ScreenSignals(1, 1, 1, 1, 1, 1, anchored=True),
        "S2": ScreenSignals(1, 0, 0, 0, 0, 0, anchored=False),
    }
    scores = score_screens(sig, ScoreWeights(), anchor_bonus=1.0)
    assert scores["S1"] == pytest.approx(2.0)
    assert scores["S2"] == pytest.approx(0.25)


def test_rank_ties_prefer_first_appearance() -> None:
    a, b = scr("S1", [(5.0, 6.0)]), scr("S2", [(1.0, 2.0)])
    assert [s.id for s in rank_screens([a, b], {"S1": 0.5, "S2": 0.5})] == ["S2", "S1"]


def _sig(anchored: bool = False) -> ScreenSignals:
    return ScreenSignals(0, 0, 0, 0, 0, 0, anchored=anchored)


def test_select_without_llm_takes_top_n_in_time_order() -> None:
    screens = [scr("S1", [(0.0, 1.0)]), scr("S2", [(1.0, 2.0)]), scr("S3", [(2.0, 3.0)])]
    scores = {"S1": 0.1, "S2": 0.9, "S3": 0.5}
    sig = {s.id: _sig() for s in screens}
    kept = select_screens(screens, scores, sig, None, max_images=2)
    assert [s.id for s in kept] == ["S2", "S3"]


def test_select_with_llm_uses_verdicts_and_anchor_override() -> None:
    screens = [
        scr("S1", [(0.0, 1.0)]),
        scr("S2", [(1.0, 2.0)]),
        scr("S3", [(2.0, 3.0)]),
        scr("S4", [(3.0, 4.0)]),
    ]
    scores = {"S1": 0.9, "S2": 0.8, "S3": 0.7, "S4": 0.6}
    sig = {"S1": _sig(), "S2": _sig(anchored=True), "S3": _sig(), "S4": _sig()}
    judgments = {
        "S1": ScreenJudgment(False, "not needed", ""),
        "S2": ScreenJudgment(False, "not needed", ""),  # anchored -> kept anyway
        "S3": None,  # failed call -> include
        # S4 not shortlisted -> not eligible
    }
    kept = select_screens(screens, scores, sig, judgments, max_images=5)
    assert [s.id for s in kept] == ["S2", "S3"]


def test_select_zero_images() -> None:
    screens = [scr("S1", [(0.0, 1.0)])]
    assert select_screens(screens, {"S1": 1.0}, {"S1": _sig()}, None, max_images=0) == []
