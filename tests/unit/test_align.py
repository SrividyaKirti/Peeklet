"""Tests for line-to-screen alignment."""

from __future__ import annotations

from peeklet.align import align_lines, screen_at
from peeklet.types import Line, Screen


def scr(sid: str, occ: list[tuple[float, float]]) -> Screen:
    return Screen(
        id=sid,
        image_jpeg=b"",
        frame_t=occ[0][0],
        ocr_text="",
        word_count=0,
        first_change=1.0,
        occurrences=occ,
    )


SCREENS = [scr("S1", [(0.0, 10.0), (30.0, 40.0)]), scr("S2", [(10.0, 20.0)])]


def test_screen_at() -> None:
    assert screen_at(SCREENS, 5.0) == "S1"
    assert screen_at(SCREENS, 10.0) == "S2"
    assert screen_at(SCREENS, 25.0) is None


def test_line_gets_largest_overlap() -> None:
    assert align_lines([Line(7.0, 15.0, "x")], SCREENS, lead_seconds=0.0) == ["S2"]


def test_lead_window_extends_line_end() -> None:
    assert align_lines([Line(8.0, 9.5, "now I'll click")], SCREENS, 1.5) == ["S1"]
    assert align_lines([Line(20.5, 29.0, "x")], SCREENS, 1.5) == ["S1"]


def test_no_overlap_gives_none() -> None:
    assert align_lines([Line(21.0, 25.0, "x")], SCREENS, 1.5) == [None]


def test_tie_prefers_screen_that_appeared_first() -> None:
    screens = [scr("S2", [(10.0, 20.0)]), scr("S1", [(0.0, 10.0)])]
    assert align_lines([Line(8.0, 12.0, "x")], screens, 0.0) == ["S1"]
