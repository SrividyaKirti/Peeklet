"""Tests for evaluation helpers (no network)."""

from __future__ import annotations

import pytest

from eval.common import (
    change_times_from_debug,
    clean_youtube_vtt,
    match_events,
    prf,
    segment_index,
    whisperx_to_vtt,
)
from peeklet.transcript import _parse_vtt

YT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000 align:start position:0%
hello<00:00:01.500><c> world</c>

00:00:03.000 --> 00:00:03.010 align:start position:0%
hello world

00:00:03.010 --> 00:00:05.000 align:start position:0%
hello world
next<00:00:03.500><c> line</c>
"""


def test_clean_youtube_vtt_strips_tags_and_rolling_duplicates() -> None:
    lines = _parse_vtt(clean_youtube_vtt(YT))
    assert [(ln.start, ln.text) for ln in lines] == [(1.0, "hello world"), (3.01, "next line")]


def test_match_events_one_to_one_within_tolerance() -> None:
    assert match_events([10.0, 10.5, 30.0], [10.2, 20.0], tolerance=2.0) == (1, 2, 1)


def test_prf_handles_zero() -> None:
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(1, 1, 0) == pytest.approx((0.5, 1.0, 2 / 3))


def test_segment_index() -> None:
    assert segment_index([10.0, 20.0], 5.0) == 0
    assert segment_index([10.0, 20.0], 10.0) == 1
    assert segment_index([10.0, 20.0], 25.0) == 2


def test_change_times_from_debug() -> None:
    debug = {
        "screens": [
            {"occurrences": [[0.0, 10.0], [30.0, 40.0]]},
            {"occurrences": [[10.0, 30.0], [30.2, 31.0]]},
        ]
    }
    assert change_times_from_debug(debug) == [10.0, 30.0]


def test_whisperx_to_vtt() -> None:
    vtt = whisperx_to_vtt([{"start": 1.617, "end": 30.305, "sentence": "Alright. We start."}])
    [ln] = _parse_vtt(vtt)
    assert (ln.start, ln.end, ln.text) == (1.617, 30.305, "Alright. We start.")
