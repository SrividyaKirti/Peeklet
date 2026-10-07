"""Tests for the streaming screen pass (OCR stubbed by colour)."""

from __future__ import annotations

import numpy as np

from peeklet.config import PeekletConfig
from peeklet.screen import WordBox
from peeklet.screens import Sample, build_screens
from peeklet.types import Checkpoint

A = (200, 20, 20)
A_RICH = (202, 20, 20)  # same screen as A, more OCR words (same pHash, different mean)
B = (20, 200, 20)
GALLERY = (0, 0, 0)


def frame(rgb: tuple[int, int, int]) -> np.ndarray:
    return np.full((64, 64, 3), rgb, dtype=np.uint8)


def boxes_for(heading: str, n_lines: int) -> list[WordBox]:
    out = [WordBox(heading, 95.0, 20, 2, 30, 6), WordBox("Home", 95.0, 0, 30, 8, 4)]
    out += [WordBox(f"word{i}", 95.0, 12, 10 + i * 5, 20, 4) for i in range(n_lines)]
    return out


def fake_ocr(f: np.ndarray) -> list[WordBox]:
    rgb = tuple(int(v) for v in f[0, 0])
    return {
        A: boxes_for("ScreenA", 12),
        A_RICH: boxes_for("ScreenA", 20),
        B: boxes_for("ScreenB", 12),
    }.get(rgb, [])  # type: ignore[arg-type]


def run(
    seq: list[tuple[int, int, int]], checkpoints: list[Checkpoint] | None = None, **cfg_kw: float
):
    samples = [Sample(t=float(i), frame=frame(rgb)) for i, rgb in enumerate(seq)]
    return build_screens(
        samples, checkpoints or [], PeekletConfig(**cfg_kw), fake_ocr, video_end=float(len(seq))
    )


def test_screen_returning_later_is_one_screen_with_two_occurrences() -> None:
    res = run([A] * 3 + [B] * 3 + [A] * 3)
    assert len(res.screens) == 2
    a = next(s for s in res.screens if "ScreenA" in s.ocr_text)
    assert a.occurrences == [(0.0, 3.0), (6.0, 9.0)]


def test_different_screens_not_merged() -> None:
    res = run([A] * 2 + [B] * 2)
    assert {s.ocr_text.split()[0] for s in res.screens} == {"ScreenA", "ScreenB"}


def test_low_info_frames_create_no_screen_and_end_occurrence() -> None:
    res = run([A] * 3 + [GALLERY] * 3 + [A] * 2)
    [a] = res.screens
    assert a.occurrences == [(0.0, 3.0), (6.0, 8.0)]
    assert (3.0, None) in res.events


def test_empty_ocr_frames_never_merge() -> None:
    def ocr_body_only(f: np.ndarray) -> list[WordBox]:
        # 12 text lines (passes the rejector) but all below the midline and right of the
        # sidebar band, so Part A (url, heading, sidebar) is empty.
        return [WordBox(f"w{i}", 95.0, 60, 110 + i * 7, 20, 4) for i in range(12)]

    big = [np.full((200, 200, 3), rgb, dtype=np.uint8) for rgb in (A, B)]
    samples = [Sample(0.0, big[0]), Sample(1.0, big[1])]
    res = build_screens(samples, [], PeekletConfig(), ocr_body_only, video_end=2.0)
    assert len(res.screens) == 2


def test_checkpoint_window_picks_richest_frame() -> None:
    seq = [A] * 12 + [A_RICH] + [A] * 8  # A_RICH at t=12 is not a change (hash match)
    res = run(seq, [Checkpoint(t=10.0, kind="speech_onset")])
    [a] = res.screens
    assert a.frame_t == 12.0
    assert a.word_count == len(boxes_for("ScreenA", 20))


def test_checkpoint_window_ties_prefer_closest_sample() -> None:
    res = run([A] * 20, [Checkpoint(t=10.0, kind="verbal_cue")])
    assert (10.0, res.screens[0].id) in res.events


def test_all_low_info_window_warns() -> None:
    res = run([GALLERY] * 20, [Checkpoint(t=10.0, kind="speech_onset")])
    assert res.screens == []
    assert any("no usable frame" in w for w in res.warnings)


def test_ocr_runs_once_per_visual_state() -> None:
    calls: list[int] = []

    def counting_ocr(f: np.ndarray) -> list[WordBox]:
        calls.append(1)
        return fake_ocr(f)

    samples = [Sample(float(i), frame(A)) for i in range(30)]
    cps = [Checkpoint(t=float(t), kind="verbal_cue") for t in (5, 10, 15, 20, 25)]
    build_screens(samples, cps, PeekletConfig(), counting_ocr, video_end=30.0)
    assert len(calls) == 1


def test_chosen_frame_is_jpeg() -> None:
    [a] = run([A] * 2).screens
    assert a.image_jpeg[:2] == b"\xff\xd8"


def _body_only_run(n: int, cp_t: float):
    def ocr_body_only(f: np.ndarray) -> list[WordBox]:
        return [WordBox(f"w{i}", 95.0, 60, 110 + i * 7, 20, 4) for i in range(12)]

    samples = [Sample(float(i), np.full((200, 200, 3), A, dtype=np.uint8)) for i in range(n)]
    return build_screens(
        samples,
        [Checkpoint(t=cp_t, kind="speech_onset")],
        PeekletConfig(),
        ocr_body_only,
        video_end=float(n),
    )


def test_checkpoint_winner_that_is_change_candidate_not_duplicated() -> None:
    res = _body_only_run(5, 0.0)
    assert len(res.screens) == 1
    assert res.screens[0].occurrences == [(0.0, 5.0)]


def test_static_empty_part_a_screen_with_mid_checkpoint_is_one_screen() -> None:
    res = _body_only_run(12, 8.0)
    assert len(res.screens) == 1
