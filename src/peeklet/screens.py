"""Streaming screen pass: candidates -> OCR + fingerprint -> global screens.

Memory stays bounded: per candidate we keep (t, screen id); full frames are kept
only as each screen's current best frame (JPEG bytes). OCR results are cached by
ChangeResult.key so a static screen is OCR'd once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from peeklet.change import ChangeDetector
from peeklet.image_utils import encode_jpeg
from peeklet.screen import FingerprintIndex, compute_fingerprint, is_low_info_frame
from peeklet.types import Screen, format_hms

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy as np

    from peeklet.config import PeekletConfig
    from peeklet.screen import Fingerprint, WordBox
    from peeklet.types import Checkpoint

    OcrFn = Callable[[np.ndarray], list[WordBox]]


@dataclass(frozen=True, slots=True)
class Sample:
    """One decoded video sample."""

    t: float
    frame: np.ndarray


@dataclass(slots=True)
class ScreenPass:
    """Result of the screen pass."""

    screens: list[Screen]
    events: list[tuple[float, str | None]]  # time-ordered; None = low-info frame
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class _Analysis:
    low_info: bool
    fingerprint: Fingerprint
    word_count: int
    ocr_text: str


@dataclass(slots=True)
class _Window:
    cp: Checkpoint
    best_key: tuple[int, float] | None = None
    best: tuple[float, _Analysis, np.ndarray, float] | None = None


class _Tracker:
    def __init__(self, cfg: PeekletConfig, ocr: OcrFn) -> None:
        self.cfg = cfg
        self.ocr = ocr
        self.cache: dict[str, _Analysis] = {}
        self.index = FingerprintIndex(
            phash_threshold=cfg.phash_threshold, ocr_field_min_chars=cfg.ocr_field_min_chars
        )
        self.screens: dict[str, Screen] = {}
        self.events: dict[float, str | None] = {}
        self.warnings: list[str] = []

    def analyze(self, frame: np.ndarray, key: str) -> _Analysis:
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        boxes = self.ocr(frame)
        result = _Analysis(
            low_info=is_low_info_frame(frame, boxes, self.cfg),
            fingerprint=compute_fingerprint(frame, boxes),  # type: ignore[arg-type]
            word_count=len(boxes),
            ocr_text=" ".join(b.text for b in boxes),
        )
        self.cache[key] = result
        return result

    def assign(self, t: float, a: _Analysis, frame: np.ndarray, change: float) -> None:
        if a.low_info:
            self.events[t] = None
            return
        sid = self.index.lookup(a.fingerprint)
        if sid is None:
            sid = f"S{len(self.screens) + 1}"
            self.index.register(a.fingerprint, screen_id=sid)
            self.screens[sid] = Screen(
                id=sid,
                image_jpeg=encode_jpeg(frame, self.cfg.jpeg_quality),
                frame_t=t,
                ocr_text=a.ocr_text,
                word_count=a.word_count,
                first_change=change,
            )
        else:
            s = self.screens[sid]
            if a.word_count > s.word_count or (a.word_count == s.word_count and t > s.frame_t):
                s.image_jpeg = encode_jpeg(frame, self.cfg.jpeg_quality)
                s.frame_t = t
                s.ocr_text = a.ocr_text
                s.word_count = a.word_count
        self.events[t] = sid

    def close(self, win: _Window) -> None:
        if win.best is None:
            self.warnings.append(
                f"no usable frame within ±{self.cfg.checkpoint_window_seconds:g}s of "
                f"{win.cp.kind} checkpoint at {format_hms(win.cp.t)}"
            )
            return
        t, a, frame, change = win.best
        self.assign(t, a, frame, change)


def build_screens(
    samples: Iterable[Sample],
    checkpoints: Sequence[Checkpoint],
    cfg: PeekletConfig,
    ocr: OcrFn,
    video_end: float,
) -> ScreenPass:
    """Run the streaming screen pass over time-ordered samples."""
    tracker = _Tracker(cfg, ocr)
    detector = ChangeDetector(cfg)
    half = cfg.checkpoint_window_seconds
    pending = sorted(checkpoints, key=lambda c: c.t)
    nxt = 0
    open_windows: list[_Window] = []

    for sample in samples:
        res = detector.update(sample.frame)
        while nxt < len(pending) and pending[nxt].t - half <= sample.t:
            open_windows.append(_Window(pending[nxt]))
            nxt += 1
        still_open: list[_Window] = []
        for win in open_windows:
            if sample.t > win.cp.t + half:
                tracker.close(win)
            else:
                still_open.append(win)
        open_windows = still_open
        active = [w for w in open_windows if abs(sample.t - w.cp.t) <= half]
        if not res.changed and not active:
            continue
        analysis = tracker.analyze(sample.frame, res.key)
        if res.changed:
            tracker.assign(sample.t, analysis, sample.frame, res.change)
        if analysis.low_info:
            continue
        for win in active:
            rank = (analysis.word_count, -abs(sample.t - win.cp.t))
            if win.best_key is None or rank > win.best_key:
                win.best_key = rank
                win.best = (sample.t, analysis, sample.frame, res.change)

    for win in open_windows:
        tracker.close(win)
    for cp in pending[nxt:]:
        tracker.close(_Window(cp))

    ordered = sorted(tracker.events.items())
    _fill_occurrences(tracker.screens, ordered, video_end)
    return ScreenPass(
        screens=list(tracker.screens.values()), events=ordered, warnings=tracker.warnings
    )


def _fill_occurrences(
    screens: dict[str, Screen], ordered: list[tuple[float, str | None]], video_end: float
) -> None:
    for i, (t, sid) in enumerate(ordered):
        if sid is None:
            continue
        end = ordered[i + 1][0] if i + 1 < len(ordered) else max(video_end, t)
        occ = screens[sid].occurrences
        if occ and abs(occ[-1][1] - t) < 1e-9:
            occ[-1] = (occ[-1][0], end)
        else:
            occ.append((t, end))
