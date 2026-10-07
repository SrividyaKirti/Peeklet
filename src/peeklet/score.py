"""Heuristic screen scoring and final selection."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from peeklet.align import screen_at

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from peeklet.config import ScoreWeights
    from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "have",
        "this",
        "that",
        "with",
        "they",
        "from",
        "what",
        "when",
        "your",
        "will",
        "there",
        "their",
        "about",
        "would",
        "which",
        "into",
        "them",
        "then",
        "than",
        "some",
        "just",
        "like",
        "here",
        "okay",
        "yeah",
        "so",
        "now",
        "going",
        "get",
        "got",
        "see",
        "look",
        "let",
        "lets",
        "it's",
        "its",
        "also",
    ]
)


def content_words(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of length >= 3 that are not stopwords."""
    return {w for w in _TOKEN_RE.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


@dataclass(frozen=True, slots=True)
class ScreenSignals:
    """Per-screen ranking signals, each in [0, 1]."""

    references: float
    text_overlap: float
    onsets: float
    verbal_cues: float
    visual_change: float
    time_on_screen: float
    anchored: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _log_scaled(counts: Mapping[str, float], ids: Sequence[str]) -> dict[str, float]:
    top = max((math.log1p(counts.get(i, 0.0)) for i in ids), default=0.0)
    return {i: (math.log1p(counts.get(i, 0.0)) / top if top > 0 else 0.0) for i in ids}


def compute_signals(
    screens: Sequence[Screen],
    lines: Sequence[Line],
    line_screens: Sequence[str | None],
    checkpoints: Sequence[Checkpoint],
) -> dict[str, ScreenSignals]:
    """Compute all ranking signals for every screen."""
    ids = [s.id for s in screens]
    refs = Counter(sid for sid in line_screens if sid is not None)
    words: dict[str, set[str]] = {i: set() for i in ids}
    for ln, sid in zip(lines, line_screens, strict=True):
        if sid is not None:
            words[sid] |= content_words(ln.text)
    kind_counts: dict[str, Counter[str]] = {
        "speech_onset": Counter(),
        "verbal_cue": Counter(),
        "action_item": Counter(),
    }
    for cp in checkpoints:
        sid = screen_at(screens, cp.t)
        if sid is not None:
            kind_counts[cp.kind][sid] += 1
    dwell = {s.id: float(sum(b - a for a, b in s.occurrences)) for s in screens}
    references = _log_scaled(refs, ids)
    onsets = _log_scaled(kind_counts["speech_onset"], ids)
    cues = _log_scaled(kind_counts["verbal_cue"], ids)
    time_on = _log_scaled(dwell, ids)
    out: dict[str, ScreenSignals] = {}
    for s in screens:
        ocr_words = content_words(s.ocr_text)
        union = words[s.id] | ocr_words
        overlap = len(words[s.id] & ocr_words) / len(union) if union else 0.0
        out[s.id] = ScreenSignals(
            references=references[s.id],
            text_overlap=overlap,
            onsets=onsets[s.id],
            verbal_cues=cues[s.id],
            visual_change=min(1.0, max(0.0, s.first_change)),
            time_on_screen=time_on[s.id],
            anchored=kind_counts["action_item"][s.id] > 0,
        )
    return out


def score_screens(
    signals: Mapping[str, ScreenSignals], weights: ScoreWeights, anchor_bonus: float
) -> dict[str, float]:
    """Weighted sum of signals, plus anchor_bonus for screens shown at an action item."""
    return {
        sid: weights.references * g.references
        + weights.text_overlap * g.text_overlap
        + weights.onsets * g.onsets
        + weights.verbal_cues * g.verbal_cues
        + weights.visual_change * g.visual_change
        + weights.time_on_screen * g.time_on_screen
        + (anchor_bonus if g.anchored else 0.0)
        for sid, g in signals.items()
    }


def first_seen(screen: Screen) -> float:
    return min((a for a, _ in screen.occurrences), default=screen.frame_t)


def rank_screens(screens: Sequence[Screen], scores: Mapping[str, float]) -> list[Screen]:
    """Highest score first; ties go to the screen that appeared first."""
    return sorted(screens, key=lambda s: (-scores[s.id], first_seen(s)))


def select_screens(
    screens: Sequence[Screen],
    scores: Mapping[str, float],
    signals: Mapping[str, ScreenSignals],
    judgments: Mapping[str, ScreenJudgment | None] | None,
    max_images: int,
) -> list[Screen]:
    """Pick at most max_images screens; result is ordered by first appearance.

    Without judgments: top by score. With judgments (keys = shortlisted ids): eligible
    if the verdict is include, the call failed (None), or the screen is anchored.
    """
    ranked = rank_screens(screens, scores)
    if judgments is not None:
        ranked = [
            s
            for s in ranked
            if s.id in judgments
            and (
                judgments[s.id] is None
                or judgments[s.id].include  # type: ignore[union-attr]
                or signals[s.id].anchored
            )
        ]
    return sorted(ranked[:max_images], key=first_seen)
