"""Transcript trigger word detection for screen-reference keyframes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment

# UI nouns and product-review vocabulary. A single high-signal word is
# enough to fire the trigger under the scoring gate, so additions here
# directly widen coverage on product-review meetings.
HIGH_SIGNAL_WORDS: frozenset[str] = frozenset(
    {
        # Original UI lexicon
        "chart",
        "graph",
        "dashboard",
        "button",
        "menu",
        "sidebar",
        "diagram",
        "screen",
        "page",
        "modal",
        "dropdown",
        "tab",
        "panel",
        "table",
        "form",
        "icon",
        "window",
        "popup",
        "field",
        "toolbar",
        "flowchart",
        "heatmap",
        # Product-review additions (PR C)
        "rollout",
        "revamp",
        "rewrite",
        "view",
        "row",
        "column",
        "filter",
        "toggle",
        "status",
        "label",
        "beta",
        "empty",
        "loading",
        "cost",
        "count",
        "banner",
        "badge",
        "card",
        "list",
        "header",
        "footer",
        "input",
        "search",
        "link",
    }
)

# Medium-signal verbs and observations. On their own they are too noisy to
# trigger (score 0.5 < 1.0); in combination with other signals they push
# ambiguous segments over the threshold.
MEDIUM_SIGNAL_WORDS: frozenset[str] = frozenset(
    {
        "notice",
        "look",
        "see",
        "click",
        "shown",
        "display",
        "hover",
        "scroll",
        "select",
        "drag",
        "zoom",
        "highlight",
        "observe",
        "watch",
    }
)

# Deictic pronouns that reference something on screen without naming it.
# Weighted lowest because they are ambiguous in isolation.
DEICTIC_WORDS: frozenset[str] = frozenset({"this", "that", "here", "there"})

_HIGH_WEIGHT = 1.0
_MEDIUM_WEIGHT = 0.5
_DEICTIC_WEIGHT = 0.3
_TRIGGER_THRESHOLD = 1.0


@dataclass(frozen=True, slots=True)
class TriggerResult:
    """A detected transcript trigger — forces a keyframe at this timestamp."""

    timestamp: float  # seconds (midpoint of the segment)
    segment: TranscriptSegment


def _score_text(text: str) -> float:
    """Return the weighted trigger score for ``text``.

    High-signal words count 1.0, medium 0.5, deictics 0.3. Only distinct
    token hits contribute so a single repeated word cannot spam the score.
    """
    words = set(re.sub(r"[^\w\s]", " ", text.lower()).split())
    score = 0.0
    score += _HIGH_WEIGHT * len(words & HIGH_SIGNAL_WORDS)
    score += _MEDIUM_WEIGHT * len(words & MEDIUM_SIGNAL_WORDS)
    score += _DEICTIC_WEIGHT * len(words & DEICTIC_WORDS)
    return score


def _is_trigger(text: str) -> bool:
    """Check if text scores at or above the trigger threshold."""
    return _score_text(text) >= _TRIGGER_THRESHOLD


def detect_triggers(segments: list[TranscriptSegment]) -> list[TriggerResult]:
    """Scan transcript segments for screen-reference trigger words."""
    triggers: list[TriggerResult] = []
    for segment in segments:
        if _is_trigger(segment.text):
            midpoint = (segment.start + segment.end) / 2
            triggers.append(TriggerResult(timestamp=midpoint, segment=segment))
    return triggers
