"""Transcript trigger word detection for screen-reference keyframes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment

HIGH_SIGNAL_WORDS: frozenset[str] = frozenset(
    {
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
    }
)

MEDIUM_SIGNAL_WORDS: frozenset[str] = frozenset(
    {
        "here",
        "this",
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
        "toggle",
        "zoom",
        "highlight",
        "observe",
        "watch",
    }
)


@dataclass(frozen=True, slots=True)
class TriggerResult:
    """A detected transcript trigger — forces a keyframe at this timestamp."""

    timestamp: float  # seconds (midpoint of the segment)
    segment: TranscriptSegment


def _is_trigger(text: str) -> bool:
    """Check if text contains enough signal words to trigger a keyframe."""
    words = set(re.sub(r"[^\w\s]", " ", text.lower()).split())
    high_matches = words & HIGH_SIGNAL_WORDS
    has_medium = bool(words & MEDIUM_SIGNAL_WORDS)
    if high_matches and has_medium:
        return True
    return len(high_matches) >= 2


def detect_triggers(segments: list[TranscriptSegment]) -> list[TriggerResult]:
    """Scan transcript segments for screen-reference trigger words."""
    triggers: list[TriggerResult] = []
    for segment in segments:
        if _is_trigger(segment.text):
            midpoint = (segment.start + segment.end) / 2
            triggers.append(TriggerResult(timestamp=midpoint, segment=segment))
    return triggers
