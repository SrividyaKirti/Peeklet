"""Verbal-cue checkpoints: transcript lines that reference something on screen.

Lexicon restored from the former ``transcript_trigger.py`` (develop, before #30).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from peeklet.types import Checkpoint

if TYPE_CHECKING:
    from peeklet.types import Line

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
DEICTIC_WORDS: frozenset[str] = frozenset({"this", "that", "here", "there"})

_HIGH_WEIGHT = 1.0
_MEDIUM_WEIGHT = 0.5
_DEICTIC_WEIGHT = 0.3
_PUNCT_RE = re.compile(r"[^\w\s]")


def cue_score(text: str) -> float:
    """Weighted count of distinct cue words (high 1.0, medium 0.5, deictic 0.3)."""
    words = set(_PUNCT_RE.sub(" ", text.lower()).split())
    return (
        _HIGH_WEIGHT * len(words & HIGH_SIGNAL_WORDS)
        + _MEDIUM_WEIGHT * len(words & MEDIUM_SIGNAL_WORDS)
        + _DEICTIC_WEIGHT * len(words & DEICTIC_WORDS)
    )


def verbal_cue_checkpoints(lines: list[Line], threshold: float = 1.0) -> list[Checkpoint]:
    """One checkpoint at the start of each line whose cue score reaches the threshold."""
    return [
        Checkpoint(t=ln.start, kind="verbal_cue", label=ln.text[:80])
        for ln in lines
        if cue_score(ln.text) >= threshold
    ]
