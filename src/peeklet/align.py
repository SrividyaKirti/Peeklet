"""Align transcript lines to the screen on display while they were spoken."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peeklet.types import Line, Screen

_EPS = 1e-9


def screen_at(screens: Sequence[Screen], t: float) -> str | None:
    """Id of the screen whose occurrence covers t (half-open intervals), else None."""
    for s in screens:
        for start, end in s.occurrences:
            if start <= t < end:
                return s.id
    return None


def align_lines(
    lines: Sequence[Line], screens: Sequence[Screen], lead_seconds: float
) -> list[str | None]:
    """At most one screen per line: largest overlap with [start, end + lead]."""
    first_seen = {s.id: min((a for a, _ in s.occurrences), default=math.inf) for s in screens}
    out: list[str | None] = []
    for ln in lines:
        lo, hi = ln.start, ln.end + lead_seconds
        best: str | None = None
        best_overlap = 0.0
        for s in screens:
            overlap = sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in s.occurrences)
            if overlap <= _EPS:
                continue
            better = overlap > best_overlap + _EPS
            tie = best is not None and abs(overlap - best_overlap) <= _EPS
            if better or (tie and first_seen[s.id] < first_seen[best]):  # type: ignore[index]
                best, best_overlap = s.id, overlap
        out.append(best)
    return out
