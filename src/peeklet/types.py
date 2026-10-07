"""Shared type definitions for Peeklet."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Region:
    """A rectangular region on screen."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(frozen=True, slots=True)
class Moment:
    """A screenshot-worthy moment in a video transcript."""

    timestamp: float  # seconds into the video
    visual_context_goal: str  # what the screenshot needs to capture
    textual_anchor: str  # the transcript line that triggers this need
    downstream_utility: str  # why the MLLM needs this image
    source: str = "llm"  # "llm" | "anchor"


@dataclass(frozen=True, slots=True)
class Line:
    """One timestamped transcript line."""

    start: float  # seconds
    end: float  # seconds
    text: str
    speaker: str | None = None
