"""Shared type definitions for Peeklet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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
class Line:
    """One timestamped transcript line."""

    start: float  # seconds
    end: float  # seconds
    text: str
    speaker: str | None = None


def format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS (truncating fractions)."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


CheckpointKind = Literal["action_item", "speech_onset", "verbal_cue"]


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A moment where Peeklet guarantees a capture and boosts the screen on display."""

    t: float
    kind: CheckpointKind
    label: str = ""


@dataclass(slots=True)
class Screen:
    """A distinct screen state, global across the whole video."""

    id: str
    image_jpeg: bytes  # chosen frame (most OCR words; ties -> latest)
    frame_t: float  # timestamp of the chosen frame
    ocr_text: str
    word_count: int
    first_change: float  # 1 - SSIM when the screen first appeared (0..1)
    occurrences: list[tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ScreenJudgment:
    """LLM verdict for one screen."""

    include: bool
    reason: str
    visual_context: str


@dataclass(frozen=True, slots=True)
class Entry:
    """One output block: consecutive lines spoken over the same screen (or none)."""

    timestamp: float
    lines: tuple[Line, ...]
    image: str | None = None
    visual_context: str | None = None
    action_items: tuple[str, ...] = ()
    screen_id: str | None = None  # internal; not serialized

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "timestamp": format_hms(self.timestamp),
            "transcript": "\n".join(f"{ln.speaker or 'Speaker'}: {ln.text}" for ln in self.lines),
            "image": self.image,
        }
        if self.image and self.visual_context:
            out["visual_context"] = self.visual_context
        if self.action_items:
            out["action_item"] = "; ".join(self.action_items)
        return out
