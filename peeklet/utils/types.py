"""Shared type definitions for Peeklet."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


class EventType(Enum):
    """Whether a frame was kept as a keyframe or skipped."""

    KEYFRAME = "KEYFRAME"
    SKIPPED = "SKIPPED"


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
class FrameMeta:
    """Metadata about a frame provided by the caller."""

    frame_id: str
    timestamp: datetime | None = None
    app_name: str | None = None
    window_title: str | None = None
    source_format: str | None = None


@dataclass(slots=True)
class FrameResult:
    """Result of processing a single frame through the pipeline."""

    frame_id: str
    event_type: EventType
    is_keyframe: bool
    perceptual_hash: str
    frame_width: int
    frame_height: int
    timestamp: datetime | None = None
    app_name: str | None = None
    window_title: str | None = None
    ssim_score: float | None = None
    change_score: float | None = None
    changed_pct: float | None = None
    changed_regions: list[Region] | None = None
    adaptive_mask: list[Region] | None = None
    source_format: str | None = None
    asset_path: str | None = None
    pii_detected: bool | None = None
    visual_reason: str | None = None
    prev_keyframe_id: str | None = None
    prev_keyframe_path: str | None = None
    # Video-specific fields (all None for image-sourced frames)
    source_video: str | None = None
    video_timestamp: float | None = None
    video_frame_number: int | None = None
    time_since_prev_keyframe: float | None = None
    audio_activity: str | None = None
    transcript_segment: str | None = None
    keyframe_index: int | None = None
    total_keyframes: int | None = None
    video_duration: float | None = None
    change_magnitude: str | None = None
