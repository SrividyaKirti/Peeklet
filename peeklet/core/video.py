"""Video frame extraction with smart sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from peeklet.utils.image import ensure_rgb_uint8


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """Metadata about a video file."""

    filename: str
    duration: float  # seconds
    fps: float
    width: int
    height: int
    total_frames: int


def _check_video_deps() -> None:
    """Raise a clear error if video dependencies are not installed."""
    try:
        import imageio.v3  # noqa: F401
    except ImportError:
        raise ImportError(
            "Video support requires additional dependencies. "
            "Install with: pip install peeklet[video]"
        ) from None


class VideoDecoder:
    """Decodes video files and extracts frames with smart sampling."""

    def __init__(self, path: Path) -> None:
        _check_video_deps()
        import imageio.v3 as iio

        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        try:
            props = iio.improps(self._path, plugin="pyav")
            meta = iio.immeta(self._path, plugin="pyav")
        except Exception as e:
            raise ValueError(f"Could not open video: {self._path}: {e}") from e

        self._fps: float = meta.get("fps", 30.0)
        self._duration: float = meta.get("duration", 0.0)
        # props.shape is (n_frames, H, W, C) for video
        h, w = props.shape[1], props.shape[2]
        self._width: int = w
        self._height: int = h
        self._total_frames: int = int(self._fps * self._duration)

    def get_metadata(self) -> VideoMeta:
        """Return metadata about the video file."""
        return VideoMeta(
            filename=self._path.name,
            duration=self._duration,
            fps=self._fps,
            width=self._width,
            height=self._height,
            total_frames=self._total_frames,
        )

    def extract_coarse_frames(
        self, sample_fps: float = 1.0
    ) -> Iterator[tuple[np.ndarray, float, int]]:
        """Extract frames at a coarse sample rate.

        Yields (frame_rgb, timestamp_seconds, frame_number) tuples.
        """
        import imageio.v3 as iio

        frame_interval = max(1, int(self._fps / sample_fps))

        for idx, frame in enumerate(iio.imiter(self._path, plugin="pyav")):
            if idx % frame_interval != 0:
                continue
            timestamp = idx / self._fps
            rgb = ensure_rgb_uint8(np.asarray(frame))
            yield rgb, timestamp, idx

    def extract_frame_range(
        self, start_frame: int, end_frame: int
    ) -> Iterator[tuple[np.ndarray, float, int]]:
        """Extract all frames in a range [start_frame, end_frame).

        Used for backfill around detected transitions.
        """
        import imageio.v3 as iio

        for idx, frame in enumerate(iio.imiter(self._path, plugin="pyav")):
            if idx < start_frame:
                continue
            if idx >= end_frame:
                break
            timestamp = idx / self._fps
            rgb = ensure_rgb_uint8(np.asarray(frame))
            yield rgb, timestamp, idx
