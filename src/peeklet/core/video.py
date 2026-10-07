"""Video decoding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peeklet.utils.image import ensure_rgb_uint8

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator


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
        raise ImportError("Video support requires imageio and av. Reinstall peeklet.") from None


class VideoDecoder:
    """Decodes video files and extracts frames."""

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

    def iter_coarse_frames_seek(
        self, sample_fps: float = 1.0, max_dim: int | None = None
    ) -> Iterator[tuple[np.ndarray, float, int]]:
        """Extract frames at a coarse sample rate using a single sequential
        pyav decode, yielding only the frames that fall on the sample grid.

        This is much faster than :meth:`extract_coarse_frames` because:
        - The codec still walks packets, but we never copy skipped frames into
          Python (no numpy materialization).
        - When ``max_dim`` is provided, downscaling is performed inside pyav's
          C code via libswscale, avoiding a separate PIL resize step and
          reducing the cost of ``to_ndarray``.

        Args:
            sample_fps: Coarse sample rate in frames per second.
            max_dim: If set, downscale frames so the largest dimension is at
                most this many pixels. Done inside pyav for speed.

        Yields (frame_rgb, timestamp_seconds, frame_number) tuples.
        """
        import av

        container = av.open(str(self._path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"  # enable multi-threaded decoding

            # Pre-compute target dimensions if downscaling
            target_w: int | None = None
            target_h: int | None = None
            if max_dim is not None and max(self._width, self._height) > max_dim:
                scale = max_dim / max(self._width, self._height)
                target_w = int(round(self._width * scale))
                target_h = int(round(self._height * scale))

            interval = 1.0 / sample_fps
            next_target = 0.0

            for frame in container.decode(stream):
                if frame.pts is None or stream.time_base is None:
                    continue
                ts = float(frame.pts * stream.time_base)
                if ts + 1e-6 < next_target:
                    continue
                # Reformat (and downscale) inside pyav before materializing.
                if target_w is not None:
                    arr = frame.to_ndarray(format="rgb24", width=target_w, height=target_h)
                else:
                    arr = frame.to_ndarray(format="rgb24")
                rgb = ensure_rgb_uint8(np.asarray(arr))
                frame_num = int(round(ts * self._fps))
                yield rgb, ts, frame_num
                next_target = ts + interval
        finally:
            container.close()
