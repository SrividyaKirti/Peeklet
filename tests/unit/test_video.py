"""Tests for the video decoder module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

iio = pytest.importorskip("imageio.v3", reason="requires peeklet[video]")
pytest.importorskip("av", reason="requires peeklet[video]")

from peeklet.video import VideoDecoder, VideoMeta  # noqa: E402


def _make_test_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> Path:
    """Create a synthetic video file for testing."""
    with iio.imopen(path, "w", plugin="pyav") as out:
        out.init_video_stream("libx264", fps=fps)
        for frame in frames:
            out.write_frame(frame)
    return path


def _solid_frame(color: tuple[int, int, int], w: int = 320, h: int = 240) -> np.ndarray:
    """Create a solid-color RGB frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class TestVideoMeta:
    def test_metadata_from_video(self, tmp_path: Path) -> None:
        frames = [_solid_frame((255, 0, 0))] * 30  # 1 second of red
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)

        decoder = VideoDecoder(video_path)
        meta = decoder.get_metadata()

        assert isinstance(meta, VideoMeta)
        assert meta.filename == "test.mp4"
        assert meta.fps == pytest.approx(30.0, abs=1.0)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.duration == pytest.approx(1.0, abs=0.2)

    def test_invalid_file_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.mp4"
        bad_file.write_bytes(b"not a video")
        with pytest.raises(ValueError, match="Could not open video"):
            VideoDecoder(bad_file)

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            VideoDecoder(Path("/nonexistent/video.mp4"))
