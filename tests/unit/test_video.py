"""Tests for the video decoder module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

iio = pytest.importorskip("imageio.v3", reason="requires peeklet[video]")
pytest.importorskip("av", reason="requires peeklet[video]")

from peeklet.config import PeekletConfig  # noqa: E402
from peeklet.core.video import VideoDecoder, VideoMeta, process_video  # noqa: E402


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


class TestCoarseExtraction:
    def test_coarse_pass_extracts_at_sample_fps(self, tmp_path: Path) -> None:
        # 3 seconds of video at 30fps = 90 frames
        frames = (
            [_solid_frame((255, 0, 0))] * 30
            + [_solid_frame((0, 255, 0))] * 30
            + [_solid_frame((0, 0, 255))] * 30
        )
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)

        decoder = VideoDecoder(video_path)
        coarse_frames = list(decoder.extract_coarse_frames(sample_fps=1.0))

        # At 1fps over ~3 seconds, expect ~3 samples
        assert len(coarse_frames) >= 2
        for frame, ts, frame_num in coarse_frames:
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8
            assert isinstance(ts, float)
            assert isinstance(frame_num, int)


class TestSmartSampling:
    def test_coarse_sampling_finds_transition(self, tmp_path: Path) -> None:
        # 3 seconds: 1.5s black, then 1.5s white (sharp transition at frame 45)
        frames = [_solid_frame((0, 0, 0))] * 45 + [_solid_frame((255, 255, 255))] * 45
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.sample_fps = 1.0

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # Should find at least 2 keyframes: first frame + the transition.
        # With coarse-only sampling at 1fps, the transition lands on the next
        # coarse sample boundary (±1s precision is fine for LLM use).
        assert len(keyframes) >= 2
        assert keyframes[0].video_frame_number == 0

    def test_no_backfill_when_no_transitions(self, tmp_path: Path) -> None:
        # 2 seconds of solid red — no transitions
        frames = [_solid_frame((255, 0, 0))] * 60
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.sample_fps = 1.0

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # Only the first frame should be a keyframe
        assert len(keyframes) == 1

    def test_results_have_video_metadata(self, tmp_path: Path) -> None:
        frames = [_solid_frame((255, 0, 0))] * 30 + [_solid_frame((0, 255, 0))] * 30
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        for kf in keyframes:
            assert kf.source_video == "test.mp4"
            assert kf.video_timestamp is not None
            assert kf.video_frame_number is not None
            assert kf.video_duration is not None
            assert kf.video_duration == pytest.approx(2.0, abs=0.5)
            assert kf.keyframe_index is not None
            assert kf.change_magnitude is not None

    def test_time_since_prev_keyframe(self, tmp_path: Path) -> None:
        frames = [_solid_frame((255, 0, 0))] * 30 + [_solid_frame((0, 255, 0))] * 30
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # First keyframe has no previous
        assert keyframes[0].time_since_prev_keyframe is None
        # Second keyframe should have a time gap
        if len(keyframes) > 1:
            assert keyframes[1].time_since_prev_keyframe is not None
            assert keyframes[1].time_since_prev_keyframe > 0


class TestAudioEnrichment:
    def test_audio_activity_populated_when_enabled(self, tmp_path: Path) -> None:
        frames = [_solid_frame((0, 0, 0))] * 30 + [_solid_frame((255, 255, 255))] * 30
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.audio_detection = True

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        for kf in keyframes:
            assert kf.audio_activity in ("speech", "silence")

    def test_audio_activity_none_when_disabled(self, tmp_path: Path) -> None:
        frames = [_solid_frame((0, 0, 0))] * 30
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.audio_detection = False

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        for kf in keyframes:
            assert kf.audio_activity is None

    def test_transcript_alignment(self, tmp_path: Path) -> None:
        frames = [_solid_frame((0, 0, 0))] * 60 + [_solid_frame((255, 255, 255))] * 60
        video_path = _make_test_video(tmp_path / "test.mp4", frames, fps=30)

        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(
            "1\n"
            "00:00:00,000 --> 00:00:01,500\n"
            "Welcome to the demo\n"
            "\n"
            "2\n"
            "00:00:02,000 --> 00:00:03,500\n"
            "Now click here\n"
            "\n"
        )

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.video.transcript_path = str(srt_path)

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # First keyframe at t=0 should match first transcript segment
        assert keyframes[0].transcript_segment is not None
        assert "Welcome" in keyframes[0].transcript_segment
