"""Integration tests for video input end-to-end."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import pytest

from peeklet.config import PeekletConfig
from peeklet.core.video import process_video


def _solid_frame(color: tuple[int, int, int], w: int = 320, h: int = 240) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _make_test_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> Path:
    with iio.imopen(path, "w", plugin="pyav") as out:
        out.init_video_stream("libx264", fps=fps)
        for frame in frames:
            out.write_frame(frame)
    return path


class TestVideoEndToEnd:
    def test_full_pipeline_produces_manifest_and_keyframes(self, tmp_path: Path) -> None:
        """Video -> process_video -> manifest.parquet + keyframe images."""
        # Use black/white for clear transitions (H.264 compresses colors)
        frames = (
            [_solid_frame((0, 0, 0))] * 45
            + [_solid_frame((255, 255, 255))] * 45
            + [_solid_frame((128, 128, 128))] * 45
        )
        video_path = _make_test_video(tmp_path / "demo.mp4", frames, fps=30)

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = False

        results = process_video(video_path, config)

        # Check results structure
        keyframes = [r for r in results if r.is_keyframe]
        assert len(keyframes) >= 2  # At least first frame + one transition

        # Check manifest was written
        manifest_path = tmp_path / "output" / "manifest.parquet"
        assert manifest_path.exists()
        table = pq.read_table(manifest_path)

        # Verify video-specific columns exist
        for col in ["source_video", "video_timestamp", "video_frame_number",
                     "keyframe_index", "total_keyframes", "video_duration",
                     "change_magnitude"]:
            assert col in table.column_names

        # Verify keyframe rows have video metadata populated
        row = table.to_pydict()
        keyframe_indices = [
            i for i, is_kf in enumerate(row["is_keyframe"]) if is_kf
        ]

        for idx in keyframe_indices:
            assert row["source_video"][idx] == "demo.mp4"
            assert row["video_timestamp"][idx] is not None
            assert row["video_frame_number"][idx] is not None
            assert row["keyframe_index"][idx] is not None
            assert row["total_keyframes"][idx] is not None
            assert row["video_duration"][idx] is not None
            assert row["change_magnitude"][idx] in ("minor", "moderate", "major")

        # Check keyframe images exist
        output_dir = tmp_path / "output"
        keyframe_images = list(output_dir.glob("step_*.png"))
        assert len(keyframe_images) >= 2

    def test_video_with_transcript(self, tmp_path: Path) -> None:
        """Video + transcript -> keyframes with transcript_segment populated."""
        frames = (
            [_solid_frame((0, 0, 0))] * 60
            + [_solid_frame((255, 255, 255))] * 60
        )
        video_path = _make_test_video(tmp_path / "demo.mp4", frames, fps=30)

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
        config.redactor.enabled = False
        config.video.transcript_path = str(srt_path)

        results = process_video(video_path, config)
        keyframes = [r for r in results if r.is_keyframe]

        # First keyframe at t=0 should have transcript
        assert keyframes[0].transcript_segment is not None

    def test_multi_video_directory(self, tmp_path: Path) -> None:
        """Multiple videos produce results with different source_video values."""
        for name, colors in [
            ("a.mp4", [(0, 0, 0), (255, 255, 255)]),
            ("b.mp4", [(128, 128, 128), (0, 0, 0)]),
        ]:
            frames = [_solid_frame(colors[0])] * 30 + [_solid_frame(colors[1])] * 30
            _make_test_video(tmp_path / name, frames, fps=30)

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = False

        # Process both videos (simulating CLI multi-video behavior)
        all_results = []
        for vf in sorted(tmp_path.glob("*.mp4")):
            results = process_video(vf, config)
            all_results.extend(results)

        videos = {r.source_video for r in all_results if r.source_video}
        assert "a.mp4" in videos
        assert "b.mp4" in videos
