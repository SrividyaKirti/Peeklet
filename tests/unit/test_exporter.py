"""Tests for Parquet export and keyframe saving."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from peeklet.core.exporter import ManifestWriter, save_keyframe
from peeklet.utils.types import EventType, FrameResult, Region


class TestSaveKeyframe:
    def test_saves_png(self, tmp_output: Path) -> None:
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        path = save_keyframe(frame, tmp_output, "frame_001", fmt="png")
        assert path.exists()
        assert path.suffix == ".png"

    def test_saves_jpg(self, tmp_output: Path) -> None:
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        path = save_keyframe(frame, tmp_output, "frame_001", fmt="jpg")
        assert path.exists()
        assert path.suffix == ".jpg"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "output"
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        path = save_keyframe(frame, out, "frame_001", fmt="png")
        assert path.exists()


class TestManifestWriter:
    def test_write_and_read_manifest(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        result = FrameResult(
            frame_id="frame_001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd1234abcd1234",
            ssim_score=0.62,
            change_score=0.38,
            changed_pct=34.0,
            changed_regions=[Region(x=100, y=200, w=300, h=150)],
            adaptive_mask=[],
            frame_width=1920,
            frame_height=1080,
            asset_path="output/frame_001.png",
        )
        writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.num_rows == 1
        assert table.column("frame_id")[0].as_py() == "frame_001"
        assert table.column("is_keyframe")[0].as_py() is True
        assert table.column("perceptual_hash")[0].as_py() == "abcd1234abcd1234"

    def test_write_skipped_frame(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        result = FrameResult(
            frame_id="frame_002",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="abcd1234abcd1234",
            frame_width=1920,
            frame_height=1080,
        )
        writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.column("is_keyframe")[0].as_py() is False
        assert table.column("asset_path")[0].as_py() is None

    def test_multiple_rows(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        for i in range(10):
            result = FrameResult(
                frame_id=f"frame_{i:03d}",
                event_type=EventType.KEYFRAME if i % 5 == 0 else EventType.SKIPPED,
                is_keyframe=i % 5 == 0,
                perceptual_hash=f"hash{i:012d}",
                frame_width=1920,
                frame_height=1080,
            )
            writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.num_rows == 10

    def test_snappy_compression(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet", compression="snappy")
        result = FrameResult(
            frame_id="frame_001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd1234abcd1234",
            frame_width=1920,
            frame_height=1080,
        )
        writer.append(result)
        writer.flush()

        meta = pq.read_metadata(tmp_output / "manifest.parquet")
        assert meta.row_group(0).column(0).compression == "SNAPPY"


class TestManifestVideoColumns:
    def test_video_columns_in_manifest(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        result = FrameResult(
            frame_id="frame_v001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd1234abcd1234",
            frame_width=1920,
            frame_height=1080,
            source_video="recording.mp4",
            video_timestamp=12.5,
            video_frame_number=375,
            time_since_prev_keyframe=3.2,
            audio_activity="speech",
            transcript_segment="Hello world",
            keyframe_index=5,
            total_keyframes=42,
            video_duration=120.0,
            change_magnitude="major",
            trigger_type="visual_change",
        )
        writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.num_rows == 1

        expected = {
            "source_video": "recording.mp4",
            "video_timestamp": 12.5,
            "video_frame_number": 375,
            "time_since_prev_keyframe": 3.2,
            "audio_activity": "speech",
            "transcript_segment": "Hello world",
            "keyframe_index": 5,
            "total_keyframes": 42,
            "video_duration": 120.0,
            "change_magnitude": "major",
            "trigger_type": "visual_change",
        }
        for col, expected_val in expected.items():
            assert col in table.column_names, f"Column '{col}' missing from manifest"
            actual = table.column(col)[0].as_py()
            assert actual == expected_val, (
                f"Column '{col}': expected {expected_val!r}, got {actual!r}"
            )

    def test_video_columns_null_for_image_frames(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        result = FrameResult(
            frame_id="frame_img001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd1234abcd1234",
            frame_width=1920,
            frame_height=1080,
        )
        writer.append(result)
        writer.flush()

        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.num_rows == 1

        video_columns = [
            "source_video",
            "video_timestamp",
            "video_frame_number",
            "time_since_prev_keyframe",
            "audio_activity",
            "transcript_segment",
            "keyframe_index",
            "total_keyframes",
            "video_duration",
            "change_magnitude",
            "trigger_type",
        ]
        for col in video_columns:
            assert col in table.column_names, f"Column '{col}' missing from manifest"
            assert table.column(col)[0].as_py() is None, (
                f"Column '{col}' should be null for image frames"
            )
