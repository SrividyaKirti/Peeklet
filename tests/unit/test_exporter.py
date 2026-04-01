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
            pii_detected=False,
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
