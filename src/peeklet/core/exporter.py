"""Parquet manifest writer and keyframe image saver."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

if TYPE_CHECKING:
    import numpy as np

    from peeklet.utils.types import FrameResult, Screen

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("frame_id", pa.string()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("event_type", pa.string()),
        pa.field("app_name", pa.string(), nullable=True),
        pa.field("window_title", pa.string(), nullable=True),
        pa.field("is_keyframe", pa.bool_()),
        pa.field("perceptual_hash", pa.string()),
        pa.field("ssim_score", pa.float64(), nullable=True),
        pa.field("change_score", pa.float64(), nullable=True),
        pa.field("changed_pct", pa.float64(), nullable=True),
        pa.field(
            "changed_regions",
            pa.list_(
                pa.struct(
                    [
                        pa.field("x", pa.int32()),
                        pa.field("y", pa.int32()),
                        pa.field("w", pa.int32()),
                        pa.field("h", pa.int32()),
                    ]
                )
            ),
            nullable=True,
        ),
        pa.field(
            "adaptive_mask",
            pa.list_(
                pa.struct(
                    [
                        pa.field("x", pa.int32()),
                        pa.field("y", pa.int32()),
                        pa.field("w", pa.int32()),
                        pa.field("h", pa.int32()),
                    ]
                )
            ),
            nullable=True,
        ),
        pa.field("frame_width", pa.int32()),
        pa.field("frame_height", pa.int32()),
        pa.field("source_format", pa.string(), nullable=True),
        pa.field("asset_path", pa.string(), nullable=True),
        pa.field("visual_reason", pa.string(), nullable=True),
        pa.field("trigger_type", pa.string(), nullable=True),
        pa.field("prev_keyframe_id", pa.string(), nullable=True),
        pa.field("prev_keyframe_path", pa.string(), nullable=True),
        pa.field("source_video", pa.string(), nullable=True),
        pa.field("video_timestamp", pa.float64(), nullable=True),
        pa.field("video_frame_number", pa.int32(), nullable=True),
        pa.field("time_since_prev_keyframe", pa.float64(), nullable=True),
        pa.field("audio_activity", pa.string(), nullable=True),
        pa.field("transcript_segment", pa.string(), nullable=True),
        pa.field("keyframe_index", pa.int32(), nullable=True),
        pa.field("total_keyframes", pa.int32(), nullable=True),
        pa.field("video_duration", pa.float64(), nullable=True),
        pa.field("change_magnitude", pa.string(), nullable=True),
    ]
)


def save_keyframe(frame: np.ndarray, output_dir: Path, frame_id: str, fmt: str = "png") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{frame_id}.{fmt}"
    Image.fromarray(frame).save(path)
    return path


class ManifestWriter:
    def __init__(self, path: Path, compression: str = "snappy") -> None:
        self._path = Path(path)
        self._compression = compression
        self._rows: list[dict] = []

    def append(self, result: FrameResult) -> None:
        regions = None
        if result.changed_regions is not None:
            regions = [r.to_dict() for r in result.changed_regions]
        mask = None
        if result.adaptive_mask is not None:
            mask = [r.to_dict() for r in result.adaptive_mask]
        self._rows.append(
            {
                "frame_id": result.frame_id,
                "timestamp": result.timestamp,
                "event_type": result.event_type.value,
                "app_name": result.app_name,
                "window_title": result.window_title,
                "is_keyframe": result.is_keyframe,
                "perceptual_hash": result.perceptual_hash,
                "ssim_score": result.ssim_score,
                "change_score": result.change_score,
                "changed_pct": result.changed_pct,
                "changed_regions": regions,
                "adaptive_mask": mask,
                "frame_width": result.frame_width,
                "frame_height": result.frame_height,
                "source_format": result.source_format,
                "asset_path": result.asset_path,
                "visual_reason": result.visual_reason,
                "trigger_type": result.trigger_type,
                "prev_keyframe_id": result.prev_keyframe_id,
                "prev_keyframe_path": result.prev_keyframe_path,
                "source_video": result.source_video,
                "video_timestamp": result.video_timestamp,
                "video_frame_number": result.video_frame_number,
                "time_since_prev_keyframe": result.time_since_prev_keyframe,
                "audio_activity": result.audio_activity,
                "transcript_segment": result.transcript_segment,
                "keyframe_index": result.keyframe_index,
                "total_keyframes": result.total_keyframes,
                "video_duration": result.video_duration,
                "change_magnitude": result.change_magnitude,
            }
        )

    def clear(self) -> None:
        """Discard all buffered rows without writing."""
        self._rows.clear()

    def flush(self) -> None:
        if not self._rows:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self._rows, schema=MANIFEST_SCHEMA)
        pq.write_table(table, self._path, compression=self._compression)
        self._rows.clear()


DEMO_SCREEN_SCHEMA = pa.schema(
    [
        pa.field("screen_id", pa.string()),
        pa.field("image_path", pa.string()),
        pa.field("first_seen_ms", pa.int64()),
        pa.field("url", pa.string()),
        pa.field("heading", pa.string()),
        pa.field("sidebar_text", pa.string()),
        pa.field("header_phash", pa.string()),  # 16-char lowercase hex
        pa.field("ocr_text", pa.string()),
    ]
)


def write_demo_manifest(screens: list[Screen], path, compression: str = "snappy") -> None:
    """Write one parquet row per unique demo screen."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "screen_id": s.screen_id,
            "image_path": s.image_path,
            "first_seen_ms": s.first_seen_ms,
            "url": s.fingerprint.url,
            "heading": s.fingerprint.heading,
            "sidebar_text": s.fingerprint.sidebar_text,
            "header_phash": f"{s.fingerprint.header_phash:016x}",
            "ocr_text": s.ocr_text,
        }
        for s in screens
    ]
    table = pa.Table.from_pylist(rows, schema=DEMO_SCREEN_SCHEMA)
    pq.write_table(table, path, compression=compression)
