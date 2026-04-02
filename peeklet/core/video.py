"""Video frame extraction with smart sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peeklet.core.audio import (
    align_transcript,
    detect_speech_segments,
    get_audio_activity,
    parse_transcript,
)
from peeklet.core.exporter import ManifestWriter
from peeklet.pipeline import Pipeline
from peeklet.utils.image import ensure_rgb_uint8

if TYPE_CHECKING:
    from collections.abc import Iterator

    from peeklet.config import PeekletConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.utils.types import FrameResult


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
        Seeks to the nearest keyframe before start_frame using pyav,
        avoiding O(n) decode of the entire video prefix.
        """
        import av

        container = av.open(str(self._path))
        stream = container.streams.video[0]

        # Seek to a point just before start_frame
        target_ts = int(start_frame / self._fps * av.time_base)
        container.seek(target_ts)

        for frame in container.decode(stream):
            idx = int(frame.pts * stream.time_base * self._fps)
            if idx < start_frame:
                continue
            if idx >= end_frame:
                break
            timestamp = idx / self._fps
            rgb = ensure_rgb_uint8(np.asarray(frame.to_ndarray(format="rgb24")))
            yield rgb, timestamp, idx

        container.close()


def _change_magnitude(result: FrameResult) -> str:
    """Derive change magnitude from SSIM and block count."""
    if result.ssim_score is None:
        return "major"  # first frame or dimension change
    n_blocks = len(result.changed_regions) if result.changed_regions else 0
    if result.ssim_score < 0.70 or n_blocks > 15:
        return "major"
    if result.ssim_score < 0.90 or n_blocks > 5:
        return "moderate"
    return "minor"


def process_video(
    path: Path,
    config: PeekletConfig,
    writer: ManifestWriter | None = None,
) -> list[FrameResult]:
    """Process a video file through smart sampling + Peeklet pipeline.

    Two-pass strategy:
    1. Coarse pass at config.video.sample_fps
    2. Backfill around SKIP->KEYFRAME transitions at native fps

    Args:
        path: Path to video file.
        config: Peeklet configuration.
        writer: Optional shared ManifestWriter for multi-video processing.
            If None, a new writer is created and flushed automatically.

    Returns list of FrameResult for all processed frames.
    """
    decoder = VideoDecoder(path)
    meta = decoder.get_metadata()
    sample_fps = config.video.sample_fps
    output_dir = Path(config.exporter.output_dir)

    owns_writer = writer is None
    if writer is None:
        writer = ManifestWriter(
            path=output_dir / "manifest.parquet",
            compression=config.exporter.parquet_compression,
        )

    # --- Pass 1: Coarse sampling to find transitions ---
    # Use a throwaway pipeline (writes to /dev/null) just for keyframe decisions
    coarse_config = config.model_copy(deep=True)
    coarse_config.exporter.output_dir = str(output_dir / ".coarse_tmp")
    coarse_pipeline = Pipeline(coarse_config)
    coarse_samples: list[tuple[np.ndarray, float, int, FrameResult]] = []

    for frame, ts, frame_num in decoder.extract_coarse_frames(sample_fps):
        frame_id = f"coarse_{frame_num:06d}"
        result = coarse_pipeline.process_frame(
            frame,
            frame_id=frame_id,
            source_format="video",
        )
        coarse_samples.append((frame, ts, frame_num, result))

    # --- Pass 2: Build final frame list with backfill ---
    all_frames: list[tuple[np.ndarray, float, int]] = []

    for i, (frame, ts, frame_num, result) in enumerate(coarse_samples):
        if i > 0:
            prev_result = coarse_samples[i - 1][3]
            prev_frame_num = coarse_samples[i - 1][2]
            # SKIP -> KEYFRAME: backfill the gap at native fps
            # Note: extract_frame_range iterates from frame 0. For long videos,
            # consider using pyav seeking for better performance.
            if not prev_result.is_keyframe and result.is_keyframe:
                for bf_frame, bf_ts, bf_num in decoder.extract_frame_range(
                    prev_frame_num + 1, frame_num
                ):
                    all_frames.append((bf_frame, bf_ts, bf_num))
        all_frames.append((frame, ts, frame_num))

    # Clean up coarse pass artifacts
    coarse_dir = output_dir / ".coarse_tmp"
    if coarse_dir.exists():
        import shutil

        shutil.rmtree(coarse_dir)

    # --- Final pass: process frames, enrich metadata, write manifest ---
    # Use a pipeline that writes keyframe images but NOT the manifest
    # (we manage the manifest ourselves for correct enrichment ordering)
    final_config = config.model_copy(deep=True)
    final_pipeline = Pipeline(final_config)
    results: list[FrameResult] = []
    keyframe_count = 0
    prev_keyframe_ts: float | None = None

    for frame, ts, frame_num in all_frames:
        frame_id = f"frame_{frame_num:06d}"
        result = final_pipeline.process_frame(
            frame,
            frame_id=frame_id,
            source_format="video",
        )

        # Enrich with video metadata
        result.source_video = meta.filename
        result.video_timestamp = ts
        result.video_frame_number = frame_num
        result.video_duration = meta.duration

        if result.is_keyframe:
            keyframe_count += 1
            new_frame_id = f"step_{keyframe_count:03d}"
            # Rename the saved keyframe image to use the step_NNN name
            if result.asset_path:
                old_path = Path(result.asset_path)
                new_path = old_path.with_name(new_frame_id + old_path.suffix)
                if old_path.exists():
                    old_path.rename(new_path)
                result.asset_path = str(new_path)
            result.frame_id = new_frame_id
            result.keyframe_index = keyframe_count
            result.change_magnitude = _change_magnitude(result)
            if prev_keyframe_ts is not None:
                result.time_since_prev_keyframe = ts - prev_keyframe_ts
            prev_keyframe_ts = ts

        results.append(result)

    # Backfill total_keyframes on all keyframe results
    for r in results:
        if r.is_keyframe:
            r.total_keyframes = keyframe_count

    # --- Audio enrichment ---
    transcript_segments: list[TranscriptSegment] | None = None
    speech_segments: list[TranscriptSegment] | None = None

    if config.video.transcript_path:
        transcript_segments = parse_transcript(Path(config.video.transcript_path))

    if config.video.audio_detection:
        try:
            speech_segments = detect_speech_segments(path)
        except Exception:
            speech_segments = None  # Audio extraction may fail for some videos

    for r in results:
        if not r.is_keyframe or r.video_timestamp is None:
            continue

        ts = r.video_timestamp

        # Transcript alignment (takes priority for audio_activity too)
        if transcript_segments:
            r.transcript_segment = align_transcript(ts, transcript_segments)
            r.audio_activity = "speech" if r.transcript_segment else "silence"
        elif speech_segments is not None:
            r.audio_activity = get_audio_activity(ts, speech_segments)
        elif config.video.audio_detection:
            r.audio_activity = "silence"

    # Write fully-enriched results to the manifest
    for r in results:
        writer.append(r)

    if owns_writer:
        writer.flush()

    return results
