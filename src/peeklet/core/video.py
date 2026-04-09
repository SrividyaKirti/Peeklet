"""Video frame extraction with smart sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from peeklet.core.audio import (
    align_transcript,
    detect_speech_segments,
    get_audio_activity,
    parse_transcript,
)
from peeklet.core.context_exporter import (
    build_context,
    timestamp_filename,
    write_context_json,
    write_context_markdown,
)
from peeklet.core.demo_filter import apply_demo_filter
from peeklet.core.exporter import ManifestWriter, save_keyframe
from peeklet.pipeline import Pipeline
from peeklet.utils.image import ensure_rgb_uint8
from peeklet.utils.types import EventType

logger = logging.getLogger(__name__)

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
        """Extract frames at a coarse sample rate using sequential decode.

        Yields (frame_rgb, timestamp_seconds, frame_number) tuples.

        Note: This decodes every frame and discards most. Prefer
        :meth:`iter_coarse_frames_seek` for long videos.
        """
        import imageio.v3 as iio

        frame_interval = max(1, int(self._fps / sample_fps))

        for idx, frame in enumerate(iio.imiter(self._path, plugin="pyav")):
            if idx % frame_interval != 0:
                continue
            timestamp = idx / self._fps
            rgb = ensure_rgb_uint8(np.asarray(frame))
            yield rgb, timestamp, idx

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
            if frame.pts is None or stream.time_base is None:
                continue
            idx = int(frame.pts * stream.time_base * self._fps)
            if idx < start_frame:
                continue
            if idx >= end_frame:
                break
            timestamp = idx / self._fps
            rgb = ensure_rgb_uint8(np.asarray(frame.to_ndarray(format="rgb24")))
            yield rgb, timestamp, idx

        container.close()

    def extract_frame_at(self, timestamp_sec: float) -> tuple[np.ndarray, float, int]:
        """Extract a single frame at (or just after) ``timestamp_sec``.

        Seeks to the nearest keyframe before the requested timestamp via
        libav and decodes forward until a frame at-or-past the target is
        found. Returns ``(frame_rgb, actual_timestamp, frame_number)``.

        Used by demo-mode's sparse OCR sweep where we only need a few
        frames spread across the video, not a continuous range.

        Raises:
            RuntimeError: if no frame can be decoded at or after the
                requested timestamp (e.g., timestamp past end of video).
        """
        import av

        container = av.open(str(self._path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            target_ts = int(timestamp_sec * av.time_base)
            container.seek(target_ts)

            for frame in container.decode(stream):
                if frame.pts is None or stream.time_base is None:
                    continue
                ts = float(frame.pts * stream.time_base)
                if ts + 1e-6 < timestamp_sec:
                    continue
                arr = frame.to_ndarray(format="rgb24")
                rgb = ensure_rgb_uint8(np.asarray(arr))
                frame_num = int(round(ts * self._fps))
                return rgb, ts, frame_num

            raise RuntimeError(
                f"No frame found at or after timestamp {timestamp_sec}s "
                f"in {self._path.name} (duration={self._duration}s)"
            )
        finally:
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


def _should_force_keyframe(
    timestamp: float, forced_timestamps: list[float], tolerance: float = 0.5
) -> bool:
    """Check if a frame timestamp is close enough to a forced timestamp."""
    return any(abs(timestamp - ft) <= tolerance for ft in forced_timestamps)


def _merge_trigger_type(
    is_visual: bool, is_transcript: bool
) -> Literal["visual_change", "transcript_trigger", "both"] | None:
    """Determine trigger_type from visual and transcript flags."""
    if is_visual and is_transcript:
        return "both"
    if is_visual:
        return "visual_change"
    if is_transcript:
        return "transcript_trigger"
    return None


def process_video(
    path: Path,
    config: PeekletConfig,
    writer: ManifestWriter | None = None,
    forced_timestamps: list[float] | None = None,
) -> list[FrameResult]:
    """Process a video file through coarse sampling + Peeklet pipeline.

    Streaming single-pass implementation:
    - Decodes one frame per ``config.video.sample_fps`` interval, no frame
      buffer accumulation (memory-bounded for arbitrarily long videos).
    - Optionally downscales frames to ``config.video.processing_max_dim``
      before they hit the pipeline (default 720) so masking, hashing, and
      SSIM stay fast on HD/4K sources.
    - Optional native-fps backfill around visual transitions can be enabled
      via ``config.video.enable_backfill`` (off by default).

    Args:
        path: Path to video file.
        config: Peeklet configuration.
        writer: Optional shared ManifestWriter for multi-video processing.
            If None, a new writer is created and flushed automatically.
        forced_timestamps: Optional list of timestamps in seconds at which
            keyframes should be forced regardless of visual change. Typically
            derived from transcript trigger words.

    Returns list of FrameResult for all processed (sampled) frames.

    Note:
        ``process_video()`` always writes ``context.json`` and ``context.md``
        to the configured output directory.
    """
    decoder = VideoDecoder(path)
    meta = decoder.get_metadata()
    sample_fps = config.video.sample_fps
    output_dir = Path(config.exporter.output_dir)
    max_dim = config.video.processing_max_dim
    forced_ts = forced_timestamps or []

    owns_writer = writer is None
    if writer is None:
        writer = ManifestWriter(
            path=output_dir / "manifest.parquet",
            compression=config.exporter.parquet_compression,
        )

    # Demo mode: bypass the coarse pass entirely. The LLM picks moments,
    # Stage B picks frames, and we write outputs directly.
    if config.demo_filter.enabled:
        demo_transcript: list[TranscriptSegment] = []
        if config.video.transcript_path:
            demo_transcript = parse_transcript(Path(config.video.transcript_path))

        demo_results = apply_demo_filter(
            decoder=decoder,
            transcript=demo_transcript,
            config=config.demo_filter,
            output_dir=output_dir,
        )

        ctx = build_context(meta.filename, meta.duration, demo_results, demo_transcript)
        write_context_json(ctx, output_dir / "context.json")
        write_context_markdown(ctx, output_dir / "context.md")

        for r in demo_results:
            writer.append(r)
        if owns_writer:
            writer.flush()

        return demo_results

    # Adaptive masking is designed for screencasts (cursor/clock noise).
    # In video mode it both adds significant overhead and tends to mask out
    # the very UI changes we want to detect. Disable it for the pipeline.
    pipeline_config = config.model_copy(deep=True)
    pipeline_config.masking.enabled = False

    # --- Single streaming pass: seek-based coarse decode + pipeline ---
    pipeline = Pipeline(pipeline_config)
    results: list[FrameResult] = []
    keyframe_count = 0
    prev_keyframe_ts: float | None = None

    # Decoder downscales inside pyav (libswscale), which is much faster than
    # decoding at full resolution and resizing in Python.
    for frame, ts, frame_num in decoder.iter_coarse_frames_seek(sample_fps, max_dim=max_dim):
        frame_id = f"frame_{frame_num:06d}"
        result = pipeline.process_frame(
            frame,
            frame_id=frame_id,
            source_format="video",
        )

        is_forced = _should_force_keyframe(ts, forced_ts)
        was_visual_keyframe = result.is_keyframe

        # Promote forced (transcript-triggered) frames to keyframes
        if is_forced and not result.is_keyframe:
            asset_path = save_keyframe(
                frame,
                output_dir,
                frame_id,
                fmt=config.exporter.keyframe_format,
            )
            result.is_keyframe = True
            result.event_type = EventType.KEYFRAME
            result.asset_path = str(asset_path)
            result.visual_reason = "Transcript trigger"
            pipeline.update_reference_state(frame, frame_id, str(asset_path))

        # Set trigger_type on every keyframe
        if result.is_keyframe:
            result.trigger_type = _merge_trigger_type(
                is_visual=was_visual_keyframe,
                is_transcript=is_forced,
            )

        # Enrich with video metadata
        result.source_video = meta.filename
        result.video_timestamp = ts
        result.video_frame_number = frame_num
        result.video_duration = meta.duration

        if result.is_keyframe:
            keyframe_count += 1
            new_frame_id = timestamp_filename(ts)
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

    # --- Context export (JSON + Markdown) ---
    transcript_for_context = transcript_segments or []
    ctx = build_context(meta.filename, meta.duration, results, transcript_for_context)
    write_context_json(ctx, output_dir / "context.json")
    write_context_markdown(ctx, output_dir / "context.md")

    # Write fully-enriched results to the manifest
    for r in results:
        writer.append(r)

    if owns_writer:
        writer.flush()

    return results
