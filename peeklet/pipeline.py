"""Pipeline orchestrator wiring all core modules with cascade logic."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from peeklet.config import PeekletConfig, load_patterns
from peeklet.core.comparator import compare_frames
from peeklet.core.exporter import ManifestWriter, save_keyframe
from peeklet.core.hasher import compute_phash, hashes_match
from peeklet.core.masking import AdaptiveMask
from peeklet.core.redactor import build_pattern_set
from peeklet.utils.types import EventType, FrameResult

if TYPE_CHECKING:
    from datetime import datetime

    import numpy as np


class Pipeline:
    """Orchestrates the full frame processing pipeline in batch mode.

    Processing cascade per frame:
    1. Apply adaptive mask
    2. Compute perceptual hash
    3. First frame → keyframe
    3b. Dimension change → keyframe
    4. Hash matches last → SKIP (ssim_score stays None)
    5. Hash differs → compute SSIM via compare_frames()
    6. SSIM > threshold → SKIP
    7. Otherwise → KEYFRAME (save image, update rolling state)
    """

    def __init__(self, config: PeekletConfig) -> None:
        self._config = config
        output_dir = Path(config.exporter.output_dir)
        self._output_dir = output_dir
        self._writer = ManifestWriter(
            path=output_dir / "manifest.parquet",
            compression=config.exporter.parquet_compression,
        )
        self._mask = AdaptiveMask(
            block_size=config.masking.block_size,
            window_size=config.masking.window_size,
            noise_threshold=config.masking.noise_threshold,
        )
        # Rolling state
        self._last_keyframe: np.ndarray | None = None  # masked version for SSIM comparison
        self._last_hash: str | None = None
        self._last_mean: float | None = None  # mean pixel value for uniform-frame disambiguation
        self._last_keyframe_id: str | None = None
        self._last_keyframe_path: str | None = None

        # Build redaction pattern set (if redactor enabled)
        custom_patterns: list = []
        if config.redactor.enabled and config.redactor.custom_patterns_file:
            try:
                custom_patterns = load_patterns(Path(config.redactor.custom_patterns_file))
            except FileNotFoundError:
                custom_patterns = []
        self._pattern_set = build_pattern_set(
            pii_types=config.redactor.pii_types if config.redactor.enabled else [],
            custom_patterns=custom_patterns,
        )

    def _make_keyframe(
        self,
        frame: np.ndarray,
        masked_frame: np.ndarray,
        frame_id: str,
        current_hash: str,
        current_mean: float,
        visual_reason: str,
        mask_regions: list,
        timestamp: datetime | None = None,
        app_name: str | None = None,
        window_title: str | None = None,
        source_format: str | None = None,
        ssim_score: float | None = None,
        change_score: float | None = None,
        changed_pct: float | None = None,
        changed_regions: list | None = None,
    ) -> FrameResult:
        """Save a keyframe and build the result, updating rolling state."""
        h, w = frame.shape[:2]
        asset_path = save_keyframe(
            frame,
            self._output_dir,
            frame_id,
            fmt=self._config.exporter.keyframe_format,
        )
        result = FrameResult(
            frame_id=frame_id,
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash=current_hash,
            frame_width=w,
            frame_height=h,
            timestamp=timestamp,
            app_name=app_name,
            window_title=window_title,
            source_format=source_format,
            adaptive_mask=mask_regions if mask_regions else None,
            ssim_score=ssim_score,
            change_score=change_score,
            changed_pct=changed_pct,
            changed_regions=changed_regions,
            asset_path=str(asset_path),
            visual_reason=visual_reason,
            prev_keyframe_id=self._last_keyframe_id,
            prev_keyframe_path=self._last_keyframe_path,
        )
        self._last_hash = current_hash
        self._last_keyframe = masked_frame
        self._last_mean = current_mean
        self._last_keyframe_id = frame_id
        self._last_keyframe_path = str(asset_path)
        self._writer.append(result)
        return result

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        timestamp: datetime | None = None,
        app_name: str | None = None,
        window_title: str | None = None,
        source_format: str | None = None,
    ) -> FrameResult:
        """Process a single frame through the pipeline cascade.

        Args:
            frame: RGB uint8 numpy array.
            frame_id: Unique identifier for this frame.
            timestamp: Optional capture timestamp.
            app_name: Optional application name.
            window_title: Optional window title.
            source_format: Optional source format string.

        Returns:
            FrameResult describing the outcome.
        """
        h, w = frame.shape[:2]

        # Step 1: Apply adaptive mask
        masked_frame, mask_regions = self._mask.apply(frame)

        # Step 2: Compute perceptual hash of masked frame
        current_hash = compute_phash(masked_frame, hash_size=self._config.hasher.hash_size)

        # Compute mean pixel value (used to disambiguate uniform frames with identical phash)
        current_mean = float(masked_frame.mean())

        # Step 3: First frame is always a keyframe
        if self._last_hash is None:
            return self._make_keyframe(
                frame, masked_frame, frame_id, current_hash, current_mean,
                visual_reason="First frame in sequence",
                mask_regions=mask_regions,
                timestamp=timestamp, app_name=app_name,
                window_title=window_title, source_format=source_format,
            )

        # Step 3b: Frame dimension change → automatic KEYFRAME
        if self._last_keyframe is not None and masked_frame.shape[:2] != self._last_keyframe.shape[:2]:
            prev_h, prev_w = self._last_keyframe.shape[:2]
            return self._make_keyframe(
                frame, masked_frame, frame_id, current_hash, current_mean,
                visual_reason=f"Dimension change ({prev_w}x{prev_h} -> {w}x{h})",
                mask_regions=mask_regions,
                timestamp=timestamp, app_name=app_name,
                window_title=window_title, source_format=source_format,
            )

        # Step 4: Hash matches AND mean pixel value is close → SKIP (ssim_score stays None)
        mean_diff = abs(current_mean - self._last_mean)  # type: ignore[operator]
        if hashes_match(current_hash, self._last_hash) and mean_diff < 5.0:
            result = FrameResult(
                frame_id=frame_id,
                event_type=EventType.SKIPPED,
                is_keyframe=False,
                perceptual_hash=current_hash,
                frame_width=w,
                frame_height=h,
                timestamp=timestamp,
                app_name=app_name,
                window_title=window_title,
                source_format=source_format,
                adaptive_mask=mask_regions if mask_regions else None,
                ssim_score=None,
                visual_reason="Hash match — visually identical to previous keyframe",
                prev_keyframe_id=self._last_keyframe_id,
                prev_keyframe_path=self._last_keyframe_path,
            )
            self._writer.append(result)
            return result

        # Step 5: Hash differs → compute SSIM
        comparison = compare_frames(
            masked_frame,
            self._last_keyframe,  # type: ignore[arg-type]
            block_size=self._config.masking.block_size,
        )

        # Step 6: SSIM above threshold → SKIP
        if comparison.ssim_score > self._config.comparator.ssim_threshold:
            result = FrameResult(
                frame_id=frame_id,
                event_type=EventType.SKIPPED,
                is_keyframe=False,
                perceptual_hash=current_hash,
                frame_width=w,
                frame_height=h,
                timestamp=timestamp,
                app_name=app_name,
                window_title=window_title,
                source_format=source_format,
                adaptive_mask=mask_regions if mask_regions else None,
                ssim_score=comparison.ssim_score,
                change_score=comparison.change_score,
                changed_pct=comparison.changed_pct,
                changed_regions=comparison.changed_regions if comparison.changed_regions else None,
                visual_reason=f"Minor change — SSIM {comparison.ssim_score:.4f} above threshold {self._config.comparator.ssim_threshold}",
                prev_keyframe_id=self._last_keyframe_id,
                prev_keyframe_path=self._last_keyframe_path,
            )
            self._writer.append(result)
            return result

        # Step 7: SSIM below threshold → KEYFRAME
        return self._make_keyframe(
            frame, masked_frame, frame_id, current_hash, current_mean,
            visual_reason=f"Significant change — SSIM {comparison.ssim_score:.4f}, {comparison.changed_pct:.1f}% of blocks changed",
            mask_regions=mask_regions,
            timestamp=timestamp, app_name=app_name,
            window_title=window_title, source_format=source_format,
            ssim_score=comparison.ssim_score,
            change_score=comparison.change_score,
            changed_pct=comparison.changed_pct,
            changed_regions=comparison.changed_regions if comparison.changed_regions else None,
        )

    def finalize(self) -> None:
        """Flush the manifest writer to disk."""
        self._writer.flush()
