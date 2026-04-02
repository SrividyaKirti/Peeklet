"""Edge and corner case tests for all Peeklet modules."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from peeklet.config import (
    ComparatorConfig,
    ExporterConfig,
    HasherConfig,
    MaskingConfig,
    PeekletConfig,
    PiiPattern,
    load_config,
)
from peeklet.core.comparator import ComparisonResult, compare_frames
from peeklet.core.exporter import ManifestWriter, save_keyframe
from peeklet.core.hasher import (
    compute_phash,
    compute_phash_tiled,
    hashes_match,
    tiled_hashes_match,
)
from peeklet.core.loader import load_frame
from peeklet.core.masking import AdaptiveMask
from peeklet.core.redactor import (
    BUILTIN_PATTERNS,
    build_pattern_set,
    find_pii_in_text,
    redact_regions,
)
from peeklet.pipeline import Pipeline
from peeklet.utils.image import compute_block_grid, crop_region, ensure_rgb_uint8
from peeklet.utils.types import EventType, FrameResult, Region

# ── Image utility edge cases ──────────────────────────────────────────────


class TestEnsureRgbUint8EdgeCases:
    def test_single_channel_3d(self) -> None:
        frame = np.full((50, 50, 1), 128, dtype=np.uint8)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (50, 50, 3)
        assert np.all(result == 128)

    def test_float_out_of_range_clips(self) -> None:
        frame = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
        result = ensure_rgb_uint8(frame)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 255  # clipped to 1.0 * 255
        assert result[0, 0, 1] == 0  # clipped to 0.0

    def test_float_zero_and_one(self) -> None:
        frame = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float64)
        result = ensure_rgb_uint8(frame)
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 255

    def test_4d_array_raises(self) -> None:
        frame = np.zeros((1, 10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            ensure_rgb_uint8(frame)

    def test_2_channel_raises(self) -> None:
        frame = np.zeros((10, 10, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 1, 3, or 4 channels"):
            ensure_rgb_uint8(frame)

    def test_rgba_strips_alpha(self) -> None:
        frame = np.zeros((10, 10, 4), dtype=np.uint8)
        frame[:, :, 0] = 100
        frame[:, :, 3] = 200  # alpha should be dropped
        result = ensure_rgb_uint8(frame)
        assert result.shape == (10, 10, 3)
        assert result[0, 0, 0] == 100

    def test_grayscale_float(self) -> None:
        frame = np.full((10, 10), 0.5, dtype=np.float64)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8
        assert 127 <= result[0, 0, 0] <= 128

    def test_1x1_pixel(self) -> None:
        frame = np.array([[[42, 43, 44]]], dtype=np.uint8)
        result = ensure_rgb_uint8(frame)
        assert result.shape == (1, 1, 3)
        np.testing.assert_array_equal(result[0, 0], [42, 43, 44])

    def test_uint16_converted(self) -> None:
        frame = np.full((10, 10, 3), 30000, dtype=np.uint16)
        result = ensure_rgb_uint8(frame)
        assert result.dtype == np.uint8


class TestCropRegionEdgeCases:
    def test_negative_coordinates_clamped(self) -> None:
        frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        region = Region(x=-10, y=-10, w=30, h=30)
        cropped = crop_region(frame, region)
        assert cropped.shape == (20, 20, 3)

    def test_region_entirely_outside(self) -> None:
        frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        region = Region(x=100, y=100, w=10, h=10)
        cropped = crop_region(frame, region)
        assert cropped.shape[0] == 0 or cropped.shape[1] == 0

    def test_zero_size_region(self) -> None:
        frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        region = Region(x=10, y=10, w=0, h=0)
        cropped = crop_region(frame, region)
        assert cropped.shape[0] == 0 or cropped.shape[1] == 0

    def test_region_covers_entire_frame(self) -> None:
        frame = np.full((50, 50, 3), 100, dtype=np.uint8)
        region = Region(x=0, y=0, w=50, h=50)
        cropped = crop_region(frame, region)
        np.testing.assert_array_equal(cropped, frame)


class TestComputeBlockGridEdgeCases:
    def test_zero_height(self) -> None:
        rows, cols = compute_block_grid(0, 100, block_size=32)
        assert rows == 0

    def test_zero_width(self) -> None:
        rows, cols = compute_block_grid(100, 0, block_size=32)
        assert cols == 0

    def test_exact_multiple(self) -> None:
        rows, cols = compute_block_grid(64, 96, block_size=32)
        assert rows == 2
        assert cols == 3

    def test_very_small_block_size(self) -> None:
        rows, cols = compute_block_grid(10, 10, block_size=1)
        assert rows == 10
        assert cols == 10

    def test_1x1_image(self) -> None:
        rows, cols = compute_block_grid(1, 1, block_size=1)
        assert rows == 1
        assert cols == 1


# ── Loader edge cases ─────────────────────────────────────────────────────


class TestLoaderEdgeCases:
    def test_load_1x1_image_file(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (1, 1), color=(42, 43, 44))
        p = tmp_path / "tiny.png"
        img.save(p)
        result = load_frame(p)
        assert result.shape == (1, 1, 3)

    def test_load_rgba_image_file(self, tmp_path: Path) -> None:
        img = Image.new("RGBA", (10, 10), color=(100, 200, 50, 128))
        p = tmp_path / "rgba.png"
        img.save(p)
        result = load_frame(p)
        assert result.shape == (10, 10, 3)
        assert result[0, 0, 0] == 100

    def test_load_grayscale_image_file(self, tmp_path: Path) -> None:
        img = Image.new("L", (20, 20), color=128)
        p = tmp_path / "gray.png"
        img.save(p)
        result = load_frame(p)
        assert result.shape == (20, 20, 3)
        assert result[0, 0, 0] == 128

    def test_load_palette_mode_image(self, tmp_path: Path) -> None:
        img = Image.new("P", (10, 10))
        p = tmp_path / "palette.png"
        img.save(p)
        result = load_frame(p)
        assert result.shape == (10, 10, 3)

    def test_load_bytearray(self) -> None:
        img = Image.new("RGB", (5, 5), (100, 100, 100))
        buf = BytesIO()
        img.save(buf, format="PNG")
        result = load_frame(bytearray(buf.getvalue()))
        assert result.shape == (5, 5, 3)

    def test_load_empty_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not load"):
            load_frame(b"")

    def test_load_truncated_png_raises(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (10, 10))
        buf = BytesIO()
        img.save(buf, format="PNG")
        truncated = buf.getvalue()[:20]
        p = tmp_path / "truncated.png"
        p.write_bytes(truncated)
        with pytest.raises(ValueError, match="Could not load"):
            load_frame(p)

    def test_load_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Unsupported source type"):
            load_frame([1, 2, 3])  # type: ignore[arg-type]

    def test_load_float_ndarray(self) -> None:
        frame = np.full((10, 10, 3), 0.5, dtype=np.float32)
        result = load_frame(frame)
        assert result.dtype == np.uint8

    def test_load_pil_grayscale(self) -> None:
        img = Image.new("L", (10, 10), color=200)
        result = load_frame(img)
        assert result.shape == (10, 10, 3)


# ── Hasher edge cases ─────────────────────────────────────────────────────


class TestHasherEdgeCases:
    def test_uniform_white_frame(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        h = compute_phash(frame)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_uniform_black_frame(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        h = compute_phash(frame)
        assert isinstance(h, str)

    def test_solid_colors_same_hash(self) -> None:
        """Different solid colors may produce same hash (no structural detail)."""
        red = np.full((100, 100, 3), 0, dtype=np.uint8)
        red[:, :, 0] = 255
        green = np.full((100, 100, 3), 0, dtype=np.uint8)
        green[:, :, 1] = 255
        # Both are uniform => same phash (this is expected; per-channel means disambiguate)
        h1 = compute_phash(red)
        h2 = compute_phash(green)
        assert h1 == h2  # proves need for mean disambiguation

    def test_1x1_frame(self) -> None:
        frame = np.array([[[128, 64, 32]]], dtype=np.uint8)
        h = compute_phash(frame)
        assert isinstance(h, str)

    def test_very_wide_frame(self) -> None:
        frame = np.full((10, 1000, 3), 128, dtype=np.uint8)
        h = compute_phash(frame)
        assert len(h) == 16

    def test_tiled_hash_1_row_frame(self) -> None:
        frame = np.full((1, 100, 3), 128, dtype=np.uint8)
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        assert len(hashes) == 1

    def test_tiled_hash_tile_height_equals_frame_height(self) -> None:
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        assert len(hashes) == 1

    def test_tiled_hash_tile_height_1(self) -> None:
        frame = np.full((5, 100, 3), 128, dtype=np.uint8)
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=1)
        assert len(hashes) == 5

    def test_hashes_match_identical(self) -> None:
        assert hashes_match("abcdef0123456789", "abcdef0123456789", tolerance=0)

    def test_hashes_match_tolerance_boundary(self) -> None:
        # 0x0 vs 0x1 = 1 bit difference
        assert hashes_match("0000000000000000", "0000000000000001", tolerance=1)
        assert not hashes_match("0000000000000000", "0000000000000001", tolerance=0)

    def test_tiled_hashes_match_one_empty(self) -> None:
        assert not tiled_hashes_match(["abc"], [], tolerance=0)
        assert not tiled_hashes_match([], ["abc"], tolerance=0)


# ── Comparator edge cases ─────────────────────────────────────────────────


class TestComparatorEdgeCases:
    def test_identical_frames_zero_change(self) -> None:
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = compare_frames(frame, frame.copy(), block_size=32)
        assert result.ssim_score > 0.999
        assert result.change_score < 0.001
        assert result.changed_pct == 0.0

    def test_inverted_frame_maximal_change(self) -> None:
        frame = np.full((64, 64, 3), 0, dtype=np.uint8)
        inverted = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = compare_frames(frame, inverted, block_size=32)
        assert result.ssim_score < 0.1
        assert result.changed_pct == 100.0

    def test_block_size_larger_than_frame(self) -> None:
        """Block size > frame size => 0 blocks, no crash."""
        frame = np.full((16, 16, 3), 128, dtype=np.uint8)
        different = np.full((16, 16, 3), 0, dtype=np.uint8)
        result = compare_frames(frame, different, block_size=32)
        assert result.changed_pct == 0.0  # 0 blocks
        assert result.changed_regions == []

    def test_block_change_threshold_boundary(self) -> None:
        """Block with mean diff exactly at threshold."""
        frame_a = np.full((32, 32, 3), 100, dtype=np.uint8)
        frame_b = np.full((32, 32, 3), 110, dtype=np.uint8)  # diff = 10.0 exactly
        result = compare_frames(frame_a, frame_b, block_size=32, block_change_threshold=10.0)
        # mean diff = 10.0, threshold checks > 10.0, so should NOT be changed
        assert len(result.changed_regions) == 0

    def test_block_change_above_threshold(self) -> None:
        frame_a = np.full((32, 32, 3), 100, dtype=np.uint8)
        frame_b = np.full((32, 32, 3), 111, dtype=np.uint8)  # diff = 11.0
        result = compare_frames(frame_a, frame_b, block_size=32, block_change_threshold=10.0)
        assert len(result.changed_regions) == 1

    def test_small_block_size(self) -> None:
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        different = np.full((10, 10, 3), 0, dtype=np.uint8)
        result = compare_frames(frame, different, block_size=2)
        assert result.changed_pct == 100.0

    def test_change_score_is_complement_of_ssim(self) -> None:
        frame_a = np.full((64, 64, 3), 100, dtype=np.uint8)
        frame_b = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = compare_frames(frame_a, frame_b, block_size=32)
        assert abs(result.change_score - (1.0 - result.ssim_score)) < 1e-6


class TestComparisonResultEdgeCases:
    def test_empty_changed_regions(self) -> None:
        result = ComparisonResult(
            ssim_score=1.0, change_score=0.0, changed_pct=0.0, changed_regions=[]
        )
        assert len(result.changed_regions) == 0

    def test_frozen_dataclass(self) -> None:
        result = ComparisonResult(
            ssim_score=0.5, change_score=0.5, changed_pct=50.0, changed_regions=[]
        )
        with pytest.raises(AttributeError):
            result.ssim_score = 0.9  # type: ignore[misc]


# ── Masking edge cases ────────────────────────────────────────────────────


class TestAdaptiveMaskEdgeCases:
    def test_frame_smaller_than_block_size(self) -> None:
        """Frame smaller than block_size => 0 grid blocks, returns unmasked."""
        mask = AdaptiveMask(block_size=100, window_size=5, noise_threshold=0.8)
        frame = np.full((50, 50, 3), 128, dtype=np.uint8)
        for _ in range(10):
            result, regions = mask.apply(frame)
        assert regions == []
        np.testing.assert_array_equal(result, frame)

    def test_window_size_1(self) -> None:
        """Degenerate window_size=1 should still work."""
        mask = AdaptiveMask(block_size=50, window_size=1, noise_threshold=0.5)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask.apply(frame)  # first frame
        # Second frame with change
        frame2 = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame2[:50, :50] = 200
        mask.apply(frame2)  # second frame, should not mask yet (history < 2 won't happen
        # actually window_size=1 means deque maxlen=1, so len(history) never reaches 2
        # let's just ensure it doesn't crash
        result, regions = mask.apply(frame)
        # With window_size=1, change_history can only hold 1 item => never >= 2
        assert regions == []

    def test_noise_threshold_0(self) -> None:
        """With threshold 0, any change frequency > 0 should mask."""
        mask = AdaptiveMask(block_size=50, window_size=3, noise_threshold=0.0)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask.apply(frame)  # first frame

        for i in range(5):
            frame2 = frame.copy()
            frame2[:50, :50] = 100 + i * 10
            result, regions = mask.apply(frame2)

        # After enough history, any changing block should be masked
        assert len(regions) > 0

    def test_noise_threshold_1(self) -> None:
        """With threshold 1.0, nothing should ever be masked (freq must exceed 1.0)."""
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=1.0)
        for i in range(20):
            frame = np.full((100, 100, 3), i * 10, dtype=np.uint8)
            _, regions = mask.apply(frame)
        assert regions == []

    def test_dimension_change_resets_state(self) -> None:
        """Changing frame dimensions should reset masking state."""
        mask = AdaptiveMask(block_size=50, window_size=5, noise_threshold=0.8)
        for i in range(10):
            frame = np.full((100, 100, 3), 128, dtype=np.uint8)
            frame[:50, :50] = i * 25
            mask.apply(frame)

        # Change dimensions
        frame_new = np.full((200, 200, 3), 128, dtype=np.uint8)
        result, regions = mask.apply(frame_new)
        assert regions == []  # reset, treats as first frame

    def test_very_large_block_returns_empty(self) -> None:
        """Block size much larger than frame => no blocks."""
        mask = AdaptiveMask(block_size=1000, window_size=5, noise_threshold=0.8)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        for _ in range(10):
            result, regions = mask.apply(frame)
        assert regions == []

    def test_single_pixel_block(self) -> None:
        """block_size=1 — every pixel is its own block."""
        mask = AdaptiveMask(block_size=1, window_size=3, noise_threshold=0.5)
        frame = np.full((5, 5, 3), 128, dtype=np.uint8)
        mask.apply(frame)
        frame2 = np.full((5, 5, 3), 200, dtype=np.uint8)
        mask.apply(frame2)
        frame3 = np.full((5, 5, 3), 50, dtype=np.uint8)
        _, regions = mask.apply(frame3)
        # All pixels changed every frame, should be masked
        assert len(regions) == 25  # 5x5 blocks


# ── Redactor edge cases ───────────────────────────────────────────────────


class TestRedactorEdgeCases:
    def test_empty_text(self) -> None:
        matches = find_pii_in_text("", BUILTIN_PATTERNS)
        assert matches == []

    def test_multiple_pii_same_type(self) -> None:
        text = "Contact john@example.com or jane@example.com"
        matches = find_pii_in_text(text, BUILTIN_PATTERNS)
        email_matches = [m for m in matches if m.pattern_name == "email"]
        assert len(email_matches) == 2

    def test_overlapping_patterns(self) -> None:
        """SSN format 123-45-6789 could partly match phone pattern."""
        text = "SSN: 123-45-6789"
        matches = find_pii_in_text(text, BUILTIN_PATTERNS)
        pattern_names = {m.pattern_name for m in matches}
        assert "ssn" in pattern_names

    def test_pattern_with_empty_regex(self) -> None:
        """A pattern with empty regex should be skipped."""
        patterns = [PiiPattern(name="empty", regex="", description="empty")]
        matches = find_pii_in_text("some text", patterns)
        assert matches == []

    def test_redact_region_at_boundary(self) -> None:
        """Region touching frame edges."""
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        region = Region(x=0, y=0, w=100, h=100)
        redacted = redact_regions(frame, [region])
        assert np.all(redacted == 0)

    def test_redact_region_partially_outside(self) -> None:
        """Region extending beyond frame bounds should be clamped."""
        frame = np.full((50, 50, 3), 200, dtype=np.uint8)
        region = Region(x=40, y=40, w=30, h=30)
        redacted = redact_regions(frame, [region])
        assert np.all(redacted[40:50, 40:50] == 0)
        assert np.all(redacted[0:30, 0:30] == 200)

    def test_redact_negative_coords(self) -> None:
        """Negative region coords should be clamped to 0."""
        frame = np.full((50, 50, 3), 200, dtype=np.uint8)
        region = Region(x=-10, y=-10, w=20, h=20)
        redacted = redact_regions(frame, [region])
        assert np.all(redacted[0:10, 0:10] == 0)

    def test_redact_multiple_regions(self) -> None:
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        regions = [Region(x=0, y=0, w=20, h=20), Region(x=50, y=50, w=20, h=20)]
        redacted = redact_regions(frame, regions)
        assert np.all(redacted[0:20, 0:20] == 0)
        assert np.all(redacted[50:70, 50:70] == 0)
        assert np.all(redacted[25:45, 25:45] == 200)

    def test_build_pattern_set_empty_types(self) -> None:
        """No types requested => empty set."""
        patterns = build_pattern_set(pii_types=[], custom_patterns=[])
        assert patterns == []

    def test_build_pattern_set_unknown_type(self) -> None:
        """Requesting unknown type => excluded (no error)."""
        patterns = build_pattern_set(pii_types=["nonexistent"], custom_patterns=[])
        assert patterns == []

    def test_address_pattern_match(self) -> None:
        text = "Located at 123 Main Street"
        matches = find_pii_in_text(text, BUILTIN_PATTERNS)
        assert any(m.pattern_name == "address" for m in matches)

    def test_credit_card_with_spaces(self) -> None:
        text = "Card: 4111 1111 1111 1111"
        matches = find_pii_in_text(text, BUILTIN_PATTERNS)
        assert any(m.pattern_name == "credit_card" for m in matches)


# ── Config edge cases ─────────────────────────────────────────────────────


class TestConfigEdgeCases:
    def test_ssim_threshold_zero(self) -> None:
        config = ComparatorConfig(ssim_threshold=0.0)
        assert config.ssim_threshold == 0.0

    def test_ssim_threshold_one(self) -> None:
        config = ComparatorConfig(ssim_threshold=1.0)
        assert config.ssim_threshold == 1.0

    def test_noise_threshold_zero(self) -> None:
        config = MaskingConfig(noise_threshold=0.0)
        assert config.noise_threshold == 0.0

    def test_noise_threshold_one(self) -> None:
        config = MaskingConfig(noise_threshold=1.0)
        assert config.noise_threshold == 1.0

    def test_block_size_1(self) -> None:
        config = MaskingConfig(block_size=1)
        assert config.block_size == 1

    def test_hash_size_1(self) -> None:
        config = HasherConfig(hash_size=1)
        assert config.hash_size == 1

    def test_min_changed_blocks_0(self) -> None:
        config = ComparatorConfig(min_changed_blocks=0)
        assert config.min_changed_blocks == 0

    def test_tile_aspect_ratio_very_small(self) -> None:
        config = HasherConfig(tile_aspect_ratio=0.01)
        assert config.tile_aspect_ratio == 0.01

    def test_concurrency_1(self) -> None:
        from peeklet.config import PipelineConfig

        config = PipelineConfig(concurrency=1)
        assert config.concurrency == 1

    def test_concurrency_0_rejected(self) -> None:
        from peeklet.config import PipelineConfig

        with pytest.raises(ValueError):
            PipelineConfig(concurrency=0)

    def test_load_config_empty_json(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text("{}")
        config = load_config(p)
        assert config == PeekletConfig()

    def test_load_config_empty_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        config = load_config(p)
        assert config == PeekletConfig()

    def test_keyframe_format_jpg(self) -> None:
        config = ExporterConfig(keyframe_format="jpg")
        assert config.keyframe_format == "jpg"

    def test_parquet_compression_options(self) -> None:
        for comp in ["snappy", "gzip", "zstd", "none"]:
            config = ExporterConfig(parquet_compression=comp)  # type: ignore[arg-type]
            assert config.parquet_compression == comp


# ── Exporter edge cases ───────────────────────────────────────────────────


class TestExporterEdgeCases:
    def test_save_keyframe_1x1(self, tmp_output: Path) -> None:
        frame = np.array([[[42, 43, 44]]], dtype=np.uint8)
        path = save_keyframe(frame, tmp_output, "tiny", fmt="png")
        assert path.exists()
        loaded = np.asarray(Image.open(path))
        assert loaded.shape[0] == 1

    def test_save_keyframe_nested_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "c"
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        path = save_keyframe(frame, out, "nested", fmt="png")
        assert path.exists()

    def test_manifest_flush_empty(self, tmp_output: Path) -> None:
        """Flushing with no rows should not create a file."""
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        writer.flush()
        assert not (tmp_output / "manifest.parquet").exists()

    def test_manifest_clear(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        result = FrameResult(
            frame_id="f1",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd",
            frame_width=100,
            frame_height=100,
        )
        writer.append(result)
        writer.clear()
        writer.flush()
        assert not (tmp_output / "manifest.parquet").exists()

    def test_manifest_gzip_compression(self, tmp_output: Path) -> None:
        writer = ManifestWriter(tmp_output / "manifest.parquet", compression="gzip")
        result = FrameResult(
            frame_id="f1",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd",
            frame_width=100,
            frame_height=100,
        )
        writer.append(result)
        writer.flush()
        meta = pq.read_metadata(tmp_output / "manifest.parquet")
        assert meta.row_group(0).column(0).compression == "GZIP"

    def test_manifest_all_nullable_fields_none(self, tmp_output: Path) -> None:
        """All optional fields as None should serialize cleanly."""
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        result = FrameResult(
            frame_id="f1",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="0" * 16,
            frame_width=100,
            frame_height=100,
        )
        writer.append(result)
        writer.flush()
        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.num_rows == 1
        assert table.column("ssim_score")[0].as_py() is None
        assert table.column("asset_path")[0].as_py() is None

    def test_manifest_overwrite(self, tmp_output: Path) -> None:
        """Flushing twice overwrites the manifest file."""
        writer = ManifestWriter(tmp_output / "manifest.parquet")
        r1 = FrameResult(
            frame_id="f1",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd",
            frame_width=100,
            frame_height=100,
        )
        writer.append(r1)
        writer.flush()

        writer2 = ManifestWriter(tmp_output / "manifest.parquet")
        r2 = FrameResult(
            frame_id="f2",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="efgh",
            frame_width=200,
            frame_height=200,
        )
        writer2.append(r2)
        writer2.flush()

        table = pq.read_table(tmp_output / "manifest.parquet")
        assert table.num_rows == 1
        assert table.column("frame_id")[0].as_py() == "f2"


# ── Types edge cases ──────────────────────────────────────────────────────


class TestTypesEdgeCases:
    def test_region_zero_area(self) -> None:
        r = Region(x=5, y=5, w=0, h=0)
        assert r.area == 0

    def test_region_to_dict(self) -> None:
        r = Region(x=1, y=2, w=3, h=4)
        assert r.to_dict() == {"x": 1, "y": 2, "w": 3, "h": 4}

    def test_event_type_values(self) -> None:
        assert EventType.KEYFRAME.value == "KEYFRAME"
        assert EventType.SKIPPED.value == "SKIPPED"

    def test_frame_result_defaults(self) -> None:
        r = FrameResult(
            frame_id="test",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="0" * 16,
            frame_width=100,
            frame_height=100,
        )
        assert r.timestamp is None
        assert r.ssim_score is None
        assert r.changed_regions is None
        assert r.source_video is None
        assert r.video_timestamp is None
        assert r.change_magnitude is None

    def test_region_frozen(self) -> None:
        r = Region(x=0, y=0, w=10, h=10)
        with pytest.raises(AttributeError):
            r.x = 5  # type: ignore[misc]


# ── Pipeline integration edge cases ───────────────────────────────────────


class TestPipelineEdgeCases:
    @pytest.fixture
    def config(self, tmp_output: Path) -> PeekletConfig:
        return PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85, "min_changed_blocks": 3},
            }
        )

    def test_single_frame_is_keyframe(self, config: PeekletConfig) -> None:
        pipeline = Pipeline(config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_id="only_frame")
        assert result.is_keyframe is True
        assert "First frame" in (result.visual_reason or "")

    def test_dimension_change_triggers_keyframe(
        self, config: PeekletConfig, tmp_output: Path
    ) -> None:
        pipeline = Pipeline(config)
        frame_a = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame_b = np.full((200, 200, 3), 128, dtype=np.uint8)

        pipeline.process_frame(frame_a, frame_id="f0")
        result = pipeline.process_frame(frame_b, frame_id="f1")

        assert result.is_keyframe is True
        assert "Dimension change" in (result.visual_reason or "")

    def test_alternating_dimensions(self, config: PeekletConfig, tmp_output: Path) -> None:
        """Rapidly alternating frame sizes should each be a keyframe."""
        pipeline = Pipeline(config)
        small = np.full((64, 64, 3), 128, dtype=np.uint8)
        large = np.full((128, 128, 3), 128, dtype=np.uint8)

        results = []
        for i in range(6):
            frame = small if i % 2 == 0 else large
            result = pipeline.process_frame(frame, frame_id=f"f{i}")
            results.append(result)

        # First frame and every dimension change
        assert all(r.is_keyframe for r in results)

    def test_very_similar_frames_skipped(self, config: PeekletConfig, tmp_output: Path) -> None:
        """Tiny per-pixel noise should be skipped."""
        pipeline = Pipeline(config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        noisy = frame.copy()
        noisy[50, 50] = [129, 128, 128]  # single pixel +1

        pipeline.process_frame(frame, frame_id="f0")
        result = pipeline.process_frame(noisy, frame_id="f1")
        assert result.is_keyframe is False

    def test_prev_keyframe_links(self, config: PeekletConfig, tmp_output: Path) -> None:
        """Skipped frames should reference the previous keyframe."""
        pipeline = Pipeline(config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        r0 = pipeline.process_frame(frame, frame_id="f0")
        r1 = pipeline.process_frame(frame, frame_id="f1")

        assert r0.is_keyframe
        assert not r1.is_keyframe
        assert r1.prev_keyframe_id == "f0"
        assert r1.prev_keyframe_path is not None

    def test_finalize_creates_manifest(self, config: PeekletConfig, tmp_output: Path) -> None:
        pipeline = Pipeline(config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        pipeline.process_frame(frame, frame_id="f0")
        pipeline.finalize()
        assert (tmp_output / "manifest.parquet").exists()

    def test_finalize_empty_pipeline(self, config: PeekletConfig, tmp_output: Path) -> None:
        """Finalizing without processing any frames should not crash."""
        pipeline = Pipeline(config)
        pipeline.finalize()
        # No manifest written (empty)
        assert not (tmp_output / "manifest.parquet").exists()

    def test_many_identical_frames(self, config: PeekletConfig, tmp_output: Path) -> None:
        """100 identical frames: first is keyframe, rest skipped."""
        pipeline = Pipeline(config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        results = [pipeline.process_frame(frame, frame_id=f"f{i}") for i in range(100)]
        assert results[0].is_keyframe
        assert all(not r.is_keyframe for r in results[1:])

    def test_metadata_passthrough(self, config: PeekletConfig, tmp_output: Path) -> None:
        """All metadata fields should pass through to result."""
        from datetime import datetime, timezone

        pipeline = Pipeline(config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = pipeline.process_frame(
            frame,
            frame_id="f0",
            timestamp=ts,
            app_name="TestApp",
            window_title="Test Window",
            source_format="png",
        )
        assert result.timestamp == ts
        assert result.app_name == "TestApp"
        assert result.window_title == "Test Window"
        assert result.source_format == "png"

    def test_ssim_none_on_hash_match_skip(self, config: PeekletConfig, tmp_output: Path) -> None:
        """When skipped at hash gate, ssim_score should be None (cascade optimization)."""
        pipeline = Pipeline(config)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        pipeline.process_frame(frame, frame_id="f0")
        result = pipeline.process_frame(frame, frame_id="f1")
        assert not result.is_keyframe
        assert result.ssim_score is None

    def test_ssim_computed_when_hash_differs(self, config: PeekletConfig, tmp_output: Path) -> None:
        """When hash differs, SSIM should be computed."""
        pipeline = Pipeline(config)
        frame_a = np.full((100, 100, 3), 50, dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 200, dtype=np.uint8)

        pipeline.process_frame(frame_a, frame_id="f0")
        result = pipeline.process_frame(frame_b, frame_id="f1")
        assert result.ssim_score is not None


class TestPipelineTiledEdgeCases:
    @pytest.fixture
    def tiled_config(self, tmp_output: Path) -> PeekletConfig:
        return PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85, "min_changed_blocks": 2},
                "hasher": {"tile_aspect_ratio": 1.5},
            }
        )

    def test_tall_to_normal_dimension_change(
        self, tiled_config: PeekletConfig, tmp_output: Path
    ) -> None:
        """Transition from tall image to normal should be a keyframe."""
        pipeline = Pipeline(tiled_config)
        tall = np.full((600, 200, 3), 128, dtype=np.uint8)
        normal = np.full((200, 200, 3), 128, dtype=np.uint8)

        pipeline.process_frame(tall, frame_id="tall")
        result = pipeline.process_frame(normal, frame_id="normal")
        assert result.is_keyframe
        assert "Dimension change" in (result.visual_reason or "")

    def test_identical_tall_frames_skipped(
        self, tiled_config: PeekletConfig, tmp_output: Path
    ) -> None:
        pipeline = Pipeline(tiled_config)
        tall = np.full((600, 200, 3), 220, dtype=np.uint8)
        pipeline.process_frame(tall, frame_id="f0")
        result = pipeline.process_frame(tall, frame_id="f1")
        assert not result.is_keyframe

    def test_change_in_last_tile_detected(
        self, tiled_config: PeekletConfig, tmp_output: Path
    ) -> None:
        """A change only in the last tile of a tall image should be detected."""
        pipeline = Pipeline(tiled_config)
        tall_a = np.full((600, 200, 3), 220, dtype=np.uint8)
        tall_b = tall_a.copy()
        # Change in last tile (bottom 200 rows)
        tall_b[500:580, 50:150] = [10, 10, 10]

        pipeline.process_frame(tall_a, frame_id="f0")
        result = pipeline.process_frame(tall_b, frame_id="f1")
        assert result.is_keyframe

