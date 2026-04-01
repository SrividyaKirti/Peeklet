"""Tests for shared type definitions."""

from peeklet.utils.types import EventType, FrameMeta, FrameResult, Region


class TestRegion:
    def test_create_region(self) -> None:
        r = Region(x=10, y=20, w=100, h=50)
        assert r.x == 10
        assert r.y == 20
        assert r.w == 100
        assert r.h == 50

    def test_region_to_dict(self) -> None:
        r = Region(x=10, y=20, w=100, h=50)
        assert r.to_dict() == {"x": 10, "y": 20, "w": 100, "h": 50}

    def test_region_area(self) -> None:
        r = Region(x=0, y=0, w=100, h=50)
        assert r.area == 5000


class TestEventType:
    def test_keyframe_value(self) -> None:
        assert EventType.KEYFRAME.value == "KEYFRAME"

    def test_skipped_value(self) -> None:
        assert EventType.SKIPPED.value == "SKIPPED"


class TestFrameMeta:
    def test_create_with_defaults(self) -> None:
        meta = FrameMeta(frame_id="frame_001")
        assert meta.frame_id == "frame_001"
        assert meta.timestamp is None
        assert meta.app_name is None
        assert meta.window_title is None
        assert meta.source_format is None

    def test_create_with_all_fields(self) -> None:
        from datetime import datetime, timezone

        ts = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        meta = FrameMeta(
            frame_id="frame_001",
            timestamp=ts,
            app_name="Chrome",
            window_title="Google - Chrome",
            source_format="png",
        )
        assert meta.app_name == "Chrome"
        assert meta.timestamp == ts


class TestFrameResult:
    def test_create_keyframe_result(self) -> None:
        result = FrameResult(
            frame_id="frame_001",
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abcd1234",
            ssim_score=0.62,
            change_score=0.38,
            changed_pct=34.0,
            changed_regions=[Region(x=100, y=200, w=300, h=150)],
            adaptive_mask=[Region(x=0, y=0, w=32, h=32)],
            frame_width=1920,
            frame_height=1080,
            pii_detected=False,
        )
        assert result.is_keyframe is True
        assert result.event_type == EventType.KEYFRAME
        assert len(result.changed_regions) == 1

    def test_create_skipped_result(self) -> None:
        result = FrameResult(
            frame_id="frame_002",
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash="abcd1234",
            frame_width=1920,
            frame_height=1080,
        )
        assert result.is_keyframe is False
        assert result.ssim_score is None
        assert result.changed_regions is None
        assert result.asset_path is None
