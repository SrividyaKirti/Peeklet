"""Tests for forced timestamp integration in video pipeline."""

from __future__ import annotations

from peeklet.core.video import _merge_trigger_type, _should_force_keyframe


class TestShouldForceKeyframe:
    def test_within_tolerance(self) -> None:
        forced = [5.0, 12.0, 20.0]
        assert _should_force_keyframe(5.3, forced, tolerance=0.5) is True

    def test_outside_tolerance(self) -> None:
        forced = [5.0, 12.0, 20.0]
        assert _should_force_keyframe(6.0, forced, tolerance=0.5) is False

    def test_empty_forced_list(self) -> None:
        assert _should_force_keyframe(5.0, [], tolerance=0.5) is False

    def test_exact_match(self) -> None:
        forced = [10.0]
        assert _should_force_keyframe(10.0, forced, tolerance=0.5) is True

    def test_inclusive_upper_boundary(self) -> None:
        """A timestamp exactly at +tolerance is included."""
        forced = [10.0]
        assert _should_force_keyframe(10.5, forced, tolerance=0.5) is True

    def test_just_outside_upper_boundary(self) -> None:
        """A timestamp just past +tolerance is excluded."""
        forced = [10.0]
        assert _should_force_keyframe(10.6, forced, tolerance=0.5) is False

    def test_inclusive_lower_boundary(self) -> None:
        """A timestamp exactly at -tolerance is included (abs symmetry)."""
        forced = [10.0]
        assert _should_force_keyframe(9.5, forced, tolerance=0.5) is True


class TestMergeTriggerType:
    def test_visual_only(self) -> None:
        assert _merge_trigger_type(is_visual=True, is_transcript=False) == "visual_change"

    def test_transcript_only(self) -> None:
        assert _merge_trigger_type(is_visual=False, is_transcript=True) == "transcript_trigger"

    def test_both(self) -> None:
        assert _merge_trigger_type(is_visual=True, is_transcript=True) == "both"

    def test_neither(self) -> None:
        assert _merge_trigger_type(is_visual=False, is_transcript=False) is None
