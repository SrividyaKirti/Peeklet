"""Tests for shared type definitions."""

from peeklet.types import (
    Checkpoint,
    Entry,
    Line,
    Moment,
    Region,
    Screen,
    ScreenJudgment,
    format_hms,
)


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


def test_moment_basic():
    seg = Moment(
        timestamp=12.5,
        visual_context_goal="Settings page open",
        textual_anchor="speaker says 'show settings'",
        downstream_utility="Capture the settings panel",
    )
    assert seg.timestamp == 12.5
    assert seg.visual_context_goal == "Settings page open"
    assert seg.textual_anchor == "speaker says 'show settings'"
    assert seg.downstream_utility == "Capture the settings panel"


def test_moment_is_frozen():
    import pytest

    seg = Moment(
        timestamp=0.0,
        visual_context_goal="x",
        textual_anchor="y",
        downstream_utility="z",
    )
    with pytest.raises((AttributeError, TypeError)):
        seg.timestamp = 1.0  # type: ignore[misc]


def test_moment_new_fields():
    from peeklet.types import Moment

    m = Moment(
        timestamp=12.5,
        visual_context_goal="Dashboard with cost breakdown",
        textual_anchor="Let me show you the estimated cost",
        downstream_utility="To extract specific cost values",
    )
    assert m.timestamp == 12.5
    assert m.visual_context_goal == "Dashboard with cost breakdown"
    assert m.textual_anchor == "Let me show you the estimated cost"
    assert m.downstream_utility == "To extract specific cost values"
    assert m.source == "llm"  # default


def test_moment_anchor_source():
    from peeklet.types import Moment

    m = Moment(
        timestamp=232.0,
        visual_context_goal="Analytics interaction breakdown",
        textual_anchor="ACTION ITEM: Fix missing assistant prompt",
        downstream_utility="Action item flagged by meeting tool",
        source="anchor",
    )
    assert m.source == "anchor"


def test_format_hms_truncates_to_whole_seconds() -> None:
    assert format_hms(0) == "00:00:00"
    assert format_hms(134.9) == "00:02:14"
    assert format_hms(3725.0) == "01:02:05"


def test_checkpoint_defaults() -> None:
    cp = Checkpoint(t=3.0, kind="speech_onset")
    assert cp.label == ""


def test_screen_occurrences_default_empty() -> None:
    s = Screen(id="S1", image_jpeg=b"x", frame_t=1.0, ocr_text="", word_count=0, first_change=1.0)
    assert s.occurrences == []


def test_entry_to_dict_full() -> None:
    e = Entry(
        timestamp=134.2,
        lines=(Line(134.2, 137.0, "Look left.", "Alice"), Line(137.0, 140.0, "See it?", None)),
        image="frame_134.jpg",
        visual_context="Policy page.",
        action_items=("Fix tag check", "Add test"),
        screen_id="S1",
    )
    assert e.to_dict() == {
        "timestamp": "00:02:14",
        "transcript": "Alice: Look left.\nSpeaker: See it?",
        "image": "frame_134.jpg",
        "visual_context": "Policy page.",
        "action_item": "Fix tag check; Add test",
    }


def test_entry_to_dict_without_image_omits_visual_context() -> None:
    e = Entry(timestamp=5.0, lines=(Line(5.0, 6.0, "Hi", "Bob"),), visual_context="ignored")
    assert e.to_dict() == {"timestamp": "00:00:05", "transcript": "Bob: Hi", "image": None}


def test_screen_judgment_is_frozen() -> None:
    import dataclasses

    import pytest

    j = ScreenJudgment(include=True, reason="r", visual_context="v")
    assert j.include
    with pytest.raises(dataclasses.FrozenInstanceError):
        j.include = False  # type: ignore[misc]
