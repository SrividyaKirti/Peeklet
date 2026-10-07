"""Tests for shared type definitions."""

from peeklet.utils.types import Moment, Region


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
    from peeklet.utils.types import Moment

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
    from peeklet.utils.types import Moment

    m = Moment(
        timestamp=232.0,
        visual_context_goal="Analytics interaction breakdown",
        textual_anchor="ACTION ITEM: Fix missing assistant prompt",
        downstream_utility="Action item flagged by meeting tool",
        source="anchor",
    )
    assert m.source == "anchor"
