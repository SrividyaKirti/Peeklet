"""Tests for transcript trigger word detection."""

from __future__ import annotations

import pytest

from peeklet.core.audio import TranscriptSegment
from peeklet.core.transcript_trigger import detect_triggers


class TestDetectTriggers:
    def test_high_plus_medium_triggers(self) -> None:
        """High-signal noun + medium-signal verb = trigger."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Now look at this dashboard"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 1
        assert triggers[0].timestamp == pytest.approx(1.5)
        assert triggers[0].segment == segments[0]

    def test_two_high_signal_triggers(self) -> None:
        """Two high-signal words alone = trigger."""
        segments = [
            TranscriptSegment(start=5.0, end=8.0, text="The chart and graph show growth"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 1

    def test_single_medium_no_trigger(self) -> None:
        """Single medium word alone = no trigger (too noisy)."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="I see what you mean"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 0

    def test_no_keywords_no_trigger(self) -> None:
        """No keywords at all = no trigger."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Welcome to the presentation"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 0

    def test_empty_segments(self) -> None:
        triggers = detect_triggers([])
        assert triggers == []

    def test_multiple_segments_multiple_triggers(self) -> None:
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Hello everyone"),
            TranscriptSegment(start=5.0, end=8.0, text="Click this button here"),
            TranscriptSegment(start=10.0, end=13.0, text="The results are in"),
            TranscriptSegment(start=15.0, end=18.0, text="Notice the sidebar menu"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 2
        timestamps = [t.timestamp for t in triggers]
        assert pytest.approx(6.5) in timestamps
        assert pytest.approx(16.5) in timestamps

    def test_trigger_timestamp_is_segment_midpoint(self) -> None:
        segments = [
            TranscriptSegment(start=10.0, end=20.0, text="Look at this chart"),
        ]
        triggers = detect_triggers(segments)
        assert triggers[0].timestamp == pytest.approx(15.0)

    def test_case_insensitive(self) -> None:
        """Keywords should match regardless of case."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="CLICK THIS BUTTON"),
        ]
        triggers = detect_triggers(segments)
        assert len(triggers) == 1
