"""Unit tests for the LLM adapter module."""

from __future__ import annotations

import pytest


def _make_segments():
    from peeklet.core.audio import TranscriptSegment

    return [
        TranscriptSegment(
            start=0.0, end=3.5, text="Hey everyone, today I'll show you the dashboard."
        ),
        TranscriptSegment(start=3.5, end=8.2, text="Let me start by signing in here."),
        TranscriptSegment(start=8.2, end=12.1, text="Okay, this is the main view after login."),
    ]


def test_format_transcript_includes_timestamps_and_text():
    from peeklet.core.llm import format_transcript_for_llm

    segments = _make_segments()
    formatted = format_transcript_for_llm(segments)

    assert "[0.0 - 3.5]" in formatted
    assert "Hey everyone" in formatted
    assert "[3.5 - 8.2]" in formatted
    assert "Let me start by signing in here." in formatted
    # One line per segment
    assert formatted.count("\n") >= len(segments) - 1


def test_parse_moments_strips_markdown_fences():
    from peeklet.core.llm import _parse_moments_json

    raw = '```json\n[{"timestamp": 12.5, "caption": "X", "reason": "Y"}]\n```'
    moments = _parse_moments_json(raw, video_duration=60.0)

    assert len(moments) == 1
    assert moments[0].timestamp == 12.5
    assert moments[0].caption == "X"
    assert moments[0].reason == "Y"


def test_parse_moments_drops_out_of_range_timestamps():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        '[{"timestamp": 5.0, "caption": "ok", "reason": "r"},'
        ' {"timestamp": 999.0, "caption": "past end", "reason": "r"},'
        ' {"timestamp": -1.0, "caption": "negative", "reason": "r"}]'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    timestamps = [m.timestamp for m in moments]
    assert timestamps == [5.0]


def test_parse_moments_sorted_by_timestamp():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        '[{"timestamp": 30.0, "caption": "c", "reason": "r"},'
        ' {"timestamp": 5.0, "caption": "c", "reason": "r"},'
        ' {"timestamp": 15.0, "caption": "c", "reason": "r"}]'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    assert [m.timestamp for m in moments] == [5.0, 15.0, 30.0]


def test_parse_moments_raises_on_unparseable_after_strip():
    from peeklet.core.llm import LLMResponseError, _parse_moments_json

    with pytest.raises(LLMResponseError):
        _parse_moments_json("this is not json at all", video_duration=60.0)


def test_parse_moments_raises_on_missing_required_keys():
    from peeklet.core.llm import LLMResponseError, _parse_moments_json

    raw = '[{"timestamp": 5.0, "caption": "no reason"}]'
    with pytest.raises(LLMResponseError):
        _parse_moments_json(raw, video_duration=60.0)


def test_build_llm_client_unknown_provider_raises():
    from peeklet.core.llm import build_llm_client

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm_client(provider="cohere", model="some-model")


def test_build_llm_client_anthropic_missing_key_raises(monkeypatch):
    from peeklet.core.llm import build_llm_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_llm_client(provider="anthropic", model="claude-haiku-4-5")


def test_build_llm_client_openai_missing_key_raises(monkeypatch):
    from peeklet.core.llm import build_llm_client

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_llm_client(provider="openai", model="gpt-4o-mini")
