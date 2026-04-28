"""Unit tests for the LLM adapter module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_segments():
    from peeklet.core.audio import TranscriptSegment

    return [
        TranscriptSegment(
            start=0.0,
            end=3.5,
            text="Hey everyone, today I'll show you the dashboard.",
            speaker="Alice",
        ),
        TranscriptSegment(
            start=3.5, end=8.2, text="Let me start by signing in here.", speaker="Bob"
        ),
        TranscriptSegment(
            start=8.2,
            end=12.1,
            text="Okay, this is the main view after login.",
            speaker="Alice",
        ),
    ]


def test_format_transcript_includes_timestamps_and_text():
    from peeklet.core.llm import format_transcript_for_llm

    segments = _make_segments()
    formatted = format_transcript_for_llm(segments)

    assert "[0.0 - 3.5]" in formatted
    assert "**Alice**:" in formatted
    assert "Hey everyone" in formatted
    assert "[3.5 - 8.2]" in formatted
    assert "**Bob**:" in formatted
    assert formatted.count("\n") >= len(segments) - 1


def test_format_transcript_omits_speaker_when_none():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.llm import format_transcript_for_llm

    segments = [TranscriptSegment(start=0.0, end=3.0, text="Hello")]
    formatted = format_transcript_for_llm(segments)
    assert "[0.0 - 3.0] Hello" in formatted
    assert "**" not in formatted


def test_parse_moments_strips_markdown_fences():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        "```json\n"
        '[{"timestamp": 12.5, "visual_context_goal": "X",'
        ' "textual_anchor": "Y", "downstream_utility": "Z"}]'
        "\n```"
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    assert len(moments) == 1
    assert moments[0].timestamp == 12.5
    assert moments[0].visual_context_goal == "X"
    assert moments[0].textual_anchor == "Y"
    assert moments[0].downstream_utility == "Z"


def test_parse_moments_drops_out_of_range_timestamps():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        "["
        '{"timestamp": 5.0, "visual_context_goal": "ok",'
        ' "textual_anchor": "t", "downstream_utility": "u"},'
        ' {"timestamp": 999.0, "visual_context_goal": "past end",'
        ' "textual_anchor": "t", "downstream_utility": "u"},'
        ' {"timestamp": -1.0, "visual_context_goal": "negative",'
        ' "textual_anchor": "t", "downstream_utility": "u"}'
        "]"
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    timestamps = [m.timestamp for m in moments]
    assert timestamps == [5.0]


def test_parse_moments_sorted_by_timestamp():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        "["
        '{"timestamp": 30.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"},'
        ' {"timestamp": 5.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"},'
        ' {"timestamp": 15.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"}'
        "]"
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    assert [m.timestamp for m in moments] == [5.0, 15.0, 30.0]


def test_parse_moments_raises_on_unparseable_after_strip():
    from peeklet.core.llm import LLMResponseError, _parse_moments_json

    with pytest.raises(LLMResponseError):
        _parse_moments_json("this is not json at all", video_duration=60.0)


def test_parse_moments_raises_on_missing_required_keys():
    from peeklet.core.llm import LLMResponseError, _parse_moments_json

    raw = '[{"timestamp": 5.0, "visual_context_goal": "no anchor or utility"}]'
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


def test_parse_moments_strips_preamble_before_json():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        "Here is the JSON you asked for:\n"
        "```json\n"
        '[{"timestamp": 7.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"}]'
        "\n```"
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    assert len(moments) == 1
    assert moments[0].timestamp == 7.0


def test_parse_moments_handles_nested_brackets_in_strings():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        '[{"timestamp": 1.0, "visual_context_goal": "uses [brackets] in goal",'
        ' "textual_anchor": "t", "downstream_utility": "u"}]'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    assert len(moments) == 1
    assert moments[0].visual_context_goal == "uses [brackets] in goal"


def test_parse_moments_raises_when_no_array():
    import pytest

    from peeklet.core.llm import LLMResponseError, _parse_moments_json

    with pytest.raises(LLMResponseError, match="no JSON array"):
        _parse_moments_json("just some prose, no array", video_duration=60.0)


def test_anthropic_client_pick_moments_calls_sdk_and_parses_response(monkeypatch):
    from peeklet.core import llm_anthropic
    from peeklet.utils.types import Moment

    fake_response = MagicMock()
    _json = (
        '[{"timestamp": 7.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"}]'
    )
    fake_response.content = [MagicMock(text=_json)]

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    monkeypatch.setattr(llm_anthropic, "anthropic", fake_anthropic)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = llm_anthropic.AnthropicClient(model="claude-haiku-4-5")
    segments = _make_segments()
    moments = client.pick_moments(segments, video_duration=60.0)

    assert moments == [
        Moment(
            timestamp=7.0,
            visual_context_goal="c",
            textual_anchor="t",
            downstream_utility="u",
        )
    ]
    fake_anthropic.Anthropic.assert_called_once()
    fake_client.messages.create.assert_called_once()
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5"
    assert "system" in call_kwargs
    assert call_kwargs["messages"][0]["role"] == "user"


def test_anthropic_client_retries_once_on_unparseable(monkeypatch):
    from peeklet.core import llm_anthropic

    bad_response = MagicMock()
    bad_response.content = [MagicMock(text="not json")]
    good_response = MagicMock()
    _json = (
        '[{"timestamp": 1.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"}]'
    )
    good_response.content = [MagicMock(text=_json)]

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [bad_response, good_response]

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    monkeypatch.setattr(llm_anthropic, "anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = llm_anthropic.AnthropicClient(model="claude-haiku-4-5")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert len(moments) == 1
    assert fake_client.messages.create.call_count == 2


def test_anthropic_client_raises_after_two_unparseable(monkeypatch):
    from peeklet.core import llm_anthropic
    from peeklet.core.llm import LLMResponseError

    bad_response = MagicMock()
    bad_response.content = [MagicMock(text="garbage")]

    fake_client = MagicMock()
    fake_client.messages.create.return_value = bad_response

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    monkeypatch.setattr(llm_anthropic, "anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = llm_anthropic.AnthropicClient(model="claude-haiku-4-5")
    with pytest.raises(LLMResponseError):
        client.pick_moments(_make_segments(), video_duration=60.0)

    assert fake_client.messages.create.call_count == 2


def test_openai_client_pick_moments_calls_sdk_and_parses_response(monkeypatch):
    from peeklet.core import llm_openai
    from peeklet.utils.types import Moment

    fake_message = MagicMock()
    fake_message.content = (
        '[{"timestamp": 11.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"}]'
    )
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openai, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = llm_openai.OpenAIClient(model="gpt-4o-mini")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert moments == [
        Moment(
            timestamp=11.0,
            visual_context_goal="c",
            textual_anchor="t",
            downstream_utility="u",
        )
    ]
    fake_client.chat.completions.create.assert_called_once()
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"


def test_openai_client_passes_base_url_when_set(monkeypatch):
    from peeklet.core import llm_openai

    fake_openai = MagicMock()
    monkeypatch.setattr(llm_openai, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    llm_openai.OpenAIClient(model="llama3")

    fake_openai.OpenAI.assert_called_once()
    call_kwargs = fake_openai.OpenAI.call_args.kwargs
    assert call_kwargs.get("base_url") == "http://localhost:11434/v1"


def test_openai_client_retries_once_on_unparseable(monkeypatch):
    from peeklet.core import llm_openai

    bad_message = MagicMock(content="not json")
    _json = (
        '[{"timestamp": 1.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"}]'
    )
    good_message = MagicMock(content=_json)
    bad_response = MagicMock(choices=[MagicMock(message=bad_message)])
    good_response = MagicMock(choices=[MagicMock(message=good_message)])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [bad_response, good_response]

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openai, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = llm_openai.OpenAIClient(model="gpt-4o-mini")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert len(moments) == 1
    assert fake_client.chat.completions.create.call_count == 2


def test_openrouter_client_constructs_with_hardcoded_base_url_and_headers(monkeypatch):
    from peeklet.core import llm_openrouter

    fake_openai = MagicMock()
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")

    fake_openai.OpenAI.assert_called_once()
    call_kwargs = fake_openai.OpenAI.call_args.kwargs
    assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert call_kwargs["api_key"] == "or-test-key"
    assert call_kwargs["default_headers"]["HTTP-Referer"] == (
        "https://github.com/SrividyaKirti/Peeklet"
    )
    assert call_kwargs["default_headers"]["X-Title"] == "Peeklet"


def test_openrouter_client_raises_when_sdk_missing(monkeypatch):
    from peeklet.core import llm_openrouter

    monkeypatch.setattr(llm_openrouter, "openai", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    with pytest.raises(RuntimeError, match=r"\[demo\] extra"):
        llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")


def test_openrouter_client_pick_moments_calls_sdk_and_parses_response(monkeypatch):
    from peeklet.core import llm_openrouter
    from peeklet.utils.types import Moment

    fake_message = MagicMock()
    fake_message.content = (
        '[{"timestamp": 13.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"}]'
    )
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    client = llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert moments == [
        Moment(
            timestamp=13.0,
            visual_context_goal="c",
            textual_anchor="t",
            downstream_utility="u",
        )
    ]
    fake_client.chat.completions.create.assert_called_once()
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-3.5-sonnet"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"


def test_openrouter_client_retries_once_on_unparseable(monkeypatch):
    from peeklet.core import llm_openrouter

    bad_message = MagicMock(content="not json")
    _json = (
        '[{"timestamp": 1.0, "visual_context_goal": "c",'
        ' "textual_anchor": "t", "downstream_utility": "u"}]'
    )
    good_message = MagicMock(content=_json)
    bad_response = MagicMock(choices=[MagicMock(message=bad_message)])
    good_response = MagicMock(choices=[MagicMock(message=good_message)])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [bad_response, good_response]

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    client = llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert len(moments) == 1
    assert fake_client.chat.completions.create.call_count == 2


def test_openrouter_client_raises_after_two_unparseable(monkeypatch):
    from peeklet.core import llm_openrouter
    from peeklet.core.llm import LLMResponseError

    bad_message = MagicMock(content="garbage")
    bad_response = MagicMock(choices=[MagicMock(message=bad_message)])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = bad_response

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    client = llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")
    with pytest.raises(LLMResponseError):
        client.pick_moments(_make_segments(), video_duration=60.0)

    assert fake_client.chat.completions.create.call_count == 2


def test_build_llm_client_openrouter_missing_key_raises(monkeypatch):
    from peeklet.core.llm import build_llm_client

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_llm_client(provider="openrouter", model="anthropic/claude-3.5-sonnet")


def test_build_llm_client_openrouter_returns_openrouter_client(monkeypatch):
    from peeklet.core import llm_openrouter
    from peeklet.core.llm import build_llm_client

    fake_openai = MagicMock()
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    client = build_llm_client(provider="openrouter", model="anthropic/claude-3.5-sonnet")

    assert isinstance(client, llm_openrouter.OpenRouterClient)


def test_format_anchors_empty_list_returns_sentinel():
    from peeklet.core.llm import format_anchors_for_llm

    assert format_anchors_for_llm([]) == "None — no guaranteed anchors in this video."


def test_format_anchors_renders_labels_with_decimal_seconds():
    from peeklet.core.llm import format_anchors_for_llm
    from peeklet.utils.types import Moment

    anchors = [
        Moment(
            timestamp=12.5,
            visual_context_goal="The risk-score modal",
            textual_anchor="**ACTION ITEM: ...",
            downstream_utility="...",
            source="anchor",
        ),
        Moment(
            timestamp=140.7,
            visual_context_goal="The Tasks page filter dropdown",
            textual_anchor="**ACTION ITEM: ...",
            downstream_utility="...",
            source="anchor",
        ),
    ]

    rendered = format_anchors_for_llm(anchors)
    assert rendered == ("[12.5] The risk-score modal\n[140.7] The Tasks page filter dropdown")


def test_format_anchors_unlabeled_anchor_degrades():
    from peeklet.core.llm import format_anchors_for_llm
    from peeklet.utils.types import Moment

    anchors = [
        Moment(
            timestamp=42.0,
            visual_context_goal="",
            textual_anchor="",
            downstream_utility="",
            source="anchor",
        ),
    ]

    assert format_anchors_for_llm(anchors) == "[42.0] (unlabeled anchor)"


def test_system_prompt_has_anchor_avoidance_and_placeholder():
    from peeklet.core.llm import SYSTEM_PROMPT

    # Old behavior must be gone — this was the root cause.
    assert "MUST include a moment at or near each such timestamp" not in SYSTEM_PROMPT
    assert "Action Item Anchors" not in SYSTEM_PROMPT

    # New behavior must be present.
    assert "Guaranteed Anchors" in SYSTEM_PROMPT
    assert "Do NOT pick any moment within ±10 seconds" in SYSTEM_PROMPT
    assert "{anchor_list}" in SYSTEM_PROMPT
    assert "Silence is a valid answer" in SYSTEM_PROMPT
