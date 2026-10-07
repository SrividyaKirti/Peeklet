"""Tests for the LLM judge layer (all SDKs mocked)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from peeklet.llm import base
from peeklet.llm.base import (
    JudgeRequest,
    JudgmentCache,
    LLMResponseError,
    build_user_text,
    has_credentials,
    judge_screens,
    parse_judgment,
)
from peeklet.types import ScreenJudgment

if TYPE_CHECKING:
    from pathlib import Path

REQ = JudgeRequest(
    "S1",
    b"\xff\xd8jpeg",
    "Billing Invoices",
    ("[00:00:01] A: open billing",),
    ("Fix invoice total",),
)


class StubClient:
    provider = "stub"
    model = "m"

    def __init__(self, result: ScreenJudgment | Exception) -> None:
        self.result = result
        self.calls = 0

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_user_text_contains_all_inputs() -> None:
    text = build_user_text("OCR WORDS", ["[00:00:01] A: hi"], ["Do X"])
    assert "OCR WORDS" in text and "[00:00:01] A: hi" in text and "Do X" in text


def test_parse_judgment_plain_and_fenced() -> None:
    raw = '{"include": true, "reason": "r", "visual_context": "v"}'
    assert parse_judgment(raw) == ScreenJudgment(True, "r", "v")
    assert parse_judgment(f"```json\n{raw}\n```") == ScreenJudgment(True, "r", "v")


@pytest.mark.parametrize("raw", ["not json", '{"include": "yes"}', '{"reason": "r"}', "[]"])
def test_parse_judgment_rejects_bad_output(raw: str) -> None:
    with pytest.raises(LLMResponseError):
        parse_judgment(raw)


def test_has_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert not has_credentials("anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    assert has_credentials("anthropic")
    assert not has_credentials("openai")


def test_cache_round_trip_and_key_sensitivity(tmp_path: Path) -> None:
    cache = JudgmentCache(tmp_path, "anthropic", "claude-haiku-4-5")
    j = ScreenJudgment(True, "r", "v")
    cache.put(REQ, j)
    assert cache.get(REQ) == j
    changed = JudgeRequest("S1", REQ.image_jpeg, "different ocr", REQ.lines, REQ.action_items)
    assert cache.get(changed) is None
    other_model = JudgmentCache(tmp_path, "anthropic", "claude-sonnet-5-5")
    assert other_model.get(REQ) is None


def test_cache_put_failure_is_non_fatal(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    client = StubClient(ScreenJudgment(True, "r", "v"))
    cache = JudgmentCache(blocker / "sub", client.provider, client.model)
    out, warnings = judge_screens(client, [REQ], cache, concurrency=1)
    assert out == {"S1": ScreenJudgment(True, "r", "v")} and warnings == []


def test_judge_screens_uses_cache(tmp_path: Path) -> None:
    client = StubClient(ScreenJudgment(True, "r", "v"))
    cache = JudgmentCache(tmp_path, client.provider, client.model)
    judge_screens(client, [REQ], cache, concurrency=2)
    out, warnings = judge_screens(client, [REQ], cache, concurrency=2)
    assert client.calls == 1
    assert out == {"S1": ScreenJudgment(True, "r", "v")} and warnings == []


def test_judge_screens_failure_gives_none_and_warning() -> None:
    out, warnings = judge_screens(StubClient(RuntimeError("boom")), [REQ], None, concurrency=1)
    assert out == {"S1": None}
    assert len(warnings) == 1 and "S1" in warnings[0]


def test_anthropic_adapter_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from peeklet.llm import anthropic_client

    fake_sdk = mock.MagicMock()
    reply = mock.MagicMock()
    reply.content = [mock.MagicMock(text='{"include": false, "reason": "r", "visual_context": ""}')]
    fake_sdk.Anthropic.return_value.messages.create.return_value = reply
    monkeypatch.setattr(anthropic_client, "anthropic", fake_sdk)
    client = anthropic_client.AnthropicClient(model="claude-haiku-4-5")
    j = client.judge_screen(b"img", "ocr", ["l1"], [])
    assert j == ScreenJudgment(False, "r", "")
    kwargs = fake_sdk.Anthropic.return_value.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["system"] == base.SYSTEM_PROMPT
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1]["type"] == "text"


def test_openai_adapter_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from peeklet.llm import openai_client

    fake_sdk = mock.MagicMock()
    choice = mock.MagicMock()
    choice.message.content = json.dumps({"include": True, "reason": "r", "visual_context": "v"})
    fake_sdk.OpenAI.return_value.chat.completions.create.return_value.choices = [choice]
    monkeypatch.setattr(openai_client, "openai", fake_sdk)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    client = openai_client.OpenAIClient(model="gpt-x")
    assert client.judge_screen(b"img", "ocr", ["l1"], []).include
    msgs = fake_sdk.OpenAI.return_value.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert user[0]["type"] == "image_url"
    assert user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_adapter_retries_once_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from peeklet.llm import anthropic_client

    fake_sdk = mock.MagicMock()
    bad, good = mock.MagicMock(), mock.MagicMock()
    bad.content = [mock.MagicMock(text="nope")]
    good.content = [mock.MagicMock(text='{"include": true, "reason": "r", "visual_context": "v"}')]
    fake_sdk.Anthropic.return_value.messages.create.side_effect = [bad, good]
    monkeypatch.setattr(anthropic_client, "anthropic", fake_sdk)
    assert anthropic_client.AnthropicClient("m").judge_screen(b"i", "o", [], []).include
