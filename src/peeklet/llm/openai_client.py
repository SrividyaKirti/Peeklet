"""OpenAI chat-completions adapter (honours OPENAI_BASE_URL) for the LLM judge."""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING, Any

from peeklet.llm.base import SYSTEM_PROMPT, LLMResponseError, build_user_text, parse_judgment

if TYPE_CHECKING:
    from peeklet.types import ScreenJudgment

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment, unused-ignore]


def chat_judge(
    client: Any,
    model: str,
    image_jpeg: bytes,
    ocr_text: str,
    lines: list[str],
    action_items: list[str],
    label: str,
) -> ScreenJudgment:
    """Shared by OpenAI and OpenRouter: one vision chat call with one JSON retry."""
    data_url = "data:image/jpeg;base64," + base64.standard_b64encode(image_jpeg).decode("ascii")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": build_user_text(ocr_text, lines, action_items)},
            ],
        },
    ]
    for attempt in (1, 2):
        response = client.chat.completions.create(model=model, messages=messages)
        raw = response.choices[0].message.content or ""
        try:
            return parse_judgment(raw)
        except LLMResponseError:
            if attempt == 2:
                raise
            logger.warning("%s returned unparseable JSON, retrying once", label)
    raise AssertionError("unreachable")


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str) -> None:
        if openai is None:  # pragma: no cover
            raise RuntimeError("The openai package is required for provider 'openai'.")
        self.model = model
        base_url = os.environ.get("OPENAI_BASE_URL")
        self._client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment:
        return chat_judge(
            self._client, self.model, image_jpeg, ocr_text, lines, action_items, "OpenAI"
        )
