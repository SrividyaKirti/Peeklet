"""OpenRouter adapter (OpenAI-compatible API) for the LLM judge."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from peeklet.llm.openai_client import chat_judge

if TYPE_CHECKING:
    from peeklet.types import ScreenJudgment

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment, unused-ignore]

_BASE_URL = "https://openrouter.ai/api/v1"
_HEADERS = {"HTTP-Referer": "https://github.com/SrividyaKirti/Peeklet", "X-Title": "Peeklet"}


class OpenRouterClient:
    provider = "openrouter"

    def __init__(self, model: str) -> None:
        if openai is None:  # pragma: no cover
            raise RuntimeError("The openai package is required for provider 'openrouter'.")
        self.model = model
        self._client = openai.OpenAI(
            base_url=_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers=_HEADERS,
        )

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment:
        return chat_judge(
            self._client, self.model, image_jpeg, ocr_text, lines, action_items, "OpenRouter"
        )
