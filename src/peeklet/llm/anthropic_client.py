"""Anthropic Messages API adapter for the LLM judge."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from peeklet.llm.base import SYSTEM_PROMPT, LLMResponseError, build_user_text, parse_judgment

if TYPE_CHECKING:
    from peeklet.types import ScreenJudgment

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment, unused-ignore]

_MAX_TOKENS = 1024


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str) -> None:
        if anthropic is None:  # pragma: no cover
            raise RuntimeError("The anthropic package is required for provider 'anthropic'.")
        self.model = model
        self._client = anthropic.Anthropic()

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment:
        content: list[Any] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(image_jpeg).decode("ascii"),
                },
            },
            {"type": "text", "text": build_user_text(ocr_text, lines, action_items)},
        ]
        for attempt in (1, 2):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            raw = "".join(getattr(b, "text", "") for b in response.content)
            try:
                return parse_judgment(raw)
            except LLMResponseError:
                if attempt == 2:
                    raise
                logger.warning("Anthropic returned unparseable JSON, retrying once")
        raise AssertionError("unreachable")
