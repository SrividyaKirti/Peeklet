"""Native OpenAI SDK adapter for the LLM Protocol.

Respects ``OPENAI_BASE_URL`` so users can route through any OpenAI-compatible
gateway (Ollama, Groq, OpenRouter, vLLM, Together, Azure, etc.).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from peeklet.core.llm import (
    SYSTEM_PROMPT,
    LLMResponseError,
    _parse_moments_json,
    format_transcript_for_llm,
)

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment
    from peeklet.utils.types import Moment

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:  # pragma: no cover - exercised when [demo] extra not installed
    openai = None  # type: ignore[assignment]


class OpenAIClient:
    """Calls the OpenAI chat completions API once per video."""

    def __init__(self, model: str) -> None:
        if openai is None:
            raise RuntimeError(
                "--demo-mode with provider 'openai' requires the [demo] extra. "
                "Install with: pip install peeklet[demo]"
            )
        self._model = model

        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            self._client = openai.OpenAI(base_url=base_url)
        else:
            self._client = openai.OpenAI()

    def pick_moments(
        self, transcript: list[TranscriptSegment], video_duration: float
    ) -> list[Moment]:
        user_message = format_transcript_for_llm(transcript)

        for attempt in (1, 2):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content or ""
            try:
                return _parse_moments_json(raw, video_duration)
            except LLMResponseError:
                if attempt == 2:
                    raise
                logger.warning("OpenAI returned unparseable JSON, retrying once")

        raise AssertionError("retry loop exited without returning")
