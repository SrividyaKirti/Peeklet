"""Native Anthropic SDK adapter for the LLM Protocol."""

from __future__ import annotations

import logging
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

# Lazy-import via module attribute so tests can patch this.
try:
    import anthropic
except ImportError:  # pragma: no cover - exercised when [demo] extra not installed
    anthropic = None  # type: ignore[assignment]

_MAX_TOKENS = 4096


class AnthropicClient:
    """Calls the Anthropic Messages API once per video."""

    def __init__(self, model: str) -> None:
        if anthropic is None:
            raise RuntimeError(
                "--demo-mode with provider 'anthropic' requires the [demo] extra. "
                "Install with: pip install peeklet[demo]"
            )
        self._model = model
        self._client = anthropic.Anthropic()

    def pick_moments(
        self, transcript: list[TranscriptSegment], video_duration: float
    ) -> list[Moment]:
        user_message = format_transcript_for_llm(transcript)

        for attempt in (1, 2):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = "".join(block.text for block in response.content if hasattr(block, "text"))
            try:
                return _parse_moments_json(raw, video_duration)
            except LLMResponseError:
                if attempt == 2:
                    raise
                logger.warning("Anthropic returned unparseable JSON, retrying once")

        # Unreachable.
        raise AssertionError("retry loop exited without returning")
