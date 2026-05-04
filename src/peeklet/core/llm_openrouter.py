"""Native OpenAI SDK adapter pointed at OpenRouter.

OpenRouter is OpenAI-wire-compatible, so we reuse the openai SDK with a
hardcoded base URL and OpenRouter's own API key. The retry+parse loop is
shared with OpenAIClient via the helper in llm_openai.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from peeklet.core.llm import SYSTEM_PROMPT, format_transcript_for_llm
from peeklet.core.llm_openai import _call_openai_chat_with_retry

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment
    from peeklet.utils.types import Moment

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:  # pragma: no cover - exercised when [demo] extra not installed
    openai = None  # type: ignore[assignment, unused-ignore]


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/SrividyaKirti/Peeklet",
    "X-Title": "Peeklet",
}


class OpenRouterClient:
    """Calls OpenRouter's OpenAI-compatible chat completions API once per video."""

    def __init__(self, model: str) -> None:
        if openai is None:
            raise RuntimeError(
                "--demo-mode with provider 'openrouter' requires the [demo] extra. "
                "Install with: pip install peeklet[demo]"
            )
        self._model = model
        self._client = openai.OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers=_DEFAULT_HEADERS,
        )

    def pick_moments(
        self,
        transcript: list[TranscriptSegment],
        video_duration: float,
        anchors: list[Moment],
    ) -> list[Moment]:
        from peeklet.core.llm import format_anchors_for_llm

        anchor_list = format_anchors_for_llm(anchors)
        system_prompt = SYSTEM_PROMPT.format(anchor_list=anchor_list)
        user_message = (
            "## Transcript\n\n"
            f"{format_transcript_for_llm(transcript)}\n\n"
            "## Anchor List (already covered — do not pick at these)\n\n"
            f"{anchor_list}"
        )
        return _call_openai_chat_with_retry(
            client=self._client,
            model=self._model,
            system_prompt=system_prompt,
            user_message=user_message,
            video_duration=video_duration,
            provider_label="OpenRouter",
        )
