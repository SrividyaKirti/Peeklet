"""LLM adapter for transcript-driven demo mode.

Thin Protocol-based wrapper around the native ``openai`` and ``anthropic``
SDKs. The whole point is to keep the trust boundary small and avoid pulling
in a third-party multi-provider router. See the design spec at
``docs/superpowers/specs/2026-04-09-transcript-driven-demo-mode-design.md``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Protocol

from peeklet.utils.types import Moment

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are reviewing a transcript from a software demo video. Your job is "
    "to identify the moments where a screenshot would help a reader understand "
    "what's happening — places where the speaker references something visual "
    "on screen, demonstrates an action, opens a UI, or moves on to a new topic "
    "with a different visual context.\n\n"
    "For each such moment, return:\n"
    "- timestamp: a float, seconds into the video\n"
    "- caption: a one-sentence description of what the screenshot should show\n"
    "- reason: a one-sentence justification quoting or paraphrasing the speaker's words\n\n"
    "Return ONLY a JSON array. Example:\n"
    '[{"timestamp": 12.5, "caption": "...", "reason": "..."}, ...]\n\n'
    "Pick as many or as few moments as the video needs. There is no minimum or maximum."
)


class LLMResponseError(RuntimeError):
    """Raised when the LLM returns output that cannot be parsed into Moments."""


class LLMClient(Protocol):
    """Internal Protocol that every concrete LLM adapter implements."""

    def pick_moments(
        self, transcript: list[TranscriptSegment], video_duration: float
    ) -> list[Moment]:
        """Send the transcript to the LLM and return parsed moments."""
        ...


def format_transcript_for_llm(segments: list[TranscriptSegment]) -> str:
    """Render the transcript as one line per segment with timestamps.

    Format: ``[start - end] text``.
    """
    lines = [f"[{s.start:.1f} - {s.end:.1f}] {s.text}" for s in segments]
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _extract_json_array(text: str, raw_for_error: str) -> str:
    """Find the substring containing a top-level JSON array.

    Walks ``text`` looking for the first ``[`` and the matching ``]``,
    accounting for nesting and strings. Tolerates LLM preambles like
    'Here is the JSON: [...]' that survive simple fence stripping.
    """
    start = text.find("[")
    if start == -1:
        raise LLMResponseError(f"LLM output contained no JSON array. Raw output:\n{raw_for_error}")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise LLMResponseError(f"LLM output had unclosed JSON array. Raw output:\n{raw_for_error}")


def _parse_moments_json(raw: str, video_duration: float) -> list[Moment]:
    """Parse the raw LLM string into a sorted list of valid Moments.

    - Strips ```` ```json ... ``` ```` markdown fences if present.
    - Drops entries with timestamps outside [0, video_duration].
    - Raises ``LLMResponseError`` if the JSON is unparseable or any entry is
      missing a required key.
    """
    cleaned = _FENCE_RE.sub("", raw).strip()
    cleaned = _extract_json_array(cleaned, raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM returned unparseable JSON. Raw output:\n{raw}") from exc

    if not isinstance(data, list):
        raise LLMResponseError(
            f"LLM did not return a JSON array. Got {type(data).__name__}.\nRaw output:\n{raw}"
        )

    moments: list[Moment] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise LLMResponseError(
                f"LLM array entry is not an object: {entry!r}\nRaw output:\n{raw}"
            )
        for key in ("timestamp", "caption", "reason"):
            if key not in entry:
                raise LLMResponseError(
                    f"LLM moment is missing required key '{key}': {entry!r}\nRaw output:\n{raw}"
                )
        try:
            ts = float(entry["timestamp"])
        except (TypeError, ValueError) as exc:
            raise LLMResponseError(f"LLM moment has non-numeric timestamp: {entry!r}") from exc

        if ts < 0 or ts > video_duration:
            logger.warning(
                "Dropping LLM moment with timestamp %.2fs outside video duration %.2fs",
                ts,
                video_duration,
            )
            continue

        moments.append(
            Moment(
                timestamp=ts,
                caption=str(entry["caption"]),
                reason=str(entry["reason"]),
            )
        )

    moments.sort(key=lambda m: m.timestamp)
    return moments


_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def build_llm_client(provider: str, model: str) -> LLMClient:
    """Validate the provider, ensure the matching env var is set, return a client.

    Imports the concrete client lazily so users who don't use ``--demo-mode``
    aren't forced to install the LLM SDKs.
    """
    if provider not in _PROVIDER_KEYS:
        raise ValueError(f"Unknown LLM provider {provider!r}. Supported: {sorted(_PROVIDER_KEYS)}.")

    env_var = _PROVIDER_KEYS[provider]
    if not os.environ.get(env_var):
        raise RuntimeError(
            f"--demo-mode with provider {provider!r} requires the {env_var} env var to be set."
        )

    if provider == "anthropic":
        from peeklet.core.llm_anthropic import AnthropicClient

        return AnthropicClient(model=model)
    if provider == "openai":
        from peeklet.core.llm_openai import OpenAIClient

        return OpenAIClient(model=model)

    # Unreachable — guarded above.
    raise AssertionError(f"unhandled provider {provider!r}")
