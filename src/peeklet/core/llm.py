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
    "You are a Video Content Analyst specializing in visual-textual "
    "alignment for multimodal AI processing.\n\n"
    "## Task\n\n"
    "Analyze the provided meeting/demo transcript to identify specific "
    "timestamps where a screenshot is essential for a downstream "
    "Multimodal LLM (MLLM) to understand the technical context being "
    "discussed.\n\n"
    "## Selection Criteria\n\n"
    "Identify a Key Moment whenever the speaker:\n"
    "1. **Navigates to a new screen or dashboard** — e.g., "
    '"Now, looking at the settings page..."\n'
    "2. **References a specific UI element** — e.g., "
    '"Note the red warning icon in the top right..."\n'
    "3. **Completes a workflow step** — e.g., "
    '"Once I click Deploy, you\'ll see the status change..."\n'
    "4. **Points to data, tables, or graphs** — e.g., "
    '"This spike in the chart represents..."\n'
    '5. **Uses deictic expressions** ("this", "that", "here", '
    '"there") referring to something visible on screen\n\n'
    "## Timing Rules\n\n"
    "- Place the timestamp **0.5-1.0 seconds after** the speaker begins "
    "the triggering sentence, to allow the UI to finish loading or "
    "animating.\n"
    "- **Avoid selecting timestamps within 15 seconds of each other** "
    "unless a major UI transition (new page, modal, or tab) occurs "
    "between them.\n"
    "- If the demo stays on one complex screen for an extended period, "
    "one screenshot is usually enough. Only add a second if the speaker "
    "references a different region or scrolls to new content.\n\n"
    "## Action Item Anchors\n\n"
    "Lines marked `ACTION ITEM` with `WATCH` links are high-priority "
    "moments flagged by the meeting tool. You MUST include a moment at "
    "or near each such timestamp. These represent confirmed points of "
    "interest that a human reviewer has validated.\n\n"
    "## Output Format\n\n"
    "Return ONLY a JSON array of objects. No preamble, no explanation.\n\n"
    '[{"timestamp": 12.5, "visual_context_goal": "...", '
    '"textual_anchor": "...", "downstream_utility": "..."}]\n\n'
    "Fields:\n"
    "- **timestamp**: float, seconds into the video\n"
    "- **visual_context_goal**: what the screenshot needs to capture "
    '(e.g., "The configuration modal for API keys")\n'
    "- **textual_anchor**: the exact transcript line that triggers this "
    "need — quote the speaker\n"
    "- **downstream_utility**: why the MLLM needs this image "
    '(e.g., "To extract parameter values not mentioned in audio")\n\n'
    "Pick as many or as few moments as the content needs."
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

    Format: ``[start - end] **Speaker**: text`` (speaker omitted when None).
    """
    lines: list[str] = []
    for s in segments:
        prefix = f"[{s.start:.1f} - {s.end:.1f}]"
        if s.speaker:
            lines.append(f"{prefix} **{s.speaker}**: {s.text}")
        else:
            lines.append(f"{prefix} {s.text}")
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
        for key in ("timestamp", "visual_context_goal", "textual_anchor", "downstream_utility"):
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
                visual_context_goal=str(entry["visual_context_goal"]),
                textual_anchor=str(entry["textual_anchor"]),
                downstream_utility=str(entry["downstream_utility"]),
            )
        )

    moments.sort(key=lambda m: m.timestamp)
    return moments


_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
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
    if provider == "openrouter":
        from peeklet.core.llm_openrouter import OpenRouterClient

        return OpenRouterClient(model=model)

    # Unreachable — guarded above.
    raise AssertionError(f"unhandled provider {provider!r}")
