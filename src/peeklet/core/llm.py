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
    "You are a Video Content Analyst specializing in visual-textual"
    " alignment for multimodal AI processing.\n\n"
    "## Task\n\n"
    "Analyze the provided meeting/demo transcript to identify specific"
    " timestamps where a screenshot is essential for a downstream"
    " Multimodal LLM (MLLM) to understand the technical context being"
    " discussed — BEYOND what is already covered by guaranteed anchor"
    " moments (listed below).\n\n"
    "## Guaranteed Anchors (do NOT pick at these)\n\n"
    "The following moments are already captured by guaranteed anchors"
    " from the meeting tool. Each anchor will produce its own screenshot"
    " at the listed timestamp. Anchor timestamps are in **decimal"
    " seconds**, the same unit used in the transcript above.\n\n"
    "{anchor_list}\n\n"
    "When picking your own moments:\n"
    "- Do NOT pick any moment within ±10 seconds of a listed anchor"
    " timestamp.\n"
    "- Do NOT pick moments that would visually duplicate what an anchor"
    " captures (e.g., if an anchor captures the Tasks page, do not pick"
    " another moment of the Tasks page unless it has materially changed"
    " — new modal, new data, scrolled region).\n\n"
    "## Selection Criteria — Complementary Moments\n\n"
    "In the transcript regions OUTSIDE the anchor windows, scan for"
    " moments where the speaker references on-screen UI that is not yet"
    " captured by an anchor. Prioritize:\n\n"
    '1. **Deictic references to visible UI** — "this chart", "that'
    ' banner", "look at the sidebar", "notice the risk score here".\n'
    "2. **UI nouns** — tab, modal, dialog, graph, table, button, panel,"
    " dropdown, sidebar, dashboard.\n"
    '3. **Demo actions** — "let me click...", "if I navigate to...",'
    ' "I\'ll open the settings...", "switching to the other tab".\n'
    "4. **Named UI elements or data values** — a specific field name, a"
    " specific column, a specific status/error/warning visible on"
    " screen.\n\n"
    "Pick one moment for each distinct UI subject the speaker names in"
    " non-anchor regions. If the same UI subject is discussed multiple"
    " times, pick the first occurrence after it appears on screen.\n\n"
    "## Timing Rules\n\n"
    "- Place the timestamp 0.5-1.0 seconds after the speaker begins the"
    " triggering sentence.\n"
    "- Avoid selecting timestamps within 15 seconds of another of your"
    " own picks unless a major UI transition occurs (new page, modal,"
    " tab).\n"
    "- In regions of pure strategic discussion (no UI references), pick"
    " nothing. Silence is a valid answer.\n\n"
    "## Output Format\n\n"
    "Return ONLY a JSON array of objects. No preamble, no explanation."
    " If no complementary moments are warranted, return an empty"
    " array [].\n\n"
    '[{{"timestamp": 12.5, "visual_context_goal": "...",'
    ' "textual_anchor": "...", "downstream_utility": "..."}}]\n\n'
    "Fields:\n"
    "- timestamp: float, seconds into the video\n"
    "- visual_context_goal: what the screenshot needs to capture"
    ' (e.g., "The risk-score modal for a Terminal Run tool call")\n'
    "- textual_anchor: the exact transcript line that triggers this"
    " need — quote the speaker\n"
    "- downstream_utility: why the MLLM needs this image"
    ' (e.g., "To extract the risk classification not mentioned in'
    ' audio")\n'
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


def format_anchors_for_llm(anchors: list[Moment]) -> str:
    """Render the anchor list as ``[<seconds>] <label>`` lines.

    Uses the same decimal-seconds format as :func:`format_transcript_for_llm`
    so the LLM compares anchor timestamps and transcript timestamps in one
    consistent unit. Empty input returns the sentinel string the prompt
    expects when no anchors are present.
    """
    if not anchors:
        return "None — no guaranteed anchors in this video."
    lines: list[str] = []
    for a in anchors:
        label = a.visual_context_goal.strip() or "(unlabeled anchor)"
        lines.append(f"[{a.timestamp:.1f}] {label}")
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
