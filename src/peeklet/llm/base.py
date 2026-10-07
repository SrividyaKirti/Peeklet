"""LLM judge + describe: decides whether a screenshot is needed and describes it."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from peeklet.types import ScreenJudgment

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

PROMPT_VERSION = "judge-v1"

SYSTEM_PROMPT = (
    "You review screenshots from a screen recording for a downstream AI that will read "
    "the transcript. For one screenshot you get the transcript lines spoken while it was "
    "on screen, the screen's OCR text, and any action items flagged at that moment.\n\n"
    "Decide whether understanding these transcript lines requires seeing this screenshot "
    "(the speaker refers to things that are only visible on screen), and describe the "
    "screen.\n\n"
    "Rules for visual_context:\n"
    "- Describe only what is visible on the screen: the application, the page or view, "
    "and the specific elements, values, messages or errors shown.\n"
    "- Do not paraphrase or summarise the transcript.\n"
    "- Copy on-screen text exactly only where the OCR text confirms it.\n"
    "- One to three sentences.\n\n"
    'Respond with only a JSON object: {"include": true|false, "reason": "<one sentence>", '
    '"visual_context": "<description>"}'
)


class LLMResponseError(RuntimeError):
    """The model's output could not be parsed into a ScreenJudgment."""


class LLMClient(Protocol):
    provider: str
    model: str

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment: ...


def build_user_text(ocr_text: str, lines: list[str], action_items: list[str]) -> str:
    """The text part of the user message (the image is sent alongside it)."""
    parts = ["## Transcript lines spoken while this screen was shown", *lines]
    if action_items:
        parts += ["", "## Action items flagged at this moment", *action_items]
    parts += ["", "## OCR text of the screen", ocr_text or "(none)"]
    return "\n".join(parts)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_judgment(raw: str) -> ScreenJudgment:
    """Parse the model's JSON reply (optionally fenced) into a ScreenJudgment."""
    text = _FENCE_RE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"not JSON: {raw[:200]!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("include"), bool):
        raise LLMResponseError(f"missing boolean 'include': {raw[:200]!r}")
    return ScreenJudgment(
        include=data["include"],
        reason=str(data.get("reason", "")),
        visual_context=str(data.get("visual_context", "")),
    )


_PROVIDER_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def has_credentials(provider: str) -> bool:
    """True if any of the provider's credential env vars is set."""
    return any(os.environ.get(v) for v in _PROVIDER_ENV.get(provider, ()))


def build_llm_client(provider: str, model: str) -> LLMClient:
    """Instantiate the adapter for provider (SDKs are imported lazily)."""
    if provider == "anthropic":
        from peeklet.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=model)
    if provider == "openai":
        from peeklet.llm.openai_client import OpenAIClient

        return OpenAIClient(model=model)
    if provider == "openrouter":
        from peeklet.llm.openrouter_client import OpenRouterClient

        return OpenRouterClient(model=model)
    raise ValueError(f"Unknown LLM provider {provider!r}")


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    screen_id: str
    image_jpeg: bytes
    ocr_text: str
    lines: tuple[str, ...]
    action_items: tuple[str, ...]


class JudgmentCache:
    """On-disk cache of judgments keyed by every input that affects the answer."""

    def __init__(self, directory: Path, provider: str, model: str) -> None:
        self._dir = Path(directory).expanduser()
        self._provider = provider
        self._model = model

    def _path(self, req: JudgeRequest) -> Path:
        payload = json.dumps(
            [
                PROMPT_VERSION,
                self._provider,
                self._model,
                hashlib.sha256(req.image_jpeg).hexdigest(),
                req.ocr_text,
                list(req.lines),
                list(req.action_items),
            ]
        )
        return self._dir / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"

    def get(self, req: JudgeRequest) -> ScreenJudgment | None:
        path = self._path(req)
        if not path.is_file():
            return None
        try:
            return parse_judgment(path.read_text())
        except (OSError, LLMResponseError):
            return None

    def put(self, req: JudgeRequest, judgment: ScreenJudgment) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(req).write_text(
            json.dumps(
                {
                    "include": judgment.include,
                    "reason": judgment.reason,
                    "visual_context": judgment.visual_context,
                }
            )
        )


def judge_screens(
    client: LLMClient,
    requests: Sequence[JudgeRequest],
    cache: JudgmentCache | None,
    concurrency: int,
) -> tuple[dict[str, ScreenJudgment | None], list[str]]:
    """Judge every request (cache first, then the API in parallel).

    Returns verdicts keyed by screen id (None where the call failed) and warnings.
    """
    results: dict[str, ScreenJudgment | None] = {}
    warnings: list[str] = []
    todo: list[JudgeRequest] = []
    for req in requests:
        hit = cache.get(req) if cache else None
        if hit is not None:
            results[req.screen_id] = hit
        else:
            todo.append(req)

    def call(req: JudgeRequest) -> tuple[JudgeRequest, ScreenJudgment | None, str | None]:
        try:
            j = client.judge_screen(
                req.image_jpeg, req.ocr_text, list(req.lines), list(req.action_items)
            )
        except Exception as exc:
            return req, None, f"LLM judge failed for screen {req.screen_id}: {exc}"
        return req, j, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for req, judgment, warning in pool.map(call, todo):
            results[req.screen_id] = judgment
            if warning:
                logger.warning(warning)
                warnings.append(warning)
            elif judgment is not None and cache:
                cache.put(req, judgment)
    return results, warnings
