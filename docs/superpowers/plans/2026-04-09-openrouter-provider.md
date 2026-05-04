# OpenRouter Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `openrouter` as a first-class `--llm-provider` choice in the Peeklet CLI, reusing the existing `openai` SDK with a hardcoded base URL.

**Architecture:** OpenRouter speaks the OpenAI chat completions wire protocol. A new thin `OpenRouterClient` reuses the `openai` SDK that the `[demo]` extra already pulls in, hardcodes `base_url="https://openrouter.ai/api/v1"`, and reads `OPENROUTER_API_KEY`. The retry+parse loop is extracted from `OpenAIClient` into a shared module-level helper that both clients call. No new dependency.

**Tech Stack:** Python 3.10+, `click`, `pydantic`, `openai>=1.0` SDK, `pytest`.

**Spec:** `docs/superpowers/specs/2026-04-09-openrouter-provider-design.md`

---

## Task 1: Extract shared retry+parse helper from OpenAIClient

**Files:**
- Modify: `src/peeklet/core/llm_openai.py`
- Test: `tests/unit/test_llm.py` (existing tests cover this — no edits needed; rerun to verify regression-free)

This task is a pure refactor. The existing `OpenAIClient.pick_moments` retry+parse loop becomes a module-level helper so the upcoming `OpenRouterClient` can call the same function. No behavior change. After this task the `test_openai_client_*` tests must still pass unmodified.

- [ ] **Step 1: Run the existing OpenAI client tests to establish a green baseline**

Run: `uv run pytest tests/unit/test_llm.py -v -k openai_client`
Expected: All 3 tests pass (`test_openai_client_pick_moments_calls_sdk_and_parses_response`, `test_openai_client_passes_base_url_when_set`, `test_openai_client_retries_once_on_unparseable`).

- [ ] **Step 2: Add the shared helper at module level in `llm_openai.py`**

Insert after the `try: import openai` block and before `class OpenAIClient`:

```python
def _call_openai_chat_with_retry(
    client: object,
    model: str,
    user_message: str,
    video_duration: float,
    *,
    provider_label: str,
) -> list[Moment]:
    """Call an OpenAI-compatible chat completions endpoint with one retry on bad JSON.

    Shared by OpenAIClient and OpenRouterClient — both speak the same wire
    protocol, so the only thing that differs is which SDK instance is passed
    in and what label appears in the retry log line.
    """
    for attempt in (1, 2):
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
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
            logger.warning("%s returned unparseable JSON, retrying once", provider_label)

    raise AssertionError("retry loop exited without returning")
```

Add the `Moment` import to the `TYPE_CHECKING` block if it isn't already there. The `Moment` import for the helper's return type annotation needs to be available at runtime *only if* you don't quote the annotation — quote it as `"list[Moment]"` if you prefer to keep the import lazy. Either is fine; pick the one that matches existing style in this file (the file already has `Moment` under `TYPE_CHECKING`, so use the quoted form).

Concretely, change the helper signature to:

```python
def _call_openai_chat_with_retry(
    client: object,
    model: str,
    user_message: str,
    video_duration: float,
    *,
    provider_label: str,
) -> "list[Moment]":
```

- [ ] **Step 3: Replace the body of `OpenAIClient.pick_moments` to call the helper**

Replace the existing method body (currently `llm_openai.py:49-70`) with:

```python
    def pick_moments(
        self, transcript: list[TranscriptSegment], video_duration: float
    ) -> list[Moment]:
        user_message = format_transcript_for_llm(transcript)
        return _call_openai_chat_with_retry(
            client=self._client,
            model=self._model,
            user_message=user_message,
            video_duration=video_duration,
            provider_label="OpenAI",
        )
```

- [ ] **Step 4: Run the OpenAI client tests — must still pass without modification**

Run: `uv run pytest tests/unit/test_llm.py -v -k openai_client`
Expected: Same 3 tests pass. If any fail, the refactor changed behavior — fix before moving on.

- [ ] **Step 5: Run the full llm test module to catch broader regressions**

Run: `uv run pytest tests/unit/test_llm.py -v`
Expected: All tests pass.

- [ ] **Step 6: Lint check**

Run: `uv run ruff check src/peeklet/core/llm_openai.py && uv run ruff format --check src/peeklet/core/llm_openai.py`
Expected: No issues.

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/core/llm_openai.py
git commit -m "refactor: extract shared OpenAI chat retry helper

Pulls the retry+parse loop out of OpenAIClient.pick_moments into a
module-level _call_openai_chat_with_retry helper. No behavior change —
prep for the OpenRouter client to share the same loop."
```

---

## Task 2: Add OpenRouterClient module

**Files:**
- Create: `src/peeklet/core/llm_openrouter.py`
- Test: `tests/unit/test_llm.py` (add tests before implementation per TDD)

- [ ] **Step 1: Write failing tests for OpenRouterClient in `tests/unit/test_llm.py`**

Append these tests to the end of `tests/unit/test_llm.py`:

```python
def test_openrouter_client_constructs_with_hardcoded_base_url_and_headers(monkeypatch):
    from peeklet.core import llm_openrouter

    fake_openai = MagicMock()
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")

    fake_openai.OpenAI.assert_called_once()
    call_kwargs = fake_openai.OpenAI.call_args.kwargs
    assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert call_kwargs["api_key"] == "or-test-key"
    assert call_kwargs["default_headers"]["HTTP-Referer"] == (
        "https://github.com/SrividyaKirti/Peeklet"
    )
    assert call_kwargs["default_headers"]["X-Title"] == "Peeklet"


def test_openrouter_client_raises_when_sdk_missing(monkeypatch):
    from peeklet.core import llm_openrouter

    monkeypatch.setattr(llm_openrouter, "openai", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    with pytest.raises(RuntimeError, match=r"\[demo\] extra"):
        llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")


def test_openrouter_client_pick_moments_calls_sdk_and_parses_response(monkeypatch):
    from peeklet.core import llm_openrouter
    from peeklet.utils.types import Moment

    fake_message = MagicMock()
    fake_message.content = '[{"timestamp": 13.0, "caption": "c", "reason": "r"}]'
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    client = llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert moments == [Moment(timestamp=13.0, caption="c", reason="r")]
    fake_client.chat.completions.create.assert_called_once()
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-3.5-sonnet"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"


def test_openrouter_client_retries_once_on_unparseable(monkeypatch):
    from peeklet.core import llm_openrouter

    bad_message = MagicMock(content="not json")
    good_message = MagicMock(content='[{"timestamp": 1.0, "caption": "c", "reason": "r"}]')
    bad_response = MagicMock(choices=[MagicMock(message=bad_message)])
    good_response = MagicMock(choices=[MagicMock(message=good_message)])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [bad_response, good_response]

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    client = llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert len(moments) == 1
    assert fake_client.chat.completions.create.call_count == 2


def test_openrouter_client_raises_after_two_unparseable(monkeypatch):
    from peeklet.core import llm_openrouter
    from peeklet.core.llm import LLMResponseError

    bad_message = MagicMock(content="garbage")
    bad_response = MagicMock(choices=[MagicMock(message=bad_message)])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = bad_response

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    client = llm_openrouter.OpenRouterClient(model="anthropic/claude-3.5-sonnet")
    with pytest.raises(LLMResponseError):
        client.pick_moments(_make_segments(), video_duration=60.0)

    assert fake_client.chat.completions.create.call_count == 2
```

- [ ] **Step 2: Run the new tests — they should fail with import error**

Run: `uv run pytest tests/unit/test_llm.py -v -k openrouter_client`
Expected: All 5 tests fail with `ModuleNotFoundError: No module named 'peeklet.core.llm_openrouter'` (or a `pytest` collection error pointing at the same).

- [ ] **Step 3: Create `src/peeklet/core/llm_openrouter.py`**

```python
"""Native OpenAI SDK adapter pointed at OpenRouter.

OpenRouter is OpenAI-wire-compatible, so we reuse the openai SDK with a
hardcoded base URL and OpenRouter's own API key. The retry+parse loop is
shared with OpenAIClient via the helper in llm_openai.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from peeklet.core.llm_openai import _call_openai_chat_with_retry, format_transcript_for_llm

if TYPE_CHECKING:
    from peeklet.core.audio import TranscriptSegment
    from peeklet.utils.types import Moment

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:  # pragma: no cover - exercised when [demo] extra not installed
    openai = None  # type: ignore[assignment]


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
        self, transcript: list[TranscriptSegment], video_duration: float
    ) -> list[Moment]:
        user_message = format_transcript_for_llm(transcript)
        return _call_openai_chat_with_retry(
            client=self._client,
            model=self._model,
            user_message=user_message,
            video_duration=video_duration,
            provider_label="OpenRouter",
        )
```

Note: `format_transcript_for_llm` lives in `peeklet.core.llm`, not `llm_openai`. Either import it from `llm_openai` (which already imports it) — that works because Python re-exports it as a module attribute — or import it directly from `peeklet.core.llm`. Prefer the direct import for clarity:

```python
from peeklet.core.llm import format_transcript_for_llm
from peeklet.core.llm_openai import _call_openai_chat_with_retry
```

Use that two-line import in the final file.

- [ ] **Step 4: Run the new tests — they should pass**

Run: `uv run pytest tests/unit/test_llm.py -v -k openrouter_client`
Expected: All 5 tests pass.

- [ ] **Step 5: Run the full llm test module to confirm no regressions**

Run: `uv run pytest tests/unit/test_llm.py -v`
Expected: All tests pass.

- [ ] **Step 6: Lint check**

Run: `uv run ruff check src/peeklet/core/llm_openrouter.py tests/unit/test_llm.py && uv run ruff format --check src/peeklet/core/llm_openrouter.py tests/unit/test_llm.py`
Expected: No issues.

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/core/llm_openrouter.py tests/unit/test_llm.py
git commit -m "feat: add OpenRouterClient using shared OpenAI helper

New thin client that points the openai SDK at OpenRouter's API with the
recommended HTTP-Referer/X-Title attribution headers. Reuses the shared
retry+parse helper so there's no copy-paste of the chat loop."
```

---

## Task 3: Wire OpenRouter into build_llm_client and config

**Files:**
- Modify: `src/peeklet/core/llm.py`
- Modify: `src/peeklet/config.py`
- Test: `tests/unit/test_llm.py` (add dispatch tests before implementation)

- [ ] **Step 1: Write failing dispatch tests in `tests/unit/test_llm.py`**

Append:

```python
def test_build_llm_client_openrouter_missing_key_raises(monkeypatch):
    from peeklet.core.llm import build_llm_client

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_llm_client(provider="openrouter", model="anthropic/claude-3.5-sonnet")


def test_build_llm_client_openrouter_returns_openrouter_client(monkeypatch):
    from peeklet.core import llm_openrouter
    from peeklet.core.llm import build_llm_client

    fake_openai = MagicMock()
    monkeypatch.setattr(llm_openrouter, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    client = build_llm_client(provider="openrouter", model="anthropic/claude-3.5-sonnet")

    assert isinstance(client, llm_openrouter.OpenRouterClient)
```

- [ ] **Step 2: Run the new tests — they should fail**

Run: `uv run pytest tests/unit/test_llm.py -v -k build_llm_client_openrouter`
Expected: Both fail. The first fails with `ValueError: Unknown LLM provider 'openrouter'`. The second fails with the same error.

- [ ] **Step 3: Add `openrouter` to `_PROVIDER_KEYS` and dispatch in `src/peeklet/core/llm.py`**

Edit `src/peeklet/core/llm.py`. Change `_PROVIDER_KEYS` from:

```python
_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
```

to:

```python
_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
```

In `build_llm_client`, add a new branch immediately after the `openai` branch and before the unreachable `AssertionError`:

```python
    if provider == "openrouter":
        from peeklet.core.llm_openrouter import OpenRouterClient

        return OpenRouterClient(model=model)
```

- [ ] **Step 4: Extend the Literal in `src/peeklet/config.py`**

Edit line 93. Change:

```python
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
```

to:

```python
    llm_provider: Literal["anthropic", "openai", "openrouter"] = "anthropic"
```

- [ ] **Step 5: Run the dispatch tests — they should pass**

Run: `uv run pytest tests/unit/test_llm.py -v -k build_llm_client_openrouter`
Expected: Both tests pass.

- [ ] **Step 6: Run the full llm and config test modules**

Run: `uv run pytest tests/unit/test_llm.py tests/unit/test_config.py -v`
Expected: All pass.

- [ ] **Step 7: Lint check**

Run: `uv run ruff check src/peeklet/core/llm.py src/peeklet/config.py && uv run ruff format --check src/peeklet/core/llm.py src/peeklet/config.py`
Expected: No issues.

- [ ] **Step 8: Commit**

```bash
git add src/peeklet/core/llm.py src/peeklet/config.py tests/unit/test_llm.py
git commit -m "feat: dispatch openrouter provider through build_llm_client

Adds OPENROUTER_API_KEY to _PROVIDER_KEYS so the centralized missing-env-var
check produces a clear error, and extends the demo_filter.llm_provider
Literal to accept the new value."
```

---

## Task 4: Expose `--llm-provider openrouter` in the CLI

**Files:**
- Modify: `src/peeklet/cli.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write a failing CLI test in `tests/unit/test_cli.py`**

Append to `tests/unit/test_cli.py` (mirroring the existing `test_cli_demo_mode_propagates_provider_and_model` at line 329):

```python
def test_cli_demo_mode_accepts_openrouter_provider(tmp_path, monkeypatch):
    """--llm-provider openrouter is accepted by Click and propagates to config."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)
    transcript = tmp_path / "t.srt"
    transcript.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")

    captured = {}

    real_process_video = video_module.process_video

    def spy_process_video(path, config, **kwargs):
        captured["provider"] = config.demo_filter.llm_provider
        captured["model"] = config.demo_filter.llm_model
        return real_process_video(path, config, **kwargs)

    monkeypatch.setattr(video_module, "process_video", spy_process_video)
    monkeypatch.setattr(video_module, "apply_demo_filter", lambda **_kw: [])

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(video_path),
            "--output",
            str(tmp_path / "out"),
            "--no-audio",
            "--transcript",
            str(transcript),
            "--demo-mode",
            "--llm-provider",
            "openrouter",
            "--llm-model",
            "anthropic/claude-3.5-sonnet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "provider": "openrouter",
        "model": "anthropic/claude-3.5-sonnet",
    }
```

- [ ] **Step 2: Run the new test — it should fail**

Run: `uv run pytest tests/unit/test_cli.py::test_cli_demo_mode_accepts_openrouter_provider -v`
Expected: Test fails because Click rejects `openrouter` as an invalid Choice. `result.exit_code == 2` with output containing `Invalid value for '--llm-provider'`.

- [ ] **Step 3: Add `openrouter` to the Click Choice in `src/peeklet/cli.py`**

Edit line 93. Change:

```python
    type=click.Choice(["anthropic", "openai"]),
```

to:

```python
    type=click.Choice(["anthropic", "openai", "openrouter"]),
```

Update the help text on lines 96-97. Change:

```python
    help="LLM provider for --demo-mode (anthropic or openai). "
    "Defaults to the value in config.demo_filter.llm_provider.",
```

to:

```python
    help="LLM provider for --demo-mode (anthropic, openai, or openrouter). "
    "OpenRouter requires OPENROUTER_API_KEY and uses namespaced model ids "
    "like 'anthropic/claude-3.5-sonnet'. "
    "Defaults to the value in config.demo_filter.llm_provider.",
```

- [ ] **Step 4: Run the new test — it should pass**

Run: `uv run pytest tests/unit/test_cli.py::test_cli_demo_mode_accepts_openrouter_provider -v`
Expected: Pass.

- [ ] **Step 5: Run the full CLI test module**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: All tests pass.

- [ ] **Step 6: Lint check**

Run: `uv run ruff check src/peeklet/cli.py tests/unit/test_cli.py && uv run ruff format --check src/peeklet/cli.py tests/unit/test_cli.py`
Expected: No issues.

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/cli.py tests/unit/test_cli.py
git commit -m "feat: accept --llm-provider openrouter in the CLI

Adds openrouter to the Click Choice for --llm-provider and updates the
help text to mention OPENROUTER_API_KEY and the namespaced model id
convention."
```

---

## Task 5: Document OpenRouter usage in the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find the existing demo-mode section in `README.md`**

Run: `uv run python -c "import re,pathlib; t=pathlib.Path('README.md').read_text(); print('demo-mode found' if '--demo-mode' in t else 'NOT FOUND')"`
Expected: `demo-mode found`. If not found, search for `--llm-provider` instead. The OpenRouter snippet goes in whichever section already documents `--demo-mode` and `--llm-provider`.

- [ ] **Step 2: Add an OpenRouter usage block beneath the existing provider examples**

Insert this block after the existing `--llm-provider openai` or `--llm-provider anthropic` example in the README:

````markdown
**Using OpenRouter** (single API key, hundreds of models):

```bash
export OPENROUTER_API_KEY=sk-or-...
peeklet \
    --input video.mp4 \
    --output out \
    --transcript transcript.srt \
    --demo-mode \
    --llm-provider openrouter \
    --llm-model anthropic/claude-3.5-sonnet
```

OpenRouter model ids are namespaced as `<vendor>/<model>` — see
[openrouter.ai/models](https://openrouter.ai/models) for the full list.
````

If the README doesn't have an existing provider-examples section, add a new `### OpenRouter` subsection under the demo-mode heading containing the same block.

- [ ] **Step 3: Spot-check rendering**

Run: `uv run python -c "import pathlib; print('OPENROUTER_API_KEY' in pathlib.Path('README.md').read_text())"`
Expected: `True`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document --llm-provider openrouter usage

Adds an OpenRouter example to the demo-mode section, including the
OPENROUTER_API_KEY env var and the namespaced model id convention."
```

---

## Task 6: Full verification before handing off

**Files:** none

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest -v`
Expected: All tests pass. If any test fails, stop and diagnose — don't push.

- [ ] **Step 2: Run lint over the whole repo**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No issues.

- [ ] **Step 3: Run mypy on the touched files**

Run: `uv run mypy src/peeklet/core/llm.py src/peeklet/core/llm_openai.py src/peeklet/core/llm_openrouter.py src/peeklet/config.py src/peeklet/cli.py`
Expected: Success, no issues found.

- [ ] **Step 4: Smoke-test the CLI help output**

Run: `uv run peeklet --help`
Expected: `--llm-provider` line lists `[anthropic|openai|openrouter]` and the help text mentions OpenRouter and `OPENROUTER_API_KEY`.

- [ ] **Step 5: Confirm the branch is ready**

Run: `git log --oneline feat/openrouter-provider ^main`
Expected: Six commits — spec, refactor, OpenRouterClient, dispatch, CLI, README.

The branch is ready for PR review. Use the `superpowers:finishing-a-development-branch` skill to choose merge/PR/cleanup.
