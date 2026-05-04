# Transcript-Driven Demo Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in `--demo-mode` flag where an LLM picks screenshot-worthy moments from a transcript and Peeklet's frame-level intelligence picks the exact frame at each moment.

**Architecture:** Two stages — Stage A sends the full transcript to a configurable LLM (native `anthropic` or `openai` SDK behind a thin internal `Protocol`) which returns `[{timestamp, caption, reason}, ...]`. Stage B walks each LLM moment, searches forward within the current transcript segment for a bidirectionally-stable frame, runs an OCR gallery check, and saves the survivors as keyframes with the LLM's caption and reason as metadata.

**Tech Stack:** Python 3.10+, openai SDK, anthropic SDK, PyAV, scikit-image (SSIM), pytesseract, pydantic, click, pytest.

**Spec:** `docs/superpowers/specs/2026-04-09-transcript-driven-demo-mode-design.md`

---

## File Structure

**Create:**
- `src/peeklet/core/llm.py` — LLMClient Protocol, AnthropicClient, OpenAIClient, build_llm_client factory, transcript formatting helper, error types. ~150 lines.
- `tests/unit/test_llm.py` — unit tests for the LLM module with mocked SDKs.
- `tests/integration/test_demo_mode_pipeline.py` — end-to-end synthetic-video test with a fake LLM client.

**Modify:**
- `src/peeklet/utils/types.py` — drop `ContentSegment`, drop `FrameResult.content_type` / `selection_reason`, add `Moment` dataclass, add `FrameResult.llm_caption` / `llm_reason`.
- `src/peeklet/config.py` — replace `DemoFilterConfig` body with the new fields from the spec.
- `src/peeklet/core/demo_filter.py` — extend with Stage B helpers + `select_frames_for_moments()` + `apply_demo_filter()`. Update the module docstring to reference the new spec.
- `src/peeklet/core/video.py` — call `apply_demo_filter()` from `process_video()` when `config.demo_filter.enabled`.
- `src/peeklet/cli.py` — add `--demo-mode`, `--llm-provider`, `--llm-model` flags with validation rules.
- `pyproject.toml` — add a new `[demo]` optional-dependency extra with `openai` + `anthropic`. Add the matching mypy overrides.
- `tests/unit/test_config.py` — replace the old `DemoFilterConfig` defaults test with one for the new fields.
- `tests/unit/test_types.py` — update for the new `Moment` type and the renamed `FrameResult` fields.
- `tests/unit/test_demo_filter.py` — extend with Stage B helper + public function tests.
- `tests/unit/test_cli.py` — add CLI tests for the new flags.
- `README.md` — document the new `--demo-mode` usage.

**Delete:** Nothing. The current state of `demo_filter.py` (Tasks 1, 4, 5 from the previous plan) is all kept; we extend it.

---

## Task 1: Replace types in `utils/types.py`

**Files:**
- Modify: `src/peeklet/utils/types.py`
- Modify: `tests/unit/test_types.py`

The previous plan added `ContentSegment` and `FrameResult.content_type` / `selection_reason`. The new design drops those and adds `Moment` and `FrameResult.llm_caption` / `llm_reason`.

- [ ] **Step 1: Write the failing tests**

Edit `tests/unit/test_types.py`. Find and DELETE the existing `test_content_segment_basic` and `test_frame_result_demo_fields_default_none` tests (added by the previous plan's Task 2 + cleanup). Then add:

```python
def test_moment_basic():
    seg = Moment(timestamp=12.5, caption="Settings page open", reason="speaker says 'show settings'")
    assert seg.timestamp == 12.5
    assert seg.caption == "Settings page open"
    assert seg.reason == "speaker says 'show settings'"


def test_moment_is_frozen():
    import pytest

    seg = Moment(timestamp=0.0, caption="x", reason="y")
    with pytest.raises((AttributeError, TypeError)):
        seg.timestamp = 1.0  # type: ignore[misc]


def test_frame_result_llm_fields_default_none():
    r = FrameResult(
        frame_id="f",
        event_type=EventType.SKIPPED,
        is_keyframe=False,
        perceptual_hash="0" * 16,
        frame_width=10,
        frame_height=10,
    )
    assert r.llm_caption is None
    assert r.llm_reason is None
```

You also need to add `Moment` to the module-level imports at the top of `tests/unit/test_types.py` alongside the existing `ContentSegment`/`FrameResult`/`EventType` imports — but **remove** any reference to `ContentSegment` from those imports since we're deleting it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_types.py -v`
Expected: imports fail with `ImportError: cannot import name 'Moment' from 'peeklet.utils.types'`. The deleted `ContentSegment` import will also fail.

- [ ] **Step 3: Implement**

Edit `src/peeklet/utils/types.py`.

**Delete** the existing `ContentSegment` dataclass (lines 37-44):

```python
@dataclass(frozen=True, slots=True)
class ContentSegment:
    """A time range classified as demo screen-share or non-demo content."""

    start_sec: float
    end_sec: float
    is_demo: bool
    text_density: float  # average words detected per sampled frame in this range
```

**Add** in its place a new `Moment` dataclass:

```python
@dataclass(frozen=True, slots=True)
class Moment:
    """An LLM-picked screenshot-worthy moment in a video transcript."""

    timestamp: float  # seconds into the video
    caption: str  # one-sentence description of what the screenshot should show
    reason: str  # one-sentence justification quoting the speaker's words
```

In `FrameResult`, **delete** the two existing demo-mode fields (lines 93-94):

```python
    content_type: Literal["demo", "non-demo"] | None = None
    selection_reason: Literal["transcript_anchor", "major_change"] | None = None
```

**Replace** them with:

```python
    # Demo-mode fields populated by the transcript-driven demo filter.
    llm_caption: str | None = None
    llm_reason: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_types.py -v`
Expected: all tests pass.

- [ ] **Step 5: Confirm nothing else references the dropped types**

Run: `uv run python -c "import peeklet.utils.types as t; assert not hasattr(t, 'ContentSegment')"`
Expected: succeeds (no output).

Run: `grep -rn "ContentSegment\|content_type\|selection_reason" src/ tests/ --include="*.py"`
Expected: a couple of remaining references in `src/peeklet/core/demo_filter.py` (the TYPE_CHECKING import) — that's fine, Task 4 will clean them up.

- [ ] **Step 6: Run the full unit test suite**

Run: `uv run pytest tests/unit/ -v`
Expected: everything passes (no regressions).

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/utils/types.py tests/unit/test_types.py
git commit -m "types: replace ContentSegment with Moment, FrameResult demo fields with LLM fields"
```

---

## Task 2: Replace `DemoFilterConfig` with the new fields

**Files:**
- Modify: `src/peeklet/config.py:83-98`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

Edit `tests/unit/test_config.py`. Find and **delete** the existing `test_demo_filter_config_defaults` test (added by the previous plan's Task 3). Then add:

```python
def test_demo_filter_config_defaults():
    cfg = PeekletConfig()
    assert cfg.demo_filter.enabled is False
    assert cfg.demo_filter.llm_provider == "anthropic"
    assert cfg.demo_filter.llm_model == "claude-haiku-4-5"
    assert cfg.demo_filter.frame_search_resolution == 360
    assert cfg.demo_filter.ssim_stability_threshold == 0.92
    assert cfg.demo_filter.forward_search_step_sec == 0.5
    assert cfg.demo_filter.forward_search_window_max_sec == 5.0
    assert cfg.demo_filter.gallery_min_words == 5


def test_demo_filter_config_rejects_unknown_provider():
    import pytest
    from pydantic import ValidationError

    from peeklet.config import DemoFilterConfig

    with pytest.raises(ValidationError):
        DemoFilterConfig(llm_provider="cohere")  # type: ignore[arg-type]
```

If `PeekletConfig` is not already imported at the top of `test_config.py`, add it.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py::test_demo_filter_config_defaults tests/unit/test_config.py::test_demo_filter_config_rejects_unknown_provider -v`
Expected: FAIL with `AttributeError` on the new field names.

- [ ] **Step 3: Implement**

Edit `src/peeklet/config.py`. Find the existing `DemoFilterConfig` (lines 83-98) and replace its body entirely:

```python
class DemoFilterConfig(BaseModel):
    """Demo-mode frame filtering settings.

    Activated via the CLI ``--demo-mode`` flag. The LLM picks screenshot-worthy
    moments from the transcript and Peeklet picks the exact frame for each
    moment using forward-search bidirectional SSIM stability plus an OCR
    gallery check. See the design spec for full details.
    """

    enabled: bool = False
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    # Frame search and selection
    frame_search_resolution: int = Field(default=360, gt=0)
    ssim_stability_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    forward_search_step_sec: float = Field(default=0.5, gt=0.0)
    forward_search_window_max_sec: float = Field(default=5.0, gt=0.0)
    # Gallery detection (reuses _count_words_in_frame)
    gallery_min_words: int = Field(default=5, ge=0)
```

`Literal` is already imported at the top of `config.py` (line 7).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: all tests pass.

- [ ] **Step 5: Lint check**

Run: `uv run ruff format --check src/peeklet/config.py tests/unit/test_config.py && uv run ruff check src/peeklet/config.py tests/unit/test_config.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/config.py tests/unit/test_config.py
git commit -m "config: replace DemoFilterConfig with transcript-driven fields"
```

---

## Task 3: Add `[demo]` extra with `openai` + `anthropic`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the new extra**

Edit `pyproject.toml`. After the existing `video` extra (around lines 36-41), add:

```toml
demo = [
    "openai>=1.0",
    "anthropic>=0.40",
]
```

So the section ends up looking like:

```toml
video = [
    "imageio[ffmpeg]>=2.31",
    "av>=14.0",
    "pydub>=0.25",
    "pytesseract>=0.3.10",
]
demo = [
    "openai>=1.0",
    "anthropic>=0.40",
]
dev = [
    ...
]
```

- [ ] **Step 2: Add mypy overrides for the new SDKs**

In the `[[tool.mypy.overrides]]` `module` list (around line 84), add `"openai.*"` and `"anthropic.*"`:

```toml
module = [
    "pyarrow.*",
    "imagehash.*",
    "skimage.*",
    "av.*",
    "imageio.*",
    "imageio_ffmpeg.*",
    "pydub.*",
    "pytesseract.*",
    "openai.*",
    "anthropic.*",
]
```

- [ ] **Step 3: Sync the lockfile**

Run: `uv sync --extra video --extra demo --extra dev`
Expected: `openai` and `anthropic` are added to `uv.lock` without errors.

- [ ] **Step 4: Verify imports**

Run: `uv run python -c "import openai, anthropic; print(openai.__version__, anthropic.__version__)"`
Expected: prints two version strings.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add openai and anthropic SDKs under [demo] extra"
```

---

## Task 4: Create the `llm.py` module skeleton — Protocol, Moment factory, transcript formatting

**Files:**
- Create: `src/peeklet/core/llm.py`
- Create: `tests/unit/test_llm.py`

This task creates the module skeleton, the Protocol, the factory, the transcript formatting helper, and the error types — but NOT yet the concrete `AnthropicClient` / `OpenAIClient` (those come in Tasks 5 and 6). Building it this way lets us TDD the contract before the concrete clients exist.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_llm.py`:

```python
"""Unit tests for the LLM adapter module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_segments():
    from peeklet.core.audio import TranscriptSegment

    return [
        TranscriptSegment(start=0.0, end=3.5, text="Hey everyone, today I'll show you the dashboard."),
        TranscriptSegment(start=3.5, end=8.2, text="Let me start by signing in here."),
        TranscriptSegment(start=8.2, end=12.1, text="Okay, this is the main view after login."),
    ]


def test_format_transcript_includes_timestamps_and_text():
    from peeklet.core.llm import format_transcript_for_llm

    segments = _make_segments()
    formatted = format_transcript_for_llm(segments)

    assert "[0.0 - 3.5]" in formatted
    assert "Hey everyone" in formatted
    assert "[3.5 - 8.2]" in formatted
    assert "Let me start by signing in here." in formatted
    # One line per segment
    assert formatted.count("\n") >= len(segments) - 1


def test_parse_moments_strips_markdown_fences():
    from peeklet.core.llm import _parse_moments_json

    raw = '```json\n[{"timestamp": 12.5, "caption": "X", "reason": "Y"}]\n```'
    moments = _parse_moments_json(raw, video_duration=60.0)

    assert len(moments) == 1
    assert moments[0].timestamp == 12.5
    assert moments[0].caption == "X"
    assert moments[0].reason == "Y"


def test_parse_moments_drops_out_of_range_timestamps():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        '[{"timestamp": 5.0, "caption": "ok", "reason": "r"},'
        ' {"timestamp": 999.0, "caption": "past end", "reason": "r"},'
        ' {"timestamp": -1.0, "caption": "negative", "reason": "r"}]'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    timestamps = [m.timestamp for m in moments]
    assert timestamps == [5.0]


def test_parse_moments_sorted_by_timestamp():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        '[{"timestamp": 30.0, "caption": "c", "reason": "r"},'
        ' {"timestamp": 5.0, "caption": "c", "reason": "r"},'
        ' {"timestamp": 15.0, "caption": "c", "reason": "r"}]'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    assert [m.timestamp for m in moments] == [5.0, 15.0, 30.0]


def test_parse_moments_raises_on_unparseable_after_strip():
    from peeklet.core.llm import LLMResponseError, _parse_moments_json

    with pytest.raises(LLMResponseError):
        _parse_moments_json("this is not json at all", video_duration=60.0)


def test_parse_moments_raises_on_missing_required_keys():
    from peeklet.core.llm import LLMResponseError, _parse_moments_json

    raw = '[{"timestamp": 5.0, "caption": "no reason"}]'
    with pytest.raises(LLMResponseError):
        _parse_moments_json(raw, video_duration=60.0)


def test_build_llm_client_unknown_provider_raises():
    from peeklet.core.llm import build_llm_client

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm_client(provider="cohere", model="some-model")


def test_build_llm_client_anthropic_missing_key_raises(monkeypatch):
    from peeklet.core.llm import build_llm_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_llm_client(provider="anthropic", model="claude-haiku-4-5")


def test_build_llm_client_openai_missing_key_raises(monkeypatch):
    from peeklet.core.llm import build_llm_client

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_llm_client(provider="openai", model="gpt-4o-mini")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm.py -v`
Expected: all tests fail with `ModuleNotFoundError: No module named 'peeklet.core.llm'`.

- [ ] **Step 3: Implement**

Create `src/peeklet/core/llm.py`:

```python
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


def _parse_moments_json(raw: str, video_duration: float) -> list[Moment]:
    """Parse the raw LLM string into a sorted list of valid Moments.

    - Strips ```` ```json ... ``` ```` markdown fences if present.
    - Drops entries with timestamps outside [0, video_duration].
    - Raises ``LLMResponseError`` if the JSON is unparseable or any entry is
      missing a required key.
    """
    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"LLM returned unparseable JSON. Raw output:\n{raw}"
        ) from exc

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
                    f"LLM moment is missing required key '{key}': {entry!r}\n"
                    f"Raw output:\n{raw}"
                )
        try:
            ts = float(entry["timestamp"])
        except (TypeError, ValueError) as exc:
            raise LLMResponseError(
                f"LLM moment has non-numeric timestamp: {entry!r}"
            ) from exc

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
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Supported: {sorted(_PROVIDER_KEYS)}."
        )

    env_var = _PROVIDER_KEYS[provider]
    if not os.environ.get(env_var):
        raise RuntimeError(
            f"--demo-mode with provider {provider!r} requires the {env_var} "
            f"env var to be set."
        )

    if provider == "anthropic":
        from peeklet.core.llm_anthropic import AnthropicClient

        return AnthropicClient(model=model)
    if provider == "openai":
        from peeklet.core.llm_openai import OpenAIClient

        return OpenAIClient(model=model)

    # Unreachable — guarded above.
    raise AssertionError(f"unhandled provider {provider!r}")
```

Note: `build_llm_client` lazy-imports `AnthropicClient` from a separate file `peeklet.core.llm_anthropic` and `OpenAIClient` from `peeklet.core.llm_openai`. Those modules don't exist yet — Tasks 5 and 6 will create them. Since `build_llm_client` only imports them inside the branches, the tests in this task that don't actually call `build_llm_client` for a configured provider won't trip on the missing modules.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm.py -v`
Expected: all 9 tests pass.

- [ ] **Step 5: Lint**

Run: `uv run ruff format --check src/peeklet/core/llm.py tests/unit/test_llm.py && uv run ruff check src/peeklet/core/llm.py tests/unit/test_llm.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/core/llm.py tests/unit/test_llm.py
git commit -m "llm: add module skeleton with Protocol, factory, transcript formatting, parser"
```

---

## Task 5: Implement `AnthropicClient`

**Files:**
- Create: `src/peeklet/core/llm_anthropic.py`
- Modify: `tests/unit/test_llm.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_llm.py`:

```python
def test_anthropic_client_pick_moments_calls_sdk_and_parses_response(monkeypatch):
    from peeklet.core import llm_anthropic
    from peeklet.utils.types import Moment

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text='[{"timestamp": 7.0, "caption": "c", "reason": "r"}]')]

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    monkeypatch.setattr(llm_anthropic, "anthropic", fake_anthropic)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = llm_anthropic.AnthropicClient(model="claude-haiku-4-5")
    segments = _make_segments()
    moments = client.pick_moments(segments, video_duration=60.0)

    assert moments == [Moment(timestamp=7.0, caption="c", reason="r")]
    fake_anthropic.Anthropic.assert_called_once()
    fake_client.messages.create.assert_called_once()
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5"
    assert "system" in call_kwargs
    assert call_kwargs["messages"][0]["role"] == "user"


def test_anthropic_client_retries_once_on_unparseable(monkeypatch):
    from peeklet.core import llm_anthropic

    bad_response = MagicMock()
    bad_response.content = [MagicMock(text="not json")]
    good_response = MagicMock()
    good_response.content = [MagicMock(text='[{"timestamp": 1.0, "caption": "c", "reason": "r"}]')]

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [bad_response, good_response]

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    monkeypatch.setattr(llm_anthropic, "anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = llm_anthropic.AnthropicClient(model="claude-haiku-4-5")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert len(moments) == 1
    assert fake_client.messages.create.call_count == 2


def test_anthropic_client_raises_after_two_unparseable(monkeypatch):
    from peeklet.core import llm_anthropic
    from peeklet.core.llm import LLMResponseError

    bad_response = MagicMock()
    bad_response.content = [MagicMock(text="garbage")]

    fake_client = MagicMock()
    fake_client.messages.create.return_value = bad_response

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    monkeypatch.setattr(llm_anthropic, "anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = llm_anthropic.AnthropicClient(model="claude-haiku-4-5")
    with pytest.raises(LLMResponseError):
        client.pick_moments(_make_segments(), video_duration=60.0)

    assert fake_client.messages.create.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm.py -v -k "anthropic"`
Expected: 3 new tests fail with `ModuleNotFoundError: No module named 'peeklet.core.llm_anthropic'`.

- [ ] **Step 3: Implement**

Create `src/peeklet/core/llm_anthropic.py`:

```python
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
    import anthropic  # type: ignore[import-not-found]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm.py -v`
Expected: all 12 tests pass.

- [ ] **Step 5: Lint**

Run: `uv run ruff format --check src/peeklet/core/llm_anthropic.py tests/unit/test_llm.py && uv run ruff check src/peeklet/core/llm_anthropic.py tests/unit/test_llm.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/core/llm_anthropic.py tests/unit/test_llm.py
git commit -m "llm: add native AnthropicClient with retry on unparseable response"
```

---

## Task 6: Implement `OpenAIClient`

**Files:**
- Create: `src/peeklet/core/llm_openai.py`
- Modify: `tests/unit/test_llm.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_llm.py`:

```python
def test_openai_client_pick_moments_calls_sdk_and_parses_response(monkeypatch):
    from peeklet.core import llm_openai
    from peeklet.utils.types import Moment

    fake_message = MagicMock()
    fake_message.content = '[{"timestamp": 11.0, "caption": "c", "reason": "r"}]'
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openai, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = llm_openai.OpenAIClient(model="gpt-4o-mini")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert moments == [Moment(timestamp=11.0, caption="c", reason="r")]
    fake_client.chat.completions.create.assert_called_once()
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"


def test_openai_client_passes_base_url_when_set(monkeypatch):
    from peeklet.core import llm_openai

    fake_openai = MagicMock()
    monkeypatch.setattr(llm_openai, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    llm_openai.OpenAIClient(model="llama3")

    fake_openai.OpenAI.assert_called_once()
    call_kwargs = fake_openai.OpenAI.call_args.kwargs
    assert call_kwargs.get("base_url") == "http://localhost:11434/v1"


def test_openai_client_retries_once_on_unparseable(monkeypatch):
    from peeklet.core import llm_openai

    bad_message = MagicMock(content="not json")
    good_message = MagicMock(content='[{"timestamp": 1.0, "caption": "c", "reason": "r"}]')
    bad_response = MagicMock(choices=[MagicMock(message=bad_message)])
    good_response = MagicMock(choices=[MagicMock(message=good_message)])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [bad_response, good_response]

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setattr(llm_openai, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = llm_openai.OpenAIClient(model="gpt-4o-mini")
    moments = client.pick_moments(_make_segments(), video_duration=60.0)

    assert len(moments) == 1
    assert fake_client.chat.completions.create.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm.py -v -k "openai"`
Expected: 3 new tests fail with `ModuleNotFoundError: No module named 'peeklet.core.llm_openai'`.

- [ ] **Step 3: Implement**

Create `src/peeklet/core/llm_openai.py`:

```python
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
    import openai  # type: ignore[import-not-found]
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

        client_kwargs: dict[str, str] = {}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm.py -v`
Expected: all 15 tests pass.

- [ ] **Step 5: Lint**

Run: `uv run ruff format --check src/peeklet/core/llm_openai.py tests/unit/test_llm.py && uv run ruff check src/peeklet/core/llm_openai.py tests/unit/test_llm.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/core/llm_openai.py tests/unit/test_llm.py
git commit -m "llm: add native OpenAIClient with OPENAI_BASE_URL support"
```

---

## Task 7: Stage B helpers — search window, stability check, gallery check

**Files:**
- Modify: `src/peeklet/core/demo_filter.py`
- Modify: `tests/unit/test_demo_filter.py`

This task adds three private helpers and their tests. The public functions come in Tasks 8 and 9. Note: the existing `demo_filter.py` has stale TYPE_CHECKING imports referencing `ContentSegment` from the old design — clean those up here.

- [ ] **Step 1: Clean up the stale imports first (no test needed)**

Edit `src/peeklet/core/demo_filter.py`. Replace lines 1-22 (the docstring, imports, and TYPE_CHECKING block):

```python
"""Demo-mode frame filtering: transcript-driven LLM moment picking + frame selection.

The LLM picks the moments from the transcript, Peeklet picks the exact frame
at each moment using forward-search bidirectional SSIM stability and an OCR
gallery check. See the design spec at
``docs/superpowers/specs/2026-04-09-transcript-driven-demo-mode-design.md``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.video import VideoDecoder
    from peeklet.utils.types import FrameResult, Moment
```

(No more `ContentSegment` import. The `# noqa: F401` comments are also gone — Tasks 8 and 9 will use these symbols, so they'll no longer be unused.)

- [ ] **Step 2: Write the failing tests**

Append to `tests/unit/test_demo_filter.py`:

```python
def test_build_search_window_caps_at_segment_end():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import _build_search_window

    seg = TranscriptSegment(start=10.0, end=12.0, text="x")
    start, end = _build_search_window(
        moment_ts=10.5,
        segment=seg,
        max_window_sec=5.0,
    )
    assert start == 10.5
    # Capped by segment end (12.0), not by max_window
    assert end == 12.0


def test_build_search_window_caps_at_max_window_when_segment_long():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import _build_search_window

    seg = TranscriptSegment(start=0.0, end=100.0, text="long monologue")
    start, end = _build_search_window(
        moment_ts=10.0,
        segment=seg,
        max_window_sec=5.0,
    )
    assert start == 10.0
    assert end == 15.0  # 10.0 + 5.0


def test_is_stable_passes_when_both_neighbors_similar():
    from peeklet.core.demo_filter import _is_stable

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    prev = np.full((100, 100, 3), 128, dtype=np.uint8)
    nxt = np.full((100, 100, 3), 128, dtype=np.uint8)

    assert _is_stable(frame, prev, nxt, threshold=0.92) is True


def test_is_stable_fails_when_one_neighbor_differs():
    from peeklet.core.demo_filter import _is_stable

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    prev = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Random noise — very different from frame, low SSIM
    rng = np.random.default_rng(42)
    nxt = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

    assert _is_stable(frame, prev, nxt, threshold=0.92) is False


def test_is_gallery_frame_returns_true_below_threshold():
    from peeklet.core.demo_filter import _is_gallery_frame

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("peeklet.core.demo_filter._count_words_in_frame", return_value=2):
        assert _is_gallery_frame(frame, downscale_dim=360, min_words=5) is True


def test_is_gallery_frame_returns_false_above_threshold():
    from peeklet.core.demo_filter import _is_gallery_frame

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("peeklet.core.demo_filter._count_words_in_frame", return_value=10):
        assert _is_gallery_frame(frame, downscale_dim=360, min_words=5) is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_demo_filter.py -v -k "build_search_window or is_stable or is_gallery_frame"`
Expected: 6 tests fail with `ImportError: cannot import name '_build_search_window'`, etc.

- [ ] **Step 4: Implement**

Append to `src/peeklet/core/demo_filter.py`:

```python
def _build_search_window(
    moment_ts: float,
    segment: TranscriptSegment,
    max_window_sec: float,
) -> tuple[float, float]:
    """Compute the (start, end) timestamps for the forward-search window.

    The window starts at the LLM's moment timestamp and ends at the earlier of:
    - The end of the transcript segment the moment falls into.
    - ``moment_ts + max_window_sec``.
    """
    end = min(segment.end, moment_ts + max_window_sec)
    return moment_ts, end


def _is_stable(
    frame: np.ndarray,
    prev: np.ndarray,
    nxt: np.ndarray,
    threshold: float,
) -> bool:
    """Bidirectional SSIM check: a frame is stable if both neighbors are similar."""
    from peeklet.core.comparator import compare_frames

    prev_result = compare_frames(frame, prev)
    next_result = compare_frames(frame, nxt)
    return prev_result.ssim_score > threshold and next_result.ssim_score > threshold


def _is_gallery_frame(frame: np.ndarray, downscale_dim: int, min_words: int) -> bool:
    """Return True if the frame has too few visible words to be demo content."""
    return _count_words_in_frame(frame, downscale_dim) < min_words
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_demo_filter.py -v`
Expected: all tests pass (the original Task 5 tests + the 6 new ones).

- [ ] **Step 6: Lint**

Run: `uv run ruff format --check src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py && uv run ruff check src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py
git commit -m "demo_filter: add Stage B helpers (window, stability, gallery)"
```

---

## Task 8: Implement `select_frames_for_moments()` (Stage B public)

**Files:**
- Modify: `src/peeklet/core/demo_filter.py`
- Modify: `tests/unit/test_demo_filter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_demo_filter.py`:

```python
def _make_decoder_for_moments(meta_duration: float = 60.0):
    """Create a decoder mock that returns a deterministic frame per timestamp."""
    decoder = MagicMock()
    decoder.get_metadata.return_value = MagicMock(
        duration=meta_duration, filename="t.mp4"
    )
    # Each call to extract_frame_at returns a flat-color frame keyed by ts
    # so SSIM comparisons between adjacent calls are predictable.
    def _fake_extract(ts: float):
        # Build a frame whose grey value is ts*10 mod 256, large enough for SSIM.
        val = int((ts * 10) % 256)
        return np.full((100, 100, 3), val, dtype=np.uint8), float(ts), int(ts * 30)

    decoder.extract_frame_at.side_effect = _fake_extract
    return decoder


def test_select_frames_for_moments_picks_first_stable_frame(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    # All frames identical → all stable → first one wins
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )

    moments = [Moment(timestamp=10.0, caption="cap", reason="reason")]
    transcript = [TranscriptSegment(start=8.0, end=12.0, text="speaking")]
    cfg = DemoFilterConfig(
        enabled=True,
        ssim_stability_threshold=0.92,
        forward_search_step_sec=0.5,
        forward_search_window_max_sec=5.0,
        gallery_min_words=0,  # disable gallery check for this test
    )

    monkeypatch.setattr(
        "peeklet.core.demo_filter.save_keyframe",
        lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
    )

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert len(results) == 1
    r = results[0]
    assert r.is_keyframe is True
    assert r.llm_caption == "cap"
    assert r.llm_reason == "reason"
    # Picked frame is somewhere in the search window [10.0, 12.0].
    # With all-identical frames, _pick_stable_index returns index 1 (the first
    # checkable position, since index 0 has no prev neighbor) → ts == 10.5.
    assert r.video_timestamp is not None
    assert 10.0 <= r.video_timestamp <= 12.0
    assert r.trigger_type == "transcript_trigger"


def test_select_frames_for_moments_drops_gallery_frames(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.zeros((100, 100, 3), dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )

    moments = [Moment(timestamp=10.0, caption="cap", reason="reason")]
    transcript = [TranscriptSegment(start=8.0, end=12.0, text="speaking")]
    cfg = DemoFilterConfig(
        enabled=True,
        gallery_min_words=5,
    )

    # Force the gallery check to report "no words" so this moment gets dropped.
    monkeypatch.setattr(
        "peeklet.core.demo_filter._count_words_in_frame",
        lambda frame, downscale_dim: 0,
    )
    monkeypatch.setattr(
        "peeklet.core.demo_filter.save_keyframe",
        lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
    )

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert results == []


def test_select_frames_for_moments_drops_moment_with_no_segment(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments()
    monkeypatch.setattr(
        "peeklet.core.demo_filter.save_keyframe",
        lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
    )

    # Moment timestamp is 50, but the only segment ends at 10
    moments = [Moment(timestamp=50.0, caption="c", reason="r")]
    transcript = [TranscriptSegment(start=0.0, end=10.0, text="x")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    # No matching transcript segment → moment dropped
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_demo_filter.py -v -k "select_frames_for_moments"`
Expected: 3 tests fail with `ImportError: cannot import name 'select_frames_for_moments'`.

- [ ] **Step 3: Update the imports at the top of `src/peeklet/core/demo_filter.py`**

Replace the existing imports section (the docstring + `from __future__ import annotations` block + TYPE_CHECKING block from Task 7) with:

```python
"""Demo-mode frame filtering: transcript-driven LLM moment picking + frame selection.

The LLM picks the moments from the transcript, Peeklet picks the exact frame
at each moment using forward-search bidirectional SSIM stability and an OCR
gallery check. See the design spec at
``docs/superpowers/specs/2026-04-09-transcript-driven-demo-mode-design.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from peeklet.core.exporter import save_keyframe
from peeklet.utils.types import EventType, FrameResult

if TYPE_CHECKING:
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.video import VideoDecoder
    from peeklet.utils.types import Moment
```

Note: `FrameResult` and `EventType` move OUT of `TYPE_CHECKING` because they're now used at runtime inside `select_frames_for_moments`. `Moment` stays in `TYPE_CHECKING` because it's only an annotation.

- [ ] **Step 4: Append the new functions**

Append to `src/peeklet/core/demo_filter.py`:

```python
def _find_segment_for_timestamp(
    ts: float, transcript: list[TranscriptSegment]
) -> TranscriptSegment | None:
    """Return the transcript segment containing ``ts``, or None."""
    for seg in transcript:
        if seg.start <= ts <= seg.end:
            return seg
    return None


def _sample_window_frames(
    decoder: VideoDecoder,
    start: float,
    end: float,
    step: float,
) -> list[tuple[np.ndarray, float, int]]:
    """Decode a small set of frames from the search window."""
    if end <= start:
        try:
            return [decoder.extract_frame_at(start)]
        except Exception as exc:
            logger.warning("Failed to extract frame at %.2fs: %s", start, exc)
            return []
    timestamps: list[float] = []
    t = start
    while t <= end + 1e-6:
        timestamps.append(round(t, 6))
        t += step
    samples: list[tuple[np.ndarray, float, int]] = []
    for ts in timestamps:
        try:
            samples.append(decoder.extract_frame_at(ts))
        except Exception as exc:
            logger.warning("Failed to extract frame at %.2fs: %s", ts, exc)
    return samples


def _pick_stable_index(
    samples: list[tuple[np.ndarray, float, int]],
    threshold: float,
) -> int:
    """Return the index of the first sample whose two neighbors are SSIM-similar.

    Falls back to index 0 if no sample passes the bidirectional check (or if
    the window has fewer than 3 samples to compare).
    """
    if len(samples) < 3:
        return 0
    for i in range(1, len(samples) - 1):
        frame, _, _ = samples[i]
        prev_frame, _, _ = samples[i - 1]
        next_frame, _, _ = samples[i + 1]
        if _is_stable(frame, prev_frame, next_frame, threshold):
            return i
    return 0


def select_frames_for_moments(
    decoder: VideoDecoder,
    moments: list[Moment],
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
    output_dir: Path,
) -> list[FrameResult]:
    """Stage B: for each LLM-picked moment, find the best actual frame.

    Walks each moment, builds a forward search window inside the current
    transcript segment, samples frames at ``forward_search_step_sec`` intervals,
    picks the first bidirectionally-stable frame, runs the gallery check, and
    saves the surviving frame as a keyframe.
    """
    output_dir = Path(output_dir)
    meta = decoder.get_metadata()
    results: list[FrameResult] = []

    for idx, moment in enumerate(moments, start=1):
        seg = _find_segment_for_timestamp(moment.timestamp, transcript)
        if seg is None:
            logger.warning(
                "LLM moment at %.2fs ('%s') has no matching transcript segment, skipping",
                moment.timestamp,
                moment.caption,
            )
            continue

        win_start, win_end = _build_search_window(
            moment_ts=moment.timestamp,
            segment=seg,
            max_window_sec=config.forward_search_window_max_sec,
        )
        samples = _sample_window_frames(
            decoder=decoder,
            start=win_start,
            end=win_end,
            step=config.forward_search_step_sec,
        )
        if not samples:
            logger.warning(
                "No frames could be extracted for moment at %.2fs, skipping",
                moment.timestamp,
            )
            continue

        picked_idx = _pick_stable_index(samples, config.ssim_stability_threshold)
        picked_frame, picked_ts, picked_frame_num = samples[picked_idx]

        if _is_gallery_frame(
            picked_frame,
            downscale_dim=config.frame_search_resolution,
            min_words=config.gallery_min_words,
        ):
            logger.warning(
                "LLM picked moment at %.2fs ('%s') but the frame is gallery-view "
                "or blank — no demo content visible. Skipping.",
                moment.timestamp,
                moment.caption,
            )
            continue

        frame_id = f"demo_{idx:04d}_{int(picked_ts):04d}s"
        asset_path = save_keyframe(picked_frame, output_dir, frame_id, fmt="jpg")

        results.append(
            FrameResult(
                frame_id=frame_id,
                event_type=EventType.KEYFRAME,
                is_keyframe=True,
                perceptual_hash="",  # not computed in demo mode
                frame_width=picked_frame.shape[1],
                frame_height=picked_frame.shape[0],
                source_format="video",
                asset_path=str(asset_path),
                trigger_type="transcript_trigger",
                source_video=meta.filename,
                video_timestamp=picked_ts,
                video_frame_number=picked_frame_num,
                video_duration=meta.duration,
                llm_caption=moment.caption,
                llm_reason=moment.reason,
                keyframe_index=idx,
            )
        )

    # Backfill total_keyframes
    for r in results:
        r.total_keyframes = len(results)
    return results
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_demo_filter.py -v`
Expected: all tests pass.

- [ ] **Step 6: Lint**

Run: `uv run ruff format --check src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py && uv run ruff check src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py
git commit -m "demo_filter: implement select_frames_for_moments (Stage B public)"
```

---

## Task 9: Implement `apply_demo_filter()` (top-level entry point)

**Files:**
- Modify: `src/peeklet/core/demo_filter.py`
- Modify: `tests/unit/test_demo_filter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_demo_filter.py`:

```python
def test_apply_demo_filter_calls_llm_then_select(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import apply_demo_filter
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    transcript = [TranscriptSegment(start=8.0, end=12.0, text="speaking")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0)

    fake_moments = [Moment(timestamp=10.0, caption="c", reason="r")]
    fake_client = MagicMock()
    fake_client.pick_moments.return_value = fake_moments

    monkeypatch.setattr(
        "peeklet.core.demo_filter.build_llm_client",
        lambda provider, model: fake_client,
    )
    monkeypatch.setattr(
        "peeklet.core.demo_filter.save_keyframe",
        lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
    )
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 200, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )

    results = apply_demo_filter(
        decoder=decoder,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    fake_client.pick_moments.assert_called_once_with(transcript, 60.0)
    assert len(results) == 1
    assert results[0].llm_caption == "c"


def test_apply_demo_filter_logs_warning_on_zero_moments(tmp_path, monkeypatch, caplog):
    import logging

    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    fake_client = MagicMock()
    fake_client.pick_moments.return_value = []  # LLM picked nothing

    monkeypatch.setattr(
        "peeklet.core.demo_filter.build_llm_client",
        lambda provider, model: fake_client,
    )

    cfg = DemoFilterConfig(enabled=True)
    with caplog.at_level(logging.WARNING):
        results = apply_demo_filter(
            decoder=decoder,
            transcript=[],
            config=cfg,
            output_dir=tmp_path,
        )

    assert results == []
    assert any("zero screenshot-worthy moments" in rec.message for rec in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_demo_filter.py -v -k "apply_demo_filter"`
Expected: 2 tests fail with `ImportError: cannot import name 'apply_demo_filter'`.

- [ ] **Step 3: Add the `build_llm_client` import**

Edit `src/peeklet/core/demo_filter.py`. After the existing `from peeklet.core.exporter import save_keyframe` line (added in Task 8), add a new line:

```python
from peeklet.core.llm import build_llm_client
```

So the runtime imports section now reads:

```python
from peeklet.core.exporter import save_keyframe
from peeklet.core.llm import build_llm_client
from peeklet.utils.types import EventType, FrameResult
```

- [ ] **Step 4: Append the new function**

Append to `src/peeklet/core/demo_filter.py`:

```python
def apply_demo_filter(
    decoder: VideoDecoder,
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
    output_dir: Path,
) -> list[FrameResult]:
    """Top-level demo-mode entry point.

    Builds the LLM client, asks it to pick screenshot-worthy moments from the
    transcript, then runs Stage B (forward-search + stability + gallery check)
    to pick the actual frames. Returns the curated keyframe list.
    """
    meta = decoder.get_metadata()
    client = build_llm_client(provider=config.llm_provider, model=config.llm_model)

    moments = client.pick_moments(transcript, meta.duration)
    logger.info("LLM picked %d screenshot-worthy moments", len(moments))

    if not moments:
        logger.warning(
            "LLM identified zero screenshot-worthy moments in this transcript."
        )
        return []

    return select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=config,
        output_dir=output_dir,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_demo_filter.py -v`
Expected: all tests pass.

- [ ] **Step 6: Lint**

Run: `uv run ruff format --check src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py && uv run ruff check src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py
git commit -m "demo_filter: add apply_demo_filter top-level entry point"
```

---

## Task 10: Wire `apply_demo_filter()` into `process_video()`

**Files:**
- Modify: `src/peeklet/core/video.py`
- Modify: `tests/unit/test_video.py`

The integration is simpler than the previous plan because the new design doesn't use the existing coarse pass at all when demo mode is on. We *replace* the keyframe path entirely with `apply_demo_filter`'s output.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_video.py`:

```python
def test_process_video_calls_demo_filter_when_enabled(tmp_path, monkeypatch):
    """When config.demo_filter.enabled is True, process_video calls apply_demo_filter."""
    from unittest.mock import MagicMock

    from peeklet.config import PeekletConfig
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "synthetic.mp4"
    write_synthetic_video(video_path, duration_sec=3, fps=10, width=64, height=64)

    transcript_path = tmp_path / "transcript.srt"
    transcript_path.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nhello world\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\ngoodbye\n"
    )

    cfg = PeekletConfig()
    cfg.exporter.output_dir = str(tmp_path / "out")
    cfg.video.audio_detection = False
    cfg.video.transcript_path = str(transcript_path)
    cfg.demo_filter.enabled = True

    fake_filtered: list = []
    mock_apply = MagicMock(return_value=fake_filtered)
    monkeypatch.setattr(video_module, "apply_demo_filter", mock_apply)

    results = video_module.process_video(video_path, cfg)

    assert mock_apply.called, "apply_demo_filter should be invoked when demo_filter.enabled"
    assert results == fake_filtered


def test_process_video_skips_demo_filter_when_disabled(tmp_path, monkeypatch):
    """When demo_filter.enabled is False, apply_demo_filter is NOT called."""
    from unittest.mock import MagicMock

    from peeklet.config import PeekletConfig
    from peeklet.core import video as video_module
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "synthetic.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)

    cfg = PeekletConfig()
    cfg.exporter.output_dir = str(tmp_path / "out")
    cfg.video.audio_detection = False
    cfg.demo_filter.enabled = False

    mock_apply = MagicMock()
    monkeypatch.setattr(video_module, "apply_demo_filter", mock_apply)

    video_module.process_video(video_path, cfg)

    assert not mock_apply.called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_video.py::test_process_video_calls_demo_filter_when_enabled -v`
Expected: FAIL with `AttributeError: module 'peeklet.core.video' has no attribute 'apply_demo_filter'`.

- [ ] **Step 3: Implement — add the imports**

Edit `src/peeklet/core/video.py`. Add `import logging` to the stdlib imports (after `from pathlib import Path`):

```python
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
```

After the existing local imports (around line 26, after `from peeklet.utils.types import EventType`), add:

```python
from peeklet.core.demo_filter import apply_demo_filter
```

Then immediately below the imports, before the `if TYPE_CHECKING:` block, add the module-level logger:

```python
logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Implement — branch on `config.demo_filter.enabled`**

In `process_video()`, locate this section:

```python
    decoder = VideoDecoder(path)
    meta = decoder.get_metadata()
    sample_fps = config.video.sample_fps
    output_dir = Path(config.exporter.output_dir)
    max_dim = config.video.processing_max_dim
    forced_ts = forced_timestamps or []

    owns_writer = writer is None
    if writer is None:
        writer = ManifestWriter(
            path=output_dir / "manifest.parquet",
            compression=config.exporter.parquet_compression,
        )
```

Immediately AFTER that block, add the demo-mode short-circuit:

```python
    # Demo mode: bypass the coarse pass entirely. The LLM picks moments,
    # Stage B picks frames, and we write outputs directly.
    if config.demo_filter.enabled:
        transcript_segments: list[TranscriptSegment] = []
        if config.video.transcript_path:
            transcript_segments = parse_transcript(Path(config.video.transcript_path))

        results = apply_demo_filter(
            decoder=decoder,
            transcript=transcript_segments,
            config=config.demo_filter,
            output_dir=output_dir,
        )

        ctx = build_context(meta.filename, meta.duration, results, transcript_segments)
        write_context_json(ctx, output_dir / "context.json")
        write_context_markdown(ctx, output_dir / "context.md")

        for r in results:
            writer.append(r)
        if owns_writer:
            writer.flush()

        return results
```

The existing coarse-pass code below remains unchanged — it runs only when demo mode is OFF.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_video.py -v`
Expected: all tests pass.

- [ ] **Step 6: Lint**

Run: `uv run ruff format --check src/peeklet/core/video.py tests/unit/test_video.py && uv run ruff check src/peeklet/core/video.py tests/unit/test_video.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/core/video.py tests/unit/test_video.py
git commit -m "video: short-circuit process_video to demo filter when enabled"
```

---

## Task 11: Add CLI flags `--demo-mode`, `--llm-provider`, `--llm-model`

**Files:**
- Modify: `src/peeklet/cli.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_cli.py`:

```python
def test_cli_demo_mode_requires_transcript(tmp_path):
    """--demo-mode without --transcript exits with a clear error."""
    from click.testing import CliRunner

    from peeklet.cli import main
    from tests.unit.helpers_video import write_synthetic_video

    video_path = tmp_path / "v.mp4"
    write_synthetic_video(video_path, duration_sec=2, fps=10, width=64, height=64)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--input", str(video_path), "--output", str(tmp_path / "out"), "--demo-mode"],
    )
    assert result.exit_code == 2
    assert "--demo-mode requires --transcript" in result.output


def test_cli_demo_mode_requires_video(tmp_path):
    """--demo-mode without a video input exits with a clear error."""
    from click.testing import CliRunner

    from peeklet.cli import main

    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    (images_dir / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    transcript = tmp_path / "t.srt"
    transcript.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            str(images_dir),
            "--output",
            str(tmp_path / "out"),
            "--mode",
            "image",
            "--transcript",
            str(transcript),
            "--demo-mode",
        ],
    )
    assert result.exit_code == 2
    assert "--demo-mode only applies to video inputs" in result.output


def test_cli_demo_mode_propagates_provider_and_model(tmp_path, monkeypatch):
    """--llm-provider and --llm-model end up on config.demo_filter."""
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
        captured["enabled"] = config.demo_filter.enabled
        captured["provider"] = config.demo_filter.llm_provider
        captured["model"] = config.demo_filter.llm_model
        return real_process_video(path, config, **kwargs)

    monkeypatch.setattr(video_module, "process_video", spy_process_video)
    # Stub out the demo filter so the test doesn't need a real LLM call.
    # Patch on video_module — that's where process_video has imported the
    # symbol. Patching df_module would not affect the already-imported reference.
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
            "openai",
            "--llm-model",
            "gpt-4o-mini",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "enabled": True,
        "provider": "openai",
        "model": "gpt-4o-mini",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli.py -v -k "demo_mode"`
Expected: 3 tests fail with `Error: No such option: --demo-mode`.

- [ ] **Step 3: Implement — add the click options**

Edit `src/peeklet/cli.py`. After the existing `--transcript` click option (around line 81), add:

```python
@click.option(
    "--demo-mode",
    "demo_mode",
    is_flag=True,
    default=False,
    help="Use the LLM to pick screenshot-worthy moments from the transcript "
    "and Peeklet to pick the actual frames. Requires --transcript and a video input.",
)
@click.option(
    "--llm-provider",
    "llm_provider",
    type=click.Choice(["anthropic", "openai"]),
    default=None,
    envvar="PEEKLET_LLM_PROVIDER",
    help="LLM provider for --demo-mode (anthropic or openai). "
    "Defaults to the value in config.demo_filter.llm_provider.",
)
@click.option(
    "--llm-model",
    "llm_model",
    type=str,
    default=None,
    envvar="PEEKLET_LLM_MODEL",
    help="LLM model identifier for --demo-mode. "
    "Defaults to the value in config.demo_filter.llm_model.",
)
```

- [ ] **Step 4: Implement — update `main()` signature**

Update the `main()` function signature (around line 83) to accept the three new parameters:

```python
def main(
    input_path: Path,
    output_dir: Path,
    config_path: Path | None,
    no_audio: bool,
    mode: str | None,
    transcript_path: Path | None,
    demo_mode: bool,
    llm_provider: str | None,
    llm_model: str | None,
) -> None:
```

- [ ] **Step 5: Implement — wire activation rules**

Inside `main()`, after `config.exporter.output_dir = str(output_dir)` and BEFORE `image_extensions = ...`, add:

```python
    if demo_mode:
        if not transcript_path:
            raise click.UsageError(
                "--demo-mode requires --transcript. Demo mode needs both a "
                "video and a transcript to filter frames effectively."
            )
        config.demo_filter.enabled = True
        if llm_provider is not None:
            config.demo_filter.llm_provider = llm_provider
        if llm_model is not None:
            config.demo_filter.llm_model = llm_model
```

After `detected_mode = _detect_mode(...)`, add:

```python
    if demo_mode and detected_mode != "video":
        raise click.UsageError("--demo-mode only applies to video inputs.")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: all tests pass.

- [ ] **Step 7: Lint**

Run: `uv run ruff format --check src/peeklet/cli.py tests/unit/test_cli.py && uv run ruff check src/peeklet/cli.py tests/unit/test_cli.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/peeklet/cli.py tests/unit/test_cli.py
git commit -m "cli: add --demo-mode, --llm-provider, --llm-model with validation"
```

---

## Task 12: End-to-end integration test

**Files:**
- Create: `tests/integration/test_demo_mode_pipeline.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_demo_mode_pipeline.py`:

```python
"""End-to-end test for the --demo-mode video pipeline.

Uses a tiny synthetic video and a fake LLM client (no real API calls). The
fake client returns a hardcoded list of moments, and the test verifies that
Stage B picks frames, applies the gallery check, and writes the manifest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.unit.helpers_video import write_synthetic_video


@pytest.fixture
def synthetic_video(tmp_path: Path) -> tuple[Path, Path]:
    video_path = tmp_path / "demo.mp4"
    write_synthetic_video(video_path, duration_sec=6, fps=10, width=64, height=64)

    transcript_path = tmp_path / "demo.srt"
    transcript_path.write_text(
        "1\n00:00:00,500 --> 00:00:02,500\nhere is the first thing\n\n"
        "2\n00:00:03,500 --> 00:00:05,500\nnow look at the second thing\n",
        encoding="utf-8",
    )
    return video_path, transcript_path


def test_demo_mode_end_to_end_with_fake_llm(
    synthetic_video: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch,
) -> None:
    from peeklet.config import PeekletConfig
    from peeklet.core import demo_filter as df_module
    from peeklet.core.video import process_video
    from peeklet.utils.types import Moment

    video_path, transcript_path = synthetic_video

    cfg = PeekletConfig()
    cfg.exporter.output_dir = str(tmp_path / "out")
    cfg.video.audio_detection = False
    cfg.video.transcript_path = str(transcript_path)
    cfg.demo_filter.enabled = True
    cfg.demo_filter.gallery_min_words = 0  # disable gallery check for synthetic video

    fake_client = MagicMock()
    fake_client.pick_moments.return_value = [
        Moment(timestamp=1.5, caption="first thing", reason="speaker says here is the first thing"),
        Moment(timestamp=4.5, caption="second thing", reason="speaker says now look at the second thing"),
    ]
    monkeypatch.setattr(
        df_module,
        "build_llm_client",
        lambda provider, model: fake_client,
    )

    results = process_video(video_path, cfg)

    assert len(results) == 2
    assert all(r.is_keyframe for r in results)
    captions = [r.llm_caption for r in results]
    assert "first thing" in captions
    assert "second thing" in captions

    # Outputs should be written
    out_dir = Path(cfg.exporter.output_dir)
    assert (out_dir / "manifest.parquet").exists()
    assert (out_dir / "context.json").exists()
    assert (out_dir / "context.md").exists()
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/integration/test_demo_mode_pipeline.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_demo_mode_pipeline.py
git commit -m "test: end-to-end demo-mode pipeline test with fake LLM client"
```

---

## Task 13: Lint, format, and full test sweep

- [ ] **Step 1: Format check**

Run: `uv run ruff format --check src/ tests/`
Expected: PASS. If anything is unformatted, run `uv run ruff format src/ tests/`.

- [ ] **Step 2: Lint check**

Run: `uv run ruff check src/ tests/`
Expected: no errors.

- [ ] **Step 3: Type check**

Run: `uv run mypy src/peeklet`
Expected: no errors.

- [ ] **Step 4: Full test suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Commit any format/lint fixes**

```bash
git status
git add -u
git diff --cached --quiet || git commit -m "style: ruff format and lint fixes"
```

---

## Task 14: Update README + final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a demo-mode section**

Edit `README.md`. Find the "Usage" or "Video processing" section that documents `--transcript`. After it, add:

````markdown
### Demo mode (transcript-driven)

For demo videos with narration, use `--demo-mode` to have an LLM pick the
screenshot-worthy moments from the transcript. Peeklet then uses its
frame-level intelligence (forward search + bidirectional SSIM stability +
OCR gallery check) to pick the exact frame for each moment.

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
peeklet --input demo.mp4 --transcript demo.srt --demo-mode --output ./out
```

`--demo-mode` requires:
- A video input (single file)
- A transcript file via `--transcript` (SRT, VTT, or Fathom-style markdown)
- The matching API key env var for the chosen LLM provider
- The `[demo]` extra installed: `pip install peeklet[demo]`
- The `[video]` extra (for the OCR gallery check): `pip install peeklet[video]`
- The `tesseract` binary (`brew install tesseract` on macOS,
  `apt install tesseract-ocr` on Linux)

**Provider selection:**

```bash
# Use a different Anthropic model
peeklet ... --demo-mode --llm-model "claude-haiku-4-5"

# Use OpenAI
peeklet ... --demo-mode --llm-provider openai --llm-model "gpt-4o-mini"

# Use a local Ollama model via OpenAI-compatible endpoint
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
peeklet ... --demo-mode --llm-provider openai --llm-model "llama3"
```

See `docs/superpowers/specs/2026-04-09-transcript-driven-demo-mode-design.md`
for the full algorithm.
````

- [ ] **Step 2: Verify the CLI help shows the new flags**

Run: `uv run peeklet --help`
Expected: `--demo-mode`, `--llm-provider`, `--llm-model` all appear in the options list.

- [ ] **Step 3: Run the full test suite one more time**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Commit and push**

```bash
git add README.md
git commit -m "docs: document --demo-mode in README"
git push -u origin feat/demo-mode-frame-filtering
```

The feature is complete. Open a PR against `develop` for review.
