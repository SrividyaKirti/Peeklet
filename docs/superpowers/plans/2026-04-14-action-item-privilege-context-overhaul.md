# Action-Item Privilege + Context Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse Fathom ACTION ITEM anchors as guaranteed captures, rewrite the LLM prompt for richer moment metadata, preserve speaker attribution, and overhaul `context.md` for downstream MLLM consumption.

**Architecture:** The change spans 4 layers bottom-up: data model (`Moment`, `TranscriptSegment`, `FrameResult`), transcript parsing (`audio.py`), LLM prompt + JSON parsing (`llm.py`), merge + privilege logic (`demo_filter.py`), and output formatting (`context_exporter.py`). Each task is self-contained and commits independently.

**Tech Stack:** Python 3.10+, pydantic (config), pytest, numpy, regex.

**Spec:** `docs/superpowers/specs/2026-04-14-action-item-privilege-context-overhaul-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/peeklet/utils/types.py` | Modify | Rename Moment fields, add `source`; rename FrameResult demo fields |
| `src/peeklet/core/audio.py` | Modify | Add `speaker` to TranscriptSegment; extract speaker in `_parse_fathom_md`; new `parse_fathom_anchors` |
| `src/peeklet/core/llm.py` | Modify | New SYSTEM_PROMPT; update `_parse_moments_json` keys; update `format_transcript_for_llm` for speaker |
| `src/peeklet/core/llm_openai.py` | No change | Uses shared `SYSTEM_PROMPT` and `_parse_moments_json` from `llm.py` |
| `src/peeklet/core/llm_anthropic.py` | No change | Same |
| `src/peeklet/core/llm_openrouter.py` | No change | Same |
| `src/peeklet/core/demo_filter.py` | Modify | New `merge_moments`; anchor privilege in `select_frames_for_moments`; wiring in `apply_demo_filter` |
| `src/peeklet/core/context_exporter.py` | Modify | Rewrite `build_context` + `write_context_markdown` for new format |
| `src/peeklet/core/video.py:331-345` | Modify | Pass raw transcript text to `apply_demo_filter` for anchor parsing |
| `tests/unit/test_types.py` | Modify | Update for renamed fields |
| `tests/unit/test_audio.py` | Modify | Tests for speaker extraction + `parse_fathom_anchors` |
| `tests/unit/test_llm.py` | Modify | Update all Moment field references + test new prompt keys |
| `tests/unit/test_demo_filter.py` | Modify | Tests for `merge_moments` + anchor privilege semantics |
| `tests/unit/test_context_exporter.py` | Modify | Tests for new context.md format |
| `tests/integration/test_demo_mode_pipeline.py` | Modify | Update Moment construction + assertions |

---

### Task 1: Rename Moment fields and add source

**Files:**
- Modify: `src/peeklet/utils/types.py:37-43`
- Modify: `tests/unit/test_types.py` (Moment tests)

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_types.py`, add a test for the new Moment shape:

```python
def test_moment_new_fields():
    from peeklet.utils.types import Moment

    m = Moment(
        timestamp=12.5,
        visual_context_goal="Dashboard with cost breakdown",
        textual_anchor="Let me show you the estimated cost",
        downstream_utility="To extract specific cost values",
    )
    assert m.timestamp == 12.5
    assert m.visual_context_goal == "Dashboard with cost breakdown"
    assert m.textual_anchor == "Let me show you the estimated cost"
    assert m.downstream_utility == "To extract specific cost values"
    assert m.source == "llm"  # default


def test_moment_anchor_source():
    from peeklet.utils.types import Moment

    m = Moment(
        timestamp=232.0,
        visual_context_goal="Analytics interaction breakdown",
        textual_anchor="ACTION ITEM: Fix missing assistant prompt",
        downstream_utility="Action item flagged by meeting tool",
        source="anchor",
    )
    assert m.source == "anchor"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_types.py::test_moment_new_fields tests/unit/test_types.py::test_moment_anchor_source -v`
Expected: FAIL — `Moment.__init__() got an unexpected keyword argument 'visual_context_goal'`

- [ ] **Step 3: Update Moment dataclass**

In `src/peeklet/utils/types.py`, replace lines 37-43:

```python
@dataclass(frozen=True, slots=True)
class Moment:
    """A screenshot-worthy moment in a video transcript."""

    timestamp: float  # seconds into the video
    visual_context_goal: str  # what the screenshot needs to capture
    textual_anchor: str  # the transcript line that triggers this need
    downstream_utility: str  # why the MLLM needs this image
    source: str = "llm"  # "llm" | "anchor"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_types.py::test_moment_new_fields tests/unit/test_types.py::test_moment_anchor_source -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/utils/types.py tests/unit/test_types.py
git commit -m "refactor: rename Moment fields to visual_context_goal/textual_anchor/downstream_utility + add source"
```

**Note:** This commit will break existing tests that construct `Moment(timestamp=..., caption=..., reason=...)`. That's expected — Tasks 5, 6, 7 fix all downstream references. Do NOT run full pytest yet.

---

### Task 2: Rename FrameResult demo fields

**Files:**
- Modify: `src/peeklet/utils/types.py:92-94`
- Modify: `tests/unit/test_types.py:164-174`

- [ ] **Step 1: Update the existing test**

In `tests/unit/test_types.py`, replace `test_frame_result_llm_fields_default_none` (lines 164-174):

```python
def test_frame_result_demo_fields_default_none():
    from peeklet.utils.types import EventType, FrameResult

    r = FrameResult(
        frame_id="f",
        event_type=EventType.SKIPPED,
        is_keyframe=False,
        perceptual_hash="0" * 16,
        frame_width=10,
        frame_height=10,
    )
    assert r.visual_context_goal is None
    assert r.textual_anchor is None
    assert r.downstream_utility is None
    assert r.moment_source is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_types.py::test_frame_result_demo_fields_default_none -v`
Expected: FAIL — `AttributeError: 'FrameResult' object has no attribute 'visual_context_goal'`

- [ ] **Step 3: Update FrameResult**

In `src/peeklet/utils/types.py`, replace lines 92-94:

```python
    # Demo-mode fields populated by the transcript-driven demo filter.
    visual_context_goal: str | None = None
    textual_anchor: str | None = None
    downstream_utility: str | None = None
    moment_source: str | None = None  # "llm" | "anchor"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_types.py::test_frame_result_demo_fields_default_none -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/utils/types.py tests/unit/test_types.py
git commit -m "refactor: rename FrameResult llm_caption/llm_reason to visual_context_goal/textual_anchor/downstream_utility/moment_source"
```

---

### Task 3: Add speaker field to TranscriptSegment and extract in Fathom parser

**Files:**
- Modify: `src/peeklet/core/audio.py:10-16, 83-84, 101-132`
- Modify: `tests/unit/test_audio.py:77-148`

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_audio.py`, add to `TestParseFathomMd`:

```python
    def test_speaker_extracted(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text(
            "++[@0:00](https://fathom.video/calls/1?timestamp=0.56)++ - **Alice**  \n"
            "Welcome everyone.  \n"
            "\n"
            "++[@0:03](https://fathom.video/calls/1?timestamp=3.0)++ - **Bob Smith**  \n"
            "Thanks for having me.  \n"
            "\n"
        )
        segments = parse_transcript(md)
        assert len(segments) == 2
        assert segments[0].speaker == "Alice"
        assert segments[1].speaker == "Bob Smith"

    def test_srt_has_no_speaker(self, tmp_path: Path) -> None:
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n"
        )
        segments = parse_transcript(srt_file)
        assert segments[0].speaker is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_audio.py::TestParseFathomMd::test_speaker_extracted tests/unit/test_audio.py::TestParseSrt::test_srt_has_no_speaker -v`
Expected: FAIL — `TranscriptSegment.__init__() got an unexpected keyword argument 'speaker'` or `AttributeError`

- [ ] **Step 3: Implement**

In `src/peeklet/core/audio.py`:

**3a.** Add `speaker` to `TranscriptSegment` (line 16):

```python
@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A timestamped segment from an SRT or VTT transcript."""

    start: float  # seconds
    end: float  # seconds
    text: str
    speaker: str | None = None
```

**3b.** Update `_FATHOM_TS_LINE_RE` (line 83-84) to capture speaker name:

```python
_FATHOM_TS_LINE_RE = re.compile(
    r"^\s*\+\+\[@\d+:\d+\]\([^)]*\?timestamp=(\d+(?:\.\d+)?)\)\+\+\s*-\s*\*\*([^*]+)\*\*\s*$"
)
```

**3c.** Update `_parse_fathom_md` to extract and pass speaker. Change `raw_segments` type to `list[tuple[float, str | None, list[str]]]` and update the parsing loop:

```python
def _parse_fathom_md(text: str) -> list[TranscriptSegment]:
    """Parse a Fathom-style markdown transcript.

    Each segment looks like::

        ++[@0:03](https://fathom.video/calls/123?timestamp=3.0)++ - **Speaker**
        Spoken text here, possibly across
        multiple lines until a blank line or the next timestamp marker.

    The numeric ``?timestamp=`` value is used for the start time (more
    precise than the visible ``MM:SS``). The end time of each segment is
    inferred from the start of the next segment.
    """
    raw_segments: list[tuple[float, str | None, list[str]]] = []
    current_start: float | None = None
    current_speaker: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = _FATHOM_TS_LINE_RE.match(line)
        if match:
            ts = float(match.group(1))
            speaker = match.group(2).strip()
            # Skip duplicate timestamp lines (Fathom often emits two in a row)
            if current_start is not None and ts == current_start and not current_lines:
                continue
            # Flush previous segment
            if current_start is not None and current_lines:
                raw_segments.append((current_start, current_speaker, current_lines))
            current_start = ts
            current_speaker = speaker
            current_lines = []
            continue
        if current_start is None:
            continue  # skip frontmatter before the first timestamp
        stripped = line.strip()
        if stripped:
            current_lines.append(stripped)

    if current_start is not None and current_lines:
        raw_segments.append((current_start, current_speaker, current_lines))

    segments: list[TranscriptSegment] = []
    for i, (start, speaker, lines) in enumerate(raw_segments):
        end = raw_segments[i + 1][0] if i + 1 < len(raw_segments) else start + 5.0
        content = " ".join(lines).strip()
        if content:
            segments.append(TranscriptSegment(start=start, end=end, text=content, speaker=speaker))
    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_audio.py -v`
Expected: ALL PASS (including existing tests — `speaker` defaults to `None` so SRT/VTT paths are unaffected)

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/core/audio.py tests/unit/test_audio.py
git commit -m "feat: extract speaker name from Fathom transcript into TranscriptSegment.speaker"
```

---

### Task 4: Add parse_fathom_anchors function

**Files:**
- Modify: `src/peeklet/core/audio.py` (new function after `_parse_fathom_md`)
- Modify: `tests/unit/test_audio.py` (new test class)

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_audio.py`, add a new test class:

```python
from peeklet.utils.types import Moment


class TestParseFathomAnchors:
    def test_extracts_action_items_with_watch_timestamps(self) -> None:
        from peeklet.core.audio import parse_fathom_anchors

        text = (
            '**ACTION ITEM: Fix missing assistant prompt - '
            '++[WATCH](https://fathom.video/calls/123?timestamp=232.9999)++**\n'
            '**ACTION ITEM: Fix missing assistant prompt - '
            '++[WATCH](https://fathom.video/calls/123?timestamp=232.9999)++**\n'
            'Some other text\n'
            '**ACTION ITEM: Investigate Policy Health guard-flag issue; fix - '
            '++[WATCH](https://fathom.video/calls/123?timestamp=455.9999)++**\n'
        )
        anchors = parse_fathom_anchors(text)

        assert len(anchors) == 2  # deduped
        assert anchors[0].timestamp == pytest.approx(232.9999)
        assert anchors[0].source == "anchor"
        assert "Fix missing assistant prompt" in anchors[0].visual_context_goal
        assert anchors[1].timestamp == pytest.approx(455.9999)

    def test_returns_empty_on_no_action_items(self) -> None:
        from peeklet.core.audio import parse_fathom_anchors

        text = "Just some regular transcript text with no action items.\n"
        assert parse_fathom_anchors(text) == []

    def test_sorted_by_timestamp(self) -> None:
        from peeklet.core.audio import parse_fathom_anchors

        text = (
            '**ACTION ITEM: Second - '
            '++[WATCH](https://fathom.video/calls/1?timestamp=500.0)++**\n'
            '**ACTION ITEM: First - '
            '++[WATCH](https://fathom.video/calls/1?timestamp=100.0)++**\n'
        )
        anchors = parse_fathom_anchors(text)
        assert [a.timestamp for a in anchors] == [100.0, 500.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_audio.py::TestParseFathomAnchors -v`
Expected: FAIL — `ImportError: cannot import name 'parse_fathom_anchors'`

- [ ] **Step 3: Implement parse_fathom_anchors**

In `src/peeklet/core/audio.py`, add after the `_FATHOM_TS_LINE_RE` definition (around line 86), before `_parse_fathom_md`:

```python
_FATHOM_ACTION_RE = re.compile(
    r"\*\*ACTION ITEM:\s*(.+?)\s*-\s*"
    r"\+\+\[WATCH\]\([^?]*\?timestamp=(\d+(?:\.\d+)?)\)\+\+\*\*"
)


def parse_fathom_anchors(text: str) -> list[Moment]:
    """Extract ACTION ITEM...WATCH markers as privileged anchor Moments.

    Fathom inlines these as::

        **ACTION ITEM: <desc> - ++[WATCH](https://fathom.video/...?timestamp=<s>)++**

    Each marker often appears twice on consecutive lines; this function
    deduplicates by (timestamp, description) before returning.

    Returns Moments sorted by timestamp with ``source="anchor"``.
    """
    from peeklet.utils.types import Moment

    seen: set[tuple[float, str]] = set()
    anchors: list[Moment] = []

    for match in _FATHOM_ACTION_RE.finditer(text):
        description = match.group(1).strip()
        timestamp = float(match.group(2))
        key = (timestamp, description)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            Moment(
                timestamp=timestamp,
                visual_context_goal=description,
                textual_anchor=match.group(0),
                downstream_utility="Action item flagged by meeting tool — guaranteed capture",
                source="anchor",
            )
        )

    anchors.sort(key=lambda m: m.timestamp)
    return anchors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_audio.py::TestParseFathomAnchors -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/core/audio.py tests/unit/test_audio.py
git commit -m "feat: parse Fathom ACTION ITEM...WATCH markers into privileged anchor Moments"
```

---

### Task 5: Update LLM prompt and JSON parsing for new Moment fields

**Files:**
- Modify: `src/peeklet/core/llm.py:25-38, 55-61, 102-155`
- Modify: `tests/unit/test_llm.py`

- [ ] **Step 1: Update tests for new field names**

In `tests/unit/test_llm.py`:

**1a.** Update `_make_segments` to include speaker:

```python
def _make_segments():
    from peeklet.core.audio import TranscriptSegment

    return [
        TranscriptSegment(
            start=0.0, end=3.5, text="Hey everyone, today I'll show you the dashboard.",
            speaker="Alice",
        ),
        TranscriptSegment(start=3.5, end=8.2, text="Let me start by signing in here.",
            speaker="Bob",
        ),
        TranscriptSegment(start=8.2, end=12.1, text="Okay, this is the main view after login.",
            speaker="Alice",
        ),
    ]
```

**1b.** Replace all `"caption"` / `"reason"` JSON keys with new names. Update these tests:

`test_format_transcript_includes_timestamps_and_text` — add assertion for speaker prefix:

```python
def test_format_transcript_includes_timestamps_and_text():
    from peeklet.core.llm import format_transcript_for_llm

    segments = _make_segments()
    formatted = format_transcript_for_llm(segments)

    assert "[0.0 - 3.5]" in formatted
    assert "**Alice**:" in formatted
    assert "Hey everyone" in formatted
    assert "[3.5 - 8.2]" in formatted
    assert "**Bob**:" in formatted
    assert formatted.count("\n") >= len(segments) - 1


def test_format_transcript_omits_speaker_when_none():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.llm import format_transcript_for_llm

    segments = [TranscriptSegment(start=0.0, end=3.0, text="Hello")]
    formatted = format_transcript_for_llm(segments)
    assert "[0.0 - 3.0] Hello" in formatted
    assert "**" not in formatted  # no speaker formatting
```

`test_parse_moments_strips_markdown_fences` — update JSON keys:

```python
def test_parse_moments_strips_markdown_fences():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        '```json\n[{"timestamp": 12.5, "visual_context_goal": "X", '
        '"textual_anchor": "Y", "downstream_utility": "Z"}]\n```'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)

    assert len(moments) == 1
    assert moments[0].timestamp == 12.5
    assert moments[0].visual_context_goal == "X"
    assert moments[0].textual_anchor == "Y"
    assert moments[0].downstream_utility == "Z"
    assert moments[0].source == "llm"
```

`test_parse_moments_drops_out_of_range_timestamps`:

```python
def test_parse_moments_drops_out_of_range_timestamps():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        '[{"timestamp": 5.0, "visual_context_goal": "ok", "textual_anchor": "t", "downstream_utility": "u"},'
        ' {"timestamp": 999.0, "visual_context_goal": "past end", "textual_anchor": "t", "downstream_utility": "u"},'
        ' {"timestamp": -1.0, "visual_context_goal": "negative", "textual_anchor": "t", "downstream_utility": "u"}]'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)
    assert [m.timestamp for m in moments] == [5.0]
```

`test_parse_moments_sorted_by_timestamp`:

```python
def test_parse_moments_sorted_by_timestamp():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        '[{"timestamp": 30.0, "visual_context_goal": "c", "textual_anchor": "t", "downstream_utility": "u"},'
        ' {"timestamp": 5.0, "visual_context_goal": "c", "textual_anchor": "t", "downstream_utility": "u"},'
        ' {"timestamp": 15.0, "visual_context_goal": "c", "textual_anchor": "t", "downstream_utility": "u"}]'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)
    assert [m.timestamp for m in moments] == [5.0, 15.0, 30.0]
```

`test_parse_moments_raises_on_missing_required_keys`:

```python
def test_parse_moments_raises_on_missing_required_keys():
    from peeklet.core.llm import LLMResponseError, _parse_moments_json

    raw = '[{"timestamp": 5.0, "visual_context_goal": "no anchor"}]'
    with pytest.raises(LLMResponseError):
        _parse_moments_json(raw, video_duration=60.0)
```

`test_parse_moments_strips_preamble_before_json`:

```python
def test_parse_moments_strips_preamble_before_json():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        "Here is the JSON you asked for:\n"
        '```json\n[{"timestamp": 7.0, "visual_context_goal": "c", '
        '"textual_anchor": "t", "downstream_utility": "u"}]\n```'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)
    assert len(moments) == 1
    assert moments[0].timestamp == 7.0
```

`test_parse_moments_handles_nested_brackets_in_strings`:

```python
def test_parse_moments_handles_nested_brackets_in_strings():
    from peeklet.core.llm import _parse_moments_json

    raw = (
        '[{"timestamp": 1.0, "visual_context_goal": "uses [brackets]", '
        '"textual_anchor": "t", "downstream_utility": "u"}]'
    )
    moments = _parse_moments_json(raw, video_duration=60.0)
    assert len(moments) == 1
    assert moments[0].visual_context_goal == "uses [brackets]"
```

Update all LLM client tests that construct fake JSON responses. The pattern is the same — replace `"caption": "c", "reason": "r"` with `"visual_context_goal": "c", "textual_anchor": "t", "downstream_utility": "u"` in every fake response string, and replace `Moment(timestamp=..., caption=..., reason=...)` assertions with `Moment(timestamp=..., visual_context_goal=..., textual_anchor=..., downstream_utility=...)`.

Affected tests (apply the same substitution pattern):
- `test_anthropic_client_pick_moments_calls_sdk_and_parses_response` (line 150, 165)
- `test_anthropic_client_retries_once_on_unparseable` (line 180)
- `test_openai_client_pick_moments_calls_sdk_and_parses_response` (line 224, 239)
- `test_openai_client_retries_once_on_unparseable` (line 266)
- `test_openrouter_client_pick_moments_calls_sdk_and_parses_response` (line 319, 334)
- `test_openrouter_client_retries_once_on_unparseable` (line 349)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_llm.py -v`
Expected: FAIL — old keys `caption`/`reason` no longer match

- [ ] **Step 3: Implement**

**3a.** Replace `SYSTEM_PROMPT` in `src/peeklet/core/llm.py:25-38`:

```python
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
    "\"Now, looking at the settings page...\"\n"
    "2. **References a specific UI element** — e.g., "
    "\"Note the red warning icon in the top right...\"\n"
    "3. **Completes a workflow step** — e.g., "
    "\"Once I click Deploy, you'll see the status change...\"\n"
    "4. **Points to data, tables, or graphs** — e.g., "
    "\"This spike in the chart represents...\"\n"
    "5. **Uses deictic expressions** (\"this\", \"that\", \"here\", "
    "\"there\") referring to something visible on screen\n\n"
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
    "(e.g., \"The configuration modal for API keys\")\n"
    "- **textual_anchor**: the exact transcript line that triggers this "
    "need — quote the speaker\n"
    "- **downstream_utility**: why the MLLM needs this image "
    "(e.g., \"To extract parameter values not mentioned in audio\")\n\n"
    "Pick as many or as few moments as the content needs."
)
```

**3b.** Update `format_transcript_for_llm` (line 55-61):

```python
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
```

**3c.** Update `_parse_moments_json` (lines 102-155) — change required keys:

Replace `for key in ("timestamp", "caption", "reason"):` with:

```python
        for key in ("timestamp", "visual_context_goal", "textual_anchor", "downstream_utility"):
```

Replace the Moment construction (lines 146-152):

```python
        moments.append(
            Moment(
                timestamp=ts,
                visual_context_goal=str(entry["visual_context_goal"]),
                textual_anchor=str(entry["textual_anchor"]),
                downstream_utility=str(entry["downstream_utility"]),
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_llm.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/peeklet/core/llm.py tests/unit/test_llm.py
git commit -m "feat: rewrite LLM prompt for richer moment schema and speaker-prefixed transcript"
```

---

### Task 6: Update demo_filter.py for new Moment/FrameResult fields + add merge_moments

**Files:**
- Modify: `src/peeklet/core/demo_filter.py:219-309, 318-353`
- Modify: `tests/unit/test_demo_filter.py`

- [ ] **Step 1: Write tests for merge_moments**

In `tests/unit/test_demo_filter.py`, add:

```python
class TestMergeMoments:
    def test_anchors_kept_llm_near_anchor_dropped(self):
        from peeklet.core.demo_filter import merge_moments
        from peeklet.utils.types import Moment

        anchors = [
            Moment(timestamp=232.0, visual_context_goal="a", textual_anchor="t",
                   downstream_utility="u", source="anchor"),
        ]
        llm_picks = [
            Moment(timestamp=230.0, visual_context_goal="b", textual_anchor="t",
                   downstream_utility="u"),  # within 5s → dropped
            Moment(timestamp=500.0, visual_context_goal="c", textual_anchor="t",
                   downstream_utility="u"),  # far away → kept
        ]
        merged = merge_moments(anchors, llm_picks, proximity_sec=5.0)

        assert len(merged) == 2
        assert merged[0].timestamp == 232.0
        assert merged[0].source == "anchor"
        assert merged[1].timestamp == 500.0
        assert merged[1].source == "llm"

    def test_all_anchors_kept_when_no_llm_picks(self):
        from peeklet.core.demo_filter import merge_moments
        from peeklet.utils.types import Moment

        anchors = [
            Moment(timestamp=100.0, visual_context_goal="a", textual_anchor="t",
                   downstream_utility="u", source="anchor"),
            Moment(timestamp=200.0, visual_context_goal="b", textual_anchor="t",
                   downstream_utility="u", source="anchor"),
        ]
        merged = merge_moments(anchors, [], proximity_sec=5.0)
        assert len(merged) == 2

    def test_sorted_by_timestamp(self):
        from peeklet.core.demo_filter import merge_moments
        from peeklet.utils.types import Moment

        anchors = [
            Moment(timestamp=500.0, visual_context_goal="a", textual_anchor="t",
                   downstream_utility="u", source="anchor"),
        ]
        llm_picks = [
            Moment(timestamp=100.0, visual_context_goal="b", textual_anchor="t",
                   downstream_utility="u"),
        ]
        merged = merge_moments(anchors, llm_picks, proximity_sec=5.0)
        assert [m.timestamp for m in merged] == [100.0, 500.0]

    def test_empty_inputs(self):
        from peeklet.core.demo_filter import merge_moments

        assert merge_moments([], [], proximity_sec=5.0) == []
```

- [ ] **Step 2: Write test for anchor privilege (skip dedup + skip tail_skip)**

```python
def test_anchor_moment_skips_dedup(tmp_path, monkeypatch):
    """Anchor moments bypass the dedup filter — two identical anchors both saved."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=600.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(timestamp=10.0, visual_context_goal="first", textual_anchor="t",
               downstream_utility="u", source="anchor"),
        Moment(timestamp=100.0, visual_context_goal="second", textual_anchor="t",
               downstream_utility="u", source="anchor"),
    ]
    transcript = [
        TranscriptSegment(start=8.0, end=12.0, text="a"),
        TranscriptSegment(start=98.0, end=102.0, text="b"),
    ]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, dedup_ssim_threshold=0.95)

    results = select_frames_for_moments(
        decoder=decoder, moments=moments, transcript=transcript,
        config=cfg, output_dir=tmp_path,
    )

    # Both saved despite identical frames — anchors skip dedup
    assert len(results) == 2


def test_anchor_moment_skips_tail_skip(tmp_path, monkeypatch):
    """Anchor moments near the end of video bypass tail_skip."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=100.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(timestamp=99.0, visual_context_goal="end anchor", textual_anchor="t",
               downstream_utility="u", source="anchor"),
    ]
    transcript = [TranscriptSegment(start=95.0, end=100.0, text="x")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, tail_skip_ratio=0.02)

    results = select_frames_for_moments(
        decoder=decoder, moments=moments, transcript=transcript,
        config=cfg, output_dir=tmp_path,
    )

    # Kept despite being in the tail — anchor privilege
    assert len(results) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_demo_filter.py::TestMergeMoments tests/unit/test_demo_filter.py::test_anchor_moment_skips_dedup tests/unit/test_demo_filter.py::test_anchor_moment_skips_tail_skip -v`
Expected: FAIL

- [ ] **Step 4: Implement merge_moments**

Add to `src/peeklet/core/demo_filter.py` before `select_frames_for_moments`:

```python
def merge_moments(
    anchors: list[Moment],
    llm_picks: list[Moment],
    proximity_sec: float = 5.0,
) -> list[Moment]:
    """Merge anchor and LLM-picked moments.

    All anchors are kept unconditionally. LLM picks within
    +/-proximity_sec of any anchor are dropped. Result is sorted
    by timestamp.
    """
    anchor_timestamps = [a.timestamp for a in anchors]
    filtered_llm: list[Moment] = []
    for pick in llm_picks:
        if any(abs(pick.timestamp - at) <= proximity_sec for at in anchor_timestamps):
            logger.info(
                "LLM pick at %.2fs dropped — within %.1fs of an anchor",
                pick.timestamp,
                proximity_sec,
            )
            continue
        filtered_llm.append(pick)

    combined = list(anchors) + filtered_llm
    combined.sort(key=lambda m: m.timestamp)
    return combined
```

- [ ] **Step 5: Update select_frames_for_moments for new field names + anchor privilege**

In `src/peeklet/core/demo_filter.py`, update `select_frames_for_moments`:

**5a.** Replace all `moment.caption` references (lines 225, 236, 270, 281) with `moment.visual_context_goal`.

**5b.** Replace `FrameResult` construction (lines 306-307):

```python
                visual_context_goal=moment.visual_context_goal,
                textual_anchor=moment.textual_anchor,
                downstream_utility=moment.downstream_utility,
                moment_source=moment.source,
```

**5c.** Add anchor privilege — make the tail-skip gate conditional on source:

```python
        is_anchor = moment.source == "anchor"

        if not is_anchor and tail_cutoff is not None and moment.timestamp >= tail_cutoff:
            logger.info(
                "Moment at %.2fs ('%s') falls in the final %.1f%% of the video "
                "(cutoff %.2fs), skipping as meeting-end noise.",
                moment.timestamp,
                moment.visual_context_goal,
                config.tail_skip_ratio * 100.0,
                tail_cutoff,
            )
            continue
```

**5d.** Make dedup conditional on source — after the gallery check, before saving:

```python
        if (
            not is_anchor
            and last_saved_frame is not None
            and config.dedup_ssim_threshold < 1.0
        ):
            dedup_score = compare_frames(picked_frame, last_saved_frame).ssim_score
            if dedup_score > config.dedup_ssim_threshold:
                logger.info(
                    "Moment at %.2fs ('%s') is a near-duplicate of the previous "
                    "keyframe (ssim=%.3f > %.3f), skipping.",
                    moment.timestamp,
                    moment.visual_context_goal,
                    dedup_score,
                    config.dedup_ssim_threshold,
                )
                continue
```

- [ ] **Step 6: Update existing tests for new field names**

All existing tests in `test_demo_filter.py` that construct `Moment(timestamp=..., caption=..., reason=...)` must change to `Moment(timestamp=..., visual_context_goal=..., textual_anchor=..., downstream_utility=...)`.

All assertions on `results[0].llm_caption` change to `results[0].visual_context_goal`.

Search for these patterns and replace:
- `caption="..."` → `visual_context_goal="..."`
- `reason="..."` → delete, add `textual_anchor="t", downstream_utility="u"`
- `r.llm_caption` → `r.visual_context_goal`
- `r.llm_reason` → `r.textual_anchor` (or remove if not asserted meaningfully)

Affected tests: `test_select_frames_for_moments_picks_first_stable_frame`, `test_select_frames_for_moments_drops_gallery_frames`, `test_select_frames_for_moments_drops_moment_with_no_segment`, `test_apply_demo_filter_calls_llm_then_select`, all dedup tests, all tail-skip tests.

- [ ] **Step 7: Run all demo_filter tests**

Run: `pytest tests/unit/test_demo_filter.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py
git commit -m "feat: add merge_moments + anchor privilege (skip dedup/tail-skip for ACTION ITEM captures)"
```

---

### Task 7: Wire anchor parsing into apply_demo_filter and video.py

**Files:**
- Modify: `src/peeklet/core/demo_filter.py:318-353` (`apply_demo_filter`)
- Modify: `src/peeklet/core/video.py:331-345`

- [ ] **Step 1: Write a test for apply_demo_filter with anchors**

In `tests/unit/test_demo_filter.py`:

```python
def test_apply_demo_filter_merges_anchors_with_llm_picks(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import apply_demo_filter
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=600.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), int(ts) % 200, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    transcript = [
        TranscriptSegment(start=230.0, end=235.0, text="Fix assistant prompt"),
        TranscriptSegment(start=498.0, end=505.0, text="Other discussion"),
    ]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0)

    fake_llm_moments = [
        Moment(timestamp=231.0, visual_context_goal="c", textual_anchor="t",
               downstream_utility="u"),  # near anchor → dropped
        Moment(timestamp=500.0, visual_context_goal="far away", textual_anchor="t",
               downstream_utility="u"),  # kept
    ]
    fake_client = MagicMock()
    fake_client.pick_moments.return_value = fake_llm_moments

    monkeypatch.setattr(
        "peeklet.core.demo_filter.build_llm_client",
        lambda provider, model: fake_client,
    )

    # Provide raw text with an action item so parse_fathom_anchors finds it
    raw_text = (
        '**ACTION ITEM: Fix assistant prompt - '
        '++[WATCH](https://fathom.video/calls/1?timestamp=232.0)++**\n'
    )

    results = apply_demo_filter(
        decoder=decoder,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
        transcript_text=raw_text,
    )

    # Should have anchor at 232 + llm pick at 500 = 2 results
    assert len(results) == 2
    sources = [r.moment_source for r in results]
    assert "anchor" in sources
    assert "llm" in sources
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_demo_filter.py::test_apply_demo_filter_merges_anchors_with_llm_picks -v`
Expected: FAIL — `apply_demo_filter() got an unexpected keyword argument 'transcript_text'`

- [ ] **Step 3: Update apply_demo_filter signature**

In `src/peeklet/core/demo_filter.py`, update `apply_demo_filter`:

```python
def apply_demo_filter(
    decoder: VideoDecoder,
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
    output_dir: Path,
    transcript_text: str = "",
) -> list[FrameResult]:
    """Top-level demo-mode entry point.

    Builds the LLM client, asks it to pick screenshot-worthy moments from the
    transcript, parses Fathom ACTION ITEM anchors from the raw transcript text,
    merges anchors with LLM picks, then runs Stage B (forward-search + stability
    + gallery check) to pick the actual frames.
    """
    from peeklet.core.audio import parse_fathom_anchors

    if pytesseract is None:
        logger.warning(
            "pytesseract is not installed — the gallery check is disabled and "
            "every LLM-picked frame will be saved without OCR filtering. "
            "Install with: pip install peeklet[demo]"
        )

    meta = decoder.get_metadata()
    client = build_llm_client(provider=config.llm_provider, model=config.llm_model)

    llm_picks = client.pick_moments(transcript, meta.duration)
    logger.info("LLM picked %d screenshot-worthy moments", len(llm_picks))

    anchors = parse_fathom_anchors(transcript_text) if transcript_text else []
    if anchors:
        logger.info("Parsed %d ACTION ITEM anchors from transcript", len(anchors))

    moments = merge_moments(anchors, llm_picks, proximity_sec=5.0)

    if not moments:
        logger.warning("No screenshot-worthy moments found (LLM + anchors).")
        return []

    return select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=config,
        output_dir=output_dir,
    )
```

- [ ] **Step 4: Update video.py to pass raw text**

In `src/peeklet/core/video.py`, update the demo-mode block (lines 331-341):

```python
    if config.demo_filter.enabled:
        demo_transcript: list[TranscriptSegment] = []
        transcript_text = ""
        if config.video.transcript_path:
            transcript_text = Path(config.video.transcript_path).read_text(encoding="utf-8")
            demo_transcript = parse_transcript(Path(config.video.transcript_path))

        demo_results = apply_demo_filter(
            decoder=decoder,
            transcript=demo_transcript,
            config=config.demo_filter,
            output_dir=output_dir,
            transcript_text=transcript_text,
        )
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_demo_filter.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/core/demo_filter.py src/peeklet/core/video.py
git commit -m "feat: wire Fathom anchor parsing into apply_demo_filter pipeline"
```

---

### Task 8: Overhaul context_exporter.py — build_context + write_context_markdown

**Files:**
- Modify: `src/peeklet/core/context_exporter.py`
- Modify: `tests/unit/test_context_exporter.py`

- [ ] **Step 1: Write tests for the new context.md format**

In `tests/unit/test_context_exporter.py`, update `TestWriteContextMarkdown`:

```python
class TestWriteContextMarkdown:
    def test_writes_markdown_with_visual_toc_and_new_header(self, tmp_path: Path) -> None:
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 65.5, "total_screenshots": 1},
            "screenshots": [
                {
                    "id": 1,
                    "file": "demo_0001_00005200ms.jpg",
                    "timestamp_s": 5.2,
                    "timestamp": "00:00:05.200",
                    "trigger": "transcript_trigger",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [1],
                    "visual_context_goal": "Dashboard with cost breakdown",
                    "textual_anchor": "Let me show you the estimated cost",
                    "downstream_utility": "To extract specific cost values",
                    "moment_source": "llm",
                }
            ],
            "transcript": [
                {
                    "id": 1,
                    "start_s": 4.5,
                    "end_s": 7.0,
                    "start": "00:00:04.500",
                    "end": "00:00:07.000",
                    "text": "Look at this chart",
                    "screenshot_ids": [1],
                    "speaker": "Alice",
                }
            ],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)

        md = out_path.read_text()
        # Header
        assert "# Meeting Context: demo.mp4" in md
        assert "Screenshots: 1" in md
        # Visual Table of Contents
        assert "Visual Table of Contents" in md
        assert "Dashboard with cost breakdown" in md
        # Speaker attribution
        assert "Alice" in md
        # Screenshot block has visual_context_goal in heading
        assert "Dashboard with cost breakdown" in md
        # Downstream utility in italics
        assert "*To extract specific cost values*" in md
        # Textual anchor as blockquote
        assert "> Let me show you the estimated cost" in md

    def test_screenshots_injected_inline_with_transcript(self, tmp_path: Path) -> None:
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 30.0, "total_screenshots": 2},
            "screenshots": [
                {
                    "id": 1,
                    "file": "demo_0001.jpg",
                    "timestamp_s": 6.0,
                    "timestamp": "00:00:06.000",
                    "trigger": "transcript_trigger",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [1],
                    "visual_context_goal": "First screen",
                    "textual_anchor": "",
                    "downstream_utility": "",
                    "moment_source": "llm",
                },
                {
                    "id": 2,
                    "file": "demo_0002.jpg",
                    "timestamp_s": 15.0,
                    "timestamp": "00:00:15.000",
                    "trigger": "transcript_trigger",
                    "change_magnitude": "minor",
                    "seconds_since_prev_screenshot": 9.0,
                    "transcript_ids": [],
                    "visual_context_goal": "Second screen",
                    "textual_anchor": "",
                    "downstream_utility": "",
                    "moment_source": "anchor",
                },
            ],
            "transcript": [
                {
                    "id": 1, "start_s": 0.0, "end_s": 5.0,
                    "start": "00:00:00.000", "end": "00:00:05.000",
                    "text": "First line spoken", "screenshot_ids": [],
                    "speaker": None,
                },
                {
                    "id": 2, "start_s": 10.0, "end_s": 14.0,
                    "start": "00:00:10.000", "end": "00:00:14.000",
                    "text": "Second line spoken", "screenshot_ids": [],
                    "speaker": "Bob",
                },
                {
                    "id": 3, "start_s": 20.0, "end_s": 25.0,
                    "start": "00:00:20.000", "end": "00:00:25.000",
                    "text": "Third line spoken", "screenshot_ids": [],
                    "speaker": None,
                },
            ],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)

        md = out_path.read_text()
        first_line = md.find("First line spoken")
        shot1 = md.find("Screenshot 1")
        second_line = md.find("Second line spoken")
        shot2 = md.find("Screenshot 2")
        third_line = md.find("Third line spoken")

        assert -1 < first_line < shot1 < second_line < shot2 < third_line

    def test_screenshots_only_when_no_transcript(self, tmp_path: Path) -> None:
        ctx = {
            "video": {"filename": "demo.mp4", "duration_s": 10.0, "total_screenshots": 1},
            "screenshots": [
                {
                    "id": 1,
                    "file": "demo_0001.jpg",
                    "timestamp_s": 2.0,
                    "timestamp": "00:00:02.000",
                    "trigger": "visual_change",
                    "change_magnitude": "major",
                    "seconds_since_prev_screenshot": None,
                    "transcript_ids": [],
                    "visual_context_goal": "Some screen",
                    "textual_anchor": "",
                    "downstream_utility": "",
                    "moment_source": "llm",
                },
            ],
            "transcript": [],
        }
        out_path = tmp_path / "context.md"
        write_context_markdown(ctx, out_path)

        md = out_path.read_text()
        assert "Screenshot 1" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_context_exporter.py::TestWriteContextMarkdown -v`
Expected: FAIL — old format doesn't include new fields

- [ ] **Step 3: Update build_context to propagate new fields**

In `src/peeklet/core/context_exporter.py`, update `build_context`:

In the screenshots dict construction (inside the `for i, r in enumerate(keyframes):` loop), add after `"seconds_since_prev_screenshot": since_prev,`:

```python
                "visual_context_goal": r.visual_context_goal or "",
                "textual_anchor": r.textual_anchor or "",
                "downstream_utility": r.downstream_utility or "",
                "moment_source": r.moment_source or "",
```

In the transcript dict construction, add after `"text": seg.text,`:

```python
                "speaker": seg.speaker,
```

- [ ] **Step 4: Rewrite _screenshot_block**

```python
def _screenshot_block(s: dict[str, Any]) -> list[str]:
    """Render a screenshot annotation block as Markdown lines."""
    goal = s.get("visual_context_goal") or (s.get("trigger") or "visual_change").replace("_", " ")
    lines = [
        f"**Screenshot {s['id']} ({s['timestamp']}) — {goal}**",
        f"![Screenshot]({s['file']})",
    ]
    utility = s.get("downstream_utility")
    if utility:
        lines.append(f"*{utility}*")
    anchor = s.get("textual_anchor")
    if anchor:
        lines.append("")
        lines.append(f"> {anchor}")
    lines.append("")
    return lines
```

- [ ] **Step 5: Rewrite write_context_markdown**

```python
def _format_short_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS for display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def write_context_markdown(ctx: dict[str, Any], path: Path | str) -> None:
    """Write context data as Markdown — chronologically interleaved transcript
    and screenshot blocks optimized for downstream MLLM consumption.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    video = ctx["video"]
    lines: list[str] = [
        f"# Meeting Context: {video['filename']}",
        "",
        f"Duration: {_format_duration(video['duration_s'])}"
        f" | Screenshots: {video['total_screenshots']}",
        "",
    ]

    screenshots = sorted(ctx["screenshots"], key=lambda s: s["timestamp_s"])

    # Visual Table of Contents
    if screenshots:
        lines.append("## Visual Table of Contents")
        lines.append("")
        lines.append("| # | Time | Visual Context |")
        lines.append("|---|------|---------------|")
        for s in screenshots:
            short_ts = _format_short_timestamp(s["timestamp_s"])
            goal = s.get("visual_context_goal") or "—"
            lines.append(f"| {s['id']} | {short_ts} | {goal} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Timeline")
    lines.append("")

    transcript = sorted(ctx["transcript"], key=lambda t: t["start_s"])

    seg_idx = 0
    shot_idx = 0
    inf = float("inf")
    while seg_idx < len(transcript) or shot_idx < len(screenshots):
        next_seg_time = transcript[seg_idx]["start_s"] if seg_idx < len(transcript) else inf
        next_shot_time = (
            screenshots[shot_idx]["timestamp_s"] if shot_idx < len(screenshots) else inf
        )

        if next_seg_time <= next_shot_time:
            seg = transcript[seg_idx]
            speaker = seg.get("speaker")
            if speaker:
                lines.append(f"**[{seg['start']} \u2192 {seg['end']}] {speaker}**")
            else:
                lines.append(f"**[{seg['start']} \u2192 {seg['end']}]**")
            lines.append(seg["text"])
            lines.append("")
            seg_idx += 1
        else:
            lines.extend(_screenshot_block(screenshots[shot_idx]))
            shot_idx += 1

    path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 6: Update TestBuildContext tests**

The `_make_keyframe_result` helper needs to set the new FrameResult fields. Add defaults:

```python
    def _make_keyframe_result(
        self,
        frame_id: str,
        timestamp: float,
        trigger_type: str = "visual_change",
        change_magnitude: str = "major",
    ) -> FrameResult:
        return FrameResult(
            frame_id=frame_id,
            event_type=EventType.KEYFRAME,
            is_keyframe=True,
            perceptual_hash="abc123",
            frame_width=1920,
            frame_height=1080,
            video_timestamp=timestamp,
            trigger_type=trigger_type,
            change_magnitude=change_magnitude,
            asset_path=f"output/{frame_id}.png",
            visual_context_goal="Test screen",
            textual_anchor="test anchor",
            downstream_utility="test utility",
            moment_source="llm",
        )
```

- [ ] **Step 7: Run all context_exporter tests**

Run: `pytest tests/unit/test_context_exporter.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/peeklet/core/context_exporter.py tests/unit/test_context_exporter.py
git commit -m "feat: overhaul context.md with visual TOC, speaker attribution, and rich screenshot blocks"
```

---

### Task 9: Update integration test

**Files:**
- Modify: `tests/integration/test_demo_mode_pipeline.py`

- [ ] **Step 1: Update Moment construction and assertions**

```python
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
    cfg.demo_filter.gallery_min_words = 0

    fake_client = MagicMock()
    fake_client.pick_moments.return_value = [
        Moment(
            timestamp=1.5,
            visual_context_goal="first thing",
            textual_anchor="speaker says here is the first thing",
            downstream_utility="verify first screen",
        ),
        Moment(
            timestamp=4.5,
            visual_context_goal="second thing",
            textual_anchor="speaker says now look at the second thing",
            downstream_utility="verify second screen",
        ),
    ]
    monkeypatch.setattr(
        df_module,
        "build_llm_client",
        lambda provider, model: fake_client,
    )

    results = process_video(video_path, cfg)

    assert len(results) == 2
    assert all(r.is_keyframe for r in results)
    goals = [r.visual_context_goal for r in results]
    assert "first thing" in goals
    assert "second thing" in goals

    out_dir = Path(cfg.exporter.output_dir)
    assert (out_dir / "manifest.parquet").exists()
    assert (out_dir / "context.json").exists()
    assert (out_dir / "context.md").exists()

    # Verify new context.md format
    md = (out_dir / "context.md").read_text()
    assert "Meeting Context:" in md
    assert "Visual Table of Contents" in md
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/integration/test_demo_mode_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_demo_mode_pipeline.py
git commit -m "test: update integration test for new Moment/FrameResult fields and context.md format"
```

---

### Task 10: Full test suite + lint verification

- [ ] **Step 1: Run ruff check**

Run: `ruff check src tests`
Expected: Clean. Fix any issues.

- [ ] **Step 2: Run ruff format check**

Run: `ruff format --check src tests`
Expected: Clean. Fix any formatting issues with `ruff format src tests`.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass (except the pre-existing `test_web_tasks.py::test_keyframe_images_saved` failure which is a fixture issue on develop).

- [ ] **Step 4: Final commit if any fixups needed**

```bash
git add -u
git commit -m "chore: lint fixups for action-item privilege PR"
```
