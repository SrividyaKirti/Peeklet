# Action-Item Privilege + Context Overhaul — Design Spec

**Date:** 2026-04-14
**Status:** Approved
**Builds on:** PR #20 (dedup + tail-skip), session notes 2026-04-10

---

## Problem

Peeklet's demo mode treats the transcript as opaque text fed to an LLM.
Fathom transcripts contain explicit `ACTION ITEM: ... WATCH@<timestamp>`
markers — human-validated points of interest — but the pipeline ignores
them. The LLM prompt is underspecified, the `Moment` schema is too thin,
`context.md` strips speaker names and renders minimal screenshot blocks,
and there is no visual table of contents for downstream MLLM consumption.

## Scope — Two PRs

### PR A: action-item privilege + prompt rewrite + context.md overhaul

Everything in this spec. Single feature branch, one PR.

### PR B: post-trigger window scoring (separate spec)

Change `_pick_stable_index` from "first SSIM-stable frame" to
"frame with most content." Consumes PR A's output unchanged. Not
covered here.

---

## 1. Data Model Changes

### 1.1 TranscriptSegment (`audio.py`)

Add optional `speaker` field:

```python
@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float   # seconds
    end: float     # seconds
    text: str
    speaker: str | None = None
```

No existing callers break — the new field defaults to `None`.

### 1.2 Moment (`types.py`)

Replace the current three-field dataclass:

```python
# Before
@dataclass(frozen=True, slots=True)
class Moment:
    timestamp: float
    caption: str
    reason: str

# After
@dataclass(frozen=True, slots=True)
class Moment:
    timestamp: float
    visual_context_goal: str
    textual_anchor: str
    downstream_utility: str
    source: str = "llm"   # "llm" | "anchor"
```

All call sites that create or read `Moment` must be updated.

### 1.3 FrameResult (`types.py`)

Replace demo-mode fields:

```python
# Before
llm_caption: str | None = None
llm_reason: str | None = None

# After
visual_context_goal: str | None = None
textual_anchor: str | None = None
downstream_utility: str | None = None
moment_source: str | None = None   # "llm" | "anchor"
```

All call sites referencing `llm_caption` / `llm_reason` must be updated.

---

## 2. Fathom Anchor Parser

### 2.1 Location

New public function in `audio.py`, co-located with `_parse_fathom_md`.

### 2.2 Interface

```python
def parse_fathom_anchors(text: str) -> list[Moment]:
    """Extract ACTION ITEM...WATCH markers as privileged anchor Moments.

    Fathom inlines these as:
        **ACTION ITEM: <desc> - ++[WATCH](https://fathom.video/...?timestamp=<s>)++**

    Each marker often appears twice on consecutive lines; this function
    deduplicates by (timestamp, description) before returning.

    Returns Moments with source="anchor".
    """
```

### 2.3 Regex

```python
_FATHOM_ACTION_RE = re.compile(
    r'\*\*ACTION ITEM:\s*(.+?)\s*-\s*'
    r'\+\+\[WATCH\]\([^?]*\?timestamp=(\d+(?:\.\d+)?)\)\+\+\*\*'
)
```

### 2.4 Output mapping

| Moment field | Value |
|---|---|
| `timestamp` | float from regex group 2 |
| `visual_context_goal` | Inferred from action-item description (regex group 1) |
| `textual_anchor` | Full matched line |
| `downstream_utility` | `"Action item flagged by meeting tool — guaranteed capture"` |
| `source` | `"anchor"` |

---

## 3. Speaker Attribution in Fathom Parser

### 3.1 Regex change

Current `_FATHOM_TS_LINE_RE` captures only the timestamp. Add a capture
group for the speaker name inside `**...**`:

```python
_FATHOM_TS_LINE_RE = re.compile(
    r'^\s*\+\+\[@\d+:\d+\]\([^)]*\?timestamp=(\d+(?:\.\d+)?)\)\+\+\s*-\s*\*\*([^*]+)\*\*\s*$'
)
```

Group 1 = timestamp, group 2 = speaker name.

### 3.2 TranscriptSegment population

`_parse_fathom_md` passes `speaker=match.group(2).strip()` when
constructing each `TranscriptSegment`.

### 3.3 LLM transcript formatting

`format_transcript_for_llm` changes from:

```
[3.0 - 15.2] Spoken text here...
```

to:

```
[3.0 - 15.2] **Gowshik T**: Spoken text here...
```

When `segment.speaker` is `None` (SRT/VTT sources), omit the prefix.

---

## 4. LLM Prompt Rewrite

### 4.1 New SYSTEM_PROMPT

Replace the current 8-line prompt (`llm.py:25-38`) with:

```
You are a Video Content Analyst specializing in visual-textual
alignment for multimodal AI processing.

## Task

Analyze the provided meeting/demo transcript to identify specific
timestamps where a screenshot is essential for a downstream
Multimodal LLM (MLLM) to understand the technical context being
discussed.

## Selection Criteria

Identify a Key Moment whenever the speaker:
1. **Navigates to a new screen or dashboard** — e.g.,
   "Now, looking at the settings page..."
2. **References a specific UI element** — e.g.,
   "Note the red warning icon in the top right..."
3. **Completes a workflow step** — e.g.,
   "Once I click Deploy, you'll see the status change..."
4. **Points to data, tables, or graphs** — e.g.,
   "This spike in the chart represents..."
5. **Uses deictic expressions** ("this", "that", "here", "there")
   referring to something visible on screen

## Timing Rules

- Place the timestamp **0.5-1.0 seconds after** the speaker begins
  the triggering sentence, to allow the UI to finish loading or
  animating.
- **Avoid selecting timestamps within 15 seconds of each other**
  unless a major UI transition (new page, modal, or tab) occurs
  between them.
- If the demo stays on one complex screen for an extended period,
  one screenshot is usually enough. Only add a second if the speaker
  references a different region or scrolls to new content.

## Action Item Anchors

Lines marked `ACTION ITEM` with `WATCH` links are high-priority
moments flagged by the meeting tool. You MUST include a moment at
or near each such timestamp. These represent confirmed points of
interest that a human reviewer has validated.

## Output Format

Return ONLY a JSON array of objects. No preamble, no explanation.

[{"timestamp": 12.5, "visual_context_goal": "...",
  "textual_anchor": "...", "downstream_utility": "..."}]

Fields:
- **timestamp**: float, seconds into the video
- **visual_context_goal**: what the screenshot needs to capture
  (e.g., "The configuration modal for API keys")
- **textual_anchor**: the exact transcript line that triggers this
  need — quote the speaker
- **downstream_utility**: why the MLLM needs this image
  (e.g., "To extract parameter values not mentioned in audio")

Pick as many or as few moments as the content needs.
```

### 4.2 JSON parsing update

`_parse_moments_json` changes required keys from
`("timestamp", "caption", "reason")` to
`("timestamp", "visual_context_goal", "textual_anchor", "downstream_utility")`.

Map parsed dict to the new `Moment` fields directly.

---

## 5. Merge Logic

### 5.1 Location

New function in `demo_filter.py`.

### 5.2 Interface

```python
def merge_moments(
    anchors: list[Moment],
    llm_picks: list[Moment],
    proximity_sec: float = 5.0,
) -> list[Moment]:
    """Merge anchor and LLM-picked moments.

    - All anchors are kept unconditionally.
    - LLM picks within +/-proximity_sec of any anchor are dropped
      (the anchor is the better-positioned capture).
    - Result is sorted by timestamp.
    """
```

### 5.3 Privilege semantics in select_frames_for_moments

Anchor moments (`source == "anchor"`) are **guaranteed captures**:

- **Skip tail_skip_ratio gate** — an action item near the end of
  a meeting is still worth capturing.
- **Skip dedup against previous keyframe** — two action items 10s
  apart on the same page are both captured because the user
  marked both as important.
- **Still subject to gallery check** — a gallery-view frame from
  an action item is still garbage; emit a warning log but skip it.

LLM-picked moments (`source == "llm"`) retain all existing filters
unchanged.

### 5.4 Wiring in apply_demo_filter

```python
# In apply_demo_filter() / the calling pipeline:
raw_text = path.read_text(...)
anchors = parse_fathom_anchors(raw_text)
segments = parse_transcript(path)
llm_picks = client.pick_moments(segments, duration)
moments = merge_moments(anchors, llm_picks, proximity_sec=5.0)
results = select_frames_for_moments(moments, ...)
```

---

## 6. context.md Overhaul

### 6.1 New format

```markdown
# Meeting Context: <filename>

Duration: Xm Ys | Screenshots: N

## Visual Table of Contents

| # | Time | Visual Context |
|---|------|---------------|
| 1 | 3:52 | Analytics — Interaction Breakdown table |
| 2 | 7:58 | Policy Health — coverage summary |

---

## Timeline

**[00:03:32 -> 00:03:50] Gowshik T**
"Let me pull up the interaction breakdown..."

**Screenshot 1 (3:52) — Analytics: Interaction Breakdown table**
![Screenshot](demo_0005_00232999ms.jpg)
*To verify which rows are missing the assistant prompt field.*

> "Fix missing assistant prompt in Analytics/Tasks"

**[00:03:50 -> 00:05:12] Vignesh Subbiah**
"Yeah, that's the bug..."
```

### 6.2 Screenshot block structure

```python
def _screenshot_block(s: dict[str, Any]) -> list[str]:
    heading = f"**Screenshot {s['id']} ({s['timestamp']}) — {s['visual_context_goal']}"
    lines = [
        heading,
        f"![Screenshot]({s['file']})",
    ]
    if s.get("downstream_utility"):
        lines.append(f"*{s['downstream_utility']}*")
    if s.get("textual_anchor"):
        lines.append("")
        lines.append(f"> {s['textual_anchor']}")
    lines.append("")
    return lines
```

### 6.3 Transcript block structure

```python
# When speaker is present:
f"**[{seg['start']} -> {seg['end']}] {seg['speaker']}**"
f"{seg['text']}"

# When speaker is None (SRT/VTT):
f"**[{seg['start']} -> {seg['end']}]**"
f"{seg['text']}"
```

### 6.4 Visual Table of Contents

Generated from the screenshots list before the timeline section.
One row per screenshot: `| id | MM:SS | visual_context_goal |`.

### 6.5 build_context changes

The `screenshots` dict in `build_context()` gains:
- `visual_context_goal` (from `FrameResult.visual_context_goal`)
- `textual_anchor` (from `FrameResult.textual_anchor`)
- `downstream_utility` (from `FrameResult.downstream_utility`)
- `moment_source` (from `FrameResult.moment_source`)

The `transcript` dict gains:
- `speaker` (from `TranscriptSegment.speaker`)

---

## 7. Files Changed

| File | Change |
|---|---|
| `src/peeklet/utils/types.py` | Moment field rename + source; FrameResult field rename |
| `src/peeklet/core/audio.py` | TranscriptSegment.speaker; speaker extraction in _parse_fathom_md; new parse_fathom_anchors |
| `src/peeklet/core/llm.py` | SYSTEM_PROMPT rewrite; _parse_moments_json key updates; format_transcript_for_llm speaker prefix |
| `src/peeklet/core/llm_anthropic.py` | Moment field renames in any usage |
| `src/peeklet/core/llm_openai.py` | Moment field renames in any usage |
| `src/peeklet/core/llm_openrouter.py` | Moment field renames in any usage |
| `src/peeklet/core/demo_filter.py` | merge_moments; anchor privilege in select_frames_for_moments; wiring |
| `src/peeklet/core/context_exporter.py` | Full rewrite of write_context_markdown; build_context gains new fields |
| `tests/unit/test_audio.py` | Tests for parse_fathom_anchors; speaker extraction |
| `tests/unit/test_demo_filter.py` | Tests for merge_moments; anchor privilege semantics |
| `tests/unit/test_llm.py` | Updated for new Moment fields and prompt |
| `tests/unit/test_context_exporter.py` | Tests for new context.md format |
| `tests/unit/test_types.py` | Updated for renamed fields |
| `tests/integration/test_demo_mode_pipeline.py` | Updated for new Moment/FrameResult fields |

---

## 8. Not in Scope

- Chapter segmentation (needs LLM or Fathom topic markers)
- OCR text extraction per frame
- Screen-share region crop (PR #8)
- Post-trigger window scoring change (PR B, separate spec)
- Descriptive filenames (keep `demo_NNNN_<ms>ms.jpg`)
- Changes to `context.json` schema (keep backward-compatible, add new fields)
- Changes to `manifest.parquet` schema
- VTT/SRT anchor support (future: `anchors.json` sidecar)
