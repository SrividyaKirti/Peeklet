# Transcript-Driven Demo Mode — Design

**Date:** 2026-04-09
**Status:** Approved (brainstorming complete)
**Branch:** `feat/demo-mode-frame-filtering`
**Supersedes:** the deleted `2026-04-08-demo-mode-frame-filtering-design.md` (visual-change-driven approach)

## Problem

Peeklet needs a way to turn a long demo video plus a transcript into a small,
high-signal set of screenshots that an LLM can consume alongside the
transcript. The previous approach started from the video and tried to filter
keyframes down using OCR + transcript anchoring. That framing was wrong: most
of the visual-change-detection machinery was doing little useful work, and a
user could roughly replicate the result with `ffmpeg` plus a single LLM
call.

The right framing is the **opposite direction**: start from the transcript,
let an LLM identify the moments where a screenshot would help a reader
understand what's happening, then use Peeklet's frame-level intelligence to
pick the *exact frame* to grab for each moment. The LLM picks the moments,
Peeklet picks the frames — neither side could do the job alone.

## Goals

- Take a video + transcript and produce a curated set of LLM-captioned
  screenshots that map directly to what the speaker is talking about
- Use an LLM to pick screenshot-worthy moments from the transcript
- Use Peeklet's existing visual-change-detection to pick stable, non-noisy
  frames at each moment (not mid-transition, not gallery view)
- Run end-to-end under ~25 seconds for a typical 60-min demo (target, not
  hard ceiling — scales linearly with LLM moment count)
- Pluggable LLM provider via thin native SDKs (no heavy multi-provider
  wrapper libraries)
- Produce richer output than before: each screenshot carries a caption and
  the speaker's reasoning quoted from the transcript

## Non-goals

- Supporting demo mode without a transcript (hard requirement)
- Supporting demo mode without an LLM API key (hard requirement)
- Replacing the existing video pipeline (demo mode is opt-in)
- Capping the number of LLM-picked moments (let the LLM decide what the video
  needs)
- Generic content classification or precise OCR text extraction
- Building a multi-provider router or proxy in our trust boundary (rules out
  LiteLLM, langchain, etc.)

## High-level architecture

Two stages. The LLM picks the moments, Peeklet picks the frames.

```
transcript ──► Stage A (LLM) ──► [{ts, caption, reason}, ...]
                                          │
                                          ▼
video ─────► Stage B (Peeklet) ──► curated keyframes with captions
```

**Stage A — LLM picks moments**
- Parse the transcript (existing code path)
- Send the full transcript with timestamps to a configurable LLM
- LLM returns a list of `{timestamp, caption, reason}` per screenshot-worthy
  moment, no cap on count
- One API call per video

**Stage B — Peeklet picks frames** (per moment)
- Search forward within the *current transcript segment* for the first frame
  that satisfies bidirectional SSIM stability:
  `SSIM(frame, prev) > threshold AND SSIM(frame, next) > threshold`
- Fall back to the frame closest to the LLM's original timestamp if no
  stable frame is found in the window
- Run an OCR text-density check on the picked frame; if it's below the
  gallery-detection threshold, drop the moment with a warning (the LLM's
  pick landed in a gallery view / blank screen / Alt-Tabbed-away region)
- Save the surviving keyframe with the LLM's caption + reason as metadata

## User-facing interface

### CLI flags

- `--demo-mode` — opt-in flag (existing concept, kept).
- `--llm-provider {anthropic,openai}` — selects the LLM SDK. Defaults to
  `anthropic`. Can also be set via `PEEKLET_LLM_PROVIDER`.
- `--llm-model TEXT` — model identifier for the chosen provider. Defaults to
  `claude-haiku-4-5` for anthropic. Can also be set via `PEEKLET_LLM_MODEL`.

### Activation rules

- `--demo-mode` requires both a video input and `--transcript <path>`.
  Error message and exit code as in the existing implementation.
- `--demo-mode` requires the API key env var that matches the chosen
  provider:
  - `anthropic` → `ANTHROPIC_API_KEY`
  - `openai` → `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL` for users
    routing through Ollama, Groq, OpenRouter, vLLM, Together, Azure, etc.)
- If the required key is missing, exit with: *"--demo-mode with provider X
  requires the X_API_KEY env var to be set."*

### Configuration block

The existing `DemoFilterConfig` from Task 3 is **replaced** with new fields
that match the transcript-first design:

```python
class DemoFilterConfig(BaseModel):
    enabled: bool = False
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    # Frame search and selection
    frame_search_resolution: int = 360   # px on the longest edge
    ssim_stability_threshold: float = 0.92
    forward_search_step_sec: float = 0.5     # sample cadence inside the window
    forward_search_window_max_sec: float = 5.0  # hard cap on per-moment window
    # Gallery detection (reuses _count_words_in_frame from Task 5)
    gallery_min_words: int = 5
```

Old OCR-sweep fields (`ocr_sample_interval_sec`, `ocr_min_words`,
`major_change_ssim`, `major_change_blocks`, `visual_change_weight`,
`text_density_weight`) are removed — they belonged to the old Stage 1/2/3
algorithm that no longer exists.

## Architecture: modules and types

### New module: `src/peeklet/core/llm.py`

Thin internal adapter wrapping `openai` and `anthropic` SDKs. ~80-120 lines
total. The whole point is to keep the trust boundary small and avoid
pulling in a third-party multi-provider router.

```python
@dataclass(frozen=True, slots=True)
class Moment:
    timestamp: float
    caption: str
    reason: str


class LLMClient(Protocol):
    def pick_moments(
        self, transcript: list[TranscriptSegment]
    ) -> list[Moment]: ...


class AnthropicClient:
    """Native Anthropic SDK implementation."""
    def __init__(self, model: str) -> None: ...
    def pick_moments(self, transcript): ...


class OpenAIClient:
    """OpenAI SDK implementation. Respects OPENAI_BASE_URL so users can
    point at Ollama / Groq / OpenRouter / vLLM / Together / Azure."""
    def __init__(self, model: str) -> None: ...
    def pick_moments(self, transcript): ...


def build_llm_client(provider: str, model: str) -> LLMClient:
    """Factory: validate the provider, ensure the matching env var is set,
    return a configured client."""
```

`Moment` lives in `utils/types.py` so other modules can import it without
pulling the LLM SDK transitively.

### New module: `src/peeklet/core/demo_filter.py` (extends existing)

The existing module (created in Task 5) is extended with the Stage B logic.
Helpers from Task 5 (`_downscale_for_ocr`, `_count_words_in_frame`) stay as
the gallery-detection backbone.

Public surface:

```python
def select_frames_for_moments(
    decoder: VideoDecoder,
    moments: list[Moment],
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
) -> list[FrameResult]:
    """Stage B: for each LLM-picked moment, find the best actual frame."""

def apply_demo_filter(
    decoder: VideoDecoder,
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
) -> list[FrameResult]:
    """Top-level entry point. Builds the LLM client, calls Stage A, then
    runs Stage B and returns the curated keyframe list."""
```

The OLD public surface from the previous spec
(`classify_content_segments`, `select_demo_keyframes`) is dropped — those
were the Stage 1/2/3 functions that no longer exist.

### Type changes in `utils/types.py`

| Type / field | Status | Notes |
|---|---|---|
| `ContentSegment` (added in Task 2) | **Remove** | The new design has no segment classification |
| `FrameResult.content_type` (added in Task 2) | **Remove** | Replaced by the captioning fields below |
| `FrameResult.selection_reason` (added in Task 2) | **Remove** | Replaced by the captioning fields below |
| `FrameResult.llm_caption: str \| None = None` | **Add** | The LLM's one-sentence description of what the screenshot shows |
| `FrameResult.llm_reason: str \| None = None` | **Add** | The LLM's quoted-from-transcript reason this moment matters |
| `Moment` dataclass | **Add** | Frozen, three fields: `timestamp: float`, `caption: str`, `reason: str` |

Type changes are mechanical; no logic depends on `content_type` /
`selection_reason` yet (Stage 2/3 logic was never built).

## Stage A — LLM picks moments

### Prompt

A single user message containing the full transcript, formatted as one line
per segment with timestamps. The system prompt fixes the role and the
output format.

**System prompt** (constant string in `llm.py`):

> You are reviewing a transcript from a software demo video. Your job is to
> identify the moments where a screenshot would help a reader understand
> what's happening — places where the speaker references something visual
> on screen, demonstrates an action, opens a UI, or moves on to a new topic
> with a different visual context.
>
> For each such moment, return:
> - `timestamp`: a float, seconds into the video
> - `caption`: a one-sentence description of what the screenshot should show
>   (e.g., "Settings page open with the Account tab selected")
> - `reason`: a one-sentence justification quoting or paraphrasing the
>   speaker's words (e.g., "Speaker says 'let me show you the account settings'")
>
> Return ONLY a JSON array. Example:
> `[{"timestamp": 12.5, "caption": "...", "reason": "..."}, ...]`
>
> Pick as many or as few moments as the video needs. There is no minimum or
> maximum.

**User message:** the formatted transcript:

```
[0.0 - 3.5] Hey everyone, today I'm going to walk you through our new dashboard.
[3.5 - 8.2] Let me start by signing in here.
[8.2 - 12.1] Okay, so this is the main view you see after login.
...
```

### LLM call

The implementation lives entirely in `OpenAIClient.pick_moments` and
`AnthropicClient.pick_moments`. Each calls its native SDK once with the
system + user messages, parses the JSON response, and returns a
`list[Moment]`.

**Parsing rules:**
- Strip any markdown code fences (` ```json ... ``` `) before parsing
- Validate every entry has the three required keys with the right types
- Drop entries with timestamps outside the video duration (with a warning)
- Sort by timestamp ascending

**Retry policy:**
- If the response is unparseable JSON: retry once with the same prompt
- If the second attempt also fails: raise `LLMResponseError` with the raw
  output included in the message for debugging

### Performance

A single Haiku call with a 60-min transcript (~10k tokens) takes ~3-8s
typically. The other providers in the same class (gpt-4o-mini, llama-3.1-8b
on Groq, etc.) land in the same range.

## Stage B — Peeklet picks frames

For each `Moment` from Stage A:

1. **Locate the transcript segment** containing the moment's timestamp.
   This bounds the forward search window: we never look past the end of
   the segment the speaker is currently in.
2. **Build the search window:** `[moment.timestamp, min(segment.end, moment.timestamp + forward_search_window_max_sec)]`.
   The window is bounded by the end of the current transcript segment and
   never extends past `forward_search_window_max_sec` (default 5s) so a
   long monologue segment doesn't blow up the per-moment cost.
3. **Sample frames** at `forward_search_step_sec` intervals (default 0.5s)
   inside the window using `VideoDecoder.extract_frame_at()` (added in
   Task 4). Decode at `frame_search_resolution` (default 360px on the
   longest edge) to keep SSIM and OCR cheap.
4. **Find the first stable frame** — for each sampled frame, also extract
   its predecessor (at `ts - forward_search_step_sec`) and successor (at
   `ts + forward_search_step_sec`). A frame is stable if both
   `SSIM(frame, prev) > ssim_stability_threshold` AND
   `SSIM(frame, next) > ssim_stability_threshold`. Pick the first stable
   frame walking forward from the moment's timestamp.
5. **Fallback:** if no stable frame is found in the window, use the
   already-sampled frame closest to the moment's original timestamp (the
   first sample in the window). No additional decode is needed.
6. **Gallery check:** run `_count_words_in_frame()` (Task 5) on the picked
   frame. If `words < gallery_min_words`, drop this moment with a warning:
   *"LLM picked moment at {ts}s ('{caption}') but the frame is gallery-view
   / blank — no demo content visible. Skipping."*
7. **Save the keyframe** at original resolution (re-extract via
   `extract_frame_at` without the downscale) and write it to the output
   directory using the existing `save_keyframe` helper.
8. **Return a `FrameResult`** with:
   - The new `llm_caption` and `llm_reason` fields populated
   - `is_keyframe = True`, `event_type = EventType.KEYFRAME`
   - The existing video metadata (`source_video`, `video_timestamp`, etc.)
   - `trigger_type = "transcript_trigger"`

The result list is sorted by timestamp and goes through the existing
context-export and manifest-write code paths unchanged.

### Per-moment cost target

| Operation | Cost (target) |
|---|---|
| 3 frame extractions @ 360p (prev / picked / next) | ~60ms |
| 2 SSIM comparisons @ 360p | ~20ms |
| OCR text-density check @ 360p | ~80-100ms |
| Final keyframe re-extraction @ original res + save | ~30ms |
| **Total per moment** | **~190-210ms** |

For 100 moments → ~20s. Plus ~5-10s for the LLM call → ~25-30s total.
Within the soft budget for typical videos.

## Failure handling

| Failure mode | Behavior |
|---|---|
| `--demo-mode` without `--transcript` | Exit 2 with the existing error message |
| `--demo-mode` without video input | Exit 2 with the existing error message |
| `--demo-mode` with unknown provider | Exit 2 with: *"Unknown LLM provider 'X'. Supported: anthropic, openai."* |
| Provider chosen but matching API key env var missing | Exit 2 with: *"--demo-mode with provider X requires X_API_KEY env var to be set."* |
| LLM SDK not installed (e.g., `pip install peeklet[video]` but not `[demo]`) | Exit 2 with: *"--demo-mode requires the [demo] extra. Install with: pip install peeklet[demo]"* |
| LLM API call fails (network, auth, rate limit) | Exit with the underlying error wrapped in a clear message |
| LLM returns unparseable JSON | Retry once. On second failure: exit with `LLMResponseError` including the raw output |
| LLM returns 0 moments | Log warning *"LLM identified zero screenshot-worthy moments in this transcript."* Save zero screenshots, write empty manifest + context |
| Gallery check drops a moment | Log warning per moment as described above |
| Tesseract binary missing | Exit 2 with the existing message from the original Task 5 |
| All Stage B moments dropped (every single one was gallery-view) | Log warning, still write empty outputs |

## Performance budget (60-min video, ~150 LLM moments)

| Stage | Cost |
|---|---|
| Video metadata + setup | ~1s |
| Parse transcript | <1s |
| Stage A: LLM call (Haiku) | ~5-10s |
| Stage B: 150 moments × ~200ms | ~30s |
| Save outputs (manifest + context) | ~2-3s |
| **Total** | **~38-45s** |

This is over the 25s soft target for a 60-min video. The honest reading is
that the 25s target is comfortably hit on **shorter videos** (15-30 min) and
on **shorter LLM outputs** (50-100 moments), but a verbose 60-min demo with
many moments will land in the 30-45s range. We are explicitly trading some
runtime for unbounded LLM moment counts, per the user's call. If runtime
becomes a real problem in practice, we can revisit by parallelizing Stage B
across moments (frame extraction is embarrassingly parallel).

## Existing committed work — adjustments needed

| Commit | What it added | Adjustment |
|---|---|---|
| Task 1 (`pytesseract` dependency) | dep + mypy override | **Keep as-is** — gallery check still uses it |
| Task 2 (`ContentSegment` + 2 `FrameResult` fields) | dataclass + 2 fields | **Modify**: delete `ContentSegment`, replace `content_type`/`selection_reason` with `llm_caption`/`llm_reason` |
| Task 3 (`DemoFilterConfig`) | config block | **Replace**: drop the OCR-sweep / scoring fields, add the new ones listed above |
| Task 4 (`VideoDecoder.extract_frame_at`) | seek helper | **Keep as-is** — Stage B's foundation |
| Task 5 (`_count_words_in_frame`, `_downscale_for_ocr`) | OCR helper | **Keep as-is** — used for gallery check |

No commits need to be reverted. The new plan layers adjustments on top.

## Out of scope (potential follow-ups)

- Parallelizing Stage B across moments (would cut total time roughly in half)
- Caching LLM responses by transcript hash so re-runs of the same demo skip
  the API call
- Adding `gemini`, `bedrock`, or other native SDK adapters — for now users
  with these providers route through `OPENAI_BASE_URL` if their gateway is
  OpenAI-compatible
- Vision-model verification (sending the picked frame back to the LLM and
  asking "does this match the caption?") — would catch mistakes but
  doubles API cost
- Per-moment "visual hint" from the LLM that Stage B uses to verify the
  frame matches expectations
- Configurable prompt template for users who want different LLM behavior
