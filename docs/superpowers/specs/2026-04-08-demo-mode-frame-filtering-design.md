# Demo-Mode Frame Filtering — Design

**Date:** 2026-04-08
**Status:** Approved (brainstorming complete)
**Branch:** `feat/ocr-content-filter` (to be created)

## Problem

When Peeklet processes a 60-minute demo video, the existing hash + SSIM
pipeline yields ~800 keyframes. Each frame is technically a visual change,
but the resulting set is far too large for an LLM to consume effectively.
Many frames capture trivial changes (cursor movement, hover states, speaker
highlight in a webcam gallery) rather than meaningful UI state transitions.

For the specific case of a **demo video paired with a transcript**, the goal
is to reduce the output to ~100-200 high-signal frames in under 60 seconds
of total processing time, without losing important moments.

## Goals

- Reduce a 60-min demo video from ~800 keyframes to ~100-200 frames
- Total processing time stays under 60 seconds
- No loss of major visual transitions (modals, view changes, navigation)
- Drop content from non-demo segments (gallery view, talking heads, blank screens)
- Frame selection is driven by what the speaker is actually narrating

## Non-goals

- Supporting demo mode without a transcript (hard requirement)
- Replacing the existing video pipeline (demo mode is opt-in)
- Precise OCR text extraction (we only need text *presence* signals)
- Generic content classification beyond demo / non-demo

## User-facing interface

### New CLI flag

`--demo-mode` (boolean opt-in flag).

### Activation rules

- `--demo-mode` requires both a video input and `--transcript <path>`.
- `--demo-mode` without `--transcript` → exit 2 with:
  *"--demo-mode requires --transcript. Demo mode needs both a video and a transcript to filter frames effectively."*
- `--demo-mode` without a video input (e.g. directory of stills) → exit 2 with:
  *"--demo-mode only applies to video inputs."*
- Without `--demo-mode`, existing video processing behavior is unchanged.

### New config section (`config.py`)

```python
class DemoFilterConfig:
    enabled: bool = False                    # set by --demo-mode
    ocr_sample_interval_sec: float = 30.0    # sparse sweep cadence
    ocr_downscale_dim: int = 360             # OCR-only longest-edge resize
    ocr_min_words: int = 5                   # min words for "demo" classification
    major_change_ssim: float = 0.70          # SSIM below this = "major"
    major_change_blocks: int = 15            # block count above this = "major"
    visual_change_weight: float = 0.7        # score weight for change magnitude
    text_density_weight: float = 0.3         # score weight for text density
```

Defaults are tuned for typical Zoom/Meet/Loom-style recordings.

## Architecture

### New module

`src/peeklet/core/demo_filter.py` — single new file, keeps existing
`video.py`, `pipeline.py`, and `comparator.py` unchanged.

### Public surface

```python
def classify_content_segments(
    decoder: VideoDecoder,
    config: DemoFilterConfig,
) -> list[ContentSegment]:
    """Stage 1: sparse OCR sweep, returns time ranges marked demo/non-demo."""

def select_demo_keyframes(
    keyframes: list[FrameResult],
    segments: list[ContentSegment],
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
) -> list[FrameResult]:
    """Stages 2+3: filter to demo segments, then transcript-anchor + major-change selection."""

def apply_demo_filter(
    decoder: VideoDecoder,
    keyframes: list[FrameResult],
    transcript: list[TranscriptSegment],
    config: DemoFilterConfig,
) -> list[FrameResult]:
    """Top-level entry point — calls the two above and stitches them together."""
```

### New types (`utils/types.py`)

```python
@dataclass
class ContentSegment:
    start_sec: float
    end_sec: float
    is_demo: bool
    text_density: float    # avg words per sampled frame in this segment
```

### New fields on `FrameResult`

- `content_type: Literal["demo", "non-demo"] | None`
- `selection_reason: Literal["transcript_anchor", "major_change"] | None`

### Integration point

In `process_video()` (`src/peeklet/core/video.py`), after the existing
keyframe enrichment loop (around line 349), if
`config.demo_filter.enabled` is true, call `apply_demo_filter()` and
replace the keyframes list with its return value before audio enrichment
and context export. Audio enrichment and context export then operate on
the filtered set unchanged.

### New dependency

- `pytesseract` added to `pyproject.toml`.
- Tesseract binary becomes a runtime requirement *only* when `--demo-mode`
  is used. Detected at the start of `apply_demo_filter()`; clear install
  instructions if missing.

## Stage 1 — Sparse OCR sweep

### Algorithm

1. **Sample timestamps:** generate timestamps at `ocr_sample_interval_sec`
   intervals (default 30s) across the full video duration. For a 60-min
   video: 0, 30, 60, ..., 3600 → 121 samples.
2. **Extract sample frames:** add a new `VideoDecoder.extract_frame_at(timestamp_sec)`
   helper that seeks to the nearest keyframe before `timestamp_sec` via libav and
   decodes forward to the requested frame. ~50ms per sample.
3. **Downscale for OCR:** resize each sample to `ocr_downscale_dim` (default
   360px on the longest edge). Detection-only — we are not reading text.
4. **Run Tesseract:** call `pytesseract.image_to_data()` (faster than
   `image_to_string`, gives word-level confidence). Count words with
   confidence > 30 and length ≥ 2 chars (filters punctuation noise).
5. **Build the timeline:** for each sample record `(timestamp, word_count)`.
   Mark `is_demo = word_count >= ocr_min_words` (default 5).
6. **Collapse into segments:** walk the sample timeline and merge consecutive
   same-classification samples into `ContentSegment` ranges. **Smoothing:**
   a single isolated sample sandwiched between two samples of the opposite
   class is flipped to match its neighbors. Avoids dropping a demo region
   because one sample landed on a transition or modal.
7. **Boundary alignment:** each segment's `start_sec` is the midpoint
   between this sample and the previous one (transitions land halfway
   between adjacent samples, not exactly on a sample point).

### Performance budget

- 121 samples × (50ms decode + 80ms OCR at 360p) ≈ **16 seconds**.

### Edge cases

- Video shorter than `ocr_sample_interval_sec` → take 3 samples evenly
  spaced (start, middle, end).
- OCR fails on a frame → treat as `is_demo = False`, log warning, continue.
- Frame extraction fails on a timestamp → skip sample, log warning, continue.
- All samples fail OCR → exit with: *"OCR pipeline failed entirely — cannot
  classify content. Check tesseract installation."*

## Stage 2 — Drop non-demo keyframes

Walk the keyframes from the existing pipeline. For each, look up which
`ContentSegment` its `video_timestamp` falls into. Drop any keyframe whose
segment is `is_demo = False`. Set `content_type = "demo"` on survivors.

Typical reduction: ~800 → ~500 (depends on how much of the video was
gallery view).

## Stage 3 — Transcript-anchored selection with major-change floor

For each transcript segment, in order:

1. **Find candidate keyframes** whose `video_timestamp` falls within
   `[segment.start, segment.end]`.
2. **If candidates exist**, score each:
   ```
   score = visual_change_weight * change_magnitude_score
         + text_density_weight  * normalized_text_density
   ```
   - `change_magnitude_score`: 1.0 = "major", 0.6 = "moderate", 0.2 = "minor"
     (uses existing `change_magnitude` field).
   - `normalized_text_density`: the keyframe's inherited `text_density`
     (see "Note on text density for scoring" below) divided by the maximum
     `text_density` observed across all Stage 1 samples. Result is in [0, 1].
     If max is 0, this term is 0.
   - Default weights: `visual_change_weight = 0.7`, `text_density_weight = 0.3`.
   - The single highest-scoring candidate is selected with
     `selection_reason = "transcript_anchor"`.
3. **Major-change floor:** any *other* candidate in the same segment that
   qualifies as "major" (`ssim < major_change_ssim` OR
   `block_count > major_change_blocks`) is *also* kept with
   `selection_reason = "major_change"`. Catches modal-popped-up-mid-sentence.
4. **Empty transcript segments** (silence with no speech): drop all
   candidate keyframes from this range. The transcript is the user's
   signal of "what matters" — silence means nothing matters.

### Note on text density for scoring

Stage 1 computes text density only on sparse samples, not on every keyframe.
For Stage 3 scoring, each keyframe **inherits the text density of the
nearest Stage 1 sample**. Re-OCR'ing every keyframe would cost too much for
a value used only as a tie-breaker among same-segment candidates.

## Expected output

For a 60-min demo with ~30 minutes of speech (~150 transcript segments) and
a few mid-sentence modals: roughly **150-180 frames** in the final output —
within the "100-200 well-chosen screenshots" target.

## Error handling

| Failure mode | Behavior |
|---|---|
| `--demo-mode` without `--transcript` | Exit 2 with the error message above |
| `--demo-mode` without video input | Exit 2 with the error message above |
| Tesseract binary missing | At start of `apply_demo_filter()`, run `pytesseract.get_tesseract_version()`. On failure exit with: *"--demo-mode requires the tesseract binary. Install with: brew install tesseract (macOS) / apt install tesseract-ocr (Linux)"* |
| OCR fails on individual frame | Treat as `is_demo = False`, log warning, continue |
| Frame extraction fails on a sample | Skip sample, log warning, continue |
| All samples fail OCR | Exit with: *"OCR pipeline failed entirely — cannot classify content. Check tesseract installation."* |
| Transcript parse fails | Surfaces from existing pipeline before demo filter runs; no new handling |
| Zero keyframes survive Stage 3 | Log warning *"Demo filter dropped all keyframes — video may not contain screen-share content"*. Still write empty manifest + context outputs |

## Performance budget (60-min video)

| Stage | Time |
|---|---|
| Existing coarse + hash + SSIM pass | ~30s |
| Stage 1: sparse OCR sweep (121 samples) | ~16s |
| Stage 2 + 3: in-memory filtering | <1s |
| Audio enrichment + context export | ~5s |
| **Total** | **~52s** |

Comfortably within the 60s budget.

## Testing strategy

### Unit tests (`tests/unit/test_demo_filter.py`)

- `classify_content_segments()` with mocked decoder + mocked Tesseract output:
  - segments built correctly from a sample timeline
  - smoothing flips isolated samples
  - very short videos use 3-sample fallback
- `select_demo_keyframes()` with synthetic keyframe lists:
  - non-demo keyframes are dropped
  - transcript-anchored selection picks the highest-scoring candidate
  - major-change floor keeps additional frames
  - empty transcript segments drop all candidates
- Scoring function: weight math and tie-breaking
- Boundary alignment: midpoint logic
- Error paths: missing tesseract, OCR failures, frame extraction failures

### Integration test (`tests/integration/test_demo_filter_pipeline.py`)

- Small synthetic video fixture (~10s, programmatically generated) mixing
  blank and text-heavy frames + hand-written transcript
- Run the full pipeline with `--demo-mode`
- Assert: keyframe count is reduced, `content_type` set on all survivors,
  `selection_reason` set, non-demo frames excluded
- Marked `@pytest.mark.requires_tesseract`, skipped gracefully if absent

### CLI tests (`tests/unit/test_cli.py` additions)

- `--demo-mode` without `--transcript` → exit code 2 with the right error
- `--demo-mode` without video → exit code 2 with the right error
- `--demo-mode` happy path → invokes demo filter pipeline

## Out of scope (potential follow-ups)

- Re-OCR of every demo keyframe for richer text-density scoring
- Pluggable OCR backend (EasyOCR, PaddleOCR)
- Configurable text-detection without Tesseract (edge-density proxy)
- Demo mode without transcript (would require alternative anchoring signal)
- GPU acceleration for OCR
