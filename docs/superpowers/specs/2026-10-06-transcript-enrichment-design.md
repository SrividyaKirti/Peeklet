# Transcript Enrichment: Screen-State–First Design

**Date:** 2026-10-06
**Status:** Draft (pending review)
**Base branch:** `develop` (6f064b6)
**Author:** Vidya + Claude

## Problem

Peeklet's goal: given a screen recording and its timestamped transcript, produce
the transcript enriched with screenshots, so a downstream LLM (which handles
images well and video poorly) can see what the speaker is talking about.

`develop` already does much of this via `--demo-mode`, but its direction has drifted:

- **Moment-first and LLM-dependent.** An LLM picks timestamps from the transcript
  (`llm.pick_moments`); frames are captured only at those moments. Every run needs
  an API key, costs money, and is non-deterministic. Screens the LLM never asks
  about are never seen.
- **The transcript is no longer the document.** The demo `context.md` /
  `context.json` list *moments* with captions; transcript segments without a
  moment disappear from the output.
- **Output is human-shaped.** Markdown for reading, not structures a program
  (Claude API call, RAG index, agent) can consume directly.
- **No image budget.** The number of screenshots is whatever the LLM picks.
- **Fathom-coupled.** Anchors and parsing assume Fathom markdown.

## Goals

- **The transcript is the document.** Every transcript segment appears in the
  output; screenshots are attached to segments.
- **Screen states are global.** A screen shown at 1:40 and again at 8:20 is one
  state with one image, referenced from both places.
- **Deterministic, local, free by default.** No LLM in the default path. Ranking
  sits behind a `Ranker` protocol so an LLM ranker can be added later.
- **Budget-aware.** Find every distinct state, rank them, keep the top N
  (default 20, caller-configurable).
- **Programmatic consumers first.** Structured JSON (for RAG/agents) and Claude
  message content blocks (for API calls).
- **Works on screen recordings and screen-shared meetings**, including meetings
  with webcam tiles alongside the shared screen, and stretches of gallery view.

## Non-Goals

- Audio transcription (a timestamped transcript is always provided).
- General camera footage / continuous-motion video.
- Human-facing Markdown output (may be added later as another renderer).
- Removing existing code paths (see "Existing code").
- An LLM ranker implementation (only the seam).

## Existing code

Left **untouched** and off the new path (removal is a later, separate decision):

- `peeklet --demo-mode` and its modules' LLM flow: `llm*.py`, `apply_demo_filter`,
  `build_demo_context`, `write_demo_context_*`.
- Image-directory mode, `Pipeline`'s Parquet manifest, the non-demo video path.

**Reused** by the new path (imported, not rewritten):

| Need | Reused from `develop` |
|---|---|
| Transcript parsing (SRT, VTT, Fathom md) + `TranscriptSegment` (with `speaker`) | `core/audio.py: parse_transcript` |
| Fathom `ACTION ITEM` anchors (optional ranking boost) | `core/audio.py: parse_fathom_anchors` |
| Video decoding, coarse sampling, seeking | `core/video.py: VideoDecoder` |
| Change detection to propose candidate frames | `Pipeline` cascade (masking → pHash → SSIM) |
| Gallery-view / low-info rejection | `core/demo_filter.py: _is_low_info_frame` (promoted to public `is_low_info_frame`) |
| OCR word boxes | `core/demo_filter.py: _ocr_word_boxes` (promoted to public `ocr_word_boxes`) |
| Global screen identity | `core/fingerprint.py: compute_fingerprint`, `FingerprintIndex` |

Promoting the two private helpers to public names is the only edit to existing
modules (old private names stay as aliases so `apply_demo_filter` is unchanged).

## Architecture

### Public API

```python
from peeklet.enrich import enrich

result = enrich(video="demo.mp4", transcript="demo.vtt", max_images=20, out_dir="./out")
result.to_json(out_dir)          # writes enriched.json + images/  (RAG, agents)
result.to_claude_content()       # list of text/image content blocks (Claude API)
```

CLI, as a **separate entry point** so the existing `peeklet` command is untouched:

```bash
peeklet-enrich demo.mp4 --transcript demo.vtt --max-images 20 --out ./out
```

### Stages and modules

All new code lives under `src/peeklet/enrich/`:

| Stage | Module | Responsibility |
|---|---|---|
| Orchestrate | `enrich/__init__.py` | `enrich()`: wires stages, applies budget, returns `EnrichedTranscript` |
| Candidates | `enrich/candidates.py` | Sample the video, run the change-detection cascade, return candidate frames with timestamps |
| States | `enrich/states.py` | Reject low-info frames; OCR + fingerprint each candidate; group into global states with occurrences; pick representative frame |
| Align | `enrich/align.py` | Link each transcript segment to the state(s) on screen while it is spoken |
| Rank | `enrich/rank.py` | `Ranker` protocol + `HeuristicRanker` |
| Render | `enrich/render.py` | `to_json`, `to_claude_content` |
| Types | `enrich/types.py` | `State`, `Segment`, `EnrichedTranscript` |
| Config | `config.py` | New `EnrichConfig` model on `PeekletConfig` |
| CLI | `enrich/cli.py` | `peeklet-enrich` entry point |

Data flow:

```
transcript ──► parse_transcript ─────────────────────────┐
                                                          ▼
video ──► candidates ──► states (reject, OCR, fingerprint, group) ──► align ──► rank ──► budget ──► EnrichedTranscript
                                                                                                   ├─► to_json
                                                                                                   └─► to_claude_content
```

## Data model

```python
@dataclass
class State:
    id: str                                  # "S1", "S2", ... by first appearance
    image_path: str                          # relative to out_dir, e.g. "images/S3.jpg"
    occurrences: list[tuple[float, float]]   # [(100.2, 131.0), (500.4, 512.9)] seconds
    ocr_text: str                            # from the representative frame
    score: float                             # set by the ranker

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None
    state_ids: list[str]                     # 0–2 entries, only kept states

@dataclass
class EnrichedTranscript:
    source_video: str
    duration: float
    states: list[State]                      # kept states only, ordered by first appearance
    segments: list[Segment]                  # every transcript segment, always
    dropped_state_count: int
    warnings: list[str]
```

## Building states

1. **Candidates.** Coarse-sample at the existing `video.sample_fps` (default 1.0) using
   `VideoDecoder`, run each sample through the existing change-detection cascade
   (adaptive masking absorbs flickering webcam tiles), and keep frames the cascade
   marks as keyframes. The first sample is always a candidate.
2. **Low-info rejection.** Each candidate passes through `is_low_info_frame`.
   Rejected frames (gallery view, blank loaders) don't become states, but they do
   **end** the current occurrence: nothing usable is on screen during that interval.
3. **Fingerprint and group.** For each surviving candidate, run OCR once
   (`ocr_word_boxes`), compute its `Fingerprint`, and look it up in a
   `FingerprintIndex` (global across the whole video). A hit adds an occurrence to
   that state. A miss creates a new state. Frames with empty Part A (OCR got
   nothing) always miss, as in develop, so unreadable frames are never merged
   together.
4. **Occurrences.** An occurrence runs from a candidate's timestamp until the next
   candidate (or rejected frame) that belongs to a different state, or the end of
   the video.
5. **Representative frame.** Among all candidate frames assigned to a state, keep
   the one with the highest OCR word count. Ties go to the latest frame, since
   UI that has finished loading usually shows more text. Only the representative
   frame is kept in memory per state, and it is written to disk only if the state
   survives the budget.

## Alignment

A segment links to every state whose occurrences overlap
`[segment.start, segment.end + lead_seconds]` (default `lead_seconds = 1.5`, which
covers narration that comes just before the action, e.g. "now I'll click save…").
If more than `max_states_per_segment` (default 2) overlap, keep the ones with the
largest overlap.

## Ranking

```python
class Ranker(Protocol):
    def score(self, states: list[State], segments: list[Segment]) -> dict[str, float]: ...
```

`HeuristicRanker` gives each state a weighted sum of these signals, each scaled to [0, 1]:

| Signal | Definition | Default weight |
|---|---|---|
| References | log(1 + number of linked segments), divided by the max across states | 0.30 |
| Spoken cues | share of linked segments containing a cue phrase (`this`, `here`, `see`, `look`, `notice`, `error`, `click`, `as you can see`; list configurable) | 0.25 |
| Text overlap | Jaccard overlap between content words in the linked segments and the state's `ocr_text` (stopwords removed) | 0.20 |
| Visual change | 1 − SSIM between this state's first candidate and the previous candidate | 0.15 |
| Time on screen | log(1 + total occurrence seconds), divided by the max across states | 0.10 |

**Fathom anchors.** If the transcript is Fathom markdown, any state on screen at an
`ACTION ITEM` anchor timestamp gets `anchor_bonus` (default +1.0) added to its
score. In effect it's always kept, unless anchored states alone exceed the budget.

Ties go to the state that appears first.

## Budget

After scoring, keep the top `max_images` states (default 20). Segments linked to a
dropped state lose that reference and are **not** re-pointed to a nearby kept state,
because a wrong image is worse than no image. `dropped_state_count` records how
many were cut. `max_images=0` gives a text-only result.

## Output formats

**`to_json(out_dir)`** writes `out_dir/enriched.json` (the `EnrichedTranscript`
serialized) and `out_dir/images/S{n}.jpg`. All paths are relative to `out_dir`.

**`to_claude_content()`** returns `list[dict]` of Anthropic content blocks:

- It walks the segments in order and merges consecutive segment text into one
  text block, each line formatted as `[mm:ss] Speaker: text`.
- The **first** time a state is referenced, the current text block ends with
  `[Screenshot S3]`, followed by a base64 `image` block.
- **Later** references append `(see Screenshot S3)` to the text and emit no image.
- Images are shrunk so the long edge is at most `max_image_edge` (default 1568px)
  before encoding.

## Configuration

A new `EnrichConfig` (pydantic, `extra="forbid"`) on `PeekletConfig`:

`max_images=20`, `lead_seconds=1.5`, `max_states_per_segment=2`,
`max_image_edge=1568`, `cue_phrases=[...]`, `ranker_weights={...}`,
`anchor_bonus=1.0`, `image_format="jpg"`.

Sampling rate is read from the existing `VideoConfig.sample_fps`. Fingerprint and rejector thresholds are read from the existing `DemoFilterConfig`
fields (`phash_threshold`, `ocr_field_min_chars`, `min_text_lines`,
`min_grid_cells`, `min_edge_ratio`, `gallery_ocr_min_dim`), so there is one
source of truth for them.

## Error handling

| Situation | Behavior |
|---|---|
| Transcript missing, unparseable, or empty | Raise `TranscriptError` (new), naming the failing line where known |
| Transcript extends past the end of the video | Warn. Those segments are kept with no states |
| All frames rejected (only gallery view) | Valid result with zero states and all segments, plus a warning |
| `max_images=0` | Valid text-only result |
| Video or demo extras missing (`imageio`, `pytesseract`, tesseract binary) | Raise with an install hint (reusing the existing dependency checks) |
| Corrupt or undecodable video | Decoder error propagates |

Warnings go into `EnrichedTranscript.warnings`, and are also logged.

## Testing

**Unit tests** (one file per module, `tests/unit/test_enrich_<module>.py`, matching the flat layout):
- `candidates`: the first frame is always included; static runs produce no candidates.
- `states`: groups identical screens that appear apart in time; doesn't merge
  different screens; empty-OCR frames never merge; rejected frames end
  occurrences; picks the representative frame by word count. OCR is stubbed with
  fixed word boxes.
- `align`: overlap including the lead, the cap of 2 states per segment, segments
  with no overlap.
- `rank`: each signal on its own, the anchor bonus, the tie-break, weights from config.
- `render`: each image is emitted once, later references are text only, merging
  of text blocks, resizing, JSON round trip.
- `enrich`: budget trimming leaves segments with empty references,
  `dropped_state_count`, `max_images=0`, warnings.

**Integration test** (`tests/integration/test_enrich.py`, needs tesseract): a
synthetic video with screen A → B → A, where the screens have distinct OCR-readable
headings, plus a flickering "webcam" tile in one corner and a 5s gallery-view-like
stretch, and a matching VTT. Asserts: exactly 2 states, A referenced in both
occurrences, the webcam tile creates no states, the gallery stretch creates no
states, and both renderers produce valid output.

**Real-recording check** (manual, not in CI): run on 2–3 real recordings, e.g. the
`UI_enhancements_Apr12026.mp4` recording used by develop's regression test (local only, not committed), a Loom-style demo,
and a Teams meeting with webcams on, and review states, alignment and ranking.
This is where the default thresholds and weights get tuned.

## Open questions

- Which real recordings to use for the manual check besides `UI_enhancements_Apr12026.mp4`.
- Whether OCR on every candidate is fast enough on hour-long recordings. If not,
  the fallback is to OCR only candidates that fail a cheap whole-frame pHash match
  against existing state representatives.
