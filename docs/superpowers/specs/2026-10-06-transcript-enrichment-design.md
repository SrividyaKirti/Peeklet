# Peeklet: Video + Transcript → Annotated Transcript

**Date:** 2026-10-07
**Status:** Draft (pending review)
**Base branch:** `develop` (6f064b6)
**Author:** Vidya + Claude

## What Peeklet does

One thing: take a **screen recording** and its **timestamped transcript**, and produce
the **transcript annotated with screenshots** of what was on screen while each part
was spoken. The result is meant for a downstream LLM, which handles images well
and video poorly.

```
demo.mp4 ──┐
           ├──► peeklet ──► out/transcript.md
demo.vtt ──┘                out/transcript.json
                            out/images/S1.jpg, S2.jpg, ...
```

Every other mode and feature in the repo is removed (see "Removals").

## Principles

- **The transcript is the document.** Every transcript segment is in the output.
  Screenshots are attached to segments.
- **Screens are global states.** A screen shown at 1:40 and again at 8:20 is one
  state with one image, referenced from both places.
- **Deterministic and local.** No LLM and no network. The same input always
  gives the same output.
- **Budgeted.** Find every distinct state, rank them, keep the top N (default 20).
- **A wrong image is worse than no image.** Segments only reference states that
  were actually on screen while they were spoken.

## Scope

**Inputs:** one video file (MP4, MOV, WebM) and one transcript (SRT, VTT, or
Fathom markdown).
**Target recordings:** screen recordings (Loom, QuickTime, OBS) and screen-shared
meetings (Zoom, Meet, Teams), including meetings with webcam tiles beside the
shared screen and stretches of gallery view.

**Non-goals:** audio transcription, camera footage or continuous-motion video,
multiple videos per run, image-directory input, LLM-based selection, PII redaction.

## Interface

**CLI** (the only command):

```bash
peeklet demo.mp4 --transcript demo.vtt --out ./out [--max-images 20] [--config peeklet.yaml]
```

**Library** (the same function the CLI wraps):

```python
from peeklet import annotate

result = annotate("demo.mp4", "demo.vtt", out_dir="./out", max_images=20)
result.to_claude_content()   # list of Anthropic text/image content blocks
```

`annotate()` writes `transcript.md`, `transcript.json` and `images/` to `out_dir`
and returns the `AnnotatedTranscript`.

## Output

### `transcript.md`

```markdown
# Annotated transcript: demo.mp4

Duration 12:34 · 14 screenshots · 3 more screens omitted (image budget)

[00:01] **Alice:** Let me show you the dashboard.

![Screen S1](images/S1.jpg)

[00:05] **Alice:** As you can see here, the chart updates live.

[00:12] **Bob:** What happens when the sync fails?

![Screen S2](images/S2.jpg)

[01:40] **Alice:** Back on the dashboard… *(screen S1, shown at 00:01)*
```

- The image goes right after the first segment that references a state.
- Later references add an inline note `*(screen S1, shown at 00:01)*` and no image.
- The speaker label is omitted when the transcript has no speakers.

### `transcript.json`

```json
{
  "video": "demo.mp4",
  "duration": 754.2,
  "dropped_state_count": 3,
  "warnings": [],
  "states": [
    {"id": "S1", "image": "images/S1.jpg", "occurrences": [[1.0, 11.8], [99.5, 131.0]],
     "ocr_text": "Dashboard Revenue ...", "score": 0.82}
  ],
  "segments": [
    {"start": 1.0, "end": 4.6, "speaker": "Alice", "text": "Let me show you the dashboard.",
     "state_ids": ["S1"]}
  ]
}
```

Image paths are relative to `out_dir`, so the folder can be moved or uploaded as a whole.

### `to_claude_content()`

Walks the segments in the same order as the Markdown and returns content blocks:

- Consecutive segment lines are merged into one text block.
- The first reference to a state ends the current text block with `[Screen S3]`,
  followed by a base64 `image` block.
- Later references add `(screen S3, shown at mm:ss)` to the text.
- Images are shrunk so the long edge is at most 1568px before encoding.

## How it works

```
transcript ──► parse ────────────────────────────────────────────────┐
                                                                     ▼
video ──► sample ──► change detection ──► reject low-info ──► OCR + fingerprint ──► group into states
                                                                     │
                                       align segments ◄──────────────┘
                                              │
                                       rank ──► budget ──► render (md, json, images)
```

1. **Parse transcript** into segments `(start, end, speaker, text)`.
2. **Sample** the video at `sample_fps` (default 1.0) with seek-based decoding.
3. **Change detection.** Each sample goes through adaptive masking, then pHash,
   then SSIM against the previous sample. Only samples that changed become
   candidates, and the first sample always does. Masking absorbs flickering
   regions such as webcam tiles.
4. **Reject low-info frames.** A candidate is rejected (gallery view, blank
   loader) only if it's low on all three of: text lines, occupied grid cells,
   and edge density. A rejected frame creates no state but ends the current
   occurrence.
5. **OCR + fingerprint.** OCR runs once per surviving candidate. The fingerprint
   is (URL, heading, sidebar text) plus a pHash of the header strip.
6. **Group into states.** Look up the fingerprint in an index covering the whole
   video. A hit adds an occurrence to that state; a miss creates a new state.
   Frames where OCR found nothing usable always create a new state, so unreadable
   frames are never merged. An occurrence runs from its candidate until the next
   candidate or rejected frame that isn't this state, or the end of the video.
7. **Representative image.** For each state, keep the candidate frame with the
   most OCR words; ties go to the latest frame. Only this frame is kept in memory.
8. **Align.** A segment links to every state whose occurrences overlap
   `[start, end + lead_seconds]` (default 1.5s, which covers narration that comes
   just before the action). It keeps at most `max_states_per_segment` (default 2),
   preferring the largest overlap.
9. **Rank.** `score_states(states, segments, config) -> dict[str, float]` is a
   plain function and the single place to change selection logic later. Each
   signal is scaled to [0, 1]:

   | Signal | Definition | Weight |
   |---|---|---|
   | References | log(1 + linked segments), divided by the max across states | 0.30 |
   | Spoken cues | share of linked segments containing a cue phrase (`this`, `here`, `see`, `look`, `notice`, `error`, `click`, `as you can see`) | 0.25 |
   | Text overlap | Jaccard overlap of content words in the linked segments vs. the state's OCR text | 0.20 |
   | Visual change | 1 − SSIM vs. the previous candidate when the state first appears | 0.15 |
   | Time on screen | log(1 + total occurrence seconds), divided by the max across states | 0.10 |

   Ties go to the state that appears first.
10. **Budget.** Keep the top `max_images` states. Segments lose references to
    dropped states and are not re-pointed elsewhere. Kept states are renumbered
    `S1…Sn` in order of first appearance.
11. **Render** Markdown, JSON and images, written only for kept states.

## Code layout after the change

```
src/peeklet/
  __init__.py      exports annotate, AnnotatedTranscript
  cli.py           the single click command
  config.py        PeekletConfig: one flat-ish model, YAML-loadable
  annotate.py      orchestrates steps 1–11
  types.py         Segment, State, AnnotatedTranscript
  transcript.py    SRT / VTT / Fathom-md parsers (from core/audio.py)
  video.py         VideoDecoder only (from core/video.py)
  change.py        masking + pHash + SSIM candidate detection (from core/masking, hasher, comparator)
  screen.py        OCR word boxes, low-info rejector, fingerprint + index (from core/demo_filter, core/fingerprint)
  states.py        grouping, occurrences, representative image
  align.py
  rank.py
  render.py        markdown, json, claude content, image resize/save
  image_utils.py   shared helpers (from utils/image.py)
```

`core/` and `utils/` go away. Existing logic is moved, not rewritten, wherever it
survives. Each moved function keeps its tests.

## Removals

| Remove | Why |
|---|---|
| Image-directory mode, `pipeline.py`, `core/loader.py` | Not video + transcript |
| `core/exporter.py`, Parquet manifest, `pyarrow` | Not consumed by an LLM |
| `--demo-mode`, `core/llm*.py`, `openai`, `anthropic` deps | Selection is local now |
| `core/context_exporter.py` | Replaced by `render.py` |
| `process_video`, transcript triggers, `trigger_type` and other `FrameResult` fields | Replaced by the new flow |
| Speech/silence detection, Fathom anchor parsing, `pydub` | Not used by the new flow |
| `--mode`, `--sensitivity`, `--quality`, `--format`, `--llm-*`, `--no-audio` flags and presets | One path, no modes |
| `scripts/`, `tests/datasets/`, `datasets` extra | Dataset tooling for the old image-gate product |
| Tests for removed code | Code is gone |
| `docs/superpowers/{specs,plans}/2026-04-*` | Describe removed modes; git history keeps them |

**Dependencies after:** `numpy`, `Pillow`, `scikit-image`, `imagehash`,
`pydantic`, `click`, `pyyaml`, `imageio[ffmpeg]`, `av`, `pytesseract`, all core
(no extras besides `dev`). The tesseract binary is a documented system
requirement.

The README is rewritten around the single path.

## Configuration

`PeekletConfig` holds only what the path uses: `sample_fps`, masking/pHash/SSIM
thresholds (current defaults kept), rejector thresholds (`min_text_lines=10`,
`min_grid_cells=12`, `min_edge_ratio=0.02`, `ocr_max_dim=1920`), fingerprint
thresholds (`phash_threshold=6`, `ocr_field_min_chars=2`), `lead_seconds=1.5`,
`max_states_per_segment=2`, `max_images=20`, `cue_phrases`, `rank_weights`,
`max_image_edge=1568`, `image_format="jpg"`. Unknown keys are rejected.
`--max-images` on the CLI overrides the config.

## Error handling

| Situation | Behavior |
|---|---|
| Transcript missing, unparseable, or empty | `TranscriptError` naming the failing line where known |
| Transcript runs past the end of the video | Warning. Those segments are kept with no states |
| Every frame rejected (only gallery view) | Valid output with all segments and no images, plus a warning |
| `max_images=0` | Valid text-only output |
| tesseract binary not found | Error at startup with an install hint (`brew install tesseract` / `apt install tesseract-ocr`) |
| Corrupt or undecodable video | Decoder error propagates |

Warnings go into `transcript.json` and are logged. The CLI prints a one-line
summary (segments, states found, states kept, output path).

## Testing

- **Unit tests**, one file per module. OCR is stubbed with fixed word boxes
  everywhere except the integration test. Existing tests for moved code
  (transcript parsers, decoder, masking, hasher, comparator, fingerprint,
  rejector) move with it.
- **States:** identical screens far apart in time are grouped; different screens
  aren't merged; empty-OCR frames are never merged; rejected frames end
  occurrences; the representative image is chosen by word count.
- **Align, rank, budget:** lead window, the 2-state cap, each ranking signal, the
  tie-break, dropped references, renumbering, `max_images=0`.
- **Render:** Markdown image-once with back-references, JSON round trip, Claude
  content blocks, resizing.
- **Integration** (needs tesseract): a synthetic video showing screen A → B → A,
  with distinct OCR-readable headings, a flickering webcam tile in one corner, a
  5s gallery-like stretch, and a matching VTT. Asserts: 2 states, A referenced at
  both times, no state from the webcam tile or the gallery stretch, and all three
  outputs are valid.
- **CLI:** a smoke test of the single command.
- **Manual, not in CI:** 2–3 real recordings (e.g. `UI_enhancements_Apr12026.mp4`,
  a local file that isn't committed) to tune thresholds and weights.

## Delivery

Work happens on `feat/transcript-enrichment` (from `develop`), and the PR targets
`develop`. Removals land first as their own commits, with the remaining tests
green, so the history shows what was cut separately from what was built. Merging
`develop` into `main` follows.

## Open questions

- Which real recordings to use for tuning, besides `UI_enhancements_Apr12026.mp4`.
- Whether OCR on every candidate is fast enough for hour-long recordings. If not,
  skip OCR for candidates whose whole-frame pHash nearly matches an existing
  state's representative image.
