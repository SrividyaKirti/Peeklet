# Peeklet: Video + Transcript → Annotated Transcript

**Date:** 2026-10-07
**Status:** Draft (pending review)
**Base branch:** `develop` (6f064b6)
**Author:** Vidya + Claude

## What Peeklet does

One thing: take a **screen recording** and its **timestamped transcript**, and produce
the **transcript annotated with screenshots and descriptions** of what was on screen
while it was spoken. The result is for a downstream LLM, which handles images well
and video poorly. Each screenshot comes with a text description, so text-only
consumers (RAG, embeddings, search, cheaper models) also get the visual context.

```
demo.mp4 ──┐
           ├──► peeklet ──► out/transcript.json
demo.vtt ──┘                out/frame_134.jpg, frame_147.jpg, ...
```

Every other mode in the repo is removed (see "Removals").

## Principles

- **The transcript is the document.** Every transcript line appears in the output.
- **Screens are global.** A screen shown at 1:40 and again at 8:20 is one screen
  with one image and one description, referenced from both places.
- **Find locally, judge with an LLM.** Local code finds every distinct screen,
  cheaply and the same way every run. An LLM looks at a shortlist of real
  screenshots next to the words spoken over them, decides whether each is needed,
  and describes it. The LLM never guesses timestamps.
- **Works without an LLM.** `--no-llm` (or no API key) falls back to heuristic-only
  selection with no descriptions.
- **Budgeted.** At most N images (default 20).
- **A wrong image is worse than no image.** Lines only reference screens that were
  actually showing while they were spoken.

## Scope

**Inputs:** one video (MP4, MOV, WebM) and one transcript (SRT, VTT, or Fathom markdown).
**Target recordings:** screen recordings, and screen-shared meetings including
webcam tiles beside the shared screen and stretches of gallery view.
**Non-goals:** audio transcription, camera footage, multiple videos per run,
image-directory input, PII redaction, highlighting regions within a screenshot.

## Interface

```bash
peeklet demo.mp4 --transcript demo.vtt --out ./out \
    [--max-images 20] [--no-llm] [--llm-provider anthropic] [--llm-model claude-haiku-4-5] \
    [--config peeklet.yaml] [--debug]
```

```python
from peeklet import annotate

entries = annotate("demo.mp4", "demo.vtt", out_dir="./out", max_images=20)
blocks = entries.to_claude_content()   # Anthropic text/image content blocks
```

## Output

### `transcript.json`

A JSON array with one entry per **block**: a run of consecutive transcript lines
spoken while the same screen was showing (or while no usable screen was showing).

```json
[
  {
    "timestamp": "00:02:14",
    "transcript": "Alice: If you look at the left-hand side, you'll see the policy that was triggered.\nAlice: This policy blocks the agent from creating an S3 bucket unless the team tag is present.",
    "image": "frame_134.jpg",
    "visual_context": "Policy detail page. Left panel lists the triggered policy 'require-team-tag'; right panel shows its rule: deny s3:CreateBucket unless tag 'team' is present."
  },
  {
    "timestamp": "00:02:27",
    "transcript": "Alice: And here you can see the remediation message we send back to Claude.",
    "image": "frame_147.jpg",
    "visual_context": "Claude Code terminal showing a PAWS remediation message asking for a 'team' tag before retrying.",
    "action_item": "Add team-tag check to S3 policy"
  },
  {
    "timestamp": "00:05:02",
    "transcript": "Bob: Can everyone hear me okay?",
    "image": null
  },
  {
    "timestamp": "00:08:20",
    "transcript": "Alice: Going back to the policy page…",
    "image": "frame_134.jpg",
    "visual_context": "Policy detail page. Left panel lists the triggered policy 'require-team-tag'; ..."
  }
]
```

Rules:
- `timestamp` is the start of the block's first line, formatted `HH:MM:SS`.
- `transcript` lines are `Speaker: text` joined by `\n`. `Speaker` is used when
  the transcript has no speaker names.
- `image` is a filename in `out/`, or `null`. Files are named
  `frame_<seconds>.jpg`, using the whole second the screen's chosen frame was
  captured. A returning screen repeats the same filename, and the file exists once.
- `visual_context` appears only when an image is present and the LLM step ran.
- `action_item` appears only on blocks containing a Fathom action-item checkpoint.
  If one block has several, they're joined with `"; "`.

### `to_claude_content()`

Renders the same entries as content blocks in this layout:

```
[00:02:14]
Alice: If you look at the left-hand side, you'll see the policy that was triggered.
Alice: This policy blocks the agent from creating an S3 bucket unless the team tag is present.

[SCREENSHOT @ 00:02:14]
<image block>
Visual context: Policy detail page. ...
```

- Consecutive text is merged into one text block.
- An action item renders as `[ACTION ITEM @ HH:MM:SS] <text>` before the screenshot.
- A screen that appeared earlier renders as
  `[SCREENSHOT @ 00:08:20 — same screen as 00:02:14]` with no image block and no
  repeated description.
- Images are shrunk so the long edge is at most 1568px.

### `debug.json` (only with `--debug`)

Every screen with its occurrences, heuristic signal values and score, shortlist
status, LLM verdict (`include`, `reason`), whether it was kept; plus all
checkpoints and run warnings. This is for tuning. Downstream consumers don't need it.

## How it works

```
transcript ──► parse ──► verbal-cue checkpoints ─────────┐
transcript ──► Fathom action-item checkpoints ───────────┤
video audio ──► speech-onset checkpoints ────────────────┤
                                                         ▼
video ──► 1 fps samples ──► change detection ──► candidates (+ best frame near each checkpoint)
                                                         │
                       reject low-info ─► OCR + fingerprint ─► group into screens
                                                         │
                                     align lines to screens
                                                         │
                         heuristic score ─► shortlist (2×N) ─► LLM judge + describe
                                                         │
                                 select ≤ N ─► blocks ─► transcript.json + frames
```

### 1. Parse the transcript

SRT, VTT, or Fathom markdown, parsed into lines of `(start, end, speaker, text)`.
This reuses develop's parsers. Empty or unparseable input raises `TranscriptError`.

### 2. Checkpoints

A checkpoint is a timestamp where Peeklet makes sure it has a good capture and
gives the screen on display a ranking boost. There are three kinds:

| Kind | Source | Effect on ranking |
|---|---|---|
| `action_item` | Fathom `ACTION ITEM … [WATCH](…?timestamp=…)` markers, via develop's `parse_fathom_anchors` | `anchor_bonus` +1.0 (in practice always shortlisted and kept) and shown in the output |
| `speech_onset` | Start of speech after ≥ `min_pause_seconds` (1.5) of silence, via develop's `detect_speech_segments` (pydub RMS) | signal "onsets" |
| `verbal_cue` | Start of a transcript line whose trigger score is ≥ 1.0, using the lexicon restored from develop's deleted `transcript_trigger.py` (UI nouns 1.0, product words 0.5, deictic words 0.3) | signal "verbal cues" |

- With no Fathom markers, there are simply no action items.
- With no audio track, or audio that can't be decoded, a warning is recorded and
  there are no speech onsets.

### 3. Candidates

- **Sampling.** Sample the whole video at 1 fps (`sample_fps`) with develop's
  seek-based decoder.
- **Change candidates.** Each sample goes through adaptive masking, then pHash,
  then SSIM against the previous sample. Samples that changed become candidates,
  and the first sample always does. Masking absorbs flickering webcam tiles.
- **Checkpoint candidates.** For each checkpoint, look at the samples within
  ±`checkpoint_window_seconds` (5), which is 11 samples at 1 fps and needs no
  extra decoding. Drop low-info ones (step 4), and add the one with the most OCR
  words as a candidate. Ties go to the sample closest to the checkpoint. If all
  11 are low-info, the checkpoint gets no capture and a warning is recorded.

### 4. Screens

1. **Low-info rejection.** Develop's triple-AND rejector (text lines, occupied grid
   cells and edge density all low) drops gallery views and blank loaders. A
   rejected frame creates no screen but ends the current screen's occurrence.
2. **OCR + fingerprint.** OCR runs once per candidate, cached per sample. The
   fingerprint is (URL, heading, sidebar text) plus a pHash of the header strip.
3. **Group.** Look up the fingerprint in an index covering the whole video. A hit
   adds an occurrence to that screen; a miss creates a new screen. Frames where
   OCR found nothing usable always create a new screen.
4. **Occurrences.** An occurrence runs from its candidate until the next candidate
   or rejected frame that isn't this screen, or the end of the video.
5. **Chosen frame.** The candidate frame with the most OCR words; ties go to the
   latest. Only this frame and its OCR text are kept in memory per screen.

### 5. Align

Each transcript line links to **at most one** screen: the one whose occurrences
overlap `[start, end + lead_seconds]` (1.5s) the most. If none overlap, the line
has no screen.

### 6. Heuristic score

`score_screens(screens, lines, checkpoints, config) -> dict[screen_id, float]`.
Each signal is scaled to [0, 1]; weights are configurable:

| Signal | Definition | Weight |
|---|---|---|
| References | log(1 + linked lines), divided by the max across screens | 0.25 |
| Text overlap | Jaccard overlap of content words in linked lines vs. the screen's OCR text | 0.20 |
| Onsets | speech-onset checkpoints landing on this screen, scaled | 0.20 |
| Verbal cues | verbal-cue checkpoints landing on this screen, scaled | 0.15 |
| Visual change | 1 − SSIM vs. the previous candidate when the screen first appears | 0.10 |
| Time on screen | log(1 + total occurrence seconds), divided by the max across screens | 0.10 |

Plus `anchor_bonus` (+1.0) for each screen showing at an action item. Ties go to the
screen that appears first.

### 7. LLM judge + describe (skipped with `--no-llm`)

- **Shortlist.** The top `shortlist_factor × max_images` screens (2 × 20 = 40) by
  heuristic score.
- **One call per shortlisted screen**, run in parallel (`llm_concurrency`, default 8).
- **Input:**
  - the chosen frame, shrunk to at most 1568px on the long edge
  - its OCR text
  - up to `llm_max_lines` (8) linked transcript lines with timestamps, preferring
    lines that contain checkpoints
  - any action-item text
- **Instructions:**
  - decide whether understanding these lines needs this screenshot
  - describe what's on the screen and only that, without paraphrasing the transcript
  - copy on-screen text exactly where the OCR text confirms it
- **Output** (JSON, validated):
  `{"include": bool, "reason": str, "visual_context": str}`.
- **Providers:** develop's multi-provider client stays (`anthropic`, `openai`,
  `openrouter`). The `LLMClient` Protocol's `pick_moments` is replaced by
  `judge_screen(image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]) -> ScreenJudgment`.
  Each adapter sends the image in its provider's vision format. Default:
  `anthropic` / `claude-haiku-4-5`.
- **Cache.** Results are stored in `~/.cache/peeklet/judgments/`, keyed by
  sha256(image bytes, OCR text, lines, action items, prompt version, provider,
  model). Re-runs cost nothing and return identical results.
- **Failures.** If a call fails after the SDK's retries, or returns unparseable
  output, that screen falls back to `include = true`, gets no `visual_context`,
  and a warning is recorded. If every call fails, the run completes as if
  `--no-llm` were set, with a warning.
- **No API key** for the chosen provider: warn once and continue as `--no-llm`.
- **Rough cost** (Haiku 4.5 at $1 / $5 per million input/output tokens): about
  2.7k tokens in and 150 out per call, so roughly $0.0035 per call and about
  $0.15 for a 40-screen shortlist.

### 8. Select

- **With LLM:** keep screens with `include = true`, highest heuristic score first,
  up to `max_images`. Action-item screens are kept even if the LLM says
  `include = false`.
- **`--no-llm`:** keep the top `max_images` by heuristic score.
- Lines linked to screens that weren't kept get `image: null` and are not
  re-pointed elsewhere. `max_images = 0` gives text-only output.

### 9. Blocks and render

- Walk the lines in order. A new block starts when the line's screen (or "none")
  differs from the previous line's.
- Write `transcript.json` and the chosen frame of every kept screen as JPEG.
- With `--debug`, also write `debug.json`.

## Code layout after the change

```
src/peeklet/
  __init__.py       exports annotate, Entries
  cli.py            the single click command
  config.py         PeekletConfig (one model, YAML-loadable, extra="forbid")
  annotate.py       orchestrates steps 1–9
  types.py          Line, Checkpoint, Screen, ScreenJudgment, Entry, Entries
  transcript.py     SRT / VTT / Fathom-md parsers + Fathom action items (from core/audio.py)
  speech.py         speech-onset detection (from core/audio.py)
  cues.py           verbal-cue lexicon + scoring (restored transcript_trigger.py)
  video.py          VideoDecoder only (from core/video.py)
  change.py         masking + pHash + SSIM (from core/masking, hasher, comparator)
  screen.py         OCR boxes, low-info rejector, fingerprint + index (from core/demo_filter, core/fingerprint)
  screens.py        candidates incl. checkpoint windows, grouping, occurrences, chosen frame
  align.py
  score.py          heuristic score
  llm/              base.py (Protocol, prompt, JSON parsing, cache), anthropic.py, openai.py, openrouter.py
  render.py         blocks, transcript.json, frames, to_claude_content, debug.json
  image_utils.py    shared helpers (from utils/image.py)
```

`core/` and `utils/` go away. Surviving logic is moved, not rewritten, and moved
functions keep their tests.

## Removals

| Remove | Why |
|---|---|
| Image-directory mode, `pipeline.py`, `core/loader.py` | Not video + transcript |
| `core/exporter.py`, Parquet manifest, `pyarrow` | Not consumed by an LLM |
| `--demo-mode`, `apply_demo_filter`, `pick_moments` and its prompt | Replaced by local screens + LLM judge |
| `core/context_exporter.py` (`context.md` / `context.json`) | Replaced by `render.py` |
| `process_video`, `FrameResult` and its trigger/demo fields, `Moment`, `MomentEntry` | Replaced by the new flow and types |
| `--mode`, `--sensitivity`, `--quality`, `--format`, `--no-audio` flags and presets | One path, no modes |
| `scripts/`, `tests/datasets/`, `datasets` extra | Tooling for the old image-gate product |
| Tests for removed code | Code is gone |
| `docs/superpowers/{specs,plans}/2026-04-*` | Describe removed modes; git history keeps them |

**Kept from develop:** transcript parsers, Fathom action items, speech detection,
the decoder, masking/pHash/SSIM, OCR boxes, the low-info rejector, the fingerprint,
and the three LLM adapters (now doing `judge_screen`). **Restored:** the
`transcript_trigger.py` lexicon (as `cues.py`).

**Dependencies after**, all core: `numpy`, `Pillow`, `scikit-image`, `imagehash`,
`pydantic`, `click`, `pyyaml`, `imageio[ffmpeg]`, `av`, `pydub`, `pytesseract`,
`anthropic`, `openai`. The tesseract binary is a documented system requirement.
The README is rewritten around the single path.

## Configuration

`PeekletConfig` holds only what the path uses, with the defaults named above:
- sampling and change detection: `sample_fps`, masking/pHash/SSIM thresholds
- checkpoints: `checkpoint_window_seconds`, `min_pause_seconds`,
  `silence_threshold_dbfs`, cue lexicon and weights, `anchor_bonus`
- screens: rejector and fingerprint thresholds
- alignment: `lead_seconds`
- scoring: `score_weights`
- LLM: `llm_provider`, `llm_model`, `shortlist_factor`, `llm_concurrency`,
  `llm_max_lines`
- output: `max_images`, `max_image_edge`, `image_format`

CLI flags override the config.

## Error handling

| Situation | Behavior |
|---|---|
| Transcript missing, unparseable, or empty | `TranscriptError` naming the failing line where known |
| Transcript runs past the end of the video | Warning; those lines are kept with `image: null` |
| No audio track / audio can't be decoded | Warning; no speech onsets |
| Every frame low-info (only gallery view) | Valid output, all `image: null`, warning |
| Checkpoint window entirely low-info | Warning; that checkpoint gets no capture |
| LLM key missing / every call fails | Warning; continues as `--no-llm` |
| A single LLM call fails or returns bad JSON | Warning; that screen gets `include = true` and no description |
| tesseract binary not found | Error at startup with an install hint |
| Corrupt or undecodable video | Decoder error propagates |

Warnings are logged, written to `debug.json` when `--debug` is set, and summarized
on the CLI's final line (lines, screens found, shortlisted, kept, warnings).

## Testing

All unit and integration tests run with `--no-llm` or a stub `LLMClient`, so CI
needs no API keys or network.

- **Unit tests**, one file per module. OCR is stubbed with fixed word boxes.
  Existing tests for moved code move with it, including develop's tests for
  speech detection, Fathom action items, and the trigger lexicon (from history).
- **Checkpoints:** onsets require the minimum pause; verbal cues use the 1.0
  threshold; window selection by word count with the tie-break; an all-low-info
  window.
- **Screens:** identical screens far apart in time are grouped; different screens
  aren't merged; empty-OCR frames aren't merged; rejected frames end
  occurrences; the chosen frame is picked by word count.
- **Align / score / select:** at most one screen per line, the lead window, each
  signal, the anchor bonus, the tie-break, the shortlist size, LLM include/exclude,
  action items overriding an exclude, `max_images = 0`, nulls for lines whose
  screen was cut.
- **LLM layer:** prompt contents, JSON validation, cache hits and misses (the key
  changes when any input changes), failure fallbacks, the missing-key fallback,
  and each adapter's request shape with mocked SDKs.
- **Render:** block boundaries, filenames, repeated images, the `visual_context`
  and `action_item` fields, the `to_claude_content` layout including
  "same screen as", and `debug.json`.
- **Integration** (needs tesseract, uses a stub LLM): a synthetic video showing
  screen A → B → A with OCR-readable headings, a flickering webcam tile, a 5s
  gallery-like stretch, a silence gap before a line saying "look at this
  dashboard", and a matching VTT. Asserts: 2 screens; A referenced at both times;
  no screen from the webcam tile or gallery stretch; the onset and cue
  checkpoints land on the right screens; and the output entries are correct.
- **CLI:** a smoke test.
- **Manual, not in CI:** 2–3 real recordings (including the local-only
  `UI_enhancements_Apr12026.mp4`) with the real LLM, to tune thresholds, weights,
  and the prompt.

## Delivery

Work happens on `feat/transcript-enrichment` (from `develop`), and the PR targets
`develop`. Removals land first as their own commits with the remaining tests green.
Then the new path is built module by module. Merging `develop` into `main` follows.

## Open questions

- Which real recordings to use for tuning, besides `UI_enhancements_Apr12026.mp4`.
- Whether OCR on every candidate is fast enough for hour-long recordings. If not,
  skip OCR for candidates whose whole-frame pHash nearly matches an existing
  screen's chosen frame.
- The restored cue lexicon includes broad words ("cost", "count", "list",
  "status"). It's low-weight here, but may need trimming after real-recording runs.
