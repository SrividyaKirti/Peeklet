# Peeklet

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

**Turn a Fathom demo video + transcript into an LLM-ready context pack.**

You give Peeklet a screen-share recording and its transcript. Peeklet hands back an
enriched transcript that is interleaved with the exact screenshots a downstream LLM
needs in order to understand what is happening on screen when the presenter says
*"as you can see here..."* or *"click this button..."*.

```
  UI_enhancements.mp4  ──┐
                          │──▶  [ peeklet --demo-mode ]  ──▶  context.md
  UI_enhancements.md    ──┘                                   context.json
   (Fathom transcript)                                        demo_*.jpg × N
                                                              manifest.parquet
```

## Why this exists

Meeting tools like Fathom give you video + transcript. Multimodal LLMs can ingest
both, but:

- **Video is expensive.** Feeding a 50-minute screen-share to a VLM costs real
  money and burns context. 95% of those frames are the same pixels as the frame
  before.
- **Transcripts alone lose the screen.** "So when I click here, you notice the
  chart updates" — the transcript is worthless without the chart.
- **Naive 1-fps screenshots are noise.** You get hundreds of near-duplicates,
  your LLM gets distracted, and the important UI transition is still missed.

Peeklet closes this gap: it uses the transcript to decide *when* a screenshot is
essential, then uses perceptual hashing + SSIM + OCR to pick *which* exact frame
to capture at that moment. The output is a chronological document any
downstream MLLM can read top-to-bottom.

## What the output looks like

```markdown
## Visual Table of Contents

| # | Time  | Visual Context                                                 |
|---|-------|----------------------------------------------------------------|
| 1 | 2:12  | Add Tasks link from Analytics to Tasks page                    |
| 2 | 2:37  | Implement Insights component in Tasks; replace log modal       |
| 3 | 3:59  | Fix missing assistant prompt in Analytics/Tasks                |
...

## Timeline

**[00:02:12.033 → 00:02:15.800] Vignesh Subbiah**
Let's have a link from here to tasks page. [...]

**Screenshot 1 (00:02:12.033) — Add Tasks link from Analytics to Tasks page** [guaranteed]
![Screenshot](demo_0001_00132033ms.jpg)

**[00:02:15.800 → 00:02:32.400] Gowshik T**
Okay, got it. So instead of opening the modal for log details [...]
```

A downstream LLM reads the timeline in order, sees the transcript in context,
and has the on-screen evidence attached where it matters.

## Quick start

```bash
pip install 'peeklet[demo,video]'
brew install tesseract        # or: apt install tesseract-ocr

export OPENROUTER_API_KEY=sk-or-...

peeklet \
    --input demo.mp4 \
    --transcript demo.md \
    --demo-mode \
    --llm-provider openrouter \
    --llm-model "google/gemini-2.5-flash" \
    --output ./out
```

Output directory contains:
- `context.md` — human- and LLM-readable interleaved transcript + screenshots
- `context.json` — structured version with bidirectional transcript/screenshot links
- `demo_NNNN_*.jpg` — the picked keyframe images
- `manifest.parquet` — DuckDB-queryable row-per-frame manifest

## How it works

Demo mode runs two stages:

**Stage A — moment picking.**
- Parse the transcript. Extract Fathom `ACTION ITEM ... [WATCH](...)` markers
  as *anchors* (guaranteed captures — these are human-validated product
  moments).
- Ask the LLM to pick *complementary* moments outside ±10s of any anchor —
  the prompt renders the anchor list so the model knows what's already
  guaranteed and avoids visually duplicating those frames.
- Merge: anchors and LLM picks are both kept. Near-anchor LLM picks emit a
  warning but are not dropped; Stage B's SSIM + pHash dedup is the sole
  near-duplicate gate.

**Stage B — frame selection.**
For each moment, Peeklet does not trust the exact timestamp. Instead it:
- Builds a search window around the moment (biased ~3s before to catch the
  UI the speaker was pointing at *before* naming the action).
- Samples frames at 0.5s intervals inside the window.
- Scores each sample by OCR word count, picks the best-content frame.
- Rejects low-information frames (gallery view, blank loader) using a
  triple-AND check: edge density, text lines, and 8×8 grid occupancy must
  all be low before a frame is dropped.
- Checks caption/image alignment — the picked frame's OCR tokens should
  overlap the moment's caption; widens the window once if not.
- Deduplicates against the previously saved keyframe using SSIM, with a
  64-bit dHash backstop (`phash_hamming_threshold`) for near-duplicates
  SSIM misses.

See `docs/superpowers/specs/2026-04-09-transcript-driven-demo-mode-design.md`
for the full algorithm.

## Supported inputs

| Input           | Formats                                              |
|-----------------|------------------------------------------------------|
| Video           | `.mp4`, `.mov`, `.webm`                              |
| Transcript      | Fathom markdown (`.md`), SRT (`.srt`), WebVTT (`.vtt`) |

Fathom markdown support includes the `++[@MM:SS](url?timestamp=N.N)++` speaker
lines and the `**ACTION ITEM: ... - ++[WATCH](...)++**` action-item markers,
which become guaranteed anchors in the output.

## LLM providers

| Provider    | Env var               | Example model                              |
|-------------|-----------------------|--------------------------------------------|
| Anthropic   | `ANTHROPIC_API_KEY`   | `claude-haiku-4-5`                         |
| OpenAI      | `OPENAI_API_KEY`      | `gpt-4o-mini`                              |
| OpenRouter  | `OPENROUTER_API_KEY`  | `google/gemini-2.5-flash`                  |

Local / self-hosted models work through the OpenAI provider by setting
`OPENAI_BASE_URL` to any OpenAI-compatible gateway (Ollama, vLLM, etc.):

```bash
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
peeklet ... --demo-mode --llm-provider openai --llm-model "llama3"
```

## CLI

```
peeklet --input <video|dir> --output <dir> [options]

  --transcript PATH               Transcript file (.md / .srt / .vtt)
  --demo-mode                     LLM picks moments; Peeklet picks frames
  --llm-provider anthropic|openai|openrouter
  --llm-model TEXT
  --config PATH                   JSON or YAML config file
  --quality fast|balanced|precise      Bundles sample_fps + resolution
  --sensitivity low|medium|high        Bundles SSIM + block thresholds
  --format png|jpg                     Keyframe image format
  --no-audio                           Disable audio detection
  --mode video|image                   Force a mode for mixed dirs
```

## Modes

### Demo mode (`--demo-mode`)

The main, recommended path. Requires a video + transcript. Described above.

### Video mode (no `--demo-mode`)

Coarse-samples the video at `sample_fps` (default 1.0) and runs the full
change-detection cascade (adaptive masking → perceptual hash → SSIM →
parquet export). Produces the same `context.md` / `context.json` /
`manifest.parquet` shape, minus LLM-picked moments. Useful for ad-hoc
screen recordings without a curated transcript.

### Image mode

Runs the same cascade on a directory of standalone screenshots. No
transcript alignment.

## Python API

```python
from pathlib import Path
from peeklet.config import PeekletConfig
from peeklet.core.video import process_video

config = PeekletConfig()
config.demo_filter.enabled = True
config.video.transcript_path = "demo.md"
config.exporter.output_dir = "./out"

results = process_video(Path("demo.mp4"), config)
keyframes = [r for r in results if r.is_keyframe]
```

Query the manifest with DuckDB:

```sql
SELECT frame_id, video_timestamp, visual_context_goal, alignment_confidence
FROM 'out/manifest.parquet'
WHERE is_keyframe = true
ORDER BY video_timestamp;
```

## Configuration

All settings are optional; defaults work. Override with a YAML/JSON file via
`--config`.

```yaml
demo_filter:
  enabled: true
  llm_provider: openrouter
  llm_model: google/gemini-2.5-flash
  forward_search_window_max_sec: 10.0
  forward_search_step_sec: 0.5
  search_window_lookback_sec: 3.0
  gallery_ocr_min_dim: 1920
  dedup_ssim_threshold: 0.95
  phash_hamming_threshold: 5
  tail_skip_ratio: 0.02

video:
  sample_fps: 1.0
  processing_max_dim: 720
  audio_detection: true

exporter:
  output_dir: ./out
  keyframe_format: jpg
```

See `src/peeklet/config.py` for the full schema and field-by-field comments.

## Project layout

```
src/peeklet/
  cli.py                  # click entry point
  config.py               # PeekletConfig + presets (pydantic)
  pipeline.py             # frame-level cascade (masking → hash → SSIM)
  core/
    video.py              # VideoDecoder + process_video orchestrator
    demo_filter.py        # Stage A/B: moment picking + frame selection
    context_exporter.py   # writes context.md / context.json
    audio.py              # Fathom md / SRT / VTT parsing + anchor extraction
    llm.py                # LLM adapter Protocol + JSON parser
    llm_anthropic.py
    llm_openai.py
    llm_openrouter.py
    hasher.py             # perceptual hashing (single + tiled)
    masking.py             # adaptive masking (non-demo mode)
    comparator.py         # SSIM + block diff
    exporter.py           # parquet manifest writer
    loader.py             # universal image loader
  utils/
    types.py              # FrameResult, Moment, AnchorRef, etc.
    image.py              # dHash, resizing, grid helpers
```

## Development

```bash
git clone https://github.com/SrividyaKirti/Peeklet.git
cd Peeklet
uv sync --all-extras --dev

uv run ruff check src/ && uv run ruff format --check src/
uv run mypy src/
uv run pytest tests/ -v
```

## Requirements

- Python 3.10+
- Core: numpy, Pillow, scikit-image, imagehash, pyarrow, pydantic, click, pyyaml
- `[video]`: imageio, av, pydub
- `[demo]`: anthropic, openai, pytesseract (+ system `tesseract` binary)

## Status

Alpha. Breaking changes between minor versions are possible until 0.1.0.
Current milestone focus: M2 VLM captions per keyframe (so each screenshot
carries its own "what's on screen" summary, not just the transcript hook).

## License

[MIT](LICENSE)
