# Video + Transcript Context Export

**Date:** 2026-04-07
**Status:** Approved

## Problem

When passing a video + transcript to an LLM for analysis (e.g., "find 5 bugs in this demo"), feeding the raw video is token-expensive and feeding just the transcript loses all visual context. The LLM needs screenshots at the right moments, linked to the transcript, to understand what was on screen when the presenter was speaking.

## Solution

Enhance Peeklet's video mode to produce LLM-ready output: timestamped screenshots extracted at key moments (visual changes + transcript trigger words), plus JSON and Markdown files that link screenshots bidirectionally with transcript segments.

## Two Features

1. **Video + Transcript mode** (enhanced): Process a video with a transcript. Extract keyframes on significant screen changes AND when the transcript references the screen. Output: screenshots + JSON + Markdown with bidirectional linking.
2. **Image directory mode** (unchanged): Existing pipeline — upload a directory of screenshots, filter to keep only those with significant visual changes.

---

## Architecture: Approach B — Minimal Pipeline Extension

Add transcript trigger detection as a separate module. Feed forced timestamps into the existing video pipeline. Add new JSON/Markdown exporters. Each piece is testable independently.

### New Modules

#### `core/transcript_trigger.py`

Weighted keyword scoring to detect when transcript references the screen.

**Word lists:**

- **High-signal** (UI/visual nouns): "chart", "graph", "dashboard", "button", "menu", "sidebar", "diagram", "screen", "page", "modal", "dropdown", "tab", "panel", "table", "form", "icon", "window", "popup", "field", "toolbar", "flowchart", "heatmap"
- **Medium-signal** (deictic/action verbs): "here", "this", "notice", "look", "see", "click", "shown", "display", "hover", "scroll", "select", "drag", "toggle", "zoom", "highlight", "observe", "watch"

**Scoring logic:**

- High + medium word in same segment = **trigger**
- 2+ high-signal words in same segment = **trigger**
- Single medium word alone = **no trigger** (too noisy)

**Interface:**

- Input: `list[TranscriptSegment]` (from existing `audio.py`)
- Output: `list[TriggerResult]` — each with timestamp, trigger phrase, and the matched segment

#### `core/context_exporter.py`

Generates JSON and Markdown output from keyframe results + transcript segments.

**JSON output** (`output/context.json`):

```json
{
  "video": {
    "filename": "demo.mp4",
    "duration_s": 120.5,
    "total_screenshots": 8
  },
  "screenshots": [
    {
      "id": 1,
      "file": "screenshot_00_00_05_200.png",
      "timestamp_s": 5.2,
      "timestamp": "00:00:05.200",
      "trigger": "visual_change",
      "change_magnitude": "major",
      "seconds_since_prev_screenshot": null,
      "transcript_ids": [2, 3]
    }
  ],
  "transcript": [
    {
      "id": 1,
      "start": "00:00:00.000",
      "end": "00:00:04.500",
      "text": "Welcome to our product demo.",
      "screenshot_ids": []
    },
    {
      "id": 2,
      "start": "00:00:04.500",
      "end": "00:00:07.000",
      "text": "As you can see on this dashboard, revenue is spiking.",
      "screenshot_ids": [1]
    }
  ]
}
```

**Markdown output** (`output/context.md`):

```markdown
# Video Summary: demo.mp4
Duration: 2m 0.5s | Screenshots: 8

---

## Screenshot 1 (00:00:05.200) — visual change
![screenshot_00_00_05_200](screenshot_00_00_05_200.png)
**Change:** major | **Since prev:** —

> "As you can see on this dashboard, revenue is spiking."

---

## Screenshot 2 (00:00:12.800) — transcript trigger
![screenshot_00_00_12_800](screenshot_00_00_12_800.png)
**Change:** minor | **Since prev:** 7.6s

> "Now click here to open the settings panel."
```

**Bidirectional linking:** Computed once from keyframe results + transcript segments, used by both exporters. A transcript segment links to a screenshot if their time ranges overlap. A screenshot links to all transcript segments that overlap its timestamp.

### Modified Modules

#### `core/video.py`

- Accept a list of forced timestamps (from trigger detection) before the two-pass pipeline runs
- During frame processing, if a frame's timestamp is within ±0.5s of a forced timestamp, save as keyframe regardless of visual change detection
- **Merge logic:** If a visual-change keyframe and a transcript-trigger keyframe land within 1s of each other, save one screenshot and mark trigger as `"both"`
- Screenshot naming: `screenshot_00_00_05_200.png` (timestamp-based, underscores for filesystem safety)
- Always output JSON + Markdown for video mode (even without transcript — transcript section is empty)

#### `utils/types.py`

- Add `trigger_type: str | None = None` to `FrameResult`
- Values: `"visual_change"`, `"transcript_trigger"`, `"both"`, `None` (for skipped frames)

#### `core/exporter.py`

- Add `trigger_type` field to Parquet manifest schema (string, nullable)

#### `cli.py`

- Wire up trigger detection: if `--transcript` is provided, scan transcript for trigger words before video processing, pass forced timestamps into `process_video()`
- Call context exporters (JSON + Markdown) after video processing completes
- No new CLI flags needed

### Unchanged Modules

- `pipeline.py` — cascade logic stays as-is
- `config.py` — no new config (trigger words hardcoded, tunable later)
- `core/audio.py` — transcript parsing already works
- `core/masking.py`, `core/hasher.py`, `core/comparator.py`, `core/loader.py` — untouched
- Image directory mode — untouched

---

## Data Flow

```
Video + Transcript (SRT/VTT)
  |
  ├── audio.py: parse_transcript() → list[TranscriptSegment]
  |
  ├── transcript_trigger.py: detect_triggers() → list[forced_timestamps]
  |
  ├── video.py: process_video(forced_timestamps=...)
  |     ├── Pass 1: Coarse sampling (visual change detection)
  |     ├── Pass 2: Backfill transitions
  |     ├── Forced keyframes at trigger timestamps
  |     └── Merge overlapping visual + trigger keyframes
  |
  ├── exporter.py: Parquet manifest (existing, + trigger_type)
  |
  └── context_exporter.py:
        ├── Compute bidirectional links
        ├── Write context.json
        └── Write context.md
```

---

## Output Files

For video mode, the output directory contains:

```
output/
  screenshot_00_00_05_200.png
  screenshot_00_00_12_800.png
  ...
  manifest.parquet          # existing Parquet manifest
  context.json              # structured LLM-ready data
  context.md                # human/LLM-readable with inline image refs
```
