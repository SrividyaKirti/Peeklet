# Video Input Support for Peeklet

**Date:** 2026-04-02
**Status:** Draft
**Scope:** Add video files as an input source, extracting meaningful keyframes for downstream LLM consumption.

## Problem

Peeklet currently only accepts directories of screenshot images. Users with demo recordings (MP4, MOV, WebM) must manually extract frames before using Peeklet. For the target use case — a Claude plugin that processes demo videos to generate insights, flag errors, and create Linear tickets — this friction must be zero.

## Goals

- Accept video files as input alongside the existing image directory mode
- Extract only meaningful keyframes using smart sampling (coarse pass + backfill)
- Preserve video-specific metadata (timestamps, audio activity, transcript alignment) for LLM consumption
- Zero system dependencies — everything installs via `pip install peeklet[video]`

## Non-Goals

- Audio transcription (users provide their own transcript files)
- Video editing or output
- Real-time / streaming video processing
- Processing mixed directories of images and videos in one run

## Supported Formats

MP4, MOV, WebM — covers the vast majority of screen recordings (Zoom, Loom, OBS, QuickTime, etc.).

## Architecture

### New Modules

#### `core/video.py` — VideoDecoder

Wraps `imageio` for frame extraction. Responsible for smart sampling only — all change detection stays in the existing pipeline.

```
VideoDecoder(path: Path, config: VideoConfig)

Methods:
  get_metadata() -> VideoMeta
    Returns: filename, duration, fps, resolution, total_frames

  extract_keyframes(pipeline: Pipeline) -> Iterator[VideoFrame]
    1. Coarse pass at configured sample_fps (default 1.0)
    2. Feed each sample to pipeline.process_frame()
    3. On SKIP -> KEYFRAME transitions between consecutive samples,
       backfill the gap at native fps to find exact transition
    4. Yield VideoFrame(frame, timestamp, frame_number) for each keyframe
```

**Backfill strategy detail:**

```
Pass 1 — Coarse (1 fps on a 60s/30fps video):
  60 samples from 1800 total frames

Pass 2 — Backfill around transitions:
  Only between SKIP -> KEYFRAME consecutive samples
  Extract the gap at native fps, feed through pipeline
  Example: frames 60(SKIP) and 90(KEYFRAME) -> extract 61-89

Typical result: ~90 frames processed instead of 1800
```

Backfill does NOT trigger for:
- Consecutive SKIPs (nothing changed)
- Consecutive KEYFRAMEs (already captured)
- KEYFRAME -> SKIP transitions (change already found)

#### `core/audio.py` — Audio Analysis

Two modes:

**Mode 1: No transcript provided (default)**
- Extract audio track using `pydub` (uses bundled ffmpeg from `imageio-ffmpeg`)
- Divide audio into chunks aligned with keyframe timestamps
- Compute RMS energy per chunk
- Above threshold = "speech", below = "silence"
- Populates `audio_activity` field in manifest

**Mode 2: User provides transcript file**
- Parse SRT or VTT files (stdlib only, no extra deps)
- For each keyframe timestamp, find overlapping transcript segment(s)
- Populates `transcript_segment` field in manifest
- `audio_activity` is also set to "speech" or "silence" based on transcript timing

### Modified Modules

#### `config.py` — New VideoConfig Section

```python
class VideoConfig:
    sample_fps: float = 1.0           # Coarse pass frame rate
    formats: list[str] = ["mp4", "mov", "webm"]
    audio_detection: bool = True      # Speech/silence detection
    transcript_path: str | None = None  # Path to SRT/VTT file
```

Added to `PeekletConfig` as `video: VideoConfig`.

#### `cli.py` — Input Detection & New Flags

```bash
# Single video
peeklet --input demo.mp4 --output ./output

# Directory of videos
peeklet --input ./recordings --output ./output

# With transcript
peeklet --input demo.mp4 --output ./output --transcript zoom.vtt

# Disable audio
peeklet --input demo.mp4 --output ./output --no-audio

# Mixed directory requires explicit mode
peeklet --input ./mixed --output ./output --mode video
peeklet --input ./mixed --output ./output --mode image
```

**Input detection logic:**
- `--input` is a file with video extension -> video mode
- `--input` is a directory with only images -> image mode
- `--input` is a directory with only videos -> video mode, process all sequentially
- `--input` is a directory with both -> require `--mode video|image`, error with guidance if missing:
  "Directory contains both image and video files. Select a mode: `--mode video` or `--mode image`"

#### `utils/types.py` — New Fields on FrameResult

```python
# Added to FrameResult:
source_video: str | None = None
video_timestamp: float | None = None       # Seconds into video
video_frame_number: int | None = None
time_since_prev_keyframe: float | None = None
audio_activity: str | None = None          # "speech" | "silence"
transcript_segment: str | None = None
keyframe_index: int | None = None          # Sequential: 1, 2, 3... (resets per video)
total_keyframes: int | None = None         # Per-video count
video_duration: float | None = None        # Total video length (seconds)
change_magnitude: str | None = None        # "minor" | "moderate" | "major"
```

All null for image-sourced frames. No breaking change to existing behavior.

#### `core/exporter.py` — Extended Parquet Schema

New columns added to the manifest:

| Field | Type | Description |
|-------|------|-------------|
| `source_video` | string | Source video filename |
| `video_timestamp` | float64 | Seconds into the video |
| `video_frame_number` | int32 | Original frame number |
| `time_since_prev_keyframe` | float64 | Gap since previous keyframe (seconds) |
| `audio_activity` | string | "speech" or "silence" |
| `transcript_segment` | string | Aligned transcript text |
| `keyframe_index` | int32 | Sequential keyframe step (1, 2, 3...) |
| `total_keyframes` | int32 | Total keyframes extracted |
| `video_duration` | float64 | Total video length (seconds) |
| `change_magnitude` | string | "minor" / "moderate" / "major" |

### Change Magnitude Derivation

Derived from SSIM score and block count after comparison:

- **minor**: SSIM > 0.90 and blocks <= 5
- **moderate**: SSIM 0.70-0.90 or blocks 6-15
- **major**: SSIM < 0.70 or blocks > 15

First keyframe is always "major" (new content).

## Packaging

New optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
video = ["imageio[ffmpeg]", "pydub"]
```

Install: `pip install peeklet[video]`

- `imageio[ffmpeg]` — video frame extraction with bundled ffmpeg binary
- `pydub` — audio energy analysis (uses bundled ffmpeg)
- SRT/VTT parsing uses stdlib only

If video mode is triggered without the extra installed:
> "Video support requires additional dependencies. Install with: `pip install peeklet[video]`"

## Multi-Video Processing

When `--input` points to a directory of video files:
- Process each video sequentially
- Each video's keyframes go into the same manifest
- `source_video` field distinguishes which video each keyframe came from
- `keyframe_index` resets per video (step 1 of video A, step 1 of video B)
- `total_keyframes` is per-video

## Pipeline Integration

The existing `Pipeline` class is unchanged. `VideoDecoder.extract_keyframes()` calls `pipeline.process_frame()` and uses the result to drive sampling decisions. The video module is purely a frame source with smart sampling — all detection, masking, hashing, comparison, and redaction logic stays in the existing cascade.

For multi-video processing, the pipeline is reset between videos (new `Pipeline` instance per video) so adaptive masking state doesn't leak across recordings.

## Testing

### Unit Tests
- `test_video.py` — VideoDecoder: metadata extraction, coarse sampling, backfill triggering, format detection, multi-video sequencing
- `test_audio.py` — RMS energy speech/silence detection, SRT parsing, VTT parsing, transcript-to-keyframe alignment
- `test_cli.py` (extend) — video mode detection, `--mode` flag, `--transcript` flag, mixed directory guidance

### Test Fixtures
- Synthetic videos (2-3 seconds) generated with `imageio` — static frames with a content change midway. Small enough to commit.
- Synthetic SRT/VTT files for transcript alignment

### Integration Tests
- End-to-end: video file -> keyframes + manifest with all new fields
- Backfill accuracy: coarse miss -> backfill finds exact transition frame
- Multi-video directory -> unified manifest with correct `source_video` attribution
- Video + transcript alignment
- Dependency-missing error message

## Example Output

Given a 90-second demo video of a user completing an onboarding flow:

```
output/
  manifest.parquet
  keyframes/
    step_001.png    # App loads (0:00)
    step_002.png    # User clicks "Sign Up" (0:08)
    step_003.png    # Form appears (0:09)
    step_004.png    # Form filled (0:34)
    step_005.png    # Success screen (0:35)
```

Manifest row for step_002:
```json
{
  "frame_id": "step_002",
  "source_video": "onboarding-demo.mp4",
  "video_timestamp": 8.234,
  "video_frame_number": 247,
  "keyframe_index": 2,
  "total_keyframes": 5,
  "video_duration": 90.0,
  "time_since_prev_keyframe": 8.234,
  "audio_activity": "speech",
  "transcript_segment": "Now click the sign up button in the top right",
  "change_magnitude": "major",
  "visual_reason": "SSIM 0.42 — major layout change, 28 blocks changed",
  "ssim_score": 0.42,
  "is_keyframe": true,
  "asset_path": "output/keyframes/step_002.png"
}
```
