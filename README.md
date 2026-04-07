# Peeklet

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

Smart screenshot change detection. Filters noise, exports structured Parquet manifests of keyframes.

Given a stream of screenshots captured every few seconds, Peeklet determines which frames represent meaningful changes (keyframes) and which are noise (cursor blinks, clock ticks, video playback). It reduces downstream OCR/LLM processing costs by 80-90%.

## Features

- **Cascade filtering** -- perceptual hashing, SSIM, and adaptive masking eliminate duplicate and noisy frames early
- **Adaptive masking** -- auto-detects and ignores high-frequency change regions (video players, ads, clocks)
- **Video support** -- two-pass smart sampling with audio/transcript alignment
- **Parquet export** -- DuckDB-ready manifests with per-frame metadata and keyframe linking
- **Zero-config defaults** -- works out of the box, fully tunable via JSON/YAML

## Installation

```bash
pip install peeklet
```

### Extras

```bash
# Video file support (mp4, mov, webm)
pip install peeklet[video]

# Development tools
pip install peeklet[dev]
```

## Quick Start

### CLI

```bash
# Process a directory of screenshots
peeklet --input ./screenshots --output ./output

# Process a video file
peeklet --input recording.mp4 --output ./output

# Use a config file
peeklet --input ./screenshots --output ./output --config config.yaml
```

### Python Library

```python
from peeklet.config import PeekletConfig
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline

config = PeekletConfig()
pipeline = Pipeline(config)

frame = load_frame("screenshot.png")
result = pipeline.process_frame(frame, frame_id="frame_001")

print(result.is_keyframe)    # True or False
print(result.event_type)     # EventType.KEYFRAME or EventType.SKIPPED
print(result.ssim_score)     # 0.0-1.0 (None for first frame)
print(result.visual_reason)  # human-readable explanation

pipeline.finalize()  # writes manifest.parquet
```

### Process a Video

```python
from pathlib import Path
from peeklet.config import PeekletConfig
from peeklet.core.video import process_video

config = PeekletConfig()
results = process_video(Path("recording.mp4"), config)

keyframes = [r for r in results if r.is_keyframe]
print(f"{len(keyframes)} keyframes from {len(results)} frames")
```

### Query Results with DuckDB

```sql
SELECT frame_id, timestamp, ssim_score, visual_reason
FROM 'output/manifest.parquet'
WHERE is_keyframe = true
ORDER BY timestamp;
```

## How It Works

Peeklet runs a cascade of increasingly expensive operations, rejecting irrelevant frames early:

```
Frame
  |
  v
[1. Adaptive Masking] -- detect and mask high-frequency regions (clocks, video, ads)
  |
  v
[2. Perceptual Hash]  -- 64-bit fingerprint, kills ~70% of frames (obvious duplicates)
  |
  v
[3. SSIM Comparison]  -- structural similarity, kills ~15% more (subtle noise)
  |
  v
[4. Parquet Export]    -- DuckDB-ready manifest with per-frame metadata
```

For video files, a two-pass strategy avoids decoding every frame:

1. **Coarse pass** at configurable sample FPS (default 1.0) finds transitions
2. **Backfill pass** around transitions decodes at native FPS to find exact change points

## API Reference

### Core Classes

#### `Pipeline`

```python
from peeklet.pipeline import Pipeline
from peeklet.config import PeekletConfig

pipeline = Pipeline(config: PeekletConfig)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `process_frame(frame, frame_id, timestamp=None, app_name=None, window_title=None, source_format=None)` | `FrameResult` | Process a single frame through the cascade |
| `finalize()` | `None` | Flush manifest to disk |

#### `PeekletConfig`

```python
from peeklet.config import PeekletConfig, load_config

# Default config
config = PeekletConfig()

# From file
config = load_config(Path("config.yaml"))
```

#### `load_frame`

```python
from peeklet.core.loader import load_frame

# Accepts: file path, bytes, PIL.Image, or numpy array
# Returns: numpy array, RGB uint8, shape (H, W, 3)
frame = load_frame("screenshot.png")
frame = load_frame(Path("screenshot.png"))
frame = load_frame(image_bytes)
frame = load_frame(pil_image)
frame = load_frame(numpy_array)
```

#### `ManifestWriter`

```python
from peeklet.core.exporter import ManifestWriter

writer = ManifestWriter(path=Path("output"), compression="snappy")
writer.append(result)  # add a FrameResult
writer.flush()         # write manifest.parquet
writer.clear()         # reset buffer
```

### Data Types

#### `FrameResult`

Returned by `Pipeline.process_frame()`. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | `str` | Unique frame identifier |
| `event_type` | `EventType` | `KEYFRAME` or `SKIPPED` |
| `is_keyframe` | `bool` | Whether this frame is a keyframe |
| `perceptual_hash` | `str` | Hex-encoded perceptual hash |
| `ssim_score` | `float \| None` | Structural similarity vs previous keyframe (0.0-1.0) |
| `change_score` | `float \| None` | `1 - ssim_score` |
| `changed_pct` | `float \| None` | Percentage of blocks that changed |
| `changed_regions` | `list[Region] \| None` | Regions that changed |
| `adaptive_mask` | `list[Region] \| None` | Regions masked as noise |
| `visual_reason` | `str \| None` | Human-readable reason for decision |
| `asset_path` | `str \| None` | Path to saved keyframe image |
| `prev_keyframe_id` | `str \| None` | Link to previous keyframe |
| `frame_width` | `int` | Image width in pixels |
| `frame_height` | `int` | Image height in pixels |

Video-specific fields (populated when processing video):

| Field | Type | Description |
|-------|------|-------------|
| `source_video` | `str \| None` | Source video filename |
| `video_timestamp` | `float \| None` | Time position in seconds |
| `video_frame_number` | `int \| None` | Frame index in video |
| `audio_activity` | `str \| None` | `"speech"` or `"silence"` |
| `transcript_segment` | `str \| None` | Aligned transcript text |
| `change_magnitude` | `str \| None` | `"major"`, `"moderate"`, or `"minor"` |

#### `Region`

```python
from peeklet.utils.types import Region

region = Region(x=0, y=0, w=32, h=32)
region.area       # 1024
region.to_dict()  # {"x": 0, "y": 0, "w": 32, "h": 32}
```

#### `EventType`

```python
from peeklet.utils.types import EventType

EventType.KEYFRAME  # "KEYFRAME"
EventType.SKIPPED   # "SKIPPED"
```

### Video Processing

#### `process_video`

```python
from peeklet.core.video import process_video, VideoDecoder, VideoMeta

# High-level: process entire video
results = process_video(Path("video.mp4"), config, writer=manifest_writer)

# Low-level: manual frame extraction
decoder = VideoDecoder(Path("video.mp4"))
meta: VideoMeta = decoder.get_metadata()
# meta.duration, meta.fps, meta.width, meta.height, meta.total_frames

for frame_array, pts, frame_number in decoder.extract_coarse_frames(sample_fps=1.0):
    result = pipeline.process_frame(frame_array, frame_id=f"frame_{frame_number}")
```

### Audio and Transcripts

```python
from peeklet.core.audio import parse_transcript, detect_speech_segments

# Parse SRT/VTT subtitle files
segments = parse_transcript(Path("subtitles.srt"))
# [TranscriptSegment(start=0.0, end=2.5, text="Hello world"), ...]

# Detect speech vs silence (requires pydub)
speech = detect_speech_segments(Path("audio.wav"))
```

## CLI Reference

```
Usage: peeklet [OPTIONS]

Options:
  --input PATH              Directory of images or video file (required)
  --output PATH             Output directory (default: ./output)
  --config PATH             JSON or YAML config file
  --no-audio                Disable audio detection for video
  --mode [video|image]      Force processing mode
  --transcript PATH         Path to SRT/VTT transcript file
  --version                 Show version and exit
  --help                    Show this message and exit
```

## Configuration

All settings are optional. Defaults work out of the box.

```yaml
# config.yaml
pipeline:
  mode: batch           # "batch" or "stream"
  concurrency: 4

masking:
  block_size: 32        # grid block size in pixels
  window_size: 15       # history window for noise detection
  noise_threshold: 0.8  # frequency threshold to mask a block (0.0-1.0)

hasher:
  algorithm: phash
  hash_size: 8          # hash resolution in bits
  tile_aspect_ratio: 1.5  # height/width ratio to trigger tiled hashing

comparator:
  ssim_threshold: 0.85  # SSIM above this = skip (0.0-1.0)
  min_changed_pct: 2.0  # minimum % of blocks changed
  min_changed_blocks: 3 # minimum block count for keyframe

exporter:
  output_dir: ./output
  keyframe_format: png        # "png" or "jpg"
  parquet_compression: snappy # "snappy", "gzip", "zstd", or "none"

input:
  supported_formats:
    - png
    - jpg
    - jpeg
    - bmp
    - tiff
    - webp
    - pdf
  sort_by: filename     # "filename" or "timestamp"

video:
  sample_fps: 1.0       # coarse sampling rate
  formats: [mp4, mov, webm]
  audio_detection: true
  transcript_path: null # path to SRT/VTT file
```

## Output Format

### Parquet Manifest

The manifest at `output/manifest.parquet` contains one row per processed frame with the full schema documented in the [API Reference](#data-types) section above. Query it with any Parquet-compatible tool:

```python
import pyarrow.parquet as pq

table = pq.read_table("output/manifest.parquet")
df = table.to_pandas()
keyframes = df[df["is_keyframe"] == True]
```

### Keyframe Images

Keyframe images are saved to the output directory in the configured format (PNG by default):

```
output/
  manifest.parquet
  frame_001.png
  frame_005.png
  frame_012.png
```

## Development

```bash
git clone https://github.com/SrividyaKirti/Peeklet.git
cd Peeklet
uv sync --all-extras --dev
```

### Run Tests

```bash
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=peeklet --cov-report=term-missing

# Skip dataset-dependent tests
uv run pytest tests/ -m "not datasets"
```

### Lint and Type Check

```bash
uv run ruff check peeklet/
uv run ruff format peeklet/
uv run mypy peeklet/
```

## Project Structure

```
peeklet/
  __init__.py          # version
  cli.py               # click CLI entry point
  config.py            # PeekletConfig and sub-configs (Pydantic)
  pipeline.py          # Pipeline orchestrator (cascade logic)
  core/
    loader.py          # load_frame() -- universal image loader
    hasher.py          # perceptual hashing (single + tiled)
    masking.py         # AdaptiveMask -- noise region detection
    comparator.py      # SSIM comparison and block-level diff
    exporter.py        # ManifestWriter and Parquet schema
    audio.py           # transcript parsing and speech detection
    video.py           # VideoDecoder and two-pass processing
  utils/
    types.py           # EventType, Region, FrameResult
    image.py           # image conversion utilities
```

## Requirements

- Python 3.10+
- Core: numpy, Pillow, scikit-image, imagehash, pyarrow, pydantic, click, pyyaml
- Optional: imageio, av, pydub (video)

## License

[MIT](LICENSE)
