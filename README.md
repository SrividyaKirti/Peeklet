# Peeklet

Smart screenshot change detection. Filters noise, redacts PII, exports structured Parquet manifests of keyframes.

## What it does

Given a stream of screenshots captured every few seconds, Peeklet determines which frames represent meaningful changes (keyframes) and which are noise (cursor blinks, clock ticks, video playback). It reduces downstream OCR/LLM processing costs by 80-90%.

## Install

```bash
pip install peeklet

# With PII redaction support
pip install peeklet[ocr]
```

## Quick start

### CLI

```bash
peeklet --input ./screenshots --output ./output --no-redact
```

### Library

```python
from peeklet.config import PeekletConfig
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline

config = PeekletConfig()
pipeline = Pipeline(config)

frame = load_frame("screenshot.png")
result = pipeline.process_frame(frame, frame_id="frame_001")
print(result.is_keyframe)  # True or False

pipeline.finalize()  # writes manifest.parquet
```

### Query results with DuckDB

```sql
SELECT * FROM 'output/manifest.parquet' WHERE is_keyframe = true ORDER BY timestamp
```

## How it works

Peeklet runs a cascade of increasingly expensive operations, killing irrelevant frames early:

1. **Adaptive masking** -- auto-detects and ignores high-frequency change regions (video, ads, clock)
2. **Perceptual hashing** -- cheap 64-bit fingerprint kills ~70% of frames (obvious duplicates)
3. **SSIM comparison** -- structural similarity kills ~15% more (minor noise)
4. **PII redaction** -- OCR + regex on changed regions only (optional, runs on keyframes only)
5. **Parquet export** -- DuckDB-ready manifest with per-frame metadata

## Configuration

Zero-config works out of the box. For tuning:

```bash
peeklet --input ./screenshots --config config.json
```

See [design spec](docs/superpowers/specs/2026-04-01-peeklet-design.md) for full config reference.

## Development

```bash
git clone https://github.com/SrividyaKirti/Peeklet.git
cd Peeklet
uv sync --all-extras --dev
uv run pytest tests/ -v
```

## License

MIT
