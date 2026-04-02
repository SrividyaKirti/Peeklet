# Tiled pHash + Absolute Block Count Detection

## Problem

The pipeline misses small but meaningful UI changes on full-page screenshots. Mind2Web captures are 1280x4171 — the entire page top-to-bottom. A zip code typed in a field changes only 855 of 1.9M pixels (0.044%). Both pHash (resizes to 8x8 globally) and SSIM (averages over all blocks) are area-weighted and miss this.

Even with viewport-sized tiling, percentage-based metrics remain too diluted:

| Metric | Full page (1280x4171) | Tiled (~1280x1043) |
|--------|----------------------|-------------------|
| Changed blocks (32x32) | 3 of 5200 = 0.06% | 3 of 1300 = 0.23% |
| SSIM | ~0.99 | ~0.96 |

## Solution: Hybrid Approach C

Tile pHash to fix the gate problem. Use absolute block count (not percentage) as the keyframe trigger for localized changes.

### 1. Tiled pHash (`hasher.py`)

**When to tile:** `height > width * tile_aspect_ratio` where `tile_aspect_ratio` defaults to 1.5.

**Tile height:** `width * 1.0` — produces roughly square tiles. A 1280x4171 image → 4 tiles: rows 0-1280, 1280-2560, 2560-3840, 3840-4171.

New functions:
- `compute_phash_tiled(frame, hash_size, tile_height) → list[str]` — slices vertically, hashes each tile
- `tiled_hashes_match(hashes_a, hashes_b, tolerance) → bool` — True only if ALL tile pairs match; any mismatch → proceed to comparison

Edge cases:
- Different tile count between frames → automatic mismatch
- Single-tile result → equivalent to current behavior

### 2. Absolute Block Count Threshold (`config.py`)

New field in `ComparatorConfig`:
- `min_changed_blocks: int = 3`

If block diff finds more than `min_changed_blocks` changed 32x32 blocks, it's a keyframe regardless of SSIM score. Rationale: 1 block could be noise, 3+ blocks (96x32px minimum) is a real UI change.

No changes to `comparator.py` — it already returns `changed_regions`. This is a pipeline-level decision using existing output.

### 3. Pipeline Cascade (`pipeline.py`)

Previous cascade:
```
pHash match → SKIP
pHash mismatch → SSIM > threshold → SKIP
                 SSIM ≤ threshold → KEYFRAME
```

New cascade:
```
Tall image? (h > w * tile_aspect_ratio)
  Yes → tiled pHash
  No  → single pHash (existing behavior)

All tile hashes match (+ mean check) → SKIP
Any tile hash differs → compare_frames()

SSIM > threshold AND changed_blocks ≤ min_changed_blocks → SKIP
SSIM ≤ threshold OR  changed_blocks > min_changed_blocks → KEYFRAME
```

Key properties:
- Non-tall images follow the existing path — zero regression risk for viewport-sized screenshots
- Block count check applies to ALL images universally after passing pHash gate
- `visual_reason` enriched: `"Localized change — SSIM 0.97 but 4 blocks changed"`

### 4. Config Additions

`HasherConfig`:
- `tile_aspect_ratio: float = 1.5` — aspect ratio threshold for tiling

`ComparatorConfig`:
- `min_changed_blocks: int = 3` — absolute block count keyframe trigger

## Files Changed

| File | Change |
|------|--------|
| `peeklet/config.py` | Add `tile_aspect_ratio` to `HasherConfig`, `min_changed_blocks` to `ComparatorConfig` |
| `peeklet/core/hasher.py` | Add `compute_phash_tiled()`, `tiled_hashes_match()` |
| `peeklet/pipeline.py` | Detect tall images, use tiled pHash, add block count keyframe trigger |
| `tests/unit/test_hasher.py` | Tests for tiled hashing |
| `tests/integration/test_pipeline_batch.py` | Integration test: tall image with small localized change detected |

No new files. No changes to `comparator.py`.
