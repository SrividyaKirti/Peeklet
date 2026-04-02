# Tiled pHash + Absolute Block Count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect small UI changes on full-page screenshots by tiling pHash for the gate check and using absolute block count as a keyframe trigger.

**Architecture:** Tile-aware hashing splits tall images (aspect ratio > 1.5) into square-ish tiles before computing pHash — any tile mismatch proceeds to SSIM. The pipeline adds a secondary keyframe trigger: if `len(changed_regions) > min_changed_blocks`, it's a keyframe even when global SSIM is above threshold. No changes to comparator internals.

**Tech Stack:** numpy, imagehash, PIL, pydantic, skimage, pytest

---

### Task 1: Config — Add `tile_aspect_ratio` and `min_changed_blocks`

**Files:**
- Modify: `peeklet/config.py:35-39` (HasherConfig) and `peeklet/config.py:42-46` (ComparatorConfig)
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests for new config fields**

Add to `tests/unit/test_config.py`:

```python
from peeklet.config import HasherConfig

# Inside class TestDefaults:
def test_default_tile_aspect_ratio(self) -> None:
    config = PeekletConfig()
    assert config.hasher.tile_aspect_ratio == 1.5

def test_default_min_changed_blocks(self) -> None:
    config = PeekletConfig()
    assert config.comparator.min_changed_blocks == 3


# Inside class TestValidation:
def test_reject_negative_tile_aspect_ratio(self) -> None:
    with pytest.raises(ValueError):
        HasherConfig(tile_aspect_ratio=-1.0)

def test_reject_negative_min_changed_blocks(self) -> None:
    with pytest.raises(ValueError):
        ComparatorConfig(min_changed_blocks=-1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config.py -v -k "tile_aspect_ratio or min_changed_blocks"`
Expected: FAIL — fields don't exist yet

- [ ] **Step 3: Add config fields**

In `peeklet/config.py`, modify `HasherConfig`:

```python
class HasherConfig(BaseModel):
    """Perceptual hashing settings."""

    algorithm: Literal["phash"] = "phash"
    hash_size: int = Field(default=8, gt=0)
    tile_aspect_ratio: float = Field(default=1.5, gt=0.0)
```

Modify `ComparatorConfig`:

```python
class ComparatorConfig(BaseModel):
    """SSIM comparison settings."""

    ssim_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_changed_pct: float = Field(default=2.0, ge=0.0)
    min_changed_blocks: int = Field(default=3, ge=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add peeklet/config.py tests/unit/test_config.py
git commit -m "feat: add tile_aspect_ratio and min_changed_blocks config fields"
```

---

### Task 2: Hasher — Add `compute_phash_tiled` and `tiled_hashes_match`

**Files:**
- Modify: `peeklet/core/hasher.py`
- Test: `tests/unit/test_hasher.py`

- [ ] **Step 1: Write failing tests for `compute_phash_tiled`**

Add to `tests/unit/test_hasher.py`:

```python
from peeklet.core.hasher import compute_phash_tiled, tiled_hashes_match


class TestComputePhashTiled:
    def test_returns_list_of_hashes(self) -> None:
        # Tall image: 100 wide, 300 tall → tile_height=100 → 3 tiles
        frame = np.zeros((300, 100, 3), dtype=np.uint8)
        frame[:100] = [255, 0, 0]
        frame[100:200] = [0, 255, 0]
        frame[200:] = [0, 0, 255]
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        assert isinstance(hashes, list)
        assert len(hashes) == 3
        assert all(isinstance(h, str) for h in hashes)

    def test_different_tiles_get_different_hashes(self) -> None:
        frame = np.zeros((300, 100, 3), dtype=np.uint8)
        frame[:100] = [255, 0, 0]
        frame[100:200] = [0, 255, 0]
        frame[200:] = [0, 0, 255]
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        # At least some tiles should differ
        assert len(set(hashes)) > 1

    def test_single_tile_when_image_not_tall_enough(self) -> None:
        # Image shorter than tile_height → single tile
        frame = np.full((80, 100, 3), 128, dtype=np.uint8)
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        assert len(hashes) == 1

    def test_last_tile_handles_remainder(self) -> None:
        # 250 tall with tile_height=100 → tiles at 0-100, 100-200, 200-250
        frame = np.full((250, 100, 3), 128, dtype=np.uint8)
        hashes = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        assert len(hashes) == 3

    def test_identical_frames_same_tiled_hashes(self) -> None:
        frame = np.full((300, 100, 3), 128, dtype=np.uint8)
        h1 = compute_phash_tiled(frame, hash_size=8, tile_height=100)
        h2 = compute_phash_tiled(frame.copy(), hash_size=8, tile_height=100)
        assert h1 == h2

    def test_small_change_in_one_tile_detected(self) -> None:
        frame_a = np.full((300, 100, 3), 200, dtype=np.uint8)
        frame_b = frame_a.copy()
        # Change a region in the middle tile (rows 100-200)
        frame_b[140:160, 40:60] = [0, 0, 0]
        h_a = compute_phash_tiled(frame_a, hash_size=8, tile_height=100)
        h_b = compute_phash_tiled(frame_b, hash_size=8, tile_height=100)
        # Tile 0 and 2 should match, tile 1 should differ
        assert h_a[0] == h_b[0]
        assert h_a[1] != h_b[1]
        assert h_a[2] == h_b[2]
```

- [ ] **Step 2: Write failing tests for `tiled_hashes_match`**

Add to `tests/unit/test_hasher.py`:

```python
class TestTiledHashesMatch:
    def test_identical_tiled_hashes_match(self) -> None:
        hashes = ["abcdef0123456789", "1234567890abcdef"]
        assert tiled_hashes_match(hashes, hashes, tolerance=0)

    def test_one_tile_differs_no_match(self) -> None:
        hashes_a = ["abcdef0123456789", "1234567890abcdef"]
        hashes_b = ["abcdef0123456789", "ffffffffffffffff"]
        assert not tiled_hashes_match(hashes_a, hashes_b, tolerance=0)

    def test_different_tile_count_no_match(self) -> None:
        hashes_a = ["abcdef0123456789", "1234567890abcdef"]
        hashes_b = ["abcdef0123456789"]
        assert not tiled_hashes_match(hashes_a, hashes_b, tolerance=0)

    def test_tolerance_applied_per_tile(self) -> None:
        hashes_a = ["0000000000000000", "0000000000000000"]
        hashes_b = ["0000000000000001", "0000000000000001"]
        assert tiled_hashes_match(hashes_a, hashes_b, tolerance=4)

    def test_empty_lists_match(self) -> None:
        assert tiled_hashes_match([], [], tolerance=0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_hasher.py -v -k "Tiled"`
Expected: FAIL — ImportError, functions don't exist

- [ ] **Step 4: Implement `compute_phash_tiled` and `tiled_hashes_match`**

In `peeklet/core/hasher.py`, add after the existing functions:

```python
import math


def compute_phash_tiled(
    frame: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    hash_size: int = 8,
    tile_height: int = 1080,
) -> list[str]:
    """Compute perceptual hashes for vertical tiles of a frame.

    Splits the frame into tiles of `tile_height` rows each (last tile may be shorter).
    Returns one hash per tile.
    """
    h = frame.shape[0]
    n_tiles = max(1, math.ceil(h / tile_height))
    hashes: list[str] = []
    for i in range(n_tiles):
        y_start = i * tile_height
        y_end = min((i + 1) * tile_height, h)
        tile = frame[y_start:y_end]
        hashes.append(compute_phash(tile, hash_size=hash_size))
    return hashes


def tiled_hashes_match(
    hashes_a: list[str], hashes_b: list[str], tolerance: int = 0
) -> bool:
    """Check if all corresponding tile hashes match.

    Returns False if tile counts differ or any tile pair exceeds tolerance.
    """
    if len(hashes_a) != len(hashes_b):
        return False
    return all(
        hashes_match(ha, hb, tolerance=tolerance)
        for ha, hb in zip(hashes_a, hashes_b)
    )
```

Also add `import math` at the top of the file.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_hasher.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add peeklet/core/hasher.py tests/unit/test_hasher.py
git commit -m "feat: add tiled perceptual hashing for tall images"
```

---

### Task 3: Pipeline — Wire up tiled pHash and block count trigger

**Files:**
- Modify: `peeklet/pipeline.py`
- Test: `tests/integration/test_pipeline_batch.py`

- [ ] **Step 1: Write failing test — tall image with small change is detected**

Add to `tests/integration/test_pipeline_batch.py`:

```python
class TestTiledDetection:
    def test_small_change_on_tall_image_detected(self, tmp_output: Path) -> None:
        """A small localized change on a tall full-page screenshot must be detected."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85, "min_changed_blocks": 2},
                "hasher": {"tile_aspect_ratio": 1.5},
            }
        )
        pipeline = Pipeline(config)

        # Simulate a full-page screenshot (200 wide, 600 tall → aspect 3.0 > 1.5)
        frame_a = np.full((600, 200, 3), 220, dtype=np.uint8)
        frame_b = frame_a.copy()
        # Small change in the middle: simulate typed text (black on light background)
        frame_b[280:310, 80:140] = [10, 10, 10]

        result_a = pipeline.process_frame(frame_a, frame_id="frame_000")
        result_b = pipeline.process_frame(frame_b, frame_id="frame_001")

        assert result_a.is_keyframe is True
        assert result_b.is_keyframe is True
        assert "block" in (result_b.visual_reason or "").lower() or "localized" in (result_b.visual_reason or "").lower()

    def test_small_change_on_tall_image_skipped_without_tiling(self, tmp_output: Path) -> None:
        """Without tiled pHash, a small change on a tall image gets skipped at the hash gate."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85, "min_changed_blocks": 2},
                # Very high aspect ratio threshold → tiling never activates
                "hasher": {"tile_aspect_ratio": 100.0},
            }
        )
        pipeline = Pipeline(config)

        frame_a = np.full((600, 200, 3), 220, dtype=np.uint8)
        frame_b = frame_a.copy()
        frame_b[280:310, 80:140] = [10, 10, 10]

        pipeline.process_frame(frame_a, frame_id="frame_000")
        result_b = pipeline.process_frame(frame_b, frame_id="frame_001")

        # Without tiling, pHash matches → skipped
        assert result_b.is_keyframe is False

    def test_identical_tall_frames_still_skipped(self, tmp_output: Path) -> None:
        """Tiling should not cause false positives on identical tall frames."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85, "min_changed_blocks": 2},
                "hasher": {"tile_aspect_ratio": 1.5},
            }
        )
        pipeline = Pipeline(config)

        frame = np.full((600, 200, 3), 220, dtype=np.uint8)

        pipeline.process_frame(frame, frame_id="frame_000")
        result = pipeline.process_frame(frame, frame_id="frame_001")

        assert result.is_keyframe is False

    def test_block_count_triggers_keyframe_even_with_high_ssim(self, tmp_output: Path) -> None:
        """Block count threshold should trigger keyframe even when SSIM is above threshold."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 32, "window_size": 5, "noise_threshold": 0.8},
                # Very high SSIM threshold — normally would skip
                "comparator": {"ssim_threshold": 0.99, "min_changed_blocks": 1},
                "hasher": {"tile_aspect_ratio": 1.5},
            }
        )
        pipeline = Pipeline(config)

        frame_a = np.full((600, 200, 3), 220, dtype=np.uint8)
        frame_b = frame_a.copy()
        # Change enough blocks to exceed min_changed_blocks=1
        frame_b[280:320, 80:150] = [10, 10, 10]

        pipeline.process_frame(frame_a, frame_id="frame_000")
        result_b = pipeline.process_frame(frame_b, frame_id="frame_001")

        assert result_b.is_keyframe is True

    def test_non_tall_image_uses_single_hash(self, tmp_output: Path) -> None:
        """Normal aspect ratio images should use single pHash (existing behavior)."""
        config = PeekletConfig.model_validate(
            {
                "redactor": {"enabled": False},
                "exporter": {"output_dir": str(tmp_output)},
                "masking": {"block_size": 50, "window_size": 5, "noise_threshold": 0.8},
                "comparator": {"ssim_threshold": 0.85},
                "hasher": {"tile_aspect_ratio": 1.5},
            }
        )
        pipeline = Pipeline(config)

        # Square image — not tall
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        pipeline.process_frame(frame, frame_id="frame_000")
        result = pipeline.process_frame(frame, frame_id="frame_001")

        # Identical frames still skipped — same behavior as before
        assert result.is_keyframe is False
        assert result.ssim_score is None  # skipped at hash gate
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_pipeline_batch.py::TestTiledDetection -v`
Expected: FAIL — `test_small_change_on_tall_image_detected` fails because tiled hashing isn't wired up

- [ ] **Step 3: Implement pipeline changes**

In `peeklet/pipeline.py`, add import:

```python
from peeklet.core.hasher import compute_phash, compute_phash_tiled, hashes_match, tiled_hashes_match
```

Then modify `process_frame` — replace the hash computation and hash-match check sections. The full updated method:

```python
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        timestamp: datetime | None = None,
        app_name: str | None = None,
        window_title: str | None = None,
        source_format: str | None = None,
    ) -> FrameResult:
        h, w = frame.shape[:2]

        # Step 1: Apply adaptive mask
        masked_frame, mask_regions = self._mask.apply(frame)

        # Step 2: Compute perceptual hash — tiled for tall images
        is_tall = h > w * self._config.hasher.tile_aspect_ratio
        if is_tall:
            tile_height = w  # roughly square tiles
            current_hashes = compute_phash_tiled(
                masked_frame,
                hash_size=self._config.hasher.hash_size,
                tile_height=tile_height,
            )
            current_hash = current_hashes[0]  # store first tile hash as representative
        else:
            current_hashes = None
            current_hash = compute_phash(masked_frame, hash_size=self._config.hasher.hash_size)

        # Compute mean pixel value (used to disambiguate uniform frames with identical phash)
        current_mean = float(masked_frame.mean())

        # Step 3: First frame is always a keyframe
        if self._last_hash is None:
            return self._make_keyframe(
                frame,
                masked_frame,
                frame_id,
                current_hash,
                current_mean,
                visual_reason="First frame in sequence",
                mask_regions=mask_regions,
                timestamp=timestamp,
                app_name=app_name,
                window_title=window_title,
                source_format=source_format,
            )

        # Step 3b: Frame dimension change → automatic KEYFRAME
        if (
            self._last_keyframe is not None
            and masked_frame.shape[:2] != self._last_keyframe.shape[:2]
        ):
            prev_h, prev_w = self._last_keyframe.shape[:2]
            return self._make_keyframe(
                frame,
                masked_frame,
                frame_id,
                current_hash,
                current_mean,
                visual_reason=f"Dimension change ({prev_w}x{prev_h} -> {w}x{h})",
                mask_regions=mask_regions,
                timestamp=timestamp,
                app_name=app_name,
                window_title=window_title,
                source_format=source_format,
            )

        # Step 4: Hash comparison — tiled or single
        mean_diff = abs(current_mean - self._last_mean)  # type: ignore[operator]
        if is_tall and current_hashes is not None and self._last_tiled_hashes is not None:
            hash_matched = tiled_hashes_match(current_hashes, self._last_tiled_hashes) and mean_diff < 5.0
        else:
            hash_matched = hashes_match(current_hash, self._last_hash) and mean_diff < 5.0

        if hash_matched:
            result = FrameResult(
                frame_id=frame_id,
                event_type=EventType.SKIPPED,
                is_keyframe=False,
                perceptual_hash=current_hash,
                frame_width=w,
                frame_height=h,
                timestamp=timestamp,
                app_name=app_name,
                window_title=window_title,
                source_format=source_format,
                adaptive_mask=mask_regions if mask_regions else None,
                ssim_score=None,
                visual_reason="Hash match — visually identical to previous keyframe",
                prev_keyframe_id=self._last_keyframe_id,
                prev_keyframe_path=self._last_keyframe_path,
            )
            self._writer.append(result)
            return result

        # Step 5: Hash differs → compute SSIM
        comparison = compare_frames(
            masked_frame,
            self._last_keyframe,  # type: ignore[arg-type]
            block_size=self._config.masking.block_size,
        )

        # Step 6: Check both SSIM threshold AND block count
        n_changed_blocks = len(comparison.changed_regions)
        block_count_exceeded = n_changed_blocks > self._config.comparator.min_changed_blocks
        ssim_below_threshold = comparison.ssim_score <= self._config.comparator.ssim_threshold

        if ssim_below_threshold or block_count_exceeded:
            # KEYFRAME — either SSIM says significant change, or enough blocks changed
            if block_count_exceeded and not ssim_below_threshold:
                reason = (
                    f"Localized change — SSIM {comparison.ssim_score:.4f}"
                    f" above threshold but {n_changed_blocks} blocks changed"
                    f" (>{self._config.comparator.min_changed_blocks})"
                )
            else:
                reason = (
                    f"Significant change — SSIM {comparison.ssim_score:.4f},"
                    f" {comparison.changed_pct:.1f}% of blocks changed"
                )
            return self._make_keyframe(
                frame,
                masked_frame,
                frame_id,
                current_hash,
                current_mean,
                visual_reason=reason,
                mask_regions=mask_regions,
                timestamp=timestamp,
                app_name=app_name,
                window_title=window_title,
                source_format=source_format,
                ssim_score=comparison.ssim_score,
                change_score=comparison.change_score,
                changed_pct=comparison.changed_pct,
                changed_regions=comparison.changed_regions if comparison.changed_regions else None,
            )

        # Step 7: SSIM above threshold AND block count below threshold → SKIP
        result = FrameResult(
            frame_id=frame_id,
            event_type=EventType.SKIPPED,
            is_keyframe=False,
            perceptual_hash=current_hash,
            frame_width=w,
            frame_height=h,
            timestamp=timestamp,
            app_name=app_name,
            window_title=window_title,
            source_format=source_format,
            adaptive_mask=mask_regions if mask_regions else None,
            ssim_score=comparison.ssim_score,
            change_score=comparison.change_score,
            changed_pct=comparison.changed_pct,
            changed_regions=comparison.changed_regions if comparison.changed_regions else None,
            visual_reason=(
                f"Minor change — SSIM {comparison.ssim_score:.4f}"
                f" above threshold {self._config.comparator.ssim_threshold},"
                f" {n_changed_blocks} blocks changed"
                f" (≤{self._config.comparator.min_changed_blocks})"
            ),
            prev_keyframe_id=self._last_keyframe_id,
            prev_keyframe_path=self._last_keyframe_path,
        )
        self._writer.append(result)
        return result
```

Also add `self._last_tiled_hashes: list[str] | None = None` to `__init__`, and update `_make_keyframe` to save tiled hashes. In `_make_keyframe`, before the `self._last_hash = current_hash` line, the caller needs to pass tiled hashes. Simplest approach: store them on self when computed.

Update `__init__` rolling state:

```python
        self._last_tiled_hashes: list[str] | None = None
```

In `process_frame`, after calling `_make_keyframe` for keyframes, update tiled hashes. The cleanest way: update `self._last_tiled_hashes` inside `process_frame` right before each `_make_keyframe` call and each skip path. Specifically, add at the end of `_make_keyframe`:

```python
        # (caller sets self._last_tiled_hashes before calling _make_keyframe)
```

Actually, simplest: set `self._last_tiled_hashes` directly in `process_frame` before returning from any keyframe path. Add this line right before every `return self._make_keyframe(...)` call and right before the skip-return when hash matched:

In `process_frame`, after computing hashes and before any return:
- Before first-frame keyframe return: `self._last_tiled_hashes = current_hashes`
- Before dimension-change keyframe return: `self._last_tiled_hashes = current_hashes`
- Before hash-match skip return: no update needed (state stays)
- Before SSIM keyframe return: `self._last_tiled_hashes = current_hashes`
- Before SSIM skip return: no update needed

So add `self._last_tiled_hashes = current_hashes` right before each `return self._make_keyframe(...)` call (3 places).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_pipeline_batch.py -v`
Expected: ALL PASS (both old tests and new TestTiledDetection)

- [ ] **Step 5: Run full test suite for regressions**

Run: `pytest tests/unit/ tests/integration/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add peeklet/pipeline.py tests/integration/test_pipeline_batch.py
git commit -m "feat: wire up tiled pHash and block count keyframe trigger in pipeline"
```

---

### Task 4: Final verification — full test suite + edge cases

- [ ] **Step 1: Run complete test suite**

Run: `pytest tests/ -v --ignore=tests/datasets`
Expected: ALL PASS

- [ ] **Step 2: Verify existing tests haven't regressed**

Run: `pytest tests/unit/ tests/integration/ -v`
Expected: Same pass count as before + new tests

- [ ] **Step 3: Commit any fixups if needed**
