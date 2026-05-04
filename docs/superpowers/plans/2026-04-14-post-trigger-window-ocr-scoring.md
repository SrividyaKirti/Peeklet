# Post-Trigger Window OCR Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SSIM stability scoring with OCR word-count scoring for demo-mode frame selection, widen the search window to 10s, and require pytesseract for demo mode.

**Architecture:** Single function swap in `demo_filter.py`. New `_pick_best_content_index` scores each sampled frame by OCR word count and picks the richest one. `apply_demo_filter` raises on missing pytesseract instead of warning. Config default widens window from 5s to 10s.

**Tech Stack:** Python 3.10+, pytesseract, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-04-14-post-trigger-window-ocr-scoring-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/peeklet/core/demo_filter.py` | Modify | New `_pick_best_content_index`; require pytesseract; remove pytesseract guard from `_is_gallery_frame`; swap scoring call |
| `src/peeklet/config.py` | Modify | `forward_search_window_max_sec` default 5.0 → 10.0 |
| `tests/unit/test_demo_filter.py` | Modify | Tests for `_pick_best_content_index`; test pytesseract requirement; update affected tests |

---

### Task 1: Add `_pick_best_content_index` and swap scoring

**Files:**
- Modify: `src/peeklet/core/demo_filter.py:173-190` (add new function near `_pick_stable_index`)
- Modify: `src/peeklet/core/demo_filter.py:287` (swap call site)
- Modify: `tests/unit/test_demo_filter.py`

- [ ] **Step 1: Write failing tests for `_pick_best_content_index`**

Add to `tests/unit/test_demo_filter.py`:

```python
class TestPickBestContentIndex:
    def test_picks_frame_with_most_words(self):
        from peeklet.core.demo_filter import _pick_best_content_index

        samples = [
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.0, 300),
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.5, 315),
            (np.zeros((100, 100, 3), dtype=np.uint8), 11.0, 330),
        ]
        word_counts = [3, 15, 8]

        with patch(
            "peeklet.core.demo_filter._count_words_in_frame",
            side_effect=word_counts,
        ):
            idx = _pick_best_content_index(samples, downscale_dim=1920)

        assert idx == 1  # frame with 15 words

    def test_tiebreak_picks_latest_frame(self):
        from peeklet.core.demo_filter import _pick_best_content_index

        samples = [
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.0, 300),
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.5, 315),
            (np.zeros((100, 100, 3), dtype=np.uint8), 11.0, 330),
        ]
        word_counts = [10, 10, 10]

        with patch(
            "peeklet.core.demo_filter._count_words_in_frame",
            side_effect=word_counts,
        ):
            idx = _pick_best_content_index(samples, downscale_dim=1920)

        assert idx == 2  # latest frame wins tie

    def test_all_zero_returns_index_zero(self):
        from peeklet.core.demo_filter import _pick_best_content_index

        samples = [
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.0, 300),
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.5, 315),
        ]
        word_counts = [0, 0]

        with patch(
            "peeklet.core.demo_filter._count_words_in_frame",
            side_effect=word_counts,
        ):
            idx = _pick_best_content_index(samples, downscale_dim=1920)

        assert idx == 0

    def test_single_sample_returns_zero(self):
        from peeklet.core.demo_filter import _pick_best_content_index

        samples = [
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.0, 300),
        ]

        with patch(
            "peeklet.core.demo_filter._count_words_in_frame",
            return_value=5,
        ):
            idx = _pick_best_content_index(samples, downscale_dim=1920)

        assert idx == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_demo_filter.py::TestPickBestContentIndex -v`
Expected: FAIL — `ImportError: cannot import name '_pick_best_content_index'`

- [ ] **Step 3: Implement `_pick_best_content_index`**

Add to `src/peeklet/core/demo_filter.py`, after `_pick_stable_index` (after line 190):

```python
def _pick_best_content_index(
    samples: list[tuple[np.ndarray, float, int]],
    downscale_dim: int,
) -> int:
    """Return the index of the frame with the most OCR-readable text.

    Scores each sampled frame by OCR word count and returns the index
    with the highest count. Ties are broken by picking the latest frame
    (higher index — more likely to be fully loaded). Returns 0 when all
    frames score zero or the sample list has a single entry.
    """
    if len(samples) <= 1:
        return 0

    best_idx = 0
    best_count = -1
    for i, (frame, _ts, _fnum) in enumerate(samples):
        count = _count_words_in_frame(frame, downscale_dim)
        if count > best_count or (count == best_count and count > 0):
            best_count = count
            best_idx = i

    return best_idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_demo_filter.py::TestPickBestContentIndex -v`
Expected: ALL PASS

- [ ] **Step 5: Swap the call in `select_frames_for_moments`**

In `src/peeklet/core/demo_filter.py`, replace line 287:

```python
        picked_idx = _pick_stable_index(samples, config.ssim_stability_threshold)
```

with:

```python
        picked_idx = _pick_best_content_index(samples, config.gallery_ocr_min_dim)
```

- [ ] **Step 6: Run demo_filter tests**

Run: `pytest tests/unit/test_demo_filter.py -v`
Expected: All pass (except the 2 pre-existing pytesseract env failures).

- [ ] **Step 7: Commit**

```bash
git add src/peeklet/core/demo_filter.py tests/unit/test_demo_filter.py
git commit -m "feat: replace SSIM stability scoring with OCR word-count scoring for frame selection"
```

---

### Task 2: Require pytesseract for demo mode + widen window

**Files:**
- Modify: `src/peeklet/core/demo_filter.py:365-370` (`apply_demo_filter`)
- Modify: `src/peeklet/core/demo_filter.py:123-133` (`_is_gallery_frame`)
- Modify: `src/peeklet/config.py:101` (window default)
- Modify: `tests/unit/test_demo_filter.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_demo_filter.py`:

```python
def test_apply_demo_filter_raises_when_pytesseract_missing(tmp_path, monkeypatch):
    """Demo mode must fail hard when pytesseract is not installed."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", None)

    cfg = DemoFilterConfig(enabled=True)
    with pytest.raises(RuntimeError, match="pytesseract"):
        apply_demo_filter(
            decoder=decoder,
            transcript=[],
            config=cfg,
            output_dir=tmp_path,
        )
```

Also add a test for the new default window size:

```python
def test_default_forward_search_window_is_10s():
    from peeklet.config import DemoFilterConfig

    cfg = DemoFilterConfig()
    assert cfg.forward_search_window_max_sec == 10.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_demo_filter.py::test_apply_demo_filter_raises_when_pytesseract_missing tests/unit/test_demo_filter.py::test_default_forward_search_window_is_10s -v`
Expected: Both FAIL — warning instead of error, default is 5.0

- [ ] **Step 3: Implement**

**3a.** In `src/peeklet/core/demo_filter.py`, replace the pytesseract warning block in `apply_demo_filter` (lines 365-370):

```python
    if pytesseract is None:
        raise RuntimeError(
            "Demo mode requires pytesseract for OCR-based frame scoring and "
            "gallery detection. Install with: pip install peeklet[demo]"
        )
```

**3b.** In `src/peeklet/core/demo_filter.py`, remove the pytesseract guard from `_is_gallery_frame` (lines 131-132). The function becomes:

```python
def _is_gallery_frame(frame: np.ndarray, downscale_dim: int, min_words: int) -> bool:
    """Return True if the frame has too few visible words to be demo content."""
    return _count_words_in_frame(frame, downscale_dim) < min_words
```

**3c.** In `src/peeklet/config.py`, change line 101:

```python
    forward_search_window_max_sec: float = Field(default=10.0, gt=0.0)
```

- [ ] **Step 4: Update existing tests affected by the pytesseract change**

The test `test_apply_demo_filter_warns_when_pytesseract_unavailable` (which asserts a warning is logged) must be replaced since the behavior is now a hard error. Replace it:

```python
def test_apply_demo_filter_raises_when_pytesseract_unavailable(tmp_path, monkeypatch):
    """Missing OCR backend must raise, not silently degrade."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", None)

    cfg = DemoFilterConfig(enabled=True)
    with pytest.raises(RuntimeError, match="pytesseract"):
        apply_demo_filter(
            decoder=decoder,
            transcript=[],
            config=cfg,
            output_dir=tmp_path,
        )
```

The test `test_is_gallery_frame_returns_false_when_pytesseract_unavailable` is now testing removed behavior. Delete it entirely — the pytesseract guard no longer exists in `_is_gallery_frame`.

Any tests that pass `forward_search_window_max_sec=5.0` explicitly are fine (they override the default). Tests that rely on the default being 5.0 need the explicit override. Check `test_build_search_window_caps_at_max_window_when_segment_long` — it passes `max_window_sec=5.0` explicitly, so it's fine.

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_demo_filter.py -v`
Expected: All pass (the 2 pre-existing pytesseract failures should now be different — `test_is_gallery_frame_returns_true_below_threshold` may now pass since the guard was removed, or fail for a different reason).

- [ ] **Step 6: Commit**

```bash
git add src/peeklet/core/demo_filter.py src/peeklet/config.py tests/unit/test_demo_filter.py
git commit -m "feat: require pytesseract for demo mode, widen search window to 10s"
```

---

### Task 3: Full test suite + lint verification

- [ ] **Step 1: Run ruff check**

Run: `ruff check src tests`
Expected: Clean.

- [ ] **Step 2: Run ruff format check**

Run: `ruff format --check src tests`
Expected: Clean.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass except pre-existing `test_web_tasks.py::test_keyframe_images_saved`.

- [ ] **Step 4: Commit fixups if needed**

```bash
git add -u
git commit -m "chore: lint fixups for post-trigger window PR"
```
