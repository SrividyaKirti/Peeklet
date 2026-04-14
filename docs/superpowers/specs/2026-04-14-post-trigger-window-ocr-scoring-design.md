# Post-Trigger Window OCR Scoring — Design Spec

**Date:** 2026-04-14
**Status:** Approved
**Depends on:** PR #21 (action-item privilege + context overhaul)

---

## Problem

`_pick_stable_index` selects the first SSIM-stable frame in the search
window. "Stable" means pixels aren't changing — loading pages, spinners,
and pre-navigation states are all stable. The Policy Health screen (the
#1 blocking bug from the April 01 test) was missed because the frame at
`t+5s` still showed the previous page. The 5s window is too short and
SSIM stability is the wrong signal.

## Solution

Replace SSIM-stability scoring with OCR word-count scoring. Pick the
frame with the most readable text in a wider 10s window. Require
pytesseract for demo mode.

---

## 1. Require pytesseract for demo mode

In `apply_demo_filter`, replace the existing pytesseract-missing warning
with a hard error:

```python
if pytesseract is None:
    raise RuntimeError(
        "Demo mode requires pytesseract. Install with: pip install peeklet[demo]"
    )
```

The `_is_gallery_frame` function's `if pytesseract is None: return False`
guard becomes unreachable in demo mode. Remove it — `_is_gallery_frame`
is only called from `select_frames_for_moments` which is only called
from `apply_demo_filter`.

## 2. Widen default window

Change `DemoFilterConfig.forward_search_window_max_sec` default from
`5.0` to `10.0`.

## 3. New scoring function

Replace `_pick_stable_index` usage with a new function:

```python
def _pick_best_content_index(
    samples: list[tuple[np.ndarray, float, int]],
    downscale_dim: int,
) -> int:
```

Logic:
- Run `_count_words_in_frame(frame, downscale_dim)` on each sample.
- Return the index with the highest word count.
- Tie-break: pick the latest frame (higher index — more likely to be
  fully loaded/navigated).
- All-zero word counts: return index 0 (same fallback as current).

## 4. Wire into select_frames_for_moments

Replace:
```python
picked_idx = _pick_stable_index(samples, config.ssim_stability_threshold)
```

With:
```python
picked_idx = _pick_best_content_index(samples, config.gallery_ocr_min_dim)
```

## 5. Retain existing functions

Keep `_pick_stable_index`, `_is_stable`, and `ssim_stability_threshold`
config. They are not used in demo mode after this change but may be used
by the non-demo video pipeline path or future work.

## 6. Config changes

| Field | Old default | New default | Notes |
|---|---|---|---|
| `forward_search_window_max_sec` | 5.0 | 10.0 | Wider window for post-navigation capture |
| `ssim_stability_threshold` | 0.92 | 0.92 (unchanged) | No longer used in demo mode |

## 7. Files changed

| File | Change |
|---|---|
| `src/peeklet/core/demo_filter.py` | New `_pick_best_content_index`; hard error on missing pytesseract; swap scoring call; remove pytesseract guard from `_is_gallery_frame` |
| `src/peeklet/config.py` | `forward_search_window_max_sec` default 5.0 → 10.0 |
| `tests/unit/test_demo_filter.py` | Tests for `_pick_best_content_index`; test pytesseract requirement; update window-dependent tests |

## 8. Not in scope

- Per-moment window size (anchor vs LLM picks)
- OCR result caching across frames
- Changes to `context.md` format
- Changes to non-demo video pipeline
