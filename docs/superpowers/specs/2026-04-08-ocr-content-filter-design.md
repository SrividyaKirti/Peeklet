# OCR Content Filter — Design

**Date:** 2026-04-08
**Status:** Approved, ready for implementation planning
**Branch:** `feat/ocr-content-filter` (to be created)

## Problem

For a 60-minute screencast, Peeklet's video pipeline currently emits ~800 keyframes. Every one of those frames *is* legitimately different from its predecessor by SSIM and block-diff, so the detector is correct — but most of them are noise from the user's perspective and from a downstream LLM's perspective:

- Talking-head video calls (Zoom/Meet/Teams gallery views) generate sustained low-amplitude pixel change for the entire call duration as active-speaker borders shift, people move in their tiles, and reconnects happen. None of these frames have any "shared content" worth showing.
- 800 frames is too noisy for users to scan and far too token-expensive to send to an LLM.

We need a filter that excludes keyframes that contain *no shared application content* — i.e. the user is in a video call but no one is screen-sharing.

## Insight

Real shared content (IDEs, browsers, slides, terminals, spreadsheets, docs) is uniformly **text-rich** — typically hundreds to thousands of OCR-recoverable characters. Webcam galleries, blank desktops, and fullscreen video have effectively zero text (maybe small name labels). There is a wide, stable gap between the two regimes, which makes "OCR character count below threshold" a near-perfect classifier for "no app is being shared" without needing face detection, layout heuristics, or trained models.

## Solution

Run OCR on every candidate keyframe. If the extracted text is shorter than a configurable threshold, **demote** the frame: it stops being a keyframe, its saved image is deleted, and it disappears from user-facing outputs. The demoted row stays in the parquet manifest with a `demoted_reason` field for auditing.

## Scope

### In scope

- Run Tesseract OCR (via `pytesseract`) on every frame the pipeline marks as a keyframe.
- Store `ocr_text` and `ocr_char_count` on each `FrameResult` and in the parquet manifest.
- If `ocr_char_count < config.video.ocr_min_text_chars`, demote the frame:
  - Delete the saved image file from disk
  - Set `is_keyframe = False`
  - Set `asset_path = None`
  - Set `demoted_reason = "low_text_content"`
  - Do **not** update pipeline reference state (the next frame compares against the prior surviving keyframe, not the demoted one)
- Forced (transcript-triggered) keyframes bypass the OCR filter entirely — explicit user trigger words mean the moment is captured regardless of on-screen content.
- `keyframe_index` and `total_keyframes` reflect only surviving keyframes.
- `context.md` only references surviving keyframes.
- Default-on (`ocr_enabled: bool = True`); opt-out via config.

### Out of scope (deferred)

- Using OCR text for chapter detection, automatic frame titles, transcript-frame text alignment, or full-text search across the screencast.
- Stability collapse / burst filtering (the broader "transitions emit too many keyframes" question).
- Perceptual-hash deduplication of repeated screens across the timeline.
- Magnitude-budget selection or any fixed-K keyframe budgeting.
- A separate `context_llm.md` output — no longer needed because `context.md` itself becomes clean enough for both human and LLM consumption.

## Architecture

### New module: `src/peeklet/core/ocr.py`

A thin wrapper around `pytesseract` with one public function:

```python
def run_ocr(frame_rgb: np.ndarray) -> str:
    """Run OCR on an RGB frame and return the extracted text."""
```

- Imports `pytesseract` lazily and raises a clear `ImportError` with install hint (`pip install peeklet[ocr]` and `brew install tesseract` / `apt install tesseract-ocr`) if unavailable. Mirrors the existing `_check_video_deps` pattern in `src/peeklet/core/video.py:48`.
- Wraps the engine choice behind a single function so swapping to PaddleOCR/EasyOCR later is a one-file change.

### `FrameResult` additions (`src/peeklet/utils/types.py`)

```python
ocr_text: str | None = None         # full OCR output, stored for audit
ocr_char_count: int | None = None   # cached for fast filtering and queries
demoted_reason: str | None = None   # "low_text_content" for now; extensible
```

`ocr_text` is preserved in the manifest even though downstream uses are out of scope. Parquet compresses it cheaply, and storing it now means the threshold can be retuned without re-running OCR.

### Config additions (`src/peeklet/config.py`)

```python
# Under config.video
ocr_enabled: bool = True
ocr_min_text_chars: int = 50
ocr_engine: Literal["tesseract"] = "tesseract"
```

If `ocr_enabled=True` but `pytesseract` is not importable, `process_video` fails loudly at startup — never mid-loop.

### `process_video` modifications (`src/peeklet/core/video.py:237`)

The OCR filter slots in immediately after a frame is marked as a keyframe and its asset is saved, but before keyframe counters are bumped:

```
... existing pipeline.process_frame ...
... existing forced-keyframe promotion ...

if result.is_keyframe:
    asset_path = save_keyframe(...)        # already happens

    if config.video.ocr_enabled and not is_forced:
        ocr_text = run_ocr(frame)
        result.ocr_text = ocr_text
        result.ocr_char_count = len(ocr_text.strip())

        if result.ocr_char_count < config.video.ocr_min_text_chars:
            # Demote
            Path(asset_path).unlink(missing_ok=True)
            result.is_keyframe = False
            result.asset_path = None
            result.demoted_reason = "low_text_content"
            # Skip reference-state update and counter bumps
            results.append(result)
            continue

    # Existing keyframe bookkeeping (counters, rename, magnitude, etc.)
    keyframe_count += 1
    ...
```

### Critical invariants

- **Reference-state isolation:** demoted frames must not become the new pipeline reference. Otherwise the next genuine screen-share frame would be SSIM-compared against a webcam gallery, always read as "huge change," and trigger a noisy keyframe immediately after every demotion.
- **Forced-frame bypass:** transcript-triggered frames are kept regardless of OCR output. The user's explicit intent takes precedence over the content classifier.
- **Manifest invariants preserved:**
  - Every `is_keyframe=True` row still has a real asset on disk.
  - Every saved image still corresponds to an `is_keyframe=True` row.
  - New: some rows have `is_keyframe=False` *and* `demoted_reason` set, with no asset. Audit queries use `WHERE demoted_reason IS NOT NULL`.

## Engine Choice: Tesseract

- **Free**, mature, fast on CPU (~100–200 ms per UI screenshot at typical keyframe resolution).
- **System binary install** only — `brew install tesseract` / `apt install tesseract-ocr`. No GPU, no model downloads.
- **Accuracy is more than sufficient** for the count-chars use case. We don't need perfect transcription; we need to distinguish "near-empty" from "full of text," and Tesseract does this trivially.
- **Optional dep:** added under a new `peeklet[ocr]` extra in `pyproject.toml`, matching the existing `peeklet[video]` pattern.

**Estimated cost on a 1-hour video:** ~800 keyframes × ~150 ms = ~2 minutes of OCR. Acceptable. Parallelization deferred until empirically needed.

## Testing

### Unit tests — `tests/test_ocr.py` (new)

- `run_ocr` returns roughly expected text for a synthetic image with rendered text (PIL `ImageDraw` to render "Hello World" on a blank canvas).
- `run_ocr` returns empty/near-empty for a blank image.
- `run_ocr` returns empty/near-empty for a noise-only image (random pixels — webcam-like).
- `run_ocr` raises a clear `ImportError` with install hint when `pytesseract` is missing.

### Unit tests — `tests/test_video_ocr_filter.py` (new)

OCR is mocked at the module boundary (`peeklet.core.ocr.run_ocr`) so tests are fast and independent of the system tesseract binary.

- A frame with OCR char count below threshold is demoted: `is_keyframe=False`, `asset_path=None`, `demoted_reason="low_text_content"`, file deleted from disk.
- A frame above threshold passes through unchanged.
- A demoted frame does **not** update pipeline reference state — the next frame is still compared against the prior surviving keyframe.
- A forced (transcript-triggered) frame with empty OCR is **kept** (filter bypassed).
- `keyframe_index` and `total_keyframes` reflect only surviving keyframes.
- `context.md` only references surviving keyframes.

### Integration test — extend `tests/test_video_pipeline.py`

One end-to-end test using a tiny synthetic video that mixes text-rich frames (rendered with PIL) and blank frames. Run the full `process_video` with `ocr_enabled=True` and a real tesseract install, gated with `pytest.importorskip("pytesseract")`. Assert the blank frames are absent from the surviving keyframe set on disk.

### Manual validation

After unit tests pass, rerun against the actual 60-minute call recording. Inspect the `ocr_char_count` distribution in the manifest, confirm the threshold cleanly separates the two regimes, and tune `ocr_min_text_chars` from the default 50 if needed.

## Rollout

### Branching

Feature branch `feat/ocr-content-filter` from `main` (or whichever branch is canonical for in-progress work when this lands). One PR at the end. Per `feedback_branching.md`: never commit directly to develop, always feature branch + PR.

### Dependency

- New `peeklet[ocr]` extra in `pyproject.toml` containing `pytesseract`.
- README install section documents the system binary requirement.
- `process_video` fails fast at startup if `ocr_enabled=True` but `pytesseract` is unimportable.

### Config defaults

- `ocr_enabled = True` — on by default. The existing output is too noisy; opt-out is the right default.
- `ocr_min_text_chars = 50` — initial guess, refined by manual validation against the call recording.

### Docs

- README feature list and config section mention OCR-based filtering.
- Short "Why some keyframes get filtered" note so users understand reduced frame counts.
- Mention the `demoted_reason` manifest field for auditing.

### Lint and verification before commit

Per `feedback_lint_before_commit.md`:

- `ruff check` clean
- `ruff format --check` clean
- Full test suite green
- Manual run against the real call recording, eyeball the surviving keyframes

### PR contents

1. New `src/peeklet/core/ocr.py` (thin tesseract wrapper)
2. New fields on `FrameResult` in `src/peeklet/utils/types.py`
3. New config fields in `src/peeklet/config.py`
4. Modified `src/peeklet/core/video.py` (demotion logic)
5. New `tests/test_ocr.py` and `tests/test_video_ocr_filter.py`
6. Extended `tests/test_video_pipeline.py` integration test
7. `pyproject.toml` `[ocr]` extra
8. README updates

## Summary

Run Tesseract OCR on every candidate keyframe. If extracted text is shorter than 50 characters, demote the frame: delete the saved image, flip `is_keyframe` to False, set `demoted_reason="low_text_content"`. Forced (transcript-trigger) frames bypass the filter. Demoted frames do not update pipeline reference state. The manifest still records demoted rows for audit. Default-on, opt-out via config. New branch, new PR, full lint and tests.
