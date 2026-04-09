"""Unit tests for demo_filter.py."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np


def _make_frame(h: int = 360, w: int = 360) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_count_words_filters_low_confidence_and_short_tokens():
    from peeklet.core.demo_filter import _count_words_in_frame

    fake_data = {
        "text": ["Settings", "x", "Save", "", "Login", "."],
        "conf": ["95", "92", "80", "-1", "20", "99"],
    }
    with patch("peeklet.core.demo_filter.pytesseract") as mock_pt:
        mock_pt.image_to_data.return_value = fake_data
        mock_pt.Output.DICT = "dict"
        count = _count_words_in_frame(_make_frame(), downscale_dim=360)

    # "Settings" (95, len 8) OK
    # "x" (92, len 1) short
    # "Save" (80, len 4) OK
    # ""  (-1, empty) empty
    # "Login" (20, len 5) low conf
    # "." (99, len 1) short
    assert count == 2


def test_count_words_handles_tesseract_exception():
    from peeklet.core.demo_filter import _count_words_in_frame

    with patch("peeklet.core.demo_filter.pytesseract") as mock_pt:
        mock_pt.image_to_data.side_effect = RuntimeError("tesseract crashed")
        mock_pt.Output.DICT = "dict"
        count = _count_words_in_frame(_make_frame(), downscale_dim=360)

    assert count == 0  # graceful fallback


def test_count_words_returns_zero_when_pytesseract_is_none():
    from peeklet.core.demo_filter import _count_words_in_frame

    with patch("peeklet.core.demo_filter.pytesseract", None):
        count = _count_words_in_frame(_make_frame(), downscale_dim=360)

    assert count == 0


def test_downscale_for_ocr_already_smaller_returns_original():
    from peeklet.core.demo_filter import _downscale_for_ocr

    frame = _make_frame(h=200, w=200)
    out = _downscale_for_ocr(frame, downscale_dim=360)
    # Already smaller than the target, so it should be returned untouched.
    assert out is frame


def test_downscale_for_ocr_square_frame_resizes_both_dims():
    from peeklet.core.demo_filter import _downscale_for_ocr

    frame = _make_frame(h=720, w=720)
    out = _downscale_for_ocr(frame, downscale_dim=360)
    assert out.shape == (360, 360, 3)


def test_downscale_for_ocr_landscape_frame_preserves_aspect():
    from peeklet.core.demo_filter import _downscale_for_ocr

    frame = _make_frame(h=540, w=960)  # 16:9 landscape
    out = _downscale_for_ocr(frame, downscale_dim=480)
    # Longest edge should match downscale_dim, height should scale proportionally
    assert out.shape[1] == 480
    assert out.shape[0] == int(round(540 * 480 / 960))


def test_downscale_for_ocr_portrait_frame_preserves_aspect():
    from peeklet.core.demo_filter import _downscale_for_ocr

    frame = _make_frame(h=960, w=540)  # 9:16 portrait
    out = _downscale_for_ocr(frame, downscale_dim=480)
    assert out.shape[0] == 480
    assert out.shape[1] == int(round(540 * 480 / 960))


def test_build_search_window_caps_at_segment_end():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import _build_search_window

    seg = TranscriptSegment(start=10.0, end=12.0, text="x")
    start, end = _build_search_window(
        moment_ts=10.5,
        segment=seg,
        max_window_sec=5.0,
    )
    assert start == 10.5
    # Capped by segment end (12.0), not by max_window
    assert end == 12.0


def test_build_search_window_caps_at_max_window_when_segment_long():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import _build_search_window

    seg = TranscriptSegment(start=0.0, end=100.0, text="long monologue")
    start, end = _build_search_window(
        moment_ts=10.0,
        segment=seg,
        max_window_sec=5.0,
    )
    assert start == 10.0
    assert end == 15.0  # 10.0 + 5.0


def test_is_stable_passes_when_both_neighbors_similar():
    from peeklet.core.demo_filter import _is_stable

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    prev = np.full((100, 100, 3), 128, dtype=np.uint8)
    nxt = np.full((100, 100, 3), 128, dtype=np.uint8)

    assert _is_stable(frame, prev, nxt, threshold=0.92) is True


def test_is_stable_fails_when_one_neighbor_differs():
    from peeklet.core.demo_filter import _is_stable

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    prev = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Random noise — very different from frame, low SSIM
    rng = np.random.default_rng(42)
    nxt = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

    assert _is_stable(frame, prev, nxt, threshold=0.92) is False


def test_is_gallery_frame_returns_true_below_threshold():
    from peeklet.core.demo_filter import _is_gallery_frame

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("peeklet.core.demo_filter._count_words_in_frame", return_value=2):
        assert _is_gallery_frame(frame, downscale_dim=360, min_words=5) is True


def test_is_gallery_frame_returns_false_above_threshold():
    from peeklet.core.demo_filter import _is_gallery_frame

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("peeklet.core.demo_filter._count_words_in_frame", return_value=10):
        assert _is_gallery_frame(frame, downscale_dim=360, min_words=5) is False
