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
