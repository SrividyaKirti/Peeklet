"""Tests for OCR text extraction."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from peeklet.utils.types import Region


class TestExtractTextRegions:
    @patch("peeklet.core.ocr._get_reader")
    def test_returns_text_and_bounding_boxes(self, mock_get_reader: MagicMock) -> None:
        """OCR returns list of (text, region) tuples."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [100, 20], [100, 40], [10, 40]], "john@example.com", 0.95),
            ([[10, 50], [80, 50], [80, 70], [10, 70]], "hello world", 0.88),
        ]
        mock_get_reader.return_value = mock_reader

        from peeklet.core.ocr import extract_text_regions

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        results = extract_text_regions(frame)

        assert len(results) == 2
        assert results[0].text == "john@example.com"
        assert results[0].region == Region(x=10, y=20, w=90, h=20)
        assert results[1].text == "hello world"

    @patch("peeklet.core.ocr._get_reader")
    def test_empty_frame_returns_empty(self, mock_get_reader: MagicMock) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_get_reader.return_value = mock_reader

        from peeklet.core.ocr import extract_text_regions

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = extract_text_regions(frame)
        assert results == []

    @patch("peeklet.core.ocr._get_reader")
    def test_filters_low_confidence(self, mock_get_reader: MagicMock) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [50, 0], [50, 20], [0, 20]], "clear text", 0.9),
            ([[0, 30], [50, 30], [50, 50], [0, 50]], "garble", 0.1),
        ]
        mock_get_reader.return_value = mock_reader

        from peeklet.core.ocr import extract_text_regions

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = extract_text_regions(frame, min_confidence=0.5)
        assert len(results) == 1
        assert results[0].text == "clear text"


class TestExtractFromRegions:
    @patch("peeklet.core.ocr._get_reader")
    def test_ocr_on_cropped_regions(self, mock_get_reader: MagicMock) -> None:
        """When changed_regions are provided, OCR only those crops."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [40, 0], [40, 10], [0, 10]], "555-123-4567", 0.92),
        ]
        mock_get_reader.return_value = mock_reader

        from peeklet.core.ocr import extract_text_from_regions

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        regions = [Region(x=50, y=80, w=60, h=40)]
        results = extract_text_from_regions(frame, regions)

        # Bounding box should be offset back to full-frame coords
        assert len(results) == 1
        assert results[0].text == "555-123-4567"
        assert results[0].region == Region(x=50, y=80, w=40, h=10)


class TestOcrUnavailable:
    @patch("peeklet.core.ocr._get_reader", side_effect=ImportError("No module named 'easyocr'"))
    def test_graceful_fallback_when_easyocr_missing(self, mock_get_reader: MagicMock) -> None:
        from peeklet.core.ocr import extract_text_regions

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ImportError, match="easyocr"):
            extract_text_regions(frame)
