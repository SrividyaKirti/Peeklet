"""Integration test: PII in a frame → redacted keyframe saved."""

from unittest.mock import MagicMock, patch

import numpy as np

from peeklet.config import PeekletConfig
from peeklet.core.ocr import OcrResult
from peeklet.pipeline import Pipeline
from peeklet.utils.types import Region


class TestPiiPipeline:
    @patch("peeklet.core.redactor.extract_text_regions")
    def test_keyframe_with_pii_is_redacted(self, mock_ocr: MagicMock, tmp_path) -> None:
        """A frame containing an email should be saved with that region blacked out."""
        mock_ocr.return_value = [
            OcrResult(
                text="user@secret.com",
                region=Region(x=10, y=10, w=80, h=15),
                confidence=0.95,
            ),
        ]

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = True
        config.redactor.pii_types = ["email"]

        pipeline = Pipeline(config)

        frame = np.full((100, 200, 3), 200, dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_id="frame_001")

        assert result.is_keyframe is True
        assert result.pii_detected is True

        # Verify saved image has the region blacked out
        from PIL import Image
        saved = np.array(Image.open(result.asset_path))
        assert np.all(saved[10:25, 10:90] == 0)

    @patch("peeklet.core.redactor.extract_text_regions")
    def test_keyframe_without_pii_not_modified(self, mock_ocr: MagicMock, tmp_path) -> None:
        mock_ocr.return_value = [
            OcrResult(text="hello world", region=Region(x=10, y=10, w=60, h=15), confidence=0.9),
        ]

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = True
        config.redactor.pii_types = ["email"]

        pipeline = Pipeline(config)

        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_id="frame_001")

        assert result.is_keyframe is True
        assert result.pii_detected is False

    @patch("peeklet.core.redactor.extract_text_regions")
    def test_redactor_disabled_skips_ocr(self, mock_ocr: MagicMock, tmp_path) -> None:
        """When redactor is disabled, no OCR is called and pii_detected stays None."""
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = False

        pipeline = Pipeline(config)

        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_id="frame_001")

        assert result.is_keyframe is True
        assert result.pii_detected is None
        mock_ocr.assert_not_called()

    @patch("peeklet.core.redactor.extract_text_regions")
    @patch("peeklet.core.redactor.extract_text_from_regions")
    def test_subsequent_keyframe_uses_changed_regions_for_ocr(
        self, mock_ocr_regions: MagicMock, mock_ocr_full: MagicMock, tmp_path
    ) -> None:
        """A second keyframe with changed_regions should OCR only those regions."""
        mock_ocr_full.return_value = []  # no PII on first frame
        mock_ocr_regions.return_value = [
            OcrResult(
                text="user@secret.com",
                region=Region(x=10, y=10, w=80, h=15),
                confidence=0.95,
            ),
        ]

        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = True
        config.redactor.pii_types = ["email"]

        pipeline = Pipeline(config)

        # First frame — becomes keyframe, uses full-frame OCR (no changed_regions)
        frame_a = np.full((100, 200, 3), 200, dtype=np.uint8)
        pipeline.process_frame(frame_a, frame_id="frame_001")
        mock_ocr_full.assert_called_once()
        mock_ocr_regions.assert_not_called()

        # Second frame — significantly different, triggers keyframe with changed_regions
        frame_b = np.full((100, 200, 3), 50, dtype=np.uint8)
        result_b = pipeline.process_frame(frame_b, frame_id="frame_002")

        assert result_b.is_keyframe is True
        assert result_b.pii_detected is True
        mock_ocr_regions.assert_called_once()
