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

    def test_redactor_disabled_skips_ocr(self, tmp_path) -> None:
        """When redactor is disabled, no OCR is called and pii_detected stays None."""
        config = PeekletConfig()
        config.exporter.output_dir = str(tmp_path / "output")
        config.redactor.enabled = False

        pipeline = Pipeline(config)

        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_id="frame_001")

        assert result.is_keyframe is True
        assert result.pii_detected is None
