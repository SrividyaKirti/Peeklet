"""Tests for PII redaction."""

from unittest.mock import MagicMock, patch

import numpy as np

from peeklet.config import PiiPattern
from peeklet.core.ocr import OcrResult
from peeklet.core.redactor import (
    BUILTIN_PATTERNS,
    build_pattern_set,
    detect_and_redact,
    find_pii_in_text,
    redact_regions,
)
from peeklet.utils.types import Region


class TestBuiltinPatterns:
    def test_email_pattern(self) -> None:
        matches = find_pii_in_text("contact me at john@example.com please", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "email" for m in matches)

    def test_phone_pattern(self) -> None:
        matches = find_pii_in_text("call 555-123-4567 now", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "phone" for m in matches)

    def test_ssn_pattern(self) -> None:
        matches = find_pii_in_text("SSN: 123-45-6789", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "ssn" for m in matches)

    def test_credit_card_pattern(self) -> None:
        matches = find_pii_in_text("card 4111-1111-1111-1111", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "credit_card" for m in matches)

    def test_ip_address_pattern(self) -> None:
        matches = find_pii_in_text("server at 192.168.1.100", BUILTIN_PATTERNS)
        assert any(m.pattern_name == "ip_address" for m in matches)

    def test_no_pii_in_clean_text(self) -> None:
        matches = find_pii_in_text("this is a normal sentence", BUILTIN_PATTERNS)
        assert matches == []


class TestBuildPatternSet:
    def test_builtin_only(self) -> None:
        patterns = build_pattern_set(pii_types=["email", "phone"], custom_patterns=[])
        assert len(patterns) == 2
        names = {p.name for p in patterns}
        assert names == {"email", "phone"}

    def test_custom_overrides_builtin(self) -> None:
        custom = [PiiPattern(name="email", regex=r"[a-z]+@acme\.com", description="Acme only")]
        patterns = build_pattern_set(pii_types=["email"], custom_patterns=custom)
        assert len(patterns) == 1
        assert patterns[0].regex == r"[a-z]+@acme\.com"

    def test_custom_adds_new_pattern(self) -> None:
        custom = [PiiPattern(name="employee_id", regex=r"EMP-\d{6}", description="Employee ID")]
        patterns = build_pattern_set(pii_types=["email", "employee_id"], custom_patterns=custom)
        names = {p.name for p in patterns}
        assert "email" in names
        assert "employee_id" in names

    def test_disabled_pattern_excluded(self) -> None:
        custom = [PiiPattern(name="ssn", enabled=False)]
        patterns = build_pattern_set(pii_types=["email", "ssn"], custom_patterns=custom)
        names = {p.name for p in patterns}
        assert "ssn" not in names
        assert "email" in names


class TestRedactRegions:
    def test_redact_draws_black_box(self) -> None:
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        region = Region(x=10, y=10, w=30, h=20)
        redacted = redact_regions(frame, [region])
        assert np.all(redacted[10:30, 10:40] == 0)
        assert np.all(redacted[0:5, 0:5] == 200)

    def test_redact_empty_regions_unchanged(self) -> None:
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        redacted = redact_regions(frame, [])
        np.testing.assert_array_equal(redacted, frame)

    def test_redact_does_not_mutate_original(self) -> None:
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        original = frame.copy()
        redact_regions(frame, [Region(x=10, y=10, w=30, h=20)])
        np.testing.assert_array_equal(frame, original)


class TestDetectAndRedact:
    @patch("peeklet.core.redactor.extract_text_regions")
    def test_detects_and_blacks_out_pii(self, mock_ocr: MagicMock) -> None:
        """Full flow: OCR finds email -> regex matches -> region blacked out."""
        mock_ocr.return_value = [
            OcrResult(text="john@example.com", region=Region(x=10, y=10, w=80, h=15), confidence=0.95),
            OcrResult(text="hello world", region=Region(x=10, y=40, w=60, h=15), confidence=0.9),
        ]

        frame = np.full((100, 200, 3), 200, dtype=np.uint8)
        patterns = build_pattern_set(pii_types=["email"], custom_patterns=[])

        redacted, pii_found = detect_and_redact(frame, patterns)

        assert pii_found is True
        # Email region should be blacked out
        assert np.all(redacted[10:25, 10:90] == 0)
        # Non-PII region should be untouched
        assert np.all(redacted[40:55, 10:70] == 200)

    @patch("peeklet.core.redactor.extract_text_regions")
    def test_no_pii_returns_original(self, mock_ocr: MagicMock) -> None:
        mock_ocr.return_value = [
            OcrResult(text="hello world", region=Region(x=10, y=10, w=60, h=15), confidence=0.9),
        ]

        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        patterns = build_pattern_set(pii_types=["email"], custom_patterns=[])

        redacted, pii_found = detect_and_redact(frame, patterns)

        assert pii_found is False
        np.testing.assert_array_equal(redacted, frame)

    @patch("peeklet.core.redactor.extract_text_regions")
    def test_no_ocr_results_returns_original(self, mock_ocr: MagicMock) -> None:
        mock_ocr.return_value = []

        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        patterns = build_pattern_set(pii_types=["email"], custom_patterns=[])

        redacted, pii_found = detect_and_redact(frame, patterns)

        assert pii_found is False
        np.testing.assert_array_equal(redacted, frame)

    @patch("peeklet.core.redactor.extract_text_from_regions")
    def test_uses_changed_regions_when_provided(self, mock_ocr_regions: MagicMock) -> None:
        """When changed_regions are passed, OCR only those areas."""
        mock_ocr_regions.return_value = [
            OcrResult(text="123-45-6789", region=Region(x=50, y=80, w=70, h=12), confidence=0.9),
        ]

        frame = np.full((200, 200, 3), 200, dtype=np.uint8)
        patterns = build_pattern_set(pii_types=["ssn"], custom_patterns=[])
        changed = [Region(x=50, y=80, w=100, h=40)]

        redacted, pii_found = detect_and_redact(frame, patterns, changed_regions=changed)

        assert pii_found is True
        mock_ocr_regions.assert_called_once_with(frame, changed)

    def test_empty_patterns_skips_ocr(self) -> None:
        """When no patterns are configured, skip OCR entirely."""
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)

        redacted, pii_found = detect_and_redact(frame, patterns=[])

        assert pii_found is False
        np.testing.assert_array_equal(redacted, frame)
