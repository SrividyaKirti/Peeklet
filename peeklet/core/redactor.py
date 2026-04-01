"""PII detection and redaction for keyframe images."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from peeklet.config import PiiPattern
from peeklet.utils.types import Region

BUILTIN_PATTERNS: list[PiiPattern] = [
    PiiPattern(name="email", regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", description="Email addresses"),
    PiiPattern(name="phone", regex=r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", description="US phone numbers"),
    PiiPattern(name="ssn", regex=r"\b\d{3}-\d{2}-\d{4}\b", description="US Social Security Numbers"),
    PiiPattern(name="credit_card", regex=r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", description="Credit card numbers"),
    PiiPattern(name="ip_address", regex=r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", description="IPv4 addresses"),
    PiiPattern(name="address", regex=r"\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b", description="US street addresses"),
]


@dataclass(frozen=True, slots=True)
class PiiMatch:
    pattern_name: str
    matched_text: str
    start: int
    end: int


def build_pattern_set(pii_types: list[str], custom_patterns: list[PiiPattern]) -> list[PiiPattern]:
    pattern_map: dict[str, PiiPattern] = {p.name: p for p in BUILTIN_PATTERNS}
    for custom in custom_patterns:
        if not custom.enabled:
            pattern_map.pop(custom.name, None)
            continue
        pattern_map[custom.name] = custom
    return [p for name, p in pattern_map.items() if name in pii_types]


def find_pii_in_text(text: str, patterns: list[PiiPattern]) -> list[PiiMatch]:
    matches: list[PiiMatch] = []
    for pattern in patterns:
        if not pattern.regex:
            continue
        for match in re.finditer(pattern.regex, text):
            matches.append(PiiMatch(pattern_name=pattern.name, matched_text=match.group(), start=match.start(), end=match.end()))
    return matches


def redact_regions(frame: np.ndarray, regions: list[Region]) -> np.ndarray:
    redacted = frame.copy()
    h, w = redacted.shape[:2]
    for region in regions:
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(w, region.x + region.w)
        y2 = min(h, region.y + region.h)
        redacted[y1:y2, x1:x2] = 0
    return redacted
