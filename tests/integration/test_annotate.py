"""End-to-end with real OCR (skipped when tesseract is not installed)."""

from __future__ import annotations

import json
import shutil
import sys
from typing import TYPE_CHECKING

import pytest

from peeklet import annotate
from tests.helpers.synth import write_demo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract")


def test_real_ocr_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys.modules["peeklet.annotate"],
        "detect_speech_segments",
        lambda *a, **k: [(0.5, 4.0), (11.0, 15.0), (27.0, 31.0)],
    )
    write_demo(tmp_path / "demo.mp4", tmp_path / "demo.vtt")
    entries = annotate(
        tmp_path / "demo.mp4", tmp_path / "demo.vtt", tmp_path / "out", use_llm=False, debug=True
    )
    data = json.loads((tmp_path / "out" / "transcript.json").read_text())
    images = [e["image"] for e in data]
    assert entries.stats.screens_found == 2, json.loads((tmp_path / "out/debug.json").read_text())
    assert images[0] == images[3] is not None
    assert images[1] not in (None, images[0])
    assert images[2] is None
