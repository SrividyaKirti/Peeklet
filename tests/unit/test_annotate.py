"""Orchestrator tests with stubbed OCR and LLM."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from peeklet import annotate
from peeklet.screen import WordBox
from peeklet.transcript import TranscriptError
from peeklet.types import ScreenJudgment
from tests.helpers.synth import write_demo

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def colour_ocr(frame: np.ndarray) -> list[WordBox]:
    """Identify synthetic screens by background tint in an empty area (right of the body)."""
    h, w = frame.shape[:2]
    tint = float(frame[int(h * 0.4), int(w * 0.85)].mean())
    if tint > 248:
        heading = "Revenue Dashboard"
    elif tint > 225:
        heading = "Billing Settings"
    else:
        return []  # gallery view
    boxes = [WordBox(heading, 95, 220, 90, 400, 48), WordBox("Home", 95, 20, 160, 60, 22)]
    boxes += [WordBox(f"row{i}", 95, 220, 180 + 34 * i, 300, 22) for i in range(12)]
    return boxes


class StubLLM:
    provider, model = "stub", "stub"

    def judge_screen(
        self, image_jpeg: bytes, ocr_text: str, lines: list[str], action_items: list[str]
    ) -> ScreenJudgment:
        return ScreenJudgment(True, "needed", f"Screen showing {ocr_text.split()[0]}")


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("demo")
    write_demo(d / "demo.mp4", d / "demo.vtt")
    return d / "demo.mp4", d / "demo.vtt"


@pytest.fixture(autouse=True)
def no_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys.modules["peeklet.annotate"],
        "detect_speech_segments",
        lambda *a, **k: [(0.5, 4.0), (11.0, 15.0), (27.0, 31.0)],
    )


def test_no_llm_end_to_end(demo: tuple[Path, Path], tmp_path: Path) -> None:
    entries = annotate(*demo, tmp_path, use_llm=False, ocr=colour_ocr)
    data = json.loads((tmp_path / "transcript.json").read_text())
    images = [e["image"] for e in data]
    assert images[0] is not None and images[0] == images[3]
    assert images[1] is not None and images[1] != images[0]
    assert images[2] is None
    assert all("visual_context" not in e for e in data)
    assert entries.stats.screens_found == 2
    for name in {i for i in images if i}:
        assert (tmp_path / name).is_file()


def test_llm_descriptions_and_cache(
    demo: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from peeklet.config import PeekletConfig

    cfg = PeekletConfig(llm_cache_dir=str(tmp_path / "cache"))
    annotate(*demo, tmp_path / "out", config=cfg, llm_client=StubLLM(), ocr=colour_ocr)
    data = json.loads((tmp_path / "out" / "transcript.json").read_text())
    assert data[0]["visual_context"].startswith("Screen showing")


def test_missing_key_falls_back_to_no_llm(
    demo: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    entries = annotate(*demo, tmp_path, ocr=colour_ocr)
    assert any("credentials" in w for w in entries.stats.warnings)


def test_max_images_zero_is_text_only(demo: tuple[Path, Path], tmp_path: Path) -> None:
    annotate(*demo, tmp_path, use_llm=False, max_images=0, ocr=colour_ocr)
    data = json.loads((tmp_path / "transcript.json").read_text())
    assert all(e["image"] is None for e in data)


def test_debug_report(demo: tuple[Path, Path], tmp_path: Path) -> None:
    annotate(*demo, tmp_path, use_llm=False, debug=True, ocr=colour_ocr)
    dbg = json.loads((tmp_path / "debug.json").read_text())
    assert {"screens", "lines", "checkpoints", "warnings", "timing"} <= dbg.keys()
    assert all("occurrences" in s and "signals" in s for s in dbg["screens"])


def test_bad_transcript_raises(demo: tuple[Path, Path], tmp_path: Path) -> None:
    bad = tmp_path / "empty.vtt"
    bad.write_text("WEBVTT\n")
    with pytest.raises(TranscriptError):
        annotate(demo[0], bad, tmp_path / "o", use_llm=False, ocr=colour_ocr)
