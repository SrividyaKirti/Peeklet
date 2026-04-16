"""End-to-end test for the --demo-mode video pipeline.

Uses a tiny synthetic video and a fake LLM client (no real API calls). The
fake client returns a hardcoded list of moments, and the test verifies that
Stage B picks frames, applies the gallery check, and writes the manifest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.unit.helpers_video import write_synthetic_video


@pytest.fixture
def synthetic_video(tmp_path: Path) -> tuple[Path, Path]:
    video_path = tmp_path / "demo.mp4"
    write_synthetic_video(video_path, duration_sec=6, fps=10, width=64, height=64)

    transcript_path = tmp_path / "demo.srt"
    transcript_path.write_text(
        "1\n00:00:00,500 --> 00:00:02,500\nhere is the first thing\n\n"
        "2\n00:00:03,500 --> 00:00:05,500\nnow look at the second thing\n",
        encoding="utf-8",
    )
    return video_path, transcript_path


def test_demo_mode_end_to_end_with_fake_llm(
    synthetic_video: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch,
) -> None:
    from peeklet.config import PeekletConfig
    from peeklet.core import demo_filter as df_module
    from peeklet.core.video import process_video
    from peeklet.utils.types import Moment

    video_path, transcript_path = synthetic_video

    cfg = PeekletConfig()
    cfg.exporter.output_dir = str(tmp_path / "out")
    cfg.video.audio_detection = False
    cfg.video.transcript_path = str(transcript_path)
    cfg.demo_filter.enabled = True

    fake_client = MagicMock()
    fake_client.pick_moments.return_value = [
        Moment(
            timestamp=1.5,
            visual_context_goal="first thing",
            textual_anchor="speaker says here is the first thing",
            downstream_utility="verify first screen",
        ),
        Moment(
            timestamp=4.5,
            visual_context_goal="second thing",
            textual_anchor="speaker says now look at the second thing",
            downstream_utility="verify second screen",
        ),
    ]
    monkeypatch.setattr(
        df_module,
        "build_llm_client",
        lambda provider, model: fake_client,
    )
    # Ensure pytesseract is seen as available (demo mode requires it).
    monkeypatch.setattr(df_module, "pytesseract", MagicMock())
    # Synthetic 64x64 frames would fail the layout rejector; bypass it for
    # the integration test so we're exercising the real pipeline wiring.
    monkeypatch.setattr(df_module, "_is_low_info_frame", lambda frame, config: False)
    # Two synthetic frames produce the same dHash, which would trip pHash
    # dedup. Hand out distinct hashes per call so both moments survive.
    _counter = {"n": 0}

    def _unique_hash(_frame):
        _counter["n"] += 1
        return _counter["n"] * 0x0123456789ABCDEF & 0xFFFFFFFFFFFFFFFF

    monkeypatch.setattr(df_module, "dhash_64", _unique_hash)

    results = process_video(video_path, cfg)

    assert len(results) == 2
    assert all(r.is_keyframe for r in results)
    goals = [r.visual_context_goal for r in results]
    assert "first thing" in goals
    assert "second thing" in goals

    # Outputs should be written
    out_dir = Path(cfg.exporter.output_dir)
    assert (out_dir / "manifest.parquet").exists()
    assert (out_dir / "context.json").exists()
    assert (out_dir / "context.md").exists()

    # Verify new context.md format
    md = (out_dir / "context.md").read_text()
    assert "Meeting Context:" in md
    assert "Visual Table of Contents" in md
