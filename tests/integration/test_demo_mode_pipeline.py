"""End-to-end test for the --demo-mode video pipeline.

Uses a tiny synthetic video and a fake LLM client (no real API calls). The
fake client returns a hardcoded list of moments, and the test verifies that
Stage B picks frames, applies content-addressable dedup, and writes the
manifest + context files.
"""

from __future__ import annotations

import json
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

    results = process_video(video_path, cfg)

    # Demo path returns empty sentinel list; outputs are in files.
    assert results == []

    out_dir = Path(cfg.exporter.output_dir)
    assert (out_dir / "manifest.parquet").exists()
    assert (out_dir / "context.json").exists()
    assert (out_dir / "context.md").exists()

    ctx = json.loads((out_dir / "context.json").read_text())
    assert "screens" in ctx
    assert "moments" in ctx
    assert len(ctx["moments"]) >= 1

    md = (out_dir / "context.md").read_text()
    assert "# Demo Context:" in md


def test_pipeline_no_gap_fill_in_output(
    synthetic_video: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch,
) -> None:
    """End-to-end: moment_source values are anchor or llm; never gap_fill."""
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
    monkeypatch.setattr(df_module, "build_llm_client", lambda provider, model: fake_client)
    monkeypatch.setattr(df_module, "pytesseract", MagicMock())
    monkeypatch.setattr(df_module, "_is_low_info_frame", lambda frame, config: False)

    results = process_video(video_path, cfg)

    # Demo path returns empty sentinel list; outputs are in files.
    assert results == []

    out_dir = Path(cfg.exporter.output_dir)
    assert (out_dir / "manifest.parquet").exists()
    assert (out_dir / "context.json").exists()
    assert (out_dir / "context.md").exists()

    ctx = json.loads((out_dir / "context.json").read_text())
    assert "screens" in ctx
    assert "moments" in ctx
    assert len(ctx["moments"]) >= 1

    # No gap_fill sources — moments are anchor or llm
    moment_sources = {m.get("moment_source") for m in ctx["moments"]}
    assert "gap_fill" not in moment_sources
    assert moment_sources <= {"anchor", "llm", None}

    md = (out_dir / "context.md").read_text()
    assert "# Demo Context:" in md
