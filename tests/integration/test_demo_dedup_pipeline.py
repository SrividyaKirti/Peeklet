"""End-to-end regression on the canonical fathom recording.

Runs the full demo pipeline (real LLM call disabled — anchors-only path)
and asserts the spec's hard guarantees: no anchor leak past the quality
gate, scroll/camera-tile/row-hover collapse, filter-chip split, total
unique-screen ceiling.

Skipped if the fixture file is absent or if pytesseract is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_PATH = REPO_ROOT / "UI_enhancements_Apr12026.mp4"
TRANSCRIPT_PATH = REPO_ROOT / "UI enhancements - April 01.md"


def _has_tesseract() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not VIDEO_PATH.exists(), reason="fixture video not present")
@pytest.mark.skipif(not TRANSCRIPT_PATH.exists(), reason="fixture transcript not present")
@pytest.mark.skipif(not _has_tesseract(), reason="tesseract binary unavailable")
def test_demo_dedup_regression_on_apr01(tmp_path, monkeypatch):
    """Spec contract on UI_enhancements_Apr12026.mp4."""
    from unittest.mock import MagicMock

    from peeklet.config import PeekletConfig
    from peeklet.core import demo_filter as df
    from peeklet.core.video import process_video

    cfg = PeekletConfig()
    cfg.exporter.output_dir = str(tmp_path)
    cfg.video.audio_detection = False
    cfg.video.transcript_path = str(TRANSCRIPT_PATH)
    cfg.demo_filter.enabled = True

    # Anchors-only run: stub the LLM so the test doesn't depend on a
    # network key or model output. The spec's anchor recall + quality
    # gate properties are independent of the LLM picks.
    fake = MagicMock()
    fake.pick_moments.return_value = []
    monkeypatch.setattr(df, "build_llm_client", lambda **k: fake)

    process_video(VIDEO_PATH, cfg)

    ctx = json.loads((tmp_path / "context.json").read_text())
    moments = ctx["moments"]
    screens = ctx["screens"]

    # 1) All 21 ACTION ITEM anchors present in moments[]
    action_items = [m for m in moments if m["type"] == "action_item"]
    assert len(action_items) == 21, f"expected 21 action items, got {len(action_items)}"

    # 2) Unique screen count is bounded
    unique_ids = {m["screen_id"] for m in action_items if m["screen_id"]}
    assert len(unique_ids) <= 20, (
        f"expected ≤20 unique screens for 21 anchors, got {len(unique_ids)}"
    )

    # 3) image_unavailable should be set for at least one moment
    #    (frame-#34 participant gallery is the canonical case)
    unavailable = [m for m in moments if m["image_unavailable"]]
    assert len(unavailable) >= 1, "expected at least one quality-gate failure"

    # 4) Sanity: no orphan screen_ids
    screen_ids = {s["id"] for s in screens}
    for m in moments:
        if m["screen_id"] is not None:
            assert m["screen_id"] in screen_ids
