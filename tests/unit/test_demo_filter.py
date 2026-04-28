"""Unit tests for demo_filter.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _pass_layout_rejector_by_default(request, monkeypatch):
    """Default: let the layout rejector pass every frame.

    Integration tests feed synthetic flat-colored frames through
    ``select_frames_for_moments``; the real layout rejector would drop
    all of them because they have zero edge density and zero OCR text.
    Tests that specifically exercise the rejector live in
    ``TestIsLowInfoFrame`` (patches the signals directly) or are marked
    ``rejector_live``.
    """
    if "TestIsLowInfoFrame" in request.node.nodeid:
        return
    if "test_low_info_rejector_accepts_real_text_frame" in request.node.nodeid:
        return
    if "test_select_frames_for_moments_drops_gallery_frames" in request.node.nodeid:
        return
    if "TestPhashDedup" in request.node.nodeid:
        return
    monkeypatch.setattr(
        "peeklet.core.demo_filter._is_low_info_frame",
        lambda frame, config: False,
    )
    # Flat-colored synthetic frames all dHash to 0 under the real algorithm,
    # which would trip pHash dedup for every pair. Hand out a random 64-bit
    # hash per call so integration tests exercise the SSIM dedup path cleanly
    # (two random 64-bit ints are ~32 bits apart, well above the dedup
    # threshold of 5).
    import os

    def _unique_hash(_frame):
        return int.from_bytes(os.urandom(8), "big")

    monkeypatch.setattr("peeklet.core.demo_filter.dhash_64", _unique_hash)


def _make_frame(h: int = 360, w: int = 360) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _tesseract_available() -> bool:
    """True iff the tesseract binary is callable. Used to gate real-OCR tests."""
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _render_text_frame(words: list[str], width: int = 1280, height: int = 720) -> np.ndarray:
    """Render a high-contrast frame with the given words drawn in large text.

    Used to feed real OCR a deterministic image with a known word count
    without needing a video fixture in the repo.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except OSError:
        font = ImageFont.load_default()
    y = 40
    for word in words:
        draw.text((40, y), word, fill="black", font=font)
        y += 60
    return np.asarray(img)


def test_count_words_filters_low_confidence_and_short_tokens():
    from peeklet.core.demo_filter import _count_words_in_frame

    fake_data = {
        "text": ["Settings", "x", "Save", "", "Login", "."],
        "conf": ["95", "92", "80", "-1", "20", "99"],
    }
    with patch("peeklet.core.demo_filter.pytesseract") as mock_pt:
        mock_pt.image_to_data.return_value = fake_data
        mock_pt.Output.DICT = "dict"
        count = _count_words_in_frame(_make_frame(), downscale_dim=360)

    # "Settings" (95, len 8) OK
    # "x" (92, len 1) short
    # "Save" (80, len 4) OK
    # ""  (-1, empty) empty
    # "Login" (20, len 5) low conf
    # "." (99, len 1) short
    assert count == 2


def test_count_words_handles_tesseract_exception():
    from peeklet.core.demo_filter import _count_words_in_frame

    with patch("peeklet.core.demo_filter.pytesseract") as mock_pt:
        mock_pt.image_to_data.side_effect = RuntimeError("tesseract crashed")
        mock_pt.Output.DICT = "dict"
        count = _count_words_in_frame(_make_frame(), downscale_dim=360)

    assert count == 0  # graceful fallback


def test_count_words_returns_zero_when_pytesseract_is_none():
    from peeklet.core.demo_filter import _count_words_in_frame

    with patch("peeklet.core.demo_filter.pytesseract", None):
        count = _count_words_in_frame(_make_frame(), downscale_dim=360)

    assert count == 0


def test_downscale_for_ocr_already_smaller_returns_original():
    from peeklet.core.demo_filter import _downscale_for_ocr

    frame = _make_frame(h=200, w=200)
    out = _downscale_for_ocr(frame, downscale_dim=360)
    # Already smaller than the target, so it should be returned untouched.
    assert out is frame


def test_downscale_for_ocr_square_frame_resizes_both_dims():
    from peeklet.core.demo_filter import _downscale_for_ocr

    frame = _make_frame(h=720, w=720)
    out = _downscale_for_ocr(frame, downscale_dim=360)
    assert out.shape == (360, 360, 3)


def test_downscale_for_ocr_landscape_frame_preserves_aspect():
    from peeklet.core.demo_filter import _downscale_for_ocr

    frame = _make_frame(h=540, w=960)  # 16:9 landscape
    out = _downscale_for_ocr(frame, downscale_dim=480)
    # Longest edge should match downscale_dim, height should scale proportionally
    assert out.shape[1] == 480
    assert out.shape[0] == int(round(540 * 480 / 960))


def test_downscale_for_ocr_portrait_frame_preserves_aspect():
    from peeklet.core.demo_filter import _downscale_for_ocr

    frame = _make_frame(h=960, w=540)  # 9:16 portrait
    out = _downscale_for_ocr(frame, downscale_dim=480)
    assert out.shape[0] == 480
    assert out.shape[1] == int(round(540 * 480 / 960))


def test_build_search_window_caps_at_segment_end():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import _build_search_window

    seg = TranscriptSegment(start=10.0, end=12.0, text="x")
    start, end = _build_search_window(
        moment_ts=10.5,
        segment=seg,
        max_window_sec=5.0,
    )
    assert start == 10.5
    # Capped by segment end (12.0), not by max_window
    assert end == 12.0


def test_build_search_window_biases_backward_with_lookback():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import _build_search_window

    seg = TranscriptSegment(start=0.0, end=100.0, text="x")
    start, end = _build_search_window(
        moment_ts=50.0,
        segment=seg,
        max_window_sec=10.0,
        lookback_sec=3.0,
    )
    # Window shifts back by lookback but width stays max_window_sec.
    assert start == pytest.approx(47.0)
    assert end == pytest.approx(57.0)


def test_build_search_window_lookback_clamped_to_segment_start():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import _build_search_window

    seg = TranscriptSegment(start=10.0, end=30.0, text="x")
    start, end = _build_search_window(
        moment_ts=11.0,
        segment=seg,
        max_window_sec=5.0,
        lookback_sec=3.0,
    )
    # Lookback would push start to 8.0 but segment begins at 10.0.
    assert start == pytest.approx(10.0)
    # End = min(30, 11 + 5 - 3) = 13.0
    assert end == pytest.approx(13.0)


def test_build_search_window_caps_at_max_window_when_segment_long():
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import _build_search_window

    seg = TranscriptSegment(start=0.0, end=100.0, text="long monologue")
    start, end = _build_search_window(
        moment_ts=10.0,
        segment=seg,
        max_window_sec=5.0,
    )
    assert start == 10.0
    assert end == 15.0  # 10.0 + 5.0


def test_is_stable_passes_when_both_neighbors_similar():
    from peeklet.core.demo_filter import _is_stable

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    prev = np.full((100, 100, 3), 128, dtype=np.uint8)
    nxt = np.full((100, 100, 3), 128, dtype=np.uint8)

    assert _is_stable(frame, prev, nxt, threshold=0.92) is True


def test_is_stable_fails_when_one_neighbor_differs():
    from peeklet.core.demo_filter import _is_stable

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    prev = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Random noise — very different from frame, low SSIM
    rng = np.random.default_rng(42)
    nxt = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

    assert _is_stable(frame, prev, nxt, threshold=0.92) is False


# Word-count-gate gallery tests removed: behavior replaced by the triple-AND
# layout rejector (see TestIsLowInfoFrame).


def _make_decoder_for_moments(meta_duration: float = 60.0):
    """Create a decoder mock that returns a deterministic frame per timestamp."""
    decoder = MagicMock()
    decoder.get_metadata.return_value = MagicMock(duration=meta_duration, filename="t.mp4")

    def _fake_extract(ts: float):
        val = int((ts * 10) % 256)
        return np.full((100, 100, 3), val, dtype=np.uint8), float(ts), int(ts * 30)

    decoder.extract_frame_at.side_effect = _fake_extract
    return decoder


def test_select_frames_for_moments_picks_first_stable_frame(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )

    moments = [
        Moment(
            timestamp=10.0, visual_context_goal="cap", textual_anchor="t", downstream_utility="u"
        )
    ]
    transcript = [TranscriptSegment(start=8.0, end=12.0, text="speaking")]
    cfg = DemoFilterConfig(
        enabled=True,
        ssim_stability_threshold=0.92,
        forward_search_step_sec=0.5,
        forward_search_window_max_sec=5.0,
        gallery_min_words=0,
    )

    monkeypatch.setattr(
        "peeklet.core.demo_filter.save_keyframe",
        lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
    )

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert len(results) == 1
    r = results[0]
    assert r.is_keyframe is True
    assert r.visual_context_goal == "cap"
    # Search window is biased backward by search_window_lookback_sec
    # (default 3s) and clamped to the containing transcript segment, so
    # any frame inside [segment.start, segment.end] is acceptable.
    assert r.video_timestamp is not None
    assert 8.0 <= r.video_timestamp <= 12.0
    assert r.trigger_type == "transcript_trigger"


def test_select_frames_for_moments_drops_gallery_frames(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.zeros((100, 100, 3), dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )

    moments = [
        Moment(
            timestamp=10.0, visual_context_goal="cap", textual_anchor="t", downstream_utility="u"
        )
    ]
    transcript = [TranscriptSegment(start=8.0, end=12.0, text="speaking")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=5)

    monkeypatch.setattr(
        "peeklet.core.demo_filter._is_low_info_frame",
        lambda frame, config: True,
    )
    monkeypatch.setattr(
        "peeklet.core.demo_filter.save_keyframe",
        lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
    )

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert results == []


def test_select_frames_for_moments_drops_moment_with_no_segment(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments()
    monkeypatch.setattr(
        "peeklet.core.demo_filter.save_keyframe",
        lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
    )

    moments = [
        Moment(timestamp=50.0, visual_context_goal="c", textual_anchor="t", downstream_utility="u")
    ]
    transcript = [TranscriptSegment(start=0.0, end=10.0, text="x")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert results == []


def test_apply_demo_filter_calls_llm_then_select(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import apply_demo_filter
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    transcript = [TranscriptSegment(start=8.0, end=12.0, text="speaking")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0)

    fake_moments = [
        Moment(timestamp=10.0, visual_context_goal="c", textual_anchor="t", downstream_utility="u")
    ]
    fake_client = MagicMock()
    fake_client.pick_moments.return_value = fake_moments

    monkeypatch.setattr(
        "peeklet.core.demo_filter.build_llm_client",
        lambda provider, model: fake_client,
    )
    monkeypatch.setattr(
        "peeklet.core.demo_filter.save_keyframe",
        lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
    )
    monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", MagicMock())
    monkeypatch.setattr(
        "peeklet.core.demo_filter._count_words_in_frame", lambda frame, downscale_dim: 0
    )
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 200, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )

    results = apply_demo_filter(
        decoder=decoder,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    fake_client.pick_moments.assert_called_once_with(transcript, 60.0)
    assert len(results) == 1
    assert results[0].visual_context_goal == "c"


def test_apply_demo_filter_logs_warning_on_zero_moments(tmp_path, monkeypatch, caplog):
    import logging

    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    fake_client = MagicMock()
    fake_client.pick_moments.return_value = []

    monkeypatch.setattr(
        "peeklet.core.demo_filter.build_llm_client",
        lambda provider, model: fake_client,
    )
    monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", MagicMock())

    cfg = DemoFilterConfig(enabled=True)
    with caplog.at_level(logging.WARNING):
        results = apply_demo_filter(
            decoder=decoder,
            transcript=[],
            config=cfg,
            output_dir=tmp_path,
        )

    assert results == []
    assert any("screenshot-worthy moments" in rec.message for rec in caplog.records)


# --- Regression tests for the demo-mode gallery-check bugs surfaced
# while testing against the UI_enhancements_Apr12026.mp4 sample. ---


def test_demo_filter_config_has_gallery_ocr_min_dim_default_at_least_1280():
    """Bug 3 regression: OCR needs near-source resolution to read screen-share text.

    Reusing ``frame_search_resolution`` (240/360/540 across quality presets) for
    OCR downscaling destroys text before Tesseract sees it on any 720p source.
    The dedicated gallery_ocr_min_dim must default to at least 1280 so a 720p
    frame is left untouched.
    """
    from peeklet.config import DemoFilterConfig

    cfg = DemoFilterConfig()
    assert hasattr(cfg, "gallery_ocr_min_dim"), (
        "DemoFilterConfig should expose gallery_ocr_min_dim independent of "
        "frame_search_resolution so OCR can run at near-source resolution."
    )
    assert cfg.gallery_ocr_min_dim >= 1280


@pytest.mark.skipif(not _tesseract_available(), reason="tesseract binary not installed")
def test_low_info_rejector_accepts_real_text_frame_at_720p():
    """Regression: a 720p frame with dashboard-density text must NOT be
    rejected as low-info at the default config thresholds. Guards against
    the OCR downscale regression that previously dropped every demo frame
    from a 720p screen-share recording.
    """
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import _is_low_info_frame

    # Dashboard-like content: many words across rows and columns so that
    # both text-line count and grid-cell dispersion clear thresholds, plus
    # enough text edges that edge density also clears.
    dashboard_rows = [
        "Settings Dashboard Analytics Users Reports Logout",
        "Home Billing Notifications Search Support Help",
        "Active Inactive Pending Archived Draft Published",
        "Create Edit Delete Import Export Refresh",
        "Name Email Role Status Updated Created",
        "Policy Tool Insights Logs Models Cost",
        "January February March April May June",
        "Monday Tuesday Wednesday Thursday Friday Saturday",
        "Alpha Beta Gamma Delta Epsilon Zeta",
        "North South East West Center Outer",
        "Red Green Blue Yellow Purple Orange",
        "Low Medium High Critical Severe Blocker",
    ]
    frame = _render_text_frame(dashboard_rows, width=1280, height=720)
    cfg = DemoFilterConfig()
    assert _is_low_info_frame(frame, cfg) is False


def test_apply_demo_filter_raises_when_pytesseract_unavailable(tmp_path, monkeypatch):
    """Missing OCR backend must raise, not silently degrade."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", None)

    cfg = DemoFilterConfig(enabled=True)
    with pytest.raises(RuntimeError, match="pytesseract"):
        apply_demo_filter(
            decoder=decoder,
            transcript=[],
            config=cfg,
            output_dir=tmp_path,
        )


# --- Dedup + tail-skip (PR A from 2026-04-10 demo-mode quality plan) ---


def _patch_save_keyframe(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "peeklet.core.demo_filter.save_keyframe",
        lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
    )


def test_select_frames_for_moments_dedups_near_duplicate_second_moment(tmp_path, monkeypatch):
    """Two moments whose candidate frames are identical → second is dropped."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=600.0)
    # Every extracted frame is identical → SSIM == 1.0 → dedup must trigger.
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(
            timestamp=10.0, visual_context_goal="first", textual_anchor="t", downstream_utility="u"
        ),
        Moment(
            timestamp=100.0,
            visual_context_goal="second",
            textual_anchor="t",
            downstream_utility="u",
        ),
    ]
    transcript = [
        TranscriptSegment(start=8.0, end=12.0, text="a"),
        TranscriptSegment(start=98.0, end=102.0, text="b"),
    ]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, dedup_ssim_threshold=0.95)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert len(results) == 1
    assert results[0].visual_context_goal == "first"


def test_select_frames_for_moments_keeps_distinct_second_moment(tmp_path, monkeypatch):
    """Two moments whose candidate frames are visually different → both kept."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=600.0)

    # Frame fill differs by timestamp bucket → low SSIM between the two moments.
    def _extract(ts: float):
        fill = 20 if ts < 50.0 else 230
        return np.full((100, 100, 3), fill, dtype=np.uint8), float(ts), int(ts * 30)

    decoder.extract_frame_at.side_effect = _extract
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(
            timestamp=10.0, visual_context_goal="first", textual_anchor="t", downstream_utility="u"
        ),
        Moment(
            timestamp=100.0,
            visual_context_goal="second",
            textual_anchor="t",
            downstream_utility="u",
        ),
    ]
    transcript = [
        TranscriptSegment(start=8.0, end=12.0, text="a"),
        TranscriptSegment(start=98.0, end=102.0, text="b"),
    ]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, dedup_ssim_threshold=0.95)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert len(results) == 2
    assert [r.visual_context_goal for r in results] == ["first", "second"]


def test_select_frames_for_moments_dedup_disabled_when_threshold_is_one(tmp_path, monkeypatch):
    """dedup_ssim_threshold=1.0 disables dedup — identical frames still both saved."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=600.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(timestamp=10.0, visual_context_goal="a", textual_anchor="t", downstream_utility="u"),
        Moment(
            timestamp=100.0, visual_context_goal="b", textual_anchor="t", downstream_utility="u"
        ),
    ]
    transcript = [
        TranscriptSegment(start=8.0, end=12.0, text="x"),
        TranscriptSegment(start=98.0, end=102.0, text="y"),
    ]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, dedup_ssim_threshold=1.0)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert len(results) == 2


def test_select_frames_for_moments_dedup_drops_all_when_threshold_is_zero(tmp_path, monkeypatch):
    """dedup_ssim_threshold=0.0 drops every moment after the first if frames overlap at all."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=600.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(timestamp=10.0, visual_context_goal="a", textual_anchor="t", downstream_utility="u"),
        Moment(
            timestamp=100.0, visual_context_goal="b", textual_anchor="t", downstream_utility="u"
        ),
        Moment(
            timestamp=200.0, visual_context_goal="c", textual_anchor="t", downstream_utility="u"
        ),
    ]
    transcript = [
        TranscriptSegment(start=8.0, end=12.0, text="x"),
        TranscriptSegment(start=98.0, end=102.0, text="y"),
        TranscriptSegment(start=198.0, end=202.0, text="z"),
    ]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, dedup_ssim_threshold=0.0)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert len(results) == 1
    assert results[0].visual_context_goal == "a"


def test_select_frames_for_moments_tail_skip_drops_moment_past_cutoff(tmp_path, monkeypatch):
    """Moment at 0.99 × duration is skipped when tail_skip_ratio=0.02."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=100.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    # 99.0 / 100.0 = 0.99 → past the 0.98 cutoff.
    moments = [
        Moment(
            timestamp=99.0, visual_context_goal="end", textual_anchor="t", downstream_utility="u"
        )
    ]
    transcript = [TranscriptSegment(start=95.0, end=100.0, text="x")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, tail_skip_ratio=0.02)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert results == []


def test_select_frames_for_moments_tail_skip_keeps_moment_before_cutoff(tmp_path, monkeypatch):
    """Moment at 0.95 × duration is kept when tail_skip_ratio=0.02."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=100.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(
            timestamp=95.0,
            visual_context_goal="near-end",
            textual_anchor="t",
            downstream_utility="u",
        )
    ]
    transcript = [TranscriptSegment(start=93.0, end=98.0, text="x")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, tail_skip_ratio=0.02)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert len(results) == 1
    assert results[0].visual_context_goal == "near-end"


def test_select_frames_for_moments_tail_skip_disabled_when_ratio_is_zero(tmp_path, monkeypatch):
    """tail_skip_ratio=0.0 disables the tail skip entirely — a 0.99-duration moment is kept."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=100.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(
            timestamp=99.0, visual_context_goal="end", textual_anchor="t", downstream_utility="u"
        )
    ]
    transcript = [TranscriptSegment(start=95.0, end=100.0, text="x")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, tail_skip_ratio=0.0)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )

    assert len(results) == 1


def test_anchor_moment_skips_dedup(tmp_path, monkeypatch):
    """Anchor moments bypass the dedup filter — two identical anchors both saved."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=600.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(
            timestamp=10.0,
            visual_context_goal="first",
            textual_anchor="t",
            downstream_utility="u",
            source="anchor",
        ),
        Moment(
            timestamp=100.0,
            visual_context_goal="second",
            textual_anchor="t",
            downstream_utility="u",
            source="anchor",
        ),
    ]
    transcript = [
        TranscriptSegment(start=8.0, end=12.0, text="a"),
        TranscriptSegment(start=98.0, end=102.0, text="b"),
    ]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, dedup_ssim_threshold=0.95)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )
    assert len(results) == 2


def test_anchor_moment_skips_tail_skip(tmp_path, monkeypatch):
    """Anchor moments near the end of video bypass tail_skip."""
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import select_frames_for_moments
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=100.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), 100, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    moments = [
        Moment(
            timestamp=99.0,
            visual_context_goal="end anchor",
            textual_anchor="t",
            downstream_utility="u",
            source="anchor",
        ),
    ]
    transcript = [TranscriptSegment(start=95.0, end=100.0, text="x")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0, tail_skip_ratio=0.02)

    results = select_frames_for_moments(
        decoder=decoder,
        moments=moments,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
    )
    assert len(results) == 1


class TestMergeMoments:
    def test_merge_moments_keeps_all_anchors_and_picks(self, caplog):
        """Pure two-tier: nothing is dropped; warnings surface near-anchor picks."""
        import logging

        from peeklet.core.demo_filter import merge_moments
        from peeklet.utils.types import Moment

        anchors = [
            Moment(
                timestamp=10.0,
                visual_context_goal="A",
                textual_anchor="",
                downstream_utility="",
                source="anchor",
            ),
            Moment(
                timestamp=50.0,
                visual_context_goal="B",
                textual_anchor="",
                downstream_utility="",
                source="anchor",
            ),
        ]
        llm_picks = [
            Moment(
                timestamp=12.0,
                visual_context_goal="near A",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
            Moment(
                timestamp=30.0,
                visual_context_goal="far",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
        ]
        with caplog.at_level(logging.WARNING):
            merged = merge_moments(anchors, llm_picks)

        assert [m.timestamp for m in merged] == [10.0, 12.0, 30.0, 50.0]
        # The 12.0 pick is within ±10s of the 10.0 anchor — must warn but not drop.
        assert any("12.0" in r.message and "10.0" in r.message for r in caplog.records)

    def test_merge_moments_sorts_by_timestamp(self):
        from peeklet.core.demo_filter import merge_moments
        from peeklet.utils.types import Moment

        anchors = [
            Moment(
                timestamp=50.0,
                visual_context_goal="",
                textual_anchor="",
                downstream_utility="",
                source="anchor",
            ),
        ]
        llm_picks = [
            Moment(
                timestamp=10.0,
                visual_context_goal="",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
            Moment(
                timestamp=30.0,
                visual_context_goal="",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
        ]
        merged = merge_moments(anchors, llm_picks)
        assert [m.timestamp for m in merged] == [10.0, 30.0, 50.0]

    def test_merge_moments_handles_empty_inputs(self):
        from peeklet.core.demo_filter import merge_moments

        assert merge_moments([], []) == []

    def test_merge_moments_logs_summary(self, caplog):
        """INFO-level summary logged on every call."""
        import logging

        from peeklet.core.demo_filter import merge_moments
        from peeklet.utils.types import Moment

        anchors = [
            Moment(
                timestamp=10.0,
                visual_context_goal="",
                textual_anchor="",
                downstream_utility="",
                source="anchor",
            ),
        ]
        llm_picks = [
            Moment(
                timestamp=30.0,
                visual_context_goal="",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
        ]
        with caplog.at_level(logging.INFO):
            merge_moments(anchors, llm_picks)

        assert any(
            "merged 2 moments" in r.message and "1 anchor" in r.message and "1 llm" in r.message
            for r in caplog.records
        )


def test_apply_demo_filter_merges_anchors_with_llm_picks(tmp_path, monkeypatch):
    from peeklet.config import DemoFilterConfig
    from peeklet.core.audio import TranscriptSegment
    from peeklet.core.demo_filter import apply_demo_filter
    from peeklet.utils.types import Moment

    decoder = _make_decoder_for_moments(meta_duration=600.0)
    decoder.extract_frame_at.side_effect = lambda ts: (
        np.full((100, 100, 3), int(ts) % 200, dtype=np.uint8),
        float(ts),
        int(ts * 30),
    )
    _patch_save_keyframe(monkeypatch, tmp_path)

    transcript = [
        TranscriptSegment(start=230.0, end=235.0, text="Fix assistant prompt"),
        TranscriptSegment(start=498.0, end=505.0, text="Other discussion"),
    ]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=0)

    fake_llm_moments = [
        Moment(
            timestamp=231.0, visual_context_goal="c", textual_anchor="t", downstream_utility="u"
        ),  # within 5s of anchor → kept (Stage B dedup gates handle near-duplicates)
        Moment(
            timestamp=500.0,
            visual_context_goal="far away",
            textual_anchor="t",
            downstream_utility="u",
        ),  # kept
    ]
    fake_client = MagicMock()
    fake_client.pick_moments.return_value = fake_llm_moments

    monkeypatch.setattr(
        "peeklet.core.demo_filter.build_llm_client",
        lambda provider, model: fake_client,
    )
    monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", MagicMock())
    monkeypatch.setattr(
        "peeklet.core.demo_filter._count_words_in_frame", lambda frame, downscale_dim: 0
    )

    raw_text = (
        "**ACTION ITEM: Fix assistant prompt - "
        "++[WATCH](https://fathom.video/calls/1?timestamp=232.0)++**\n"
    )

    results = apply_demo_filter(
        decoder=decoder,
        transcript=transcript,
        config=cfg,
        output_dir=tmp_path,
        transcript_text=raw_text,
    )

    # Should have anchor at 232 + llm pick at 231 + llm pick at 500 = 3 results
    assert len(results) == 3
    sources = [r.moment_source for r in results]
    assert "anchor" in sources
    assert "llm" in sources


def test_default_forward_search_window_is_10s():
    from peeklet.config import DemoFilterConfig

    cfg = DemoFilterConfig()
    assert cfg.forward_search_window_max_sec == 10.0


class TestPickBestContentIndex:
    def test_picks_frame_with_most_words(self):
        from peeklet.core.demo_filter import _pick_best_content_index

        samples = [
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.0, 300),
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.5, 315),
            (np.zeros((100, 100, 3), dtype=np.uint8), 11.0, 330),
        ]
        with patch("peeklet.core.demo_filter._count_words_in_frame", side_effect=[3, 15, 8]):
            idx = _pick_best_content_index(samples, downscale_dim=1920)
        assert idx == 1

    def test_tiebreak_picks_latest_frame(self):
        from peeklet.core.demo_filter import _pick_best_content_index

        samples = [
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.0, 300),
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.5, 315),
            (np.zeros((100, 100, 3), dtype=np.uint8), 11.0, 330),
        ]
        with patch("peeklet.core.demo_filter._count_words_in_frame", side_effect=[10, 10, 10]):
            idx = _pick_best_content_index(samples, downscale_dim=1920)
        assert idx == 2

    def test_all_zero_returns_index_zero(self):
        from peeklet.core.demo_filter import _pick_best_content_index

        samples = [
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.0, 300),
            (np.zeros((100, 100, 3), dtype=np.uint8), 10.5, 315),
        ]
        with patch("peeklet.core.demo_filter._count_words_in_frame", side_effect=[0, 0]):
            idx = _pick_best_content_index(samples, downscale_dim=1920)
        assert idx == 0

    def test_single_sample_returns_zero(self):
        from peeklet.core.demo_filter import _pick_best_content_index

        samples = [(np.zeros((100, 100, 3), dtype=np.uint8), 10.0, 300)]
        with patch("peeklet.core.demo_filter._count_words_in_frame", return_value=5):
            idx = _pick_best_content_index(samples, downscale_dim=1920)
        assert idx == 0


class TestOcrWordBoxes:
    def test_returns_empty_when_pytesseract_missing(self, monkeypatch):
        from peeklet.core import demo_filter

        monkeypatch.setattr(demo_filter, "pytesseract", None)
        result = demo_filter._ocr_word_boxes(
            np.zeros((10, 10, 3), dtype=np.uint8), downscale_dim=100
        )
        assert result == []

    def test_filters_low_confidence_and_short_words(self, monkeypatch):
        from peeklet.core import demo_filter

        fake_data = {
            "text": ["hello", "x", "world", "noise"],
            "conf": ["90", "80", "85", "10"],
            "left": [10, 50, 100, 200],
            "top": [5, 5, 5, 5],
            "width": [40, 5, 45, 30],
            "height": [12, 12, 12, 12],
        }

        class FakeTess:
            class Output:
                DICT = "dict"

            @staticmethod
            def image_to_data(img, output_type):
                return fake_data

        monkeypatch.setattr(demo_filter, "pytesseract", FakeTess)
        boxes = demo_filter._ocr_word_boxes(
            np.zeros((100, 300, 3), dtype=np.uint8), downscale_dim=1000
        )
        texts = [b.text for b in boxes]
        assert texts == ["hello", "world"]
        assert boxes[0].x == 10 and boxes[0].y == 5
        assert boxes[0].w == 40 and boxes[0].h == 12

    def test_count_words_still_reflects_box_count(self, monkeypatch):
        from peeklet.core import demo_filter

        fake_data = {
            "text": ["aa", "bb", "cc"],
            "conf": ["90", "90", "90"],
            "left": [0, 10, 20],
            "top": [0, 0, 0],
            "width": [5, 5, 5],
            "height": [10, 10, 10],
        }

        class FakeTess:
            class Output:
                DICT = "dict"

            @staticmethod
            def image_to_data(img, output_type):
                return fake_data

        monkeypatch.setattr(demo_filter, "pytesseract", FakeTess)
        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        assert demo_filter._count_words_in_frame(frame, downscale_dim=1000) == 3


class TestCountTextLines:
    def _box(self, y, h=10):
        from peeklet.core.demo_filter import WordBox

        return WordBox(text="x", conf=90.0, x=0, y=y, w=20, h=h)

    def test_empty_returns_zero(self):
        from peeklet.core.demo_filter import _count_text_lines

        assert _count_text_lines([]) == 0

    def test_single_row_words_count_as_one_line(self):
        from peeklet.core.demo_filter import _count_text_lines

        boxes = [self._box(y=10), self._box(y=12), self._box(y=11)]
        assert _count_text_lines(boxes) == 1

    def test_widely_separated_rows_count_separately(self):
        from peeklet.core.demo_filter import _count_text_lines

        boxes = [self._box(y=10), self._box(y=100), self._box(y=200)]
        assert _count_text_lines(boxes) == 3

    def test_dense_ui_produces_many_lines(self):
        from peeklet.core.demo_filter import _count_text_lines

        boxes = [self._box(y=20 * i) for i in range(20)]
        assert _count_text_lines(boxes) == 20


class TestCountOccupiedGridCells:
    def _box(self, x, y, w=10, h=10):
        from peeklet.core.demo_filter import WordBox

        return WordBox(text="x", conf=90.0, x=x, y=y, w=w, h=h)

    def test_empty_returns_zero(self):
        from peeklet.core.demo_filter import _count_occupied_grid_cells

        assert _count_occupied_grid_cells([], (800, 1280, 3)) == 0

    def test_all_boxes_in_one_cell(self):
        from peeklet.core.demo_filter import _count_occupied_grid_cells

        boxes = [self._box(x=15, y=15), self._box(x=17, y=17)]
        assert _count_occupied_grid_cells(boxes, (800, 1280, 3)) == 1

    def test_dispersed_boxes_fill_many_cells(self):
        from peeklet.core.demo_filter import _count_occupied_grid_cells

        boxes = [self._box(x=100 * i + 50, y=80 * i + 40) for i in range(8)]
        assert _count_occupied_grid_cells(boxes, (800, 1280, 3)) == 8

    def test_handles_out_of_bounds_gracefully(self):
        from peeklet.core.demo_filter import _count_occupied_grid_cells

        boxes = [self._box(x=10_000, y=10_000)]
        assert _count_occupied_grid_cells(boxes, (800, 1280, 3)) == 1


class TestEdgePixelRatio:
    def test_flat_frame_has_near_zero_edges(self):
        from peeklet.core.demo_filter import _edge_pixel_ratio

        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert _edge_pixel_ratio(frame) < 0.001

    def test_high_contrast_checkerboard_has_many_edges(self):
        from peeklet.core.demo_filter import _edge_pixel_ratio

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(10):
            for j in range(10):
                if (i + j) % 2 == 0:
                    frame[i * 20 : (i + 1) * 20, j * 20 : (j + 1) * 20] = 255
        assert _edge_pixel_ratio(frame) > 0.05

    def test_grayscale_input_supported(self):
        from peeklet.core.demo_filter import _edge_pixel_ratio

        frame = np.zeros((100, 100), dtype=np.uint8)
        assert _edge_pixel_ratio(frame) == 0.0


class TestIsLowInfoFrame:
    def _cfg(self, **overrides):
        from peeklet.config import DemoFilterConfig

        return DemoFilterConfig(**overrides)

    def test_rejects_when_all_three_signals_fail(self, monkeypatch):
        from peeklet.core import demo_filter

        monkeypatch.setattr(demo_filter, "_ocr_word_boxes", lambda *a, **kw: [])
        monkeypatch.setattr(demo_filter, "_count_text_lines", lambda boxes: 3)
        monkeypatch.setattr(demo_filter, "_count_occupied_grid_cells", lambda boxes, shape: 4)
        monkeypatch.setattr(demo_filter, "_edge_pixel_ratio", lambda frame: 0.001)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert demo_filter._is_low_info_frame(frame, self._cfg()) is True

    def test_passes_when_only_text_lines_pass(self, monkeypatch):
        from peeklet.core import demo_filter

        monkeypatch.setattr(demo_filter, "_ocr_word_boxes", lambda *a, **kw: [])
        monkeypatch.setattr(demo_filter, "_count_text_lines", lambda boxes: 20)
        monkeypatch.setattr(demo_filter, "_count_occupied_grid_cells", lambda boxes, shape: 4)
        monkeypatch.setattr(demo_filter, "_edge_pixel_ratio", lambda frame: 0.001)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert demo_filter._is_low_info_frame(frame, self._cfg()) is False

    def test_passes_when_only_edge_density_passes(self, monkeypatch):
        from peeklet.core import demo_filter

        monkeypatch.setattr(demo_filter, "_ocr_word_boxes", lambda *a, **kw: [])
        monkeypatch.setattr(demo_filter, "_count_text_lines", lambda boxes: 3)
        monkeypatch.setattr(demo_filter, "_count_occupied_grid_cells", lambda boxes, shape: 4)
        monkeypatch.setattr(demo_filter, "_edge_pixel_ratio", lambda frame: 0.05)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert demo_filter._is_low_info_frame(frame, self._cfg()) is False

    def test_passes_when_only_grid_cells_pass(self, monkeypatch):
        from peeklet.core import demo_filter

        monkeypatch.setattr(demo_filter, "_ocr_word_boxes", lambda *a, **kw: [])
        monkeypatch.setattr(demo_filter, "_count_text_lines", lambda boxes: 3)
        monkeypatch.setattr(demo_filter, "_count_occupied_grid_cells", lambda boxes, shape: 20)
        monkeypatch.setattr(demo_filter, "_edge_pixel_ratio", lambda frame: 0.001)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert demo_filter._is_low_info_frame(frame, self._cfg()) is False


class _StubSsim:
    def __init__(self, score):
        self.ssim_score = score


class TestPhashDedup:
    def _decoder(self):
        class FakeDecoder:
            def get_metadata(self):
                from types import SimpleNamespace

                return SimpleNamespace(
                    filename="fake.mp4",
                    duration=60.0,
                    frame_count=1800,
                    fps=30.0,
                )

            def extract_frame_at(self, ts):
                return np.zeros((100, 100, 3), dtype=np.uint8), ts, int(ts * 30)

        return FakeDecoder()

    def test_phash_duplicate_is_dropped(self, tmp_path, monkeypatch):
        from peeklet.config import DemoFilterConfig
        from peeklet.core import demo_filter
        from peeklet.core.audio import TranscriptSegment
        from peeklet.utils.types import Moment

        monkeypatch.setattr(demo_filter, "_is_low_info_frame", lambda f, c: False)
        monkeypatch.setattr("peeklet.core.comparator.compare_frames", lambda a, b: _StubSsim(0.0))
        monkeypatch.setattr(demo_filter, "dhash_64", lambda f: 0xABCD1234)
        monkeypatch.setattr(demo_filter, "hamming_distance", lambda a, b: 0)
        monkeypatch.setattr(
            demo_filter,
            "save_keyframe",
            lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
        )

        moments = [
            Moment(
                timestamp=10.0,
                visual_context_goal="first",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
            Moment(
                timestamp=20.0,
                visual_context_goal="duplicate",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
        ]
        transcript = [
            TranscriptSegment(start=9.0, end=25.0, text="x", speaker=None),
        ]
        results = demo_filter.select_frames_for_moments(
            decoder=self._decoder(),
            moments=moments,
            transcript=transcript,
            config=DemoFilterConfig(),
            output_dir=tmp_path,
        )
        assert len(results) == 1

    def test_anchor_bypasses_phash_dedup(self, tmp_path, monkeypatch):
        from peeklet.config import DemoFilterConfig
        from peeklet.core import demo_filter
        from peeklet.core.audio import TranscriptSegment
        from peeklet.utils.types import Moment

        monkeypatch.setattr(demo_filter, "_is_low_info_frame", lambda f, c: False)
        monkeypatch.setattr("peeklet.core.comparator.compare_frames", lambda a, b: _StubSsim(0.0))
        monkeypatch.setattr(demo_filter, "dhash_64", lambda f: 0xABCD1234)
        monkeypatch.setattr(demo_filter, "hamming_distance", lambda a, b: 0)
        monkeypatch.setattr(
            demo_filter,
            "save_keyframe",
            lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
        )

        moments = [
            Moment(
                timestamp=10.0,
                visual_context_goal="first",
                textual_anchor="",
                downstream_utility="",
                source="anchor",
            ),
            Moment(
                timestamp=20.0,
                visual_context_goal="second anchor",
                textual_anchor="",
                downstream_utility="",
                source="anchor",
            ),
        ]
        transcript = [
            TranscriptSegment(start=9.0, end=25.0, text="x", speaker=None),
        ]
        results = demo_filter.select_frames_for_moments(
            decoder=self._decoder(),
            moments=moments,
            transcript=transcript,
            config=DemoFilterConfig(),
            output_dir=tmp_path,
        )
        assert len(results) == 2


class TestNormalizeTokens:
    def test_lowercases_and_drops_stopwords(self):
        from peeklet.core.demo_filter import _normalize_tokens

        tokens = _normalize_tokens("The Settings Dashboard")
        assert tokens == {"settings", "dashboard"}

    def test_drops_short_tokens(self):
        from peeklet.core.demo_filter import _normalize_tokens

        # "ab" is <3 chars; "an" is a stopword; "log" survives.
        assert _normalize_tokens("ab an log") == {"log"}

    def test_empty_input(self):
        from peeklet.core.demo_filter import _normalize_tokens

        assert _normalize_tokens("") == set()


class TestCaptionImageAlignment:
    def _base_moment(self, ts: float = 10.0):
        from peeklet.utils.types import Moment

        return Moment(
            timestamp=ts,
            visual_context_goal="Invoice approval dashboard",
            textual_anchor="clicks approve button",
            downstream_utility="u",
        )

    def _make_config(self, **overrides):
        from peeklet.config import DemoFilterConfig

        defaults = dict(
            enabled=True,
            forward_search_step_sec=0.5,
            forward_search_window_max_sec=4.0,
            search_window_lookback_sec=2.0,
            gallery_min_words=0,
            dedup_ssim_threshold=1.0,
            phash_hamming_threshold=0,
            max_seconds_between_keyframes=0.0,
        )
        defaults.update(overrides)
        return DemoFilterConfig(**defaults)

    def test_content_confidence_when_ocr_matches_caption(self, tmp_path, monkeypatch):
        from peeklet.core import demo_filter
        from peeklet.core.audio import TranscriptSegment

        decoder = _make_decoder_for_moments(meta_duration=60.0)
        monkeypatch.setattr(
            demo_filter,
            "save_keyframe",
            lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
        )
        monkeypatch.setattr(
            demo_filter,
            "_ocr_text_and_tokens",
            lambda frame, downscale_dim: (
                "Invoice Approval Dashboard",
                {"invoice", "approval", "dashboard"},
            ),
        )

        results = demo_filter.select_frames_for_moments(
            decoder=decoder,
            moments=[self._base_moment()],
            transcript=[TranscriptSegment(start=0.0, end=30.0, text="x")],
            config=self._make_config(),
            output_dir=tmp_path,
        )

        assert len(results) == 1
        assert results[0].alignment_confidence == "content"

    def test_temporal_only_when_no_overlap_even_after_widening(self, tmp_path, monkeypatch):
        from peeklet.core import demo_filter
        from peeklet.core.audio import TranscriptSegment

        decoder = _make_decoder_for_moments(meta_duration=60.0)
        monkeypatch.setattr(
            demo_filter,
            "save_keyframe",
            lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
        )
        monkeypatch.setattr(
            demo_filter,
            "_ocr_text_and_tokens",
            lambda frame, downscale_dim: ("Unrelated Toolbar", {"unrelated", "toolbar"}),
        )

        results = demo_filter.select_frames_for_moments(
            decoder=decoder,
            moments=[self._base_moment()],
            transcript=[TranscriptSegment(start=0.0, end=30.0, text="x")],
            config=self._make_config(),
            output_dir=tmp_path,
        )

        assert len(results) == 1
        # Frame is still kept — the rejector only drops low-info frames,
        # not caption-misaligned ones. But the confidence is downgraded.
        assert results[0].alignment_confidence == "temporal_only"

    def test_widening_recovers_caption_aligned_frame(self, tmp_path, monkeypatch):
        from peeklet.core import demo_filter
        from peeklet.core.audio import TranscriptSegment

        decoder = _make_decoder_for_moments(meta_duration=60.0)
        monkeypatch.setattr(
            demo_filter,
            "save_keyframe",
            lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
        )

        calls = {"n": 0}

        def _tokens(frame, downscale_dim):
            # First call (initial window): no overlap. Second call (widened
            # retry): matches the caption.
            calls["n"] += 1
            if calls["n"] == 1:
                return ("unrelated", {"unrelated"})
            return ("invoice dashboard", {"invoice", "dashboard"})

        monkeypatch.setattr(demo_filter, "_ocr_text_and_tokens", _tokens)

        results = demo_filter.select_frames_for_moments(
            decoder=decoder,
            moments=[self._base_moment()],
            transcript=[TranscriptSegment(start=0.0, end=30.0, text="x")],
            config=self._make_config(),
            output_dir=tmp_path,
        )

        assert len(results) == 1
        assert results[0].alignment_confidence == "content"
        assert calls["n"] == 2

    def test_skips_retry_when_window_already_spans_segment(self, tmp_path, monkeypatch):
        from peeklet.core import demo_filter
        from peeklet.core.audio import TranscriptSegment

        decoder = _make_decoder_for_moments(meta_duration=60.0)
        monkeypatch.setattr(
            demo_filter,
            "save_keyframe",
            lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
        )

        calls = {"n": 0}

        def _tokens(frame, downscale_dim):
            calls["n"] += 1
            return ("unrelated", {"unrelated"})

        monkeypatch.setattr(demo_filter, "_ocr_text_and_tokens", _tokens)

        # Tiny segment — initial window is clamped to it entirely, so the
        # widened retry would produce the same window and is skipped.
        results = demo_filter.select_frames_for_moments(
            decoder=decoder,
            moments=[self._base_moment(ts=10.0)],
            transcript=[TranscriptSegment(start=9.9, end=10.1, text="x")],
            config=self._make_config(),
            output_dir=tmp_path,
        )

        assert len(results) == 1
        assert results[0].alignment_confidence == "temporal_only"
        assert calls["n"] == 1

    def test_empty_caption_defaults_to_content(self, tmp_path, monkeypatch):
        from peeklet.core import demo_filter
        from peeklet.core.audio import TranscriptSegment
        from peeklet.utils.types import Moment

        decoder = _make_decoder_for_moments(meta_duration=60.0)
        monkeypatch.setattr(
            demo_filter,
            "save_keyframe",
            lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
        )
        monkeypatch.setattr(
            demo_filter,
            "_ocr_text_and_tokens",
            lambda frame, downscale_dim: ("", set()),
        )

        moment = Moment(
            timestamp=10.0,
            visual_context_goal="",
            textual_anchor="",
            downstream_utility="u",
        )
        results = demo_filter.select_frames_for_moments(
            decoder=decoder,
            moments=[moment],
            transcript=[TranscriptSegment(start=0.0, end=30.0, text="x")],
            config=self._make_config(),
            output_dir=tmp_path,
        )

        assert len(results) == 1
        assert results[0].alignment_confidence == "content"

    def test_frame_results_carry_ocr_text_and_tokens(self, tmp_path, monkeypatch):
        from peeklet.core import demo_filter
        from peeklet.core.audio import TranscriptSegment

        decoder = _make_decoder_for_moments(meta_duration=60.0)
        monkeypatch.setattr(
            demo_filter,
            "save_keyframe",
            lambda frame, output_dir, frame_id, fmt="jpg": tmp_path / f"{frame_id}.jpg",
        )
        monkeypatch.setattr(
            demo_filter,
            "_ocr_text_and_tokens",
            lambda frame, downscale_dim: (
                "Invoice Approval Dashboard",
                {"invoice", "approval", "dashboard"},
            ),
        )

        results = demo_filter.select_frames_for_moments(
            decoder=decoder,
            moments=[self._base_moment()],
            transcript=[TranscriptSegment(start=0.0, end=30.0, text="x")],
            config=self._make_config(),
            output_dir=tmp_path,
        )

        assert len(results) == 1
        assert results[0].ocr_text == "Invoice Approval Dashboard"
        assert set(results[0].ocr_tokens or []) == {"invoice", "approval", "dashboard"}
        assert results[0].alignment_confidence == "content"
