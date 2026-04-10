"""Unit tests for demo_filter.py."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


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


def test_is_gallery_frame_returns_true_below_threshold():
    from peeklet.core.demo_filter import _is_gallery_frame

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("peeklet.core.demo_filter._count_words_in_frame", return_value=2):
        assert _is_gallery_frame(frame, downscale_dim=360, min_words=5) is True


def test_is_gallery_frame_returns_false_above_threshold():
    from peeklet.core.demo_filter import _is_gallery_frame

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("peeklet.core.demo_filter._count_words_in_frame", return_value=10):
        assert _is_gallery_frame(frame, downscale_dim=360, min_words=5) is False


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

    moments = [Moment(timestamp=10.0, caption="cap", reason="reason")]
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
    assert r.llm_caption == "cap"
    assert r.llm_reason == "reason"
    # With all-identical frames, _pick_stable_index returns index 1 (first
    # checkable position) → ts == 10.5. Just assert the picked frame is in
    # the search window.
    assert r.video_timestamp is not None
    assert 10.0 <= r.video_timestamp <= 12.0
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

    moments = [Moment(timestamp=10.0, caption="cap", reason="reason")]
    transcript = [TranscriptSegment(start=8.0, end=12.0, text="speaking")]
    cfg = DemoFilterConfig(enabled=True, gallery_min_words=5)

    monkeypatch.setattr(
        "peeklet.core.demo_filter._count_words_in_frame",
        lambda frame, downscale_dim: 0,
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

    moments = [Moment(timestamp=50.0, caption="c", reason="r")]
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

    fake_moments = [Moment(timestamp=10.0, caption="c", reason="r")]
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
    assert results[0].llm_caption == "c"


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

    cfg = DemoFilterConfig(enabled=True)
    with caplog.at_level(logging.WARNING):
        results = apply_demo_filter(
            decoder=decoder,
            transcript=[],
            config=cfg,
            output_dir=tmp_path,
        )

    assert results == []
    assert any("zero screenshot-worthy moments" in rec.message for rec in caplog.records)


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


def test_is_gallery_frame_returns_false_when_pytesseract_unavailable():
    """Bug 2 regression: missing OCR must not silently reject every frame.

    When pytesseract is unavailable we cannot determine whether a frame is
    gallery view, so the safe default is to let the frame through and let the
    LLM-picked moment win. The previous behavior returned True (== gallery)
    for every frame because ``_count_words_in_frame`` returned 0 < min_words.
    """
    from peeklet.core.demo_filter import _is_gallery_frame

    frame = _make_frame(h=720, w=1280)
    with patch("peeklet.core.demo_filter.pytesseract", None):
        assert _is_gallery_frame(frame, downscale_dim=1920, min_words=5) is False


@pytest.mark.skipif(not _tesseract_available(), reason="tesseract binary not installed")
def test_is_gallery_frame_accepts_real_text_frame_at_720p():
    """Bug 3 regression: a 720p frame with clearly readable text must NOT be
    flagged as gallery view at the default ``gallery_ocr_min_dim``.

    This is the end-to-end OCR test that would have caught the silent rejection
    of every demo frame from a 720p screen-share recording.
    """
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import _is_gallery_frame

    frame = _render_text_frame(
        ["Settings", "Dashboard", "Analytics", "Users", "Tokens", "Reports", "Logout"],
        width=1280,
        height=720,
    )
    cfg = DemoFilterConfig()
    assert (
        _is_gallery_frame(
            frame,
            downscale_dim=cfg.gallery_ocr_min_dim,
            min_words=cfg.gallery_min_words,
        )
        is False
    )


def test_apply_demo_filter_warns_when_pytesseract_unavailable(tmp_path, monkeypatch, caplog):
    """Bug 2 regression: a missing OCR backend must produce a startup warning,
    not a silent zero-keyframes run.
    """
    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = _make_decoder_for_moments(meta_duration=60.0)
    fake_client = MagicMock()
    fake_client.pick_moments.return_value = []
    monkeypatch.setattr(
        "peeklet.core.demo_filter.build_llm_client",
        lambda provider, model: fake_client,
    )
    monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", None)

    cfg = DemoFilterConfig(enabled=True)
    with caplog.at_level(logging.WARNING):
        apply_demo_filter(
            decoder=decoder,
            transcript=[],
            config=cfg,
            output_dir=tmp_path,
        )

    assert any(
        "OCR" in rec.message and "gallery" in rec.message.lower() for rec in caplog.records
    ), f"expected an OCR-unavailable warning, got: {[r.message for r in caplog.records]}"


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
        Moment(timestamp=10.0, caption="first", reason="r"),
        Moment(timestamp=100.0, caption="second", reason="r"),
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
    assert results[0].llm_caption == "first"


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
        Moment(timestamp=10.0, caption="first", reason="r"),
        Moment(timestamp=100.0, caption="second", reason="r"),
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
    assert [r.llm_caption for r in results] == ["first", "second"]


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
        Moment(timestamp=10.0, caption="a", reason="r"),
        Moment(timestamp=100.0, caption="b", reason="r"),
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
        Moment(timestamp=10.0, caption="a", reason="r"),
        Moment(timestamp=100.0, caption="b", reason="r"),
        Moment(timestamp=200.0, caption="c", reason="r"),
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
    assert results[0].llm_caption == "a"


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
    moments = [Moment(timestamp=99.0, caption="end", reason="r")]
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

    moments = [Moment(timestamp=95.0, caption="near-end", reason="r")]
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
    assert results[0].llm_caption == "near-end"


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

    moments = [Moment(timestamp=99.0, caption="end", reason="r")]
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
