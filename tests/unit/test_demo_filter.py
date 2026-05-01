"""Unit tests for demo_filter.py."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _pass_layout_rejector_by_default(request, monkeypatch):
    """Default: let the layout rejector pass every frame in this module.

    Tests that specifically exercise the rejector live in
    ``TestIsLowInfoFrame`` (patches the signals directly) — those are
    skipped here.
    """
    if "TestIsLowInfoFrame" in request.node.nodeid:
        return
    if "test_low_info_rejector_accepts_real_text_frame" in request.node.nodeid:
        return
    monkeypatch.setattr(
        "peeklet.core.demo_filter._is_low_info_frame",
        lambda frame, config: False,
    )


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
    from unittest.mock import MagicMock

    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = MagicMock()
    monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", None)

    cfg = DemoFilterConfig(enabled=True)
    with pytest.raises(RuntimeError, match="pytesseract"):
        apply_demo_filter(
            decoder=decoder,
            transcript=[],
            config=cfg,
            output_dir=tmp_path,
        )


def test_apply_demo_filter_logs_warning_on_zero_moments(tmp_path, monkeypatch, caplog):
    import logging
    from unittest.mock import MagicMock

    from peeklet.config import DemoFilterConfig
    from peeklet.core.demo_filter import apply_demo_filter

    decoder = MagicMock()
    meta = MagicMock()
    meta.duration = 60.0
    meta.filename = "t.mp4"
    decoder.get_metadata.return_value = meta

    fake_client = MagicMock()
    fake_client.pick_moments.return_value = []

    monkeypatch.setattr(
        "peeklet.core.demo_filter.build_llm_client",
        lambda **k: fake_client,
    )
    monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", MagicMock())

    cfg = DemoFilterConfig(enabled=True)
    with caplog.at_level(logging.WARNING):
        screens, moments = apply_demo_filter(
            decoder=decoder,
            transcript=[],
            config=cfg,
            output_dir=tmp_path,
        )

    assert screens == []
    assert moments == []
    assert any("no anchors and no LLM picks" in rec.message for rec in caplog.records)


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


class TestApplyDemoFilterLinearPass:
    def _build_decoder(self, frames_by_ts):
        from unittest.mock import MagicMock

        decoder = MagicMock()
        meta = MagicMock()
        meta.duration = 100.0
        meta.filename = "test.mp4"
        decoder.get_metadata.return_value = meta

        def extract(t):
            # Match nearest known timestamp to t; default to a uniform frame
            best = min(frames_by_ts.keys(), key=lambda k: abs(k - t))
            return frames_by_ts[best], t, int(t * 30)

        decoder.extract_frame_at.side_effect = extract
        return decoder

    def test_two_moments_same_screen_share_screen_id(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import numpy as np

        from peeklet.config import DemoFilterConfig
        from peeklet.core.audio import TranscriptSegment
        from peeklet.core.demo_filter import apply_demo_filter
        from peeklet.utils.types import Moment

        # Both moments will see the same painted frame with same OCR boxes.
        frame = np.full((1000, 1600, 3), 200, dtype=np.uint8)
        decoder = self._build_decoder({1.0: frame, 5.0: frame})
        transcript = [TranscriptSegment(start=0, end=10, text="...")]

        # Stub OCR + layout rejector
        boxes = [
            self._wb("https://app.fathom.video/calls/1", 200, 20, 400, 18),
            self._wb("Dashboard", 300, 120, 600, 48),
            self._wb("Home", 20, 200, 80, 20),
        ]
        monkeypatch.setattr("peeklet.core.demo_filter._ocr_word_boxes", lambda f, d: boxes)
        monkeypatch.setattr("peeklet.core.demo_filter._is_low_info_frame", lambda f, c: False)

        client = MagicMock()
        client.pick_moments.return_value = [
            Moment(
                timestamp=1.0, visual_context_goal="a", textual_anchor="a", downstream_utility="a"
            ),
            Moment(
                timestamp=5.0, visual_context_goal="b", textual_anchor="b", downstream_utility="b"
            ),
        ]
        monkeypatch.setattr("peeklet.core.demo_filter.build_llm_client", lambda **k: client)
        monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", MagicMock())

        screens, moments = apply_demo_filter(
            decoder=decoder,
            transcript=transcript,
            config=DemoFilterConfig(),
            output_dir=tmp_path,
            transcript_text="",
        )
        assert len(screens) == 1
        assert len(moments) == 2
        assert moments[0].screen_id == moments[1].screen_id == screens[0].screen_id

    def _wb(self, text, x, y, w, h):
        from collections import namedtuple

        WB = namedtuple("WB", ["text", "conf", "x", "y", "w", "h"])
        return WB(text, 99.0, x, y, w, h)

    def test_anchors_appear_with_type_action_item(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import numpy as np

        from peeklet.config import DemoFilterConfig
        from peeklet.core.audio import TranscriptSegment
        from peeklet.core.demo_filter import apply_demo_filter

        frame = np.full((1000, 1600, 3), 200, dtype=np.uint8)
        decoder = self._build_decoder({1.0: frame})
        transcript = [TranscriptSegment(start=0, end=10, text="...")]

        boxes = [
            self._wb("https://app.fathom.video/calls/1", 200, 20, 400, 18),
            self._wb("Dashboard", 300, 120, 600, 48),
        ]
        monkeypatch.setattr("peeklet.core.demo_filter._ocr_word_boxes", lambda f, d: boxes)
        monkeypatch.setattr("peeklet.core.demo_filter._is_low_info_frame", lambda f, c: False)

        anchor_text = (
            "**ACTION ITEM: Configure bug filter - "
            "++[WATCH](https://fathom.video/calls/1?timestamp=1.0)++**"
        )
        client = MagicMock()
        client.pick_moments.return_value = []
        monkeypatch.setattr("peeklet.core.demo_filter.build_llm_client", lambda **k: client)
        monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", MagicMock())

        screens, moments = apply_demo_filter(
            decoder=decoder,
            transcript=transcript,
            config=DemoFilterConfig(),
            output_dir=tmp_path,
            transcript_text=anchor_text,
        )
        assert len(moments) == 1
        assert moments[0].type == "action_item"

    def test_anchor_with_failing_quality_gate_emits_image_unavailable(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import numpy as np

        from peeklet.config import DemoFilterConfig
        from peeklet.core.audio import TranscriptSegment
        from peeklet.core.demo_filter import apply_demo_filter

        frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
        decoder = self._build_decoder({1.0: frame})
        transcript = [TranscriptSegment(start=0, end=10, text="...")]

        monkeypatch.setattr("peeklet.core.demo_filter._ocr_word_boxes", lambda f, d: [])
        monkeypatch.setattr(
            "peeklet.core.demo_filter._is_low_info_frame", lambda f, c: True
        )  # always fail

        anchor_text = (
            "**ACTION ITEM: Galleria - ++[WATCH](https://fathom.video/calls/1?timestamp=1.0)++**"
        )
        client = MagicMock()
        client.pick_moments.return_value = []
        monkeypatch.setattr("peeklet.core.demo_filter.build_llm_client", lambda **k: client)
        monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", MagicMock())

        screens, moments = apply_demo_filter(
            decoder=decoder,
            transcript=transcript,
            config=DemoFilterConfig(),
            output_dir=tmp_path,
            transcript_text=anchor_text,
        )
        assert len(screens) == 0
        assert len(moments) == 1
        assert moments[0].image_unavailable is True
        assert moments[0].screen_id is None

    def test_safety_fallback_keeps_unreadable_frames_distinct(self, tmp_path, monkeypatch):
        """Empty Part A → no collapse; both moments save separate images."""
        from unittest.mock import MagicMock

        import numpy as np

        from peeklet.config import DemoFilterConfig
        from peeklet.core.audio import TranscriptSegment
        from peeklet.core.demo_filter import apply_demo_filter
        from peeklet.utils.types import Moment

        # Two distinct frames so file paths differ
        f1 = np.full((1000, 1600, 3), 50, dtype=np.uint8)
        f2 = np.full((1000, 1600, 3), 200, dtype=np.uint8)
        decoder = self._build_decoder({1.0: f1, 5.0: f2})
        transcript = [TranscriptSegment(start=0, end=10, text="...")]

        # No OCR text → empty Part A
        monkeypatch.setattr("peeklet.core.demo_filter._ocr_word_boxes", lambda f, d: [])
        monkeypatch.setattr("peeklet.core.demo_filter._is_low_info_frame", lambda f, c: False)

        client = MagicMock()
        client.pick_moments.return_value = [
            Moment(
                timestamp=1.0, visual_context_goal="a", textual_anchor="a", downstream_utility="a"
            ),
            Moment(
                timestamp=5.0, visual_context_goal="b", textual_anchor="b", downstream_utility="b"
            ),
        ]
        monkeypatch.setattr("peeklet.core.demo_filter.build_llm_client", lambda **k: client)
        monkeypatch.setattr("peeklet.core.demo_filter.pytesseract", MagicMock())

        screens, moments = apply_demo_filter(
            decoder=decoder,
            transcript=transcript,
            config=DemoFilterConfig(),
            output_dir=tmp_path,
            transcript_text="",
        )
        assert len(screens) == 2  # safety fallback prevents collapse
        assert moments[0].screen_id != moments[1].screen_id
