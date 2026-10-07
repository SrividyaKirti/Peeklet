"""Unit tests for screen.py."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _pass_layout_rejector_by_default(request, monkeypatch):
    """Default: stub the layout rejector to always pass.

    Tests that need to exercise the REAL rejector mark themselves
    with ``@pytest.mark.rejector_live`` to skip this stub.
    """
    if request.node.get_closest_marker("rejector_live"):
        return
    monkeypatch.setattr(
        "peeklet.screen._is_low_info_frame",
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
    from peeklet.screen import _ocr_word_boxes

    fake_data = {
        "text": ["Settings", "x", "Save", "", "Login", "."],
        "conf": ["95", "92", "80", "-1", "20", "99"],
    }
    with patch("peeklet.screen.pytesseract") as mock_pt:
        mock_pt.image_to_data.return_value = fake_data
        mock_pt.Output.DICT = "dict"
        count = len(_ocr_word_boxes(_make_frame(), downscale_dim=360))

    # "Settings" (95, len 8) OK
    # "x" (92, len 1) short
    # "Save" (80, len 4) OK
    # ""  (-1, empty) empty
    # "Login" (20, len 5) low conf
    # "." (99, len 1) short
    assert count == 2


def test_count_words_handles_tesseract_exception():
    from peeklet.screen import _ocr_word_boxes

    with patch("peeklet.screen.pytesseract") as mock_pt:
        mock_pt.image_to_data.side_effect = RuntimeError("tesseract crashed")
        mock_pt.Output.DICT = "dict"
        count = len(_ocr_word_boxes(_make_frame(), downscale_dim=360))

    assert count == 0  # graceful fallback


def test_count_words_returns_zero_when_pytesseract_is_none():
    from peeklet.screen import _ocr_word_boxes

    with patch("peeklet.screen.pytesseract", None):
        count = len(_ocr_word_boxes(_make_frame(), downscale_dim=360))

    assert count == 0


def test_downscale_for_ocr_already_smaller_returns_original():
    from peeklet.screen import _downscale_for_ocr

    frame = _make_frame(h=200, w=200)
    out = _downscale_for_ocr(frame, downscale_dim=360)
    # Already smaller than the target, so it should be returned untouched.
    assert out is frame


def test_downscale_for_ocr_square_frame_resizes_both_dims():
    from peeklet.screen import _downscale_for_ocr

    frame = _make_frame(h=720, w=720)
    out = _downscale_for_ocr(frame, downscale_dim=360)
    assert out.shape == (360, 360, 3)


def test_downscale_for_ocr_landscape_frame_preserves_aspect():
    from peeklet.screen import _downscale_for_ocr

    frame = _make_frame(h=540, w=960)  # 16:9 landscape
    out = _downscale_for_ocr(frame, downscale_dim=480)
    # Longest edge should match downscale_dim, height should scale proportionally
    assert out.shape[1] == 480
    assert out.shape[0] == int(round(540 * 480 / 960))


def test_downscale_for_ocr_portrait_frame_preserves_aspect():
    from peeklet.screen import _downscale_for_ocr

    frame = _make_frame(h=960, w=540)  # 9:16 portrait
    out = _downscale_for_ocr(frame, downscale_dim=480)
    assert out.shape[0] == 480
    assert out.shape[1] == int(round(540 * 480 / 960))


@pytest.mark.rejector_live
@pytest.mark.skipif(not _tesseract_available(), reason="tesseract binary not installed")
def test_low_info_rejector_accepts_real_text_frame_at_720p():
    """Regression: a 720p frame with dashboard-density text must NOT be
    rejected as low-info at the default config thresholds. Guards against
    the OCR downscale regression that previously dropped every demo frame
    from a 720p screen-share recording.
    """
    from peeklet.config import PeekletConfig
    from peeklet.screen import _is_low_info_frame

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
    cfg = PeekletConfig()
    assert _is_low_info_frame(frame, cfg) is False


class TestOcrWordBoxes:
    def test_returns_empty_when_pytesseract_missing(self, monkeypatch):
        from peeklet import screen

        monkeypatch.setattr(screen, "pytesseract", None)
        result = screen._ocr_word_boxes(np.zeros((10, 10, 3), dtype=np.uint8), downscale_dim=100)
        assert result == []

    def test_filters_low_confidence_and_short_words(self, monkeypatch):
        from peeklet import screen

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

        monkeypatch.setattr(screen, "pytesseract", FakeTess)
        boxes = screen._ocr_word_boxes(np.zeros((100, 300, 3), dtype=np.uint8), downscale_dim=1000)
        texts = [b.text for b in boxes]
        assert texts == ["hello", "world"]
        assert boxes[0].x == 10 and boxes[0].y == 5
        assert boxes[0].w == 40 and boxes[0].h == 12

    def test_count_words_still_reflects_box_count(self, monkeypatch):
        from peeklet import screen

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

        monkeypatch.setattr(screen, "pytesseract", FakeTess)
        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        assert len(screen._ocr_word_boxes(frame, downscale_dim=1000)) == 3


class TestCountTextLines:
    def _box(self, y, h=10):
        from peeklet.screen import WordBox

        return WordBox(text="x", conf=90.0, x=0, y=y, w=20, h=h)

    def test_empty_returns_zero(self):
        from peeklet.screen import _count_text_lines

        assert _count_text_lines([]) == 0

    def test_single_row_words_count_as_one_line(self):
        from peeklet.screen import _count_text_lines

        boxes = [self._box(y=10), self._box(y=12), self._box(y=11)]
        assert _count_text_lines(boxes) == 1

    def test_widely_separated_rows_count_separately(self):
        from peeklet.screen import _count_text_lines

        boxes = [self._box(y=10), self._box(y=100), self._box(y=200)]
        assert _count_text_lines(boxes) == 3

    def test_dense_ui_produces_many_lines(self):
        from peeklet.screen import _count_text_lines

        boxes = [self._box(y=20 * i) for i in range(20)]
        assert _count_text_lines(boxes) == 20


class TestCountOccupiedGridCells:
    def _box(self, x, y, w=10, h=10):
        from peeklet.screen import WordBox

        return WordBox(text="x", conf=90.0, x=x, y=y, w=w, h=h)

    def test_empty_returns_zero(self):
        from peeklet.screen import _count_occupied_grid_cells

        assert _count_occupied_grid_cells([], (800, 1280, 3)) == 0

    def test_all_boxes_in_one_cell(self):
        from peeklet.screen import _count_occupied_grid_cells

        boxes = [self._box(x=15, y=15), self._box(x=17, y=17)]
        assert _count_occupied_grid_cells(boxes, (800, 1280, 3)) == 1

    def test_dispersed_boxes_fill_many_cells(self):
        from peeklet.screen import _count_occupied_grid_cells

        boxes = [self._box(x=100 * i + 50, y=80 * i + 40) for i in range(8)]
        assert _count_occupied_grid_cells(boxes, (800, 1280, 3)) == 8

    def test_handles_out_of_bounds_gracefully(self):
        from peeklet.screen import _count_occupied_grid_cells

        boxes = [self._box(x=10_000, y=10_000)]
        assert _count_occupied_grid_cells(boxes, (800, 1280, 3)) == 1


class TestEdgePixelRatio:
    def test_flat_frame_has_near_zero_edges(self):
        from peeklet.screen import _edge_pixel_ratio

        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert _edge_pixel_ratio(frame) < 0.001

    def test_high_contrast_checkerboard_has_many_edges(self):
        from peeklet.screen import _edge_pixel_ratio

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(10):
            for j in range(10):
                if (i + j) % 2 == 0:
                    frame[i * 20 : (i + 1) * 20, j * 20 : (j + 1) * 20] = 255
        assert _edge_pixel_ratio(frame) > 0.05

    def test_grayscale_input_supported(self):
        from peeklet.screen import _edge_pixel_ratio

        frame = np.zeros((100, 100), dtype=np.uint8)
        assert _edge_pixel_ratio(frame) == 0.0


@pytest.mark.rejector_live
class TestIsLowInfoFrame:
    def _cfg(self, **overrides):
        from peeklet.config import PeekletConfig

        return PeekletConfig(**overrides)

    def test_rejects_when_all_three_signals_fail(self, monkeypatch):
        from peeklet import screen

        monkeypatch.setattr(screen, "_ocr_word_boxes", lambda *a, **kw: [])
        monkeypatch.setattr(screen, "_count_text_lines", lambda boxes: 3)
        monkeypatch.setattr(screen, "_count_occupied_grid_cells", lambda boxes, shape: 4)
        monkeypatch.setattr(screen, "_edge_pixel_ratio", lambda frame: 0.001)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert screen._is_low_info_frame(frame, self._cfg()) is True

    def test_passes_when_only_text_lines_pass(self, monkeypatch):
        from peeklet import screen

        monkeypatch.setattr(screen, "_ocr_word_boxes", lambda *a, **kw: [])
        monkeypatch.setattr(screen, "_count_text_lines", lambda boxes: 20)
        monkeypatch.setattr(screen, "_count_occupied_grid_cells", lambda boxes, shape: 4)
        monkeypatch.setattr(screen, "_edge_pixel_ratio", lambda frame: 0.001)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert screen._is_low_info_frame(frame, self._cfg()) is False

    def test_passes_when_only_edge_density_passes(self, monkeypatch):
        from peeklet import screen

        monkeypatch.setattr(screen, "_ocr_word_boxes", lambda *a, **kw: [])
        monkeypatch.setattr(screen, "_count_text_lines", lambda boxes: 3)
        monkeypatch.setattr(screen, "_count_occupied_grid_cells", lambda boxes, shape: 4)
        monkeypatch.setattr(screen, "_edge_pixel_ratio", lambda frame: 0.05)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert screen._is_low_info_frame(frame, self._cfg()) is False

    def test_passes_when_only_grid_cells_pass(self, monkeypatch):
        from peeklet import screen

        monkeypatch.setattr(screen, "_ocr_word_boxes", lambda *a, **kw: [])
        monkeypatch.setattr(screen, "_count_text_lines", lambda boxes: 3)
        monkeypatch.setattr(screen, "_count_occupied_grid_cells", lambda boxes, shape: 20)
        monkeypatch.setattr(screen, "_edge_pixel_ratio", lambda frame: 0.001)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert screen._is_low_info_frame(frame, self._cfg()) is False
