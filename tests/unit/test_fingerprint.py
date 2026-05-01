"""Unit tests for content-addressable screen fingerprinting."""

from __future__ import annotations

from collections import namedtuple

import numpy as np


def test_normalize_text_lowercases_and_collapses_whitespace():
    from peeklet.core.fingerprint import _normalize_text

    assert _normalize_text("  Hello   World  ") == "hello world"


def test_normalize_text_strips_non_alphanumeric_except_spaces():
    from peeklet.core.fingerprint import _normalize_text

    assert _normalize_text("Hello, World! 123.") == "hello world 123"


def test_normalize_text_empty_input_returns_empty_string():
    from peeklet.core.fingerprint import _normalize_text

    assert _normalize_text("") == ""
    assert _normalize_text("   \t\n  ") == ""


def test_normalize_url_preserves_slashes_and_dots():
    from peeklet.core.fingerprint import _normalize_url

    assert _normalize_url("https://Example.COM/path/to?q=1") == "https://example.com/path/to"


def test_normalize_url_strips_query_and_fragment():
    from peeklet.core.fingerprint import _normalize_url

    assert _normalize_url("https://example.com/x?a=b&c=d#frag") == "https://example.com/x"


def test_normalize_url_collapses_whitespace_inside_token():
    from peeklet.core.fingerprint import _normalize_url

    assert _normalize_url("  https://example.com  ") == "https://example.com"


WB = namedtuple("WB", ["text", "x", "y", "w", "h"])


def _frame(h: int = 1000, w: int = 1600) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


URL_RE_FIXTURE = "https://app.fathom.video/calls/123"


def test_extract_url_picks_url_token_in_top_5_percent():
    from peeklet.core.fingerprint import extract_url

    h, w = 1000, 1600
    boxes = [
        WB(URL_RE_FIXTURE, x=200, y=20, w=400, h=18),
        WB("Dashboard", x=60, y=200, w=200, h=40),  # below top 5%
    ]
    assert extract_url(boxes, frame_h=h, frame_w=w) == "https://app.fathom.video/calls/123"


def test_extract_url_returns_empty_when_no_url_pattern_in_band():
    from peeklet.core.fingerprint import extract_url

    boxes = [WB("Welcome", x=10, y=10, w=200, h=18)]
    assert extract_url(boxes, frame_h=1000, frame_w=1600) == ""


def test_extract_url_ignores_url_outside_top_band():
    from peeklet.core.fingerprint import extract_url

    boxes = [WB("https://example.com", x=10, y=900, w=200, h=18)]  # bottom of frame
    assert extract_url(boxes, frame_h=1000, frame_w=1600) == ""


def test_extract_url_runs_normalize_url_on_match():
    from peeklet.core.fingerprint import extract_url

    boxes = [WB("HTTPS://Example.COM/PATH?q=1", x=10, y=10, w=200, h=18)]
    assert extract_url(boxes, frame_h=1000, frame_w=1600) == "https://example.com/path"


def test_extract_heading_picks_longest_large_font_in_top_half():
    from peeklet.core.fingerprint import extract_heading

    h, w = 1000, 1600
    boxes = [
        WB("Dashboard Settings", x=300, y=120, w=600, h=48),  # tall = large font
        WB("ok", x=300, y=140, w=20, h=12),  # small, ignored
        WB("Footer text here", x=300, y=900, w=400, h=48),  # bottom half, ignored
    ]
    assert extract_heading(boxes, frame_h=h, frame_w=w) == "dashboard settings"


def test_extract_heading_returns_empty_when_no_text_above_midline():
    from peeklet.core.fingerprint import extract_heading

    boxes = [WB("Footer", x=10, y=900, w=200, h=48)]
    assert extract_heading(boxes, frame_h=1000, frame_w=1600) == ""


def test_extract_sidebar_concatenates_left_15_percent():
    from peeklet.core.fingerprint import extract_sidebar_text

    h, w = 1000, 1600
    # 15% of 1600 = 240
    boxes = [
        WB("Home", x=20, y=200, w=80, h=20),
        WB("Settings", x=20, y=240, w=140, h=20),
        WB("Main content", x=400, y=200, w=200, h=20),  # outside sidebar
    ]
    assert extract_sidebar_text(boxes, frame_h=h, frame_w=w) == "home settings"


def test_extract_sidebar_returns_empty_when_no_left_text():
    from peeklet.core.fingerprint import extract_sidebar_text

    boxes = [WB("Centered", x=800, y=300, w=200, h=20)]
    assert extract_sidebar_text(boxes, frame_h=1000, frame_w=1600) == ""


def test_compute_part_b_identical_frames_have_zero_distance():
    from peeklet.core.fingerprint import compute_part_b, hamming_distance_64

    frame = np.random.default_rng(seed=1).integers(0, 256, (1000, 1600, 3), dtype=np.uint8)
    h1 = compute_part_b(frame)
    h2 = compute_part_b(frame.copy())
    assert hamming_distance_64(h1, h2) == 0


def test_compute_part_b_random_frames_have_large_distance():
    from peeklet.core.fingerprint import compute_part_b, hamming_distance_64

    rng = np.random.default_rng(seed=42)
    f1 = rng.integers(0, 256, (1000, 1600, 3), dtype=np.uint8)
    f2 = rng.integers(0, 256, (1000, 1600, 3), dtype=np.uint8)
    assert hamming_distance_64(compute_part_b(f1), compute_part_b(f2)) > 10


def test_compute_part_b_returns_64_bit_int():
    from peeklet.core.fingerprint import compute_part_b

    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
    h = compute_part_b(frame)
    assert isinstance(h, int)
    assert 0 <= h < (1 << 64)


def test_compute_part_b_uses_only_header_strip():
    """Pixels outside the header strip should not influence the hash.

    Both frames carry identical *structured* content inside the strip and
    different *structured* content outside. If the hash leaked outside
    pixels, the Hamming distance would be > 0.
    """
    from peeklet.core.fingerprint import compute_part_b, hamming_distance_64

    rng = np.random.default_rng(seed=7)
    h, w = 1000, 1600
    y0, y1 = int(h * 0.08), int(h * 0.22)
    strip_pixels = rng.integers(0, 256, (y1 - y0, w, 3), dtype=np.uint8)

    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[y0:y1] = strip_pixels
    b = np.zeros((h, w, 3), dtype=np.uint8)
    b[y0:y1] = strip_pixels
    # Differ only OUTSIDE the strip with structured noise so a leak would
    # show up as nonzero Hamming distance.
    outside_noise = rng.integers(0, 256, (h - y1, w, 3), dtype=np.uint8)
    b[y1:] = outside_noise

    assert hamming_distance_64(compute_part_b(a), compute_part_b(b)) == 0


def test_hamming_distance_64_basics():
    from peeklet.core.fingerprint import hamming_distance_64

    assert hamming_distance_64(0, 0) == 0
    assert hamming_distance_64(0, 1) == 1
    assert hamming_distance_64(0xFFFFFFFFFFFFFFFF, 0) == 64


_GOLDEN_RNG_SEED = 1234
_PINNED_GOLDEN_HASH = 0xD0093D73ADCBD603


def test_compute_part_b_structured_frame_golden():
    """Pin resize filter, DCT norm, scan order, and bit-endianness.

    A *structured* input is used so any change to the recipe (different
    resize, different DCT norm, different median behavior, different bit
    endianness) shifts the hash. An all-zero input would NOT catch
    those changes — every pHash recipe produces 0 on uniform input.

    If this hash changes, persisted manifest entries become incompatible
    — bump a schema version when the change is intentional.
    """
    from peeklet.core.fingerprint import compute_part_b

    rng = np.random.default_rng(seed=_GOLDEN_RNG_SEED)
    frame = rng.integers(0, 256, (1000, 1600, 3), dtype=np.uint8)
    assert compute_part_b(frame) == _PINNED_GOLDEN_HASH


def test_compute_fingerprint_combines_parts():
    from peeklet.core.fingerprint import Fingerprint, compute_fingerprint

    h, w = 1000, 1600
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    boxes = [
        WB("https://app.fathom.video/calls/1", x=200, y=20, w=400, h=18),
        WB("Dashboard", x=300, y=120, w=600, h=48),
        WB("Home", x=20, y=200, w=80, h=20),
    ]
    fp = compute_fingerprint(frame, boxes)
    assert isinstance(fp, Fingerprint)
    assert fp.url == "https://app.fathom.video/calls/1"
    assert fp.heading == "dashboard"
    assert fp.sidebar_text == "home"
    assert isinstance(fp.header_phash, int)


def test_is_match_same_part_a_close_phash_returns_true():
    from peeklet.core.fingerprint import Fingerprint, is_match

    a = Fingerprint(url="u", heading="h", sidebar_text="s", header_phash=0)
    b = Fingerprint(url="u", heading="h", sidebar_text="s", header_phash=0b111111)  # 6 bits
    assert is_match(a, b, phash_threshold=6) is True


def test_is_match_same_part_a_distant_phash_returns_false():
    from peeklet.core.fingerprint import Fingerprint, is_match

    a = Fingerprint(url="u", heading="h", sidebar_text="s", header_phash=0)
    b = Fingerprint(url="u", heading="h", sidebar_text="s", header_phash=0xFF)  # 8 bits
    assert is_match(a, b, phash_threshold=6) is False


def test_is_match_different_part_a_close_phash_returns_false():
    from peeklet.core.fingerprint import Fingerprint, is_match

    a = Fingerprint(url="u", heading="h", sidebar_text="s", header_phash=0)
    b = Fingerprint(url="u2", heading="h", sidebar_text="s", header_phash=0)
    assert is_match(a, b, phash_threshold=6) is False


def test_part_a_is_empty_returns_true_when_all_three_below_min_chars():
    from peeklet.core.fingerprint import Fingerprint, part_a_is_empty

    fp = Fingerprint(url="", heading="", sidebar_text="a", header_phash=0)
    assert part_a_is_empty(fp, min_chars=2) is True


def test_part_a_is_empty_returns_false_when_one_field_meets_min_chars():
    from peeklet.core.fingerprint import Fingerprint, part_a_is_empty

    fp = Fingerprint(url="", heading="dashboard", sidebar_text="", header_phash=0)
    assert part_a_is_empty(fp, min_chars=2) is False
