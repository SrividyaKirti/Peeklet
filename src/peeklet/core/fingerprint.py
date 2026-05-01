"""Content-addressable screen fingerprinting for demo-mode dedup.

Each unique screen captured by the demo pipeline is identified by a
two-part fingerprint:

* Part A — a normalized tuple of (url, heading, sidebar_text) extracted
  from OCR boxes in fixed regions of the frame. Tuple equality after
  normalization gates membership in a screen "bucket".
* Part B — a 64-bit DCT perceptual hash of the header strip
  (y in [0.08*H, 0.22*H]). Frames in the same Part-A bucket are
  considered the same screen iff their Part-B hashes are within
  ``phash_threshold`` Hamming distance.

See ``docs/superpowers/specs/2026-04-30-content-addressable-screen-dedup-design.md``.
"""

from __future__ import annotations

import re
from typing import Protocol

_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Lowercase, drop non-alphanumeric (except space), collapse whitespace.

    Used for the ``heading`` and ``sidebar_text`` fields of Part A. Empty
    or whitespace-only input returns an empty string.
    """
    if not text:
        return ""
    lowered = text.lower()
    stripped = _NON_ALNUM_RE.sub(" ", lowered)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()
    return collapsed


_URL_KEEP_RE = re.compile(r"[^a-z0-9/. :]+")


def _normalize_url(text: str) -> str:
    """Lowercase, keep ``/`` and ``.``, drop query/fragment, collapse whitespace.

    URLs need slashes and dots preserved so ``/path/to`` and ``example.com``
    survive normalization. Anything after ``?`` or ``#`` is discarded — query
    strings are noise for screen identity (timestamp params on Fathom URLs
    would otherwise prevent any URL match).
    """
    if not text:
        return ""
    lowered = text.lower().strip()
    cut = re.split(r"[?#]", lowered, maxsplit=1)[0]
    kept = _URL_KEEP_RE.sub(" ", cut)
    collapsed = _WHITESPACE_RE.sub(" ", kept).strip()
    return collapsed.replace(" ", "")


class _BoxLike(Protocol):
    """Minimal protocol for an OCR word-box used by Part-A extraction.

    Matches ``demo_filter.WordBox`` and any namedtuple with the same five
    attributes. Defining it here keeps fingerprint independent of the
    demo_filter module.
    """

    text: str
    x: int
    y: int
    w: int
    h: int


_TOP_BAND_FRACTION = 0.05
_HEADING_TOP_FRACTION = 0.5
_SIDEBAR_LEFT_FRACTION = 0.15
# Spec says "largest font cluster"; this is the operational definition —
# any box within 25% of the tallest box's height is "in the cluster".
# Adjustable without touching extraction logic.
_HEADING_FONT_CLUSTER_TOLERANCE = 0.75
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def extract_url(
    boxes: list[_BoxLike],
    frame_h: int,
    frame_w: int,
) -> str:
    """Return the first URL-pattern token whose box falls in the top 5% of the frame."""
    band = frame_h * _TOP_BAND_FRACTION
    for b in boxes:
        if b.y + b.h > band:
            continue
        match = _URL_RE.search(b.text)
        if match:
            return _normalize_url(match.group(0))
    return ""


def extract_heading(
    boxes: list[_BoxLike],
    frame_h: int,
    frame_w: int,
) -> str:
    """Pick the longest text token at the largest font size in the top half.

    "Largest font size" is approximated by box height. The largest
    height value found above the midline defines the cluster; among
    boxes within 25% of that max height, the one with the longest
    ``text`` wins.
    """
    midline = frame_h * _HEADING_TOP_FRACTION
    candidates = [b for b in boxes if b.y + b.h <= midline]
    if not candidates:
        return ""
    max_h = max(b.h for b in candidates)
    if max_h <= 0:
        return ""
    cluster = [b for b in candidates if b.h >= max_h * _HEADING_FONT_CLUSTER_TOLERANCE]
    cluster.sort(key=lambda b: (-len(b.text), b.x))
    return _normalize_text(cluster[0].text)


def extract_sidebar_text(
    boxes: list[_BoxLike],
    frame_h: int,
    frame_w: int,
) -> str:
    """Concatenate OCR text whose box is fully inside the left 15% of the frame."""
    cutoff = frame_w * _SIDEBAR_LEFT_FRACTION
    sidebar = [b for b in boxes if b.x + b.w <= cutoff]
    if not sidebar:
        return ""
    sidebar.sort(key=lambda b: (b.y, b.x))
    joined = " ".join(b.text for b in sidebar)
    return _normalize_text(joined)
