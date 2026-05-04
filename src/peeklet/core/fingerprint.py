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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

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
_HEADER_STRIP_Y_RANGE = (0.08, 0.22)
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def extract_url(
    boxes: Sequence[_BoxLike],
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
    boxes: Sequence[_BoxLike],
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
    boxes: Sequence[_BoxLike],
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


def hamming_distance_64(a: int, b: int) -> int:
    """Number of differing bits between two non-negative 64-bit ints."""
    return (a ^ b).bit_count()


def _crop_header_strip(frame: np.ndarray) -> np.ndarray:
    """Crop the [8% .. 22%] vertical band, full width."""
    h = frame.shape[0]
    y0 = int(round(h * _HEADER_STRIP_Y_RANGE[0]))
    y1 = max(y0 + 1, int(round(h * _HEADER_STRIP_Y_RANGE[1])))
    return frame[y0:y1, :]


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    return (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(
        np.float32
    )


def compute_part_b(frame: np.ndarray) -> int:
    """Compute a 64-bit DCT pHash of the header strip.

    Pipeline:
      1. Crop y∈[8%..22%] full width.
      2. Convert to grayscale.
      3. Resize to 32×32 via PIL (BILINEAR).
      4. Type-II DCT along both axes (scipy.fft).
      5. Take the top-left 8×8 low-frequency block.
      6. Use the median of the 63 non-DC coefficients as the threshold;
         bits above the median are 1, below are 0. The DC coefficient
         is excluded from the median (it dominates and would skew the
         threshold).

    The DC drop + median follows the standard pHash recipe and makes the
    hash invariant to overall brightness shifts.
    """
    from PIL import Image
    from scipy.fft import dct  # type: ignore[import-untyped]

    strip = _crop_header_strip(frame)
    if strip.size == 0:
        return 0
    gray = _to_grayscale(strip)
    img = Image.fromarray(gray.astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    coeffs = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
    block = coeffs[:8, :8]
    flat = block.ravel()
    non_dc = flat[1:]  # 63 elements, drops DC at [0,0]
    threshold = float(np.median(non_dc))
    bits = (flat > threshold).astype(np.uint8)
    packed = np.packbits(bits, bitorder="big")
    return int.from_bytes(packed.tobytes(), "big")


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """A screen's content-addressable identity.

    Part A = (url, heading, sidebar_text), all normalized. Part B is the
    header-strip 64-bit DCT pHash. Two fingerprints are the same screen
    iff Part A is equal AND Hamming(Part B) ≤ threshold.
    """

    url: str
    heading: str
    sidebar_text: str
    header_phash: int

    @property
    def part_a(self) -> tuple[str, str, str]:
        return (self.url, self.heading, self.sidebar_text)


def compute_fingerprint(frame: np.ndarray, boxes: Sequence[_BoxLike]) -> Fingerprint:
    """Compute Part A + Part B for ``frame`` given its OCR word boxes."""
    h, w = frame.shape[:2]
    return Fingerprint(
        url=extract_url(boxes, frame_h=h, frame_w=w),
        heading=extract_heading(boxes, frame_h=h, frame_w=w),
        sidebar_text=extract_sidebar_text(boxes, frame_h=h, frame_w=w),
        header_phash=compute_part_b(frame),
    )


def is_match(a: Fingerprint, b: Fingerprint, phash_threshold: int) -> bool:
    """True iff Part A is equal and Part B is within ``phash_threshold`` bits."""
    if a.part_a != b.part_a:
        return False
    return hamming_distance_64(a.header_phash, b.header_phash) <= phash_threshold


def part_a_is_empty(fp: Fingerprint, min_chars: int) -> bool:
    """True if every Part A field is shorter than ``min_chars``.

    The dedup pipeline uses this to bypass collapse on frames where OCR
    failed completely — otherwise unrelated unreadable frames would all
    cluster under the same empty key and false-collapse.
    """
    return all(len(field) < min_chars for field in fp.part_a)


class FingerprintIndex:
    """Hash-map screen lookup keyed on normalized Part A.

    Each bucket holds (header_phash, screen_id) tuples. Lookup scans the
    bucket linearly; in practice a bucket holds 1–3 entries because Part
    A pins identity tightly and Part B only resolves ambiguity within
    the bucket.
    """

    def __init__(self, *, phash_threshold: int, ocr_field_min_chars: int) -> None:
        self._buckets: dict[tuple[str, str, str], list[tuple[int, str]]] = {}
        self._phash_threshold = phash_threshold
        self._min_chars = ocr_field_min_chars

    def lookup(self, fp: Fingerprint) -> str | None:
        """Return the screen_id of an existing matching screen, or None.

        Always misses when Part A is empty — see ``part_a_is_empty``.
        """
        if part_a_is_empty(fp, self._min_chars):
            return None
        bucket = self._buckets.get(fp.part_a)
        if not bucket:
            return None
        for phash, screen_id in bucket:
            if hamming_distance_64(phash, fp.header_phash) <= self._phash_threshold:
                return screen_id
        return None

    def register(self, fp: Fingerprint, *, screen_id: str) -> None:
        """Append ``(phash, screen_id)`` under the Part A key.

        Empty-Part-A entries are still registered so that callers using
        ``register`` directly observe a coherent state — the safety
        bypass lives in ``lookup``, not here.
        """
        self._buckets.setdefault(fp.part_a, []).append((fp.header_phash, screen_id))
