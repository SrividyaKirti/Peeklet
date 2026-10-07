"""Screen analysis: OCR word boxes, the low-information frame rejector, and fingerprinting."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peeklet.config import DemoFilterConfig

logger = logging.getLogger(__name__)

# Imported lazily so test files can patch ``screen.pytesseract``.
try:
    import pytesseract
except ImportError:  # pragma: no cover - exercised in install-error path
    pytesseract = None

_MIN_WORD_LENGTH = 2
_MIN_WORD_CONFIDENCE = 30


class WordBox(NamedTuple):
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int


def _downscale_for_ocr(frame: np.ndarray, downscale_dim: int) -> np.ndarray:
    """Resize ``frame`` so its longest edge equals ``downscale_dim``.

    Returns the original frame if it is already smaller. Uses Pillow for
    a high-quality resize without pulling in extra dependencies.
    """
    from PIL import Image

    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= downscale_dim:
        return frame
    scale = downscale_dim / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.asarray(img)


def _ocr_word_boxes(frame: np.ndarray, downscale_dim: int) -> list[WordBox]:
    """Run Tesseract once and return accepted word boxes.

    A word is accepted if its confidence >= ``_MIN_WORD_CONFIDENCE`` and its
    stripped text length >= ``_MIN_WORD_LENGTH``. Coordinates are in the
    downscaled frame's pixel space so all layout signals share one
    coordinate system from a single OCR call.
    """
    if pytesseract is None:
        return []

    downscaled = _downscale_for_ocr(frame, downscale_dim)
    try:
        data = pytesseract.image_to_data(downscaled, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        logger.warning("OCR failed on frame: %s", exc)
        return []

    texts = data.get("text", [])
    confs = data.get("conf", [])
    n = len(texts)
    # Geometry fields may be absent in mocked OCR responses; default to zeros
    # so legacy callers that only populate text+conf still produce word counts.
    lefts = data.get("left") or [0] * n
    tops = data.get("top") or [0] * n
    widths = data.get("width") or [0] * n
    heights = data.get("height") or [0] * n
    boxes: list[WordBox] = []
    for text, conf, x, y, w, h in zip(texts, confs, lefts, tops, widths, heights, strict=False):
        if not text or len(text.strip()) < _MIN_WORD_LENGTH:
            continue
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_val < _MIN_WORD_CONFIDENCE:
            continue
        boxes.append(
            WordBox(
                text=text,
                conf=conf_val,
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
            )
        )
    return boxes


def _count_text_lines(boxes: list[WordBox]) -> int:
    """Cluster word boxes by y-center into distinct text lines.

    Uses a tolerance of half the median box height so a single typographic
    row stays one line even when words have minor y jitter from Tesseract.
    """
    if not boxes:
        return 0
    median_h = float(np.median([b.h for b in boxes]))
    tolerance = max(1.0, median_h / 2.0)
    centers = sorted(b.y + b.h / 2.0 for b in boxes)
    lines = 1
    current = centers[0]
    for c in centers[1:]:
        if c - current > tolerance:
            lines += 1
            current = c
    return lines


_GRID_DIM = 8


def _count_occupied_grid_cells(boxes: list[WordBox], frame_shape: tuple[int, ...]) -> int:
    """Count distinct cells in an 8x8 grid that contain a word-box center.

    Frame shape follows numpy convention (H, W, ...). Out-of-bounds centers
    are clamped to the nearest edge cell so an OCR box that extends beyond
    the downscaled frame still counts once.
    """
    if not boxes:
        return 0
    h, w = frame_shape[:2]
    cell_h = max(1, h // _GRID_DIM)
    cell_w = max(1, w // _GRID_DIM)
    occupied: set[tuple[int, int]] = set()
    for b in boxes:
        cx = b.x + b.w / 2.0
        cy = b.y + b.h / 2.0
        col = min(_GRID_DIM - 1, max(0, int(cx // cell_w)))
        row = min(_GRID_DIM - 1, max(0, int(cy // cell_h)))
        occupied.add((row, col))
    return len(occupied)


_EDGE_MAGNITUDE_THRESHOLD = 30.0


def _edge_pixel_ratio(frame: np.ndarray) -> float:
    """Fraction of pixels whose |∂x|+|∂y| gradient exceeds a fixed threshold.

    Uses ``np.gradient`` on the grayscale frame — numpy-only, no cv2
    dependency. Gallery views (flat colored tiles) land near zero;
    dashboard UIs with chrome sit well above 0.02.
    """
    if frame.ndim == 3:
        gray = (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(
            np.float32
        )
    else:
        gray = frame.astype(np.float32)
    gy, gx = np.gradient(gray)
    mag = np.abs(gx) + np.abs(gy)
    edge_pixels = int((mag > _EDGE_MAGNITUDE_THRESHOLD).sum())
    total = gray.size
    return edge_pixels / total if total else 0.0


def _is_low_info_frame(frame: np.ndarray, config: DemoFilterConfig) -> bool:
    """Composite low-information rejector (triple-AND).

    Rejects a frame only if **all three** independent signals fall under
    their thresholds. A legit minimalist UI will pass on at least one axis.
    """
    boxes = _ocr_word_boxes(frame, config.gallery_ocr_min_dim)
    num_lines = _count_text_lines(boxes)
    num_cells = _count_occupied_grid_cells(boxes, frame.shape)
    edge_ratio = _edge_pixel_ratio(frame)
    if edge_ratio >= config.min_edge_ratio:
        return False
    if num_lines >= config.min_text_lines:
        return False
    if num_cells >= config.min_grid_cells:
        return False
    logger.info(
        "Low-info frame rejected: lines=%d cells=%d edge_ratio=%.4f",
        num_lines,
        num_cells,
        edge_ratio,
    )
    return True


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

    Matches ``screen.WordBox`` and any namedtuple with the same five
    attributes. Defining it here keeps fingerprinting independent of the
    OCR code above.
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


def _to_grayscale_float(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    return (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(
        np.float32
    )


def compute_part_b(frame: np.ndarray) -> int:
    """Compute a 64-bit DCT pHash of the header strip.

    Steps:
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
    gray = _to_grayscale_float(strip)
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

    The dedup logic uses this to bypass collapse on frames where OCR
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
