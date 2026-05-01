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
