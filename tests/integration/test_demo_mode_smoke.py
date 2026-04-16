"""Integration smoke test for demo-mode dedup correctness.

Synthetic 30-second 3-scene scenario. Asserts:
1. No two saved frames share a dHash (unique dedup).
2. An anchor colliding with a prior distinct frame merges, not saves.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from peeklet.config import DemoFilterConfig
from peeklet.core import demo_filter
from peeklet.core.audio import TranscriptSegment
from peeklet.utils.image import dhash_64
from peeklet.utils.types import Moment

_SCENE_PARAMS = {
    "A": (42, 20),  # (rng_seed, base_brightness)
    "B": (99, 100),
    "C": (7, 180),
}


def _make_scene_frame(seed: int, base: int) -> np.ndarray:
    """Deterministic seeded-noise frame. Non-uniform pixels give a non-trivial dHash."""
    h, w = 120, 160
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 60, (h, w), dtype=np.uint8)
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    return gray[:, :, np.newaxis].repeat(3, axis=2)


class _FakeDecoder:
    """Three-scene synthetic decoder: A for 0-10s, B for 10-20s, C for 20-30s.

    Uses seeded-noise frames per scene so dHash is unique across scenes while
    remaining constant within a scene (same frame returned for any ts in that range).
    """

    def __init__(self):
        self.duration = 30.0
        self._frames = {
            "A": _make_scene_frame(*_SCENE_PARAMS["A"]),
            "B": _make_scene_frame(*_SCENE_PARAMS["B"]),
            "C": _make_scene_frame(*_SCENE_PARAMS["C"]),
        }

    def get_metadata(self):
        return SimpleNamespace(filename="synthetic.mp4", duration=self.duration)

    def extract_frame_at(self, ts):
        if ts < 10.0:
            frame = self._frames["A"]
        elif ts < 20.0:
            frame = self._frames["B"]
        else:
            frame = self._frames["C"]
        return frame, ts, int(ts * 30)


def _patch_ocr(monkeypatch):
    """Token-returning stub keyed off pixel value so each scene has unique tokens."""

    def fake_ocr(frame, dim):
        v = int(frame[0, 0, 0])
        tokens = {f"scene_{v}_token_{i}" for i in range(8)}
        return " ".join(sorted(tokens)), tokens

    monkeypatch.setattr(demo_filter, "_is_low_info_frame", lambda f, c: False)
    monkeypatch.setattr(demo_filter, "_ocr_text_and_tokens", fake_ocr)


@pytest.mark.smoke
class TestDemoModeSmoke:
    def test_no_duplicate_dhashes(self, tmp_path, monkeypatch):
        _patch_ocr(monkeypatch)

        seg = TranscriptSegment(start=0.0, end=30.0, text="...", speaker=None)
        moments = [
            Moment(
                timestamp=2.0,
                visual_context_goal="A1",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
            Moment(
                timestamp=5.0,
                visual_context_goal="A2",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
            Moment(
                timestamp=12.0,
                visual_context_goal="B1",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
            Moment(
                timestamp=22.0,
                visual_context_goal="C1",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
        ]
        cfg = DemoFilterConfig(
            # Disable gap-fill and lookback so each moment's search window stays
            # within its own scene. This isolates the Jaccard dedup from gap-fill
            # and cross-scene lookback interactions, making assertions deterministic.
            max_seconds_between_keyframes=0.0,
            search_window_lookback_sec=0.0,
            dedup_jaccard_threshold=0.95,
            min_ocr_tokens_for_jaccard=5,
        )

        results = demo_filter.select_frames_for_moments(
            decoder=_FakeDecoder(),
            moments=moments,
            transcript=[seg],
            config=cfg,
            output_dir=tmp_path,
        )

        # Verify unique dHashes (no dedup bypass)
        from PIL import Image

        hashes = []
        for r in results:
            img = np.asarray(Image.open(r.asset_path))
            hashes.append(dhash_64(img))
        assert len(set(hashes)) == len(hashes), f"Duplicate dHashes found in {len(results)} results"

        # A2 should dedup against A1 (same scene tokens). Expect 3 unique
        # scene frames (A, B, C), not 4. Gap-fill disabled so no extra frames.
        assert len(results) == 3

    def test_anchor_merge_not_double_save(self, tmp_path, monkeypatch):
        _patch_ocr(monkeypatch)

        seg = TranscriptSegment(start=0.0, end=30.0, text="...", speaker=None)
        moments = [
            Moment(
                timestamp=2.0,
                visual_context_goal="A1 llm",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
            Moment(
                timestamp=3.0,
                visual_context_goal="A1 anchor",
                textual_anchor="we'll track this",
                downstream_utility="",
                source="anchor",
            ),
        ]
        cfg = DemoFilterConfig(max_seconds_between_keyframes=0.0)

        results = demo_filter.select_frames_for_moments(
            decoder=_FakeDecoder(),
            moments=moments,
            transcript=[seg],
            config=cfg,
            output_dir=tmp_path,
        )

        assert len(results) == 1
        assert len(results[0].anchors) == 1
        assert results[0].anchors[0].textual_anchor == "we'll track this"

    def test_schema_version_in_context_json(self, tmp_path, monkeypatch):
        _patch_ocr(monkeypatch)

        seg = TranscriptSegment(start=0.0, end=30.0, text="test", speaker="Vidya")
        moments = [
            Moment(
                timestamp=5.0,
                visual_context_goal="A",
                textual_anchor="",
                downstream_utility="",
                source="llm",
            ),
        ]
        cfg = DemoFilterConfig(max_seconds_between_keyframes=0.0)

        results = demo_filter.select_frames_for_moments(
            decoder=_FakeDecoder(),
            moments=moments,
            transcript=[seg],
            config=cfg,
            output_dir=tmp_path,
        )

        from peeklet.core.context_exporter import build_context

        ctx = build_context(
            filename="synthetic.mp4",
            duration_s=30.0,
            results=results,
            segments=[seg],
        )
        assert ctx["schema_version"] == 2
