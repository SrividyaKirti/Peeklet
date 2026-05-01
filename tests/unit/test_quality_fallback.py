"""Unit tests for the bounded quality-fallback capture helper."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


def _cfg():
    from peeklet.config import DemoFilterConfig

    return DemoFilterConfig(
        quality_fallback_max_attempts=8,
        quality_fallback_half_window_seconds=4.0,
        quality_fallback_step_seconds=1.0,
    )


def test_returns_t0_frame_when_initial_passes(monkeypatch):
    from peeklet.core.demo_filter import _quality_capture

    decoder = MagicMock()
    decoder.extract_frame_at.side_effect = lambda t: (
        np.full((100, 100, 3), 200, dtype=np.uint8),
        t,
        int(t * 30),
    )

    monkeypatch.setattr(
        "peeklet.core.demo_filter._is_low_info_frame",
        lambda frame, cfg: False,
    )

    result = _quality_capture(decoder, t=10.0, config=_cfg())
    assert result is not None
    frame, ts, fnum = result
    assert ts == 10.0
    decoder.extract_frame_at.assert_called_once_with(10.0)


def test_falls_back_to_neighbor_when_initial_fails(monkeypatch):
    from peeklet.core.demo_filter import _quality_capture

    decoder = MagicMock()
    decoder.extract_frame_at.side_effect = lambda t: (
        np.full((100, 100, 3), 200, dtype=np.uint8),
        t,
        int(t * 30),
    )

    # Initial timestamp fails, every subsequent call passes.
    call_state = {"n": 0}

    def low_info(_frame, _cfg):
        call_state["n"] += 1
        return call_state["n"] == 1

    monkeypatch.setattr("peeklet.core.demo_filter._is_low_info_frame", low_info)

    result = _quality_capture(decoder, t=10.0, config=_cfg())
    assert result is not None
    _frame, ts, _fnum = result
    # Order is +1, -1, +2, -2, ... so first fallback is t=11.
    assert ts == 11.0


def test_returns_none_after_all_attempts_fail(monkeypatch):
    from peeklet.core.demo_filter import _quality_capture

    decoder = MagicMock()
    decoder.extract_frame_at.side_effect = lambda t: (
        np.zeros((100, 100, 3), dtype=np.uint8),
        t,
        int(t * 30),
    )
    monkeypatch.setattr(
        "peeklet.core.demo_filter._is_low_info_frame",
        lambda frame, cfg: True,
    )

    assert _quality_capture(decoder, t=10.0, config=_cfg()) is None


def test_skips_attempts_below_zero(monkeypatch):
    """Don't request frames at negative timestamps; just skip those slots."""
    from peeklet.core.demo_filter import _quality_capture

    decoder = MagicMock()
    decoder.extract_frame_at.side_effect = lambda t: (
        np.full((100, 100, 3), 200, dtype=np.uint8),
        t,
        int(t * 30),
    )
    monkeypatch.setattr(
        "peeklet.core.demo_filter._is_low_info_frame",
        lambda frame, cfg: True,  # always fail to exhaust all attempts
    )

    _quality_capture(decoder, t=2.0, config=_cfg())
    requested = [c.args[0] for c in decoder.extract_frame_at.call_args_list]
    assert all(t >= 0 for t in requested), f"got negative timestamps: {requested}"
