"""Test helpers for generating tiny synthetic videos."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def write_synthetic_video(
    path: Path,
    duration_sec: int = 5,
    fps: int = 10,
    width: int = 64,
    height: int = 64,
) -> Path:
    """Write a tiny solid-color video that changes color every second.

    Used by demo-filter and video tests. Color cycles through R/G/B/Y/M
    so each second is visually distinct.
    """
    import imageio.v3 as iio

    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    frames = []
    for sec in range(duration_sec):
        color = palette[sec % len(palette)]
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        for _ in range(fps):
            frames.append(frame)
    iio.imwrite(path, np.stack(frames), plugin="pyav", fps=fps, codec="libx264")
    return path
