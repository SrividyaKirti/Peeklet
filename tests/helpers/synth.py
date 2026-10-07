"""Synthetic screen-share demo: A (10s) -> B (10s) -> gallery (5s) -> A (15s)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from pathlib import Path

W, H, FPS = 1280, 720, 2


def _screen(heading: str, body_prefix: str, background: tuple[int, int, int]) -> np.ndarray:
    img = Image.new("RGB", (W, H), background)
    d = ImageDraw.Draw(img)
    big = ImageFont.load_default(size=48)
    small = ImageFont.load_default(size=22)
    d.text((220, 90), heading, fill="black", font=big)
    for i, item in enumerate(["Home", "Reports", "Settings", "Billing"]):
        d.text((20, 160 + 40 * i), item, fill="black", font=small)
    for i in range(12):
        d.text(
            (220, 180 + 34 * i),
            f"{body_prefix} row {i + 1}: value {i * 17}",
            fill="black",
            font=small,
        )
    return np.asarray(img)


def _gallery() -> np.ndarray:
    img = Image.new("RGB", (W, H), (40, 40, 40))
    d = ImageDraw.Draw(img)
    for i, colour in enumerate([(90, 60, 60), (60, 90, 60), (60, 60, 90), (90, 90, 60)]):
        x, y = (i % 2) * (W // 2), (i // 2) * (H // 2)
        d.rectangle([x + 10, y + 10, x + W // 2 - 10, y + H // 2 - 10], fill=colour)
    return np.asarray(img)


def _with_webcam(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = frame.copy()
    out[H - 82 : H - 10, W - 106 : W - 10] = rng.integers(0, 255, (72, 96, 3), dtype=np.uint8)
    return out


def write_demo(video_path: Path, vtt_path: Path) -> None:
    import imageio.v3 as iio

    rng = np.random.default_rng(7)
    a = _screen("Revenue Dashboard", "Revenue", (255, 255, 255))
    b = _screen("Billing Settings", "Invoice", (236, 236, 236))  # tint lets stub OCR tell A/B apart
    g = _gallery()
    plan = [(a, 10), (b, 10), (g, 5), (a, 15)]
    frames = [_with_webcam(f, rng) for f, secs in plan for _ in range(secs * FPS)]
    iio.imwrite(video_path, np.stack(frames), plugin="pyav", fps=FPS, codec="libx264")
    vtt_path.write_text(
        "WEBVTT\n\n"
        "00:00:00.500 --> 00:00:04.000\nWelcome, this is the revenue dashboard.\n\n"
        "00:00:11.000 --> 00:00:15.000\nNow look at the billing settings page.\n\n"
        "00:00:21.000 --> 00:00:23.000\nCan everyone hear me okay?\n\n"
        "00:00:27.000 --> 00:00:31.000\nBack on the dashboard, notice the totals.\n"
    )
