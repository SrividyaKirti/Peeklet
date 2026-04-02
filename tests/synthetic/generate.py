"""Generate synthetic screenshot sequences for testing.

Each sequence has known ground truth: which frames are keyframes and
what regions changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_idle_sequence(output_dir: Path, num_frames: int = 20) -> dict:
    """Frames where only the 'clock' region changes (top-right corner).
    Expected: frame 0 is keyframe, all others skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = {"keyframes": [0], "description": "Idle — only clock changes"}
    for i in range(num_frames):
        frame = _make_desktop_base()
        _draw_text(frame, f"12:{i:02d}", x=880, y=5, color=(200, 200, 200))
        Image.fromarray(frame).save(output_dir / f"frame_{i:03d}.png")
    _save_ground_truth(output_dir, ground_truth)
    return ground_truth


def generate_app_switch_sequence(output_dir: Path) -> dict:
    """Frames simulating an app switch at frame 5.
    Expected: frames 0 and 5 are keyframes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = {"keyframes": [0, 5], "description": "App switch at frame 5"}
    for i in range(10):
        if i < 5:
            frame = _make_desktop_base(bg_color=(40, 40, 60))
            _draw_text(frame, "Excel - Budget.xlsx", x=10, y=10, color=(255, 255, 255))
        else:
            frame = _make_desktop_base(bg_color=(255, 255, 255))
            _draw_text(frame, "Chrome - Google", x=10, y=10, color=(0, 0, 0))
        _draw_text(frame, f"12:{i:02d}", x=880, y=5, color=(200, 200, 200))
        Image.fromarray(frame).save(output_dir / f"frame_{i:03d}.png")
    _save_ground_truth(output_dir, ground_truth)
    return ground_truth


def generate_form_fill_sequence(output_dir: Path) -> dict:
    """Frames simulating typing in a form field.
    Expected: frames 0, 3, 6, 9, 12 are keyframes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    text_states = [
        "",
        "",
        "",
        "Joh",
        "Joh",
        "Joh",
        "John D",
        "John D",
        "John D",
        "John Doe",
        "John Doe",
        "John Doe",
        "John Doe, 42",
        "John Doe, 42",
        "John Doe, 42",
    ]
    keyframes = [0, 3, 6, 9, 12]
    ground_truth = {
        "keyframes": keyframes,
        "description": "Form fill — text appears every 3 frames",
    }
    for i, text in enumerate(text_states):
        frame = _make_desktop_base(bg_color=(245, 245, 245))
        _draw_text(frame, "Name:", x=50, y=100, color=(0, 0, 0))
        frame[130:160, 50:400] = [255, 255, 255]
        if text:
            _draw_text(frame, text, x=55, y=133, color=(0, 0, 0))
        Image.fromarray(frame).save(output_dir / f"frame_{i:03d}.png")
    _save_ground_truth(output_dir, ground_truth)
    return ground_truth


def generate_pip_video_sequence(output_dir: Path, num_frames: int = 20) -> dict:
    """Article with PiP video. Video changes every frame (noise).
    Expected: frame 0 is keyframe, rest skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = {"keyframes": [0], "description": "PiP video playing — should be masked"}
    rng = np.random.default_rng(42)
    for i in range(num_frames):
        frame = _make_desktop_base(bg_color=(250, 250, 250))
        for line in range(5):
            _draw_text(
                frame,
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                x=30,
                y=60 + line * 30,
                color=(30, 30, 30),
            )
        video_region = rng.integers(0, 256, (150, 200, 3), dtype=np.uint8)
        frame[450:600, 740:940] = video_region
        Image.fromarray(frame).save(output_dir / f"frame_{i:03d}.png")
    _save_ground_truth(output_dir, ground_truth)
    return ground_truth


def generate_all(output_dir: Path) -> None:
    """Generate all synthetic sequences."""
    generate_idle_sequence(output_dir / "idle")
    generate_app_switch_sequence(output_dir / "app_switch")
    generate_form_fill_sequence(output_dir / "form_fill")
    generate_pip_video_sequence(output_dir / "pip_video")


def _make_desktop_base(
    width: int = 960, height: int = 600, bg_color: tuple[int, int, int] = (50, 50, 70)
) -> np.ndarray:
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
    frame[570:600, :] = [30, 30, 30]  # taskbar
    return frame


def _draw_text(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, fill=color)
    frame[:] = np.asarray(img)


def _save_ground_truth(output_dir: Path, ground_truth: dict) -> None:
    (output_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/synthetic/output")
    generate_all(out)
    print(f"Generated synthetic sequences in {out}")
