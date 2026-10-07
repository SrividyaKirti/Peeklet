"""Tests for blocks, output files and Claude content rendering."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import numpy as np

from peeklet.image_utils import decode_jpeg, encode_jpeg
from peeklet.render import Entries, RunStats, build_entries, image_filenames, write_outputs
from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment

if TYPE_CHECKING:
    from pathlib import Path

JPEG = encode_jpeg(np.full((2000, 3000, 3), 100, dtype=np.uint8))


def scr(sid: str, t: float) -> Screen:
    return Screen(
        id=sid,
        image_jpeg=JPEG,
        frame_t=t,
        ocr_text="",
        word_count=1,
        first_change=1.0,
        occurrences=[(t, t + 1)],
    )


LINES = [
    Line(134.2, 137.0, "Look at the left.", "Alice"),
    Line(137.0, 140.0, "This blocks the bucket.", "Alice"),
    Line(147.0, 150.0, "Here is the remediation.", "Alice"),
    Line(302.0, 304.0, "Can you hear me?", "Bob"),
    Line(500.0, 503.0, "Back to the policy.", "Alice"),
]
SCREENS = ["S1", "S1", "S2", None, "S1"]


def test_image_filenames_use_whole_seconds_and_dedupe() -> None:
    files = image_filenames([scr("S1", 134.9), scr("S2", 134.2), scr("S3", 147.0)])
    # Sorted by frame time: S2 (134.2) claims frame_134.jpg first.
    assert files == {"S2": "frame_134.jpg", "S1": "frame_134_2.jpg", "S3": "frame_147.jpg"}


def test_blocks_break_on_screen_change_and_repeat_filename() -> None:
    files = {"S1": "frame_134.jpg", "S2": "frame_147.jpg"}
    judg = {"S1": ScreenJudgment(True, "r", "Policy page."), "S2": None}
    cps = [Checkpoint(148.0, "action_item", "Add team tag")]
    entries = build_entries(LINES, SCREENS, files, judg, cps)
    assert [e.to_dict() for e in entries] == [
        {
            "timestamp": "00:02:14",
            "transcript": "Alice: Look at the left.\nAlice: This blocks the bucket.",
            "image": "frame_134.jpg",
            "visual_context": "Policy page.",
        },
        {
            "timestamp": "00:02:27",
            "transcript": "Alice: Here is the remediation.",
            "image": "frame_147.jpg",
            "action_item": "Add team tag",
        },
        {"timestamp": "00:05:02", "transcript": "Bob: Can you hear me?", "image": None},
        {
            "timestamp": "00:08:20",
            "transcript": "Alice: Back to the policy.",
            "image": "frame_134.jpg",
            "visual_context": "Policy page.",
        },
    ]


def test_lines_on_dropped_screens_get_null_image() -> None:
    entries = build_entries(LINES[:2], ["S9", "S9"], {}, {}, [])
    assert [e.image for e in entries] == [None]


def test_write_outputs(tmp_path: Path) -> None:
    files = {"S1": "frame_134.jpg"}
    entries = build_entries(LINES[:1], ["S1"], files, {}, [])
    write_outputs(entries, [scr("S1", 134.0)], files, tmp_path)
    data = json.loads((tmp_path / "transcript.json").read_text())
    assert data[0]["image"] == "frame_134.jpg"
    assert (tmp_path / "frame_134.jpg").read_bytes() == JPEG


def test_claude_content_layout(tmp_path: Path) -> None:
    files = {"S1": "frame_134.jpg", "S2": "frame_147.jpg"}
    judg = {"S1": ScreenJudgment(True, "r", "Policy page."), "S2": None}
    entries = build_entries(
        LINES, SCREENS, files, judg, [Checkpoint(148.0, "action_item", "Add team tag")]
    )
    write_outputs(entries, [scr("S1", 134.0), scr("S2", 147.0)], files, tmp_path)
    blocks = Entries(entries, tmp_path, RunStats(5, 2, 2, 2, [])).to_claude_content()
    kinds = [b["type"] for b in blocks]
    assert kinds == ["text", "image", "text", "image", "text"]
    assert blocks[0]["text"].startswith("[00:02:14]\nAlice: Look at the left.")
    assert blocks[0]["text"].endswith("[SCREENSHOT @ 00:02:14]")
    assert blocks[2]["text"].startswith("Visual context: Policy page.")
    assert "[ACTION ITEM @ 00:02:27] Add team tag" in blocks[2]["text"]
    assert "[SCREENSHOT @ 00:08:20 — same screen as 00:02:14]" in blocks[4]["text"]
    img = decode_jpeg(base64.b64decode(blocks[1]["source"]["data"]))
    assert max(img.shape[:2]) == 1568
