"""Blocks, output files, Claude content blocks and the debug report."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from peeklet.image_utils import decode_jpeg, encode_jpeg
from peeklet.types import Entry, format_hms

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from peeklet.types import Checkpoint, Line, Screen, ScreenJudgment


def image_filenames(screens: Sequence[Screen]) -> dict[str, str]:
    """frame_<whole seconds>.jpg per screen; suffix _2, _3... on collisions."""
    files: dict[str, str] = {}
    taken: set[str] = set()
    for s in sorted(screens, key=lambda s: (s.frame_t, s.id)):
        stem = f"frame_{int(s.frame_t)}"
        name, n = f"{stem}.jpg", 2
        while name in taken:
            name, n = f"{stem}_{n}.jpg", n + 1
        taken.add(name)
        files[s.id] = name
    return files


def build_entries(
    lines: Sequence[Line],
    line_screens: Sequence[str | None],
    files: Mapping[str, str],
    judgments: Mapping[str, ScreenJudgment | None],
    action_items: Sequence[Checkpoint],
) -> list[Entry]:
    """Group consecutive lines on the same kept screen (or none) into entries."""
    labels: dict[int, list[str]] = defaultdict(list)
    for cp in action_items:
        idx = max((i for i, ln in enumerate(lines) if ln.start <= cp.t), default=0)
        labels[idx].append(cp.label)

    entries: list[Entry] = []
    block: list[int] = []
    block_key: str | None = None

    def flush() -> None:
        if not block:
            return
        judgment = judgments.get(block_key) if block_key else None
        entries.append(
            Entry(
                timestamp=lines[block[0]].start,
                lines=tuple(lines[i] for i in block),
                image=files.get(block_key) if block_key else None,
                visual_context=judgment.visual_context if judgment else None,
                action_items=tuple(lab for i in block for lab in labels.get(i, [])),
                screen_id=block_key,
            )
        )

    for i, sid in enumerate(line_screens):
        key = sid if sid is not None and sid in files else None
        if block and key != block_key:
            flush()
            block = []
        block_key = key
        block.append(i)
    flush()
    return entries


@dataclass(slots=True)
class RunStats:
    lines: int
    screens_found: int
    shortlisted: int
    kept: int
    warnings: list[str]


@dataclass(slots=True)
class Entries:
    """The annotated transcript plus where its images live."""

    items: list[Entry]
    out_dir: Path
    stats: RunStats
    max_image_edge: int = 1568

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Entry:
        return self.items[i]

    def to_claude_content(self) -> list[dict[str, Any]]:
        """Anthropic content blocks: text, then each new screenshot once."""
        blocks: list[dict[str, Any]] = []
        buf: list[str] = []
        seen: dict[str, str] = {}

        def flush_text() -> None:
            if buf:
                blocks.append({"type": "text", "text": "\n\n".join(buf)})
                buf.clear()

        for e in self.items:
            hms = format_hms(e.timestamp)
            buf.append(f"[{hms}]\n" + e.to_dict()["transcript"])
            buf.extend(f"[ACTION ITEM @ {hms}] {label}" for label in e.action_items)
            if not e.image or e.screen_id is None:
                continue
            if e.screen_id in seen:
                buf.append(f"[SCREENSHOT @ {hms} — same screen as {seen[e.screen_id]}]")
                continue
            buf.append(f"[SCREENSHOT @ {hms}]")
            flush_text()
            raw = (self.out_dir / e.image).read_bytes()
            small = encode_jpeg(decode_jpeg(raw), quality=90, max_edge=self.max_image_edge)
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(small).decode("ascii"),
                    },
                }
            )
            if e.visual_context:
                buf.append(f"Visual context: {e.visual_context}")
            seen[e.screen_id] = hms
        flush_text()
        return blocks


def write_outputs(
    entries: list[Entry], kept: Sequence[Screen], files: Mapping[str, str], out_dir: Path
) -> None:
    """Write transcript.json and the chosen frame of every kept screen."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.json").write_text(
        json.dumps([e.to_dict() for e in entries], indent=2, ensure_ascii=False) + "\n"
    )
    for s in kept:
        (out_dir / files[s.id]).write_bytes(s.image_jpeg)


def write_debug(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
