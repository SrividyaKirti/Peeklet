"""Shared helpers for Peeklet evaluation scripts (not part of the package)."""

from __future__ import annotations

import os
import re
import urllib.request
from bisect import bisect_right
from pathlib import Path
from typing import Any

_CUE_RE = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")
_TAG_RE = re.compile(r"<[^>]+>")


def _secs(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _vtt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    return (
        f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d}.{ms % 1000:03d}"
    )


def clean_youtube_vtt(text: str) -> str:
    """Strip inline timing tags and YouTube's rolling duplicate lines; keep one line per cue."""
    out = ["WEBVTT", ""]
    last = ""
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        cue_idx = None
        match = None
        for i, ln in enumerate(lines):
            match = _CUE_RE.search(ln)
            if match:
                cue_idx = i
                break
        if match is None or cue_idx is None:
            continue
        start, end = _secs(match.group(1)), _secs(match.group(2))
        if end - start < 0.05:
            continue
        body = [_TAG_RE.sub("", ln).strip() for ln in lines[cue_idx + 1 :]]
        body = [b for b in body if b]
        new = body[-1] if body else ""
        if not new or new == last:
            continue
        last = new
        out += [f"{_vtt_ts(start)} --> {_vtt_ts(end)}", new, ""]
    return "\n".join(out)


def whisperx_to_vtt(items: list[dict[str, Any]]) -> str:
    """GUIDE transcript JSON ([{start, end, sentence}]) -> WebVTT."""
    out = ["WEBVTT", ""]
    for it in items:
        text = str(it.get("sentence", it.get("text", ""))).strip()
        if text:
            out += [f"{_vtt_ts(float(it['start']))} --> {_vtt_ts(float(it['end']))}", text, ""]
    return "\n".join(out)


def match_events(pred: list[float], truth: list[float], tolerance: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching; returns (true positives, false positives, false negatives)."""
    unused = sorted(pred)
    tp = 0
    for t in sorted(truth):
        best = min(unused, key=lambda p: abs(p - t), default=None)
        if best is not None and abs(best - t) <= tolerance:
            unused.remove(best)
            tp += 1
    return tp, len(pred) - tp, len(truth) - tp


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def segment_index(boundaries: list[float], t: float) -> int:
    """Index of the ground-truth segment containing t (boundaries are segment starts)."""
    return bisect_right(boundaries, t)


def change_times_from_debug(debug: dict[str, Any], min_t: float = 1.0) -> list[float]:
    """Predicted screen-change times: occurrence starts (excluding the video start), deduped."""
    starts = sorted(float(o[0]) for s in debug["screens"] for o in s["occurrences"])
    out: list[float] = []
    for t in starts:
        if t >= min_t and (not out or t - out[-1] > 0.5):
            out.append(t)
    return out


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def hf_token() -> str | None:
    """HF_TOKEN env var, else ~/.cache/huggingface/token."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    path = Path.home() / ".cache" / "huggingface" / "token"
    return path.read_text().strip() if path.is_file() else None


def download(url: str, dest: Path, headers: dict[str, str] | None = None) -> Path:
    """Download url to dest unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers=headers or {})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    return dest
