"""LPM evaluation: change detection and alignment vs hand-labelled slide transitions.

Usage: uv run --with yt-dlp python -m eval.lpm_eval --limit 10 --max-minutes 30
Data (CC BY-NC-SA 4.0; videos under YouTube terms) goes to eval/data/lpm/, never committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from eval.common import (
    change_times_from_debug,
    clean_youtube_vtt,
    download,
    ffmpeg_exe,
    match_events,
    prf,
    segment_index,
)
from peeklet import annotate

CSV_URL = "https://raw.githubusercontent.com/dondongwon/LPMDataset/main/dataset/raw_video_links.csv"
DATA = Path("eval/data/lpm")
RESULTS = Path("eval/results")


def pick_videos(rows: list[dict[str, str]], limit: int, max_minutes: float) -> list[dict[str, str]]:
    """Shortest eligible video per speaker first (diversity), then the next shortest."""
    eligible = [r for r in rows if float(r["seconds"]) <= max_minutes * 60 and r["youtube_url"]]
    eligible.sort(key=lambda r: float(r["seconds"]))
    picked, speakers = [], set()
    for r in eligible:
        if r["speaker"] not in speakers:
            picked.append(r)
            speakers.add(r["speaker"])
    for r in eligible:
        if r not in picked:
            picked.append(r)
    return picked[:limit]


def fetch(row: dict[str, str]) -> tuple[Path, Path] | None:
    vid = row["video_id"].removesuffix(".mp4")
    out = DATA / vid
    video = out / f"{vid}.mp4"
    common = ["yt-dlp", "--ffmpeg-location", ffmpeg_exe(), "-o", str(out / f"{vid}.%(ext)s")]
    if not video.exists():
        subprocess.run(
            [
                *common,
                "-f",
                "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
                "--merge-output-format",
                "mp4",
                row["youtube_url"],
            ],
            check=False,
        )
    subs = sorted(out.glob("*.vtt"))
    if not subs:
        subprocess.run(
            [
                *common,
                "--skip-download",
                "--write-subs",
                "--sub-langs",
                "en.*",
                "--sub-format",
                "vtt",
                row["youtube_url"],
            ],
            check=False,
        )
        subs = sorted(out.glob("*.vtt"))
    if not subs:
        subprocess.run(
            [
                *common,
                "--skip-download",
                "--write-auto-subs",
                "--sub-langs",
                "en",
                "--sub-format",
                "vtt",
                row["youtube_url"],
            ],
            check=False,
        )
        subs = sorted(out.glob("*.vtt"))
    if not video.exists() or not subs:
        return None
    clean = out / "clean.vtt"
    clean.write_text(clean_youtube_vtt(subs[0].read_text(encoding="utf-8")))
    return video, clean


def evaluate(row: dict[str, str], video: Path, vtt: Path, tolerance: float) -> dict[str, Any]:
    duration = float(row["seconds"])
    raw = [float(x) for x in row["Answer.startTimeList"].split("|")[1:] if x.strip()]
    truth = sorted({round(t, 2) for t in raw if 0.0 < t < duration})
    out = RESULTS / "lpm" / video.stem
    t0 = time.monotonic()
    annotate(video, vtt, out, max_images=10_000, use_llm=False, debug=True)
    runtime = time.monotonic() - t0
    debug = json.loads((out / "debug.json").read_text())
    tp, fp, fn = match_events(change_times_from_debug(debug), truth, tolerance)
    p, r, f = prf(tp, fp, fn)
    frame_t = {s["id"]: s["frame_t"] for s in debug["screens"]}
    lines = debug["lines"]
    with_screen = [ln for ln in lines if ln["screen_id"]]
    correct = sum(
        segment_index(truth, (ln["start"] + ln["end"]) / 2)
        == segment_index(truth, frame_t[ln["screen_id"]])
        for ln in with_screen
    )
    return {
        "video_id": video.stem,
        "speaker": row["speaker"],
        "duration_s": duration,
        "runtime_s": round(runtime, 1),
        "truth_changes": len(truth),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f, 3),
        "line_coverage": round(len(with_screen) / len(lines), 3) if lines else 0.0,
        "line_screen_accuracy": round(correct / len(with_screen), 3) if with_screen else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--tolerance", type=float, default=2.0)
    args = ap.parse_args()
    csv_path = download(CSV_URL, DATA / "raw_video_links.csv")
    rows = list(csv.DictReader(csv_path.open()))
    results = []
    for row in pick_videos(rows, args.limit, args.max_minutes):
        got = fetch(row)
        if got is None:
            print(f"skip {row['video_id']}: download or captions unavailable", file=sys.stderr)
            continue
        try:
            res = evaluate(row, *got, tolerance=args.tolerance)
        except Exception as exc:
            print(f"skip {row['video_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(json.dumps(res))
        results.append(res)
    if not results:
        print("no videos evaluated", file=sys.stderr)
        return 1
    mean = {
        k: round(sum(r[k] for r in results) / len(results), 3)
        for k in ("precision", "recall", "f1", "line_coverage", "line_screen_accuracy")
    }
    report = {
        "date": str(date.today()),
        "tolerance_s": args.tolerance,
        "videos": results,
        "mean": mean,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"lpm_{date.today()}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nMEAN over {len(results)} videos: {mean}\nreport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
