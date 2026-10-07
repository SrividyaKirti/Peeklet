"""GUIDE evaluation: real software recordings with think-aloud narration.

Usage: uv run python -m eval.guide_eval --limit 5 [--llm]
Requires a Hugging Face token with access to kixlab/GuideBench (HF_TOKEN or
~/.cache/huggingface/token). Data (CC BY 4.0) goes to eval/data/guide/, never committed.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from eval.common import (
    change_times_from_debug,
    download,
    help_switch_recall,
    hf_token,
    near_duplicate_count,
    whisperx_to_vtt,
)
from peeklet import annotate

HF = "https://huggingface.co/datasets/kixlab/GuideBench/resolve/main"
DATA = Path("eval/data/guide")
RESULTS = Path("eval/results")


def gallery(out: Path, video_id: str) -> None:
    entries = json.loads((out / "transcript.json").read_text())
    rows = []
    for e in entries:
        img = (
            f'<img src="{html.escape(e["image"])}" width="640">'
            if e["image"]
            else "<em>no image</em>"
        )
        vc = f"<p><b>Visual context:</b> {html.escape(e.get('visual_context', ''))}</p>"
        text = html.escape(e["transcript"]).replace("\n", "<br>")
        rows.append(f"<section><h3>{e['timestamp']}</h3><p>{text}</p>{img}{vc}</section>")
    (out / "index.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>{video_id}</title>"
        "<style>body{font-family:sans-serif;max-width:900px;margin:auto}"
        "section{border-bottom:1px solid #ccc;padding:12px 0}</style>" + "".join(rows)
    )


def evaluate(
    video_id: str,
    url: str,
    headers: dict[str, str],
    help_items: list[dict[str, Any]],
    use_llm: bool,
) -> dict[str, Any]:
    vdir = DATA / video_id
    video = download(url, vdir / f"{video_id}.mp4")
    tx = download(
        f"{HF}/transcriptions/transcripts_{video_id}.json", vdir / "transcript.json", headers
    )
    vtt = vdir / "transcript.vtt"
    vtt.write_text(whisperx_to_vtt(json.loads(tx.read_text())))
    out = RESULTS / "guide" / video_id
    t0 = time.monotonic()
    entries = annotate(video, vtt, out, use_llm=use_llm, debug=True)
    runtime = time.monotonic() - t0
    debug = json.loads((out / "debug.json").read_text())
    lines = debug["lines"]
    duration_min = max(ln["end"] for ln in lines) / 60 if lines else 0.0
    unique_imgs = list({e.image: None for e in entries if e.image})
    segments = [
        (float(h["start_time"]), float(h["end_time"]))
        for h in help_items
        if h["video_id"] == video_id and h["type"] == "1_explicit"
    ]
    hits, total = help_switch_recall(change_times_from_debug(debug), segments)
    lines_with_image = sum(len(e.lines) for e in entries if e.image)
    line_image_coverage = round(lines_with_image / len(lines), 3) if lines else 0.0
    gallery(out, video_id)
    return {
        "video_id": video_id,
        "duration_min": round(duration_min, 1),
        "runtime_s": round(runtime, 1),
        "runtime_per_video_hour_s": (
            round(runtime / (duration_min / 60), 1) if duration_min else None
        ),
        "screen_pass_s": round(debug["timing"]["seconds_screen_pass"], 1),
        "lines": len(lines),
        "line_image_coverage": line_image_coverage,
        "screens_found": entries.stats.screens_found,
        "screens_per_min": (
            round(entries.stats.screens_found / duration_min, 2) if duration_min else None
        ),
        "kept": entries.stats.kept,
        "near_duplicate_kept": near_duplicate_count(
            [(out / name).read_bytes() for name in unique_imgs]
        ),
        "llm_include_rate": _include_rate(debug) if use_llm else None,
        "help_switch_hits": hits,
        "help_switch_total": total,
        "warnings": len(entries.stats.warnings),
        "gallery": str(out / "index.html"),
    }


def _include_rate(debug: dict[str, Any]) -> float | None:
    judged = [s["judgment"] for s in debug["screens"] if s["shortlisted"] and s["judgment"]]
    return round(sum(j["include"] for j in judged) / len(judged), 3) if judged else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--video-id", action="append", default=[])
    ap.add_argument("--llm", action="store_true", help="Run the LLM judge (costs money).")
    args = ap.parse_args()
    token = hf_token()
    if not token:
        print(
            "No Hugging Face token: set HF_TOKEN or run `hf auth login`, and accept the "
            "terms at https://huggingface.co/datasets/kixlab/GuideBench",
            file=sys.stderr,
        )
        return 2
    headers = {"Authorization": f"Bearer {token}"}
    urls_csv = download(f"{HF}/videos/video_urls.csv", DATA / "video_urls.csv", headers)
    help_json = download(
        f"{HF}/annotations/3_help_annotations.json", DATA / "3_help_annotations.json", headers
    )
    rows = list(csv.DictReader(urls_csv.open()))
    help_items = json.loads(help_json.read_text())
    if args.video_id:
        rows = [r for r in rows if r["video_id"] in set(args.video_id)]
    else:
        rows = sorted(rows, key=lambda r: r["video_id"])[: args.limit]
    results = []
    for r in rows:
        try:
            res = evaluate(r["video_id"], r["file_name"], headers, help_items, args.llm)
        except Exception as exc:
            print(f"skip {r['video_id']}: {exc}", file=sys.stderr)
            continue
        print(json.dumps(res))
        results.append(res)
    if not results:
        print("no videos evaluated", file=sys.stderr)
        return 1
    hits = sum(r["help_switch_hits"] for r in results)
    total = sum(r["help_switch_total"] for r in results)
    summary = {
        "videos": len(results),
        "mean_runtime_per_video_hour_s": round(
            sum(r["runtime_per_video_hour_s"] or 0 for r in results) / len(results), 1
        ),
        "mean_line_image_coverage": round(
            sum(r["line_image_coverage"] for r in results) / len(results), 3
        ),
        "mean_screens_per_min": round(
            sum(r["screens_per_min"] or 0 for r in results) / len(results), 2
        ),
        "help_switch_recall": round(hits / total, 3) if total else None,
    }
    report = {"date": str(date.today()), "llm": args.llm, "videos": results, "summary": summary}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"guide_{date.today()}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSUMMARY: {summary}\nreport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
