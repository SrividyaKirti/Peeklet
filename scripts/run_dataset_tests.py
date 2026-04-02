"""Run Peeklet pipeline on downloaded dataset and save results for the HTML explorer.

Usage:
    python scripts/run_dataset_tests.py [--output results/]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from peeklet.config import PeekletConfig
from peeklet.core.loader import load_frame
from peeklet.pipeline import Pipeline

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "tests" / "datasets" / "data" / "web_tasks"


def process_all_tasks(output_dir: Path) -> dict:
    """Process all downloaded tasks and save results."""
    index_file = DATA_DIR / "index.json"
    if not index_file.exists():
        print("Error: No dataset found. Run: python scripts/setup_test_data.py")
        return {}

    index = json.loads(index_file.read_text())
    task_ids = index["task_ids"]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": index["source"],
        "split": index["split"],
        "tasks": [],
        "total_frames": 0,
        "total_keyframes": 0,
        "total_time_ms": 0,
    }

    for task_id in task_ids:
        task_dir = DATA_DIR / task_id
        meta = json.loads((task_dir / "metadata.json").read_text())
        screenshots = sorted(task_dir.glob("step_*.png"))

        task_output = output_dir / task_id
        task_output.mkdir(parents=True, exist_ok=True)

        config = PeekletConfig.model_validate({
            "redactor": {"enabled": False},
            "exporter": {"output_dir": str(task_output)},
        })
        pipeline = Pipeline(config)

        task_results = []
        start = time.perf_counter()

        for i, img_path in enumerate(screenshots):
            frame = load_frame(img_path)
            result = pipeline.process_frame(
                frame,
                frame_id=f"step_{i:03d}",
                app_name=meta.get("website", ""),
                window_title=meta.get("confirmed_task", ""),
            )
            action = meta["actions"][i] if i < len(meta["actions"]) else None

            task_results.append({
                "frame_id": result.frame_id,
                "event_type": result.event_type.value,
                "is_keyframe": result.is_keyframe,
                "perceptual_hash": result.perceptual_hash,
                "ssim_score": result.ssim_score,
                "change_score": result.change_score,
                "changed_pct": result.changed_pct,
                "changed_regions": (
                    [r.to_dict() for r in result.changed_regions]
                    if result.changed_regions else None
                ),
                "adaptive_mask": (
                    [r.to_dict() for r in result.adaptive_mask]
                    if result.adaptive_mask else None
                ),
                "asset_path": result.asset_path,
                "action": action,
                "source_image": str(img_path.name),
            })

        pipeline.finalize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        keyframe_count = sum(1 for r in task_results if r["is_keyframe"])
        task_summary = {
            "task_id": task_id,
            "website": meta.get("website", ""),
            "domain": meta.get("domain", ""),
            "confirmed_task": meta.get("confirmed_task", ""),
            "num_frames": len(task_results),
            "num_keyframes": keyframe_count,
            "skip_rate": 1 - (keyframe_count / len(task_results)) if task_results else 0,
            "processing_time_ms": round(elapsed_ms, 1),
            "frames": task_results,
        }

        # Save per-task results JSON
        (task_output / "results.json").write_text(json.dumps(task_summary, indent=2))

        summary["tasks"].append({
            "task_id": task_id,
            "website": task_summary["website"],
            "domain": task_summary["domain"],
            "confirmed_task": task_summary["confirmed_task"],
            "num_frames": task_summary["num_frames"],
            "num_keyframes": task_summary["num_keyframes"],
            "skip_rate": task_summary["skip_rate"],
            "processing_time_ms": task_summary["processing_time_ms"],
        })
        summary["total_frames"] += len(task_results)
        summary["total_keyframes"] += keyframe_count
        summary["total_time_ms"] += round(elapsed_ms, 1)

        print(
            f"  {task_id}: {len(task_results)} frames, "
            f"{keyframe_count} keyframes, "
            f"{task_summary['skip_rate']:.0%} skip rate, "
            f"{elapsed_ms:.0f}ms"
        )

    # Save global summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    total_skip = 1 - (summary["total_keyframes"] / summary["total_frames"]) if summary["total_frames"] else 0
    print(f"\nTotal: {summary['total_frames']} frames, {summary['total_keyframes']} keyframes, {total_skip:.0%} skip rate")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Peeklet on dataset and save results")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()
    process_all_tasks(Path(args.output))


if __name__ == "__main__":
    main()
