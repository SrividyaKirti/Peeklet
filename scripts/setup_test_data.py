"""Download and prepare Mind2Web dataset for Peeklet testing.

Usage:
    python scripts/setup_test_data.py [--num-tasks 10] [--split test_task]

Downloads screenshot sequences from Hugging Face's Multimodal-Mind2Web
dataset and organizes them into tests/datasets/data/web_tasks/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "datasets" / "data" / "web_tasks"
HF_DATASET = "osunlp/Multimodal-Mind2Web"


def download_tasks(num_tasks: int = 10, split: str = "test_task") -> None:
    """Download Mind2Web tasks and save as screenshot sequences."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. Install with:")
        print("  pip install 'peeklet[datasets]'")
        sys.exit(1)

    print(f"Streaming {split} split from {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split=split, streaming=True)

    # Group rows by annotation_id (each row is one action step)
    # Streaming mode: iterate and collect only until we have enough tasks
    tasks: dict[str, list] = {}
    for row in ds:
        task_id = row["annotation_id"]
        if task_id not in tasks:
            if len(tasks) >= num_tasks:
                # Check if this row belongs to an existing task
                continue
            tasks[task_id] = []
        tasks[task_id].append(row)

    # Sort each task's steps by action index
    for task_id in tasks:
        tasks[task_id].sort(key=lambda r: int(r["target_action_index"]))

    selected_ids = list(tasks.keys())[:num_tasks]
    print(f"Collected {len(selected_ids)} tasks from stream...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for task_id in selected_ids:
        steps = tasks[task_id]
        task_dir = DATA_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "annotation_id": task_id,
            "website": steps[0].get("website", ""),
            "domain": steps[0].get("domain", ""),
            "subdomain": steps[0].get("subdomain", ""),
            "confirmed_task": steps[0].get("confirmed_task", ""),
            "num_steps": len(steps),
            "actions": [],
        }

        for i, step in enumerate(steps):
            # Save screenshot
            screenshot = step["screenshot"]  # PIL Image
            img_path = task_dir / f"step_{i:03d}.png"
            screenshot.save(img_path)

            # Collect action metadata
            metadata["actions"].append({
                "step_index": i,
                "action_uid": step.get("action_uid", ""),
                "operation": step.get("operation", {}),
                "target_action_reprs": step.get("target_action_reprs", ""),
            })

        # Save task metadata
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"  Saved {task_id}: {len(steps)} steps")

    # Write index file
    index = {
        "source": HF_DATASET,
        "split": split,
        "num_tasks": len(selected_ids),
        "task_ids": selected_ids,
    }
    (DATA_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nDone! {len(selected_ids)} tasks saved to {DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Mind2Web test data")
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks to download")
    parser.add_argument("--split", default="test_task", help="Dataset split to use")
    args = parser.parse_args()
    download_tasks(num_tasks=args.num_tasks, split=args.split)


if __name__ == "__main__":
    main()
