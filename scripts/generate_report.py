"""Generate an HTML report from Peeklet pipeline results.

Usage:
    python scripts/generate_report.py [results_dir] [-o report.html]

Reads summary.json and per-task results.json from the results directory.
Produces an HTML file that references images via relative paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def load_results(results_dir: Path, images_dir: Path) -> dict:
    """Load summary and all per-task results, copying images to report images dir."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run scripts/run_dataset_tests.py first.")
        sys.exit(1)

    summary = json.loads(summary_path.read_text())
    images_dir.mkdir(parents=True, exist_ok=True)

    for task in summary["tasks"]:
        task_dir = results_dir / task["task_id"]
        task_results_path = task_dir / "results.json"
        if task_results_path.exists():
            task_detail = json.loads(task_results_path.read_text())
            task["frames"] = task_detail["frames"]

            # Source images from dataset
            data_task_dir = (
                Path(__file__).resolve().parent.parent
                / "tests" / "datasets" / "data" / "web_tasks" / task["task_id"]
            )
            task_images = images_dir / task["task_id"]
            task_images.mkdir(parents=True, exist_ok=True)

            for frame in task["frames"]:
                # Copy source screenshot
                src_img = data_task_dir / frame["source_image"]
                if src_img.exists():
                    dest = task_images / frame["source_image"]
                    shutil.copy2(src_img, dest)
                    frame["source_image_path"] = f"images/{task['task_id']}/{frame['source_image']}"

                # Copy keyframe image
                if frame["asset_path"]:
                    asset = Path(frame["asset_path"])
                    if asset.exists():
                        kf_name = f"kf_{frame['frame_id']}.png"
                        dest = task_images / kf_name
                        shutil.copy2(asset, dest)
                        frame["keyframe_image_path"] = f"images/{task['task_id']}/{kf_name}"

    return summary


def generate_html(summary: dict) -> str:
    """Generate the complete HTML report string."""
    tasks_json = json.dumps(summary["tasks"])
    total_skip_rate = (
        1 - (summary["total_keyframes"] / summary["total_frames"])
        if summary["total_frames"] else 0
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Peeklet Results Explorer</title>
<style>
:root {{
    --bg: #1a1a2e;
    --bg-card: #16213e;
    --bg-sidebar: #0f3460;
    --text: #e6e6e6;
    --text-muted: #8888aa;
    --accent: #e94560;
    --accent-green: #4ecca3;
    --accent-yellow: #ffd700;
    --border: #2a2a4a;
}}
[data-theme="light"] {{
    --bg: #f5f5f5;
    --bg-card: #ffffff;
    --bg-sidebar: #e8e8e8;
    --text: #222222;
    --text-muted: #666666;
    --accent: #d63031;
    --accent-green: #00b894;
    --accent-yellow: #f39c12;
    --border: #dddddd;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    background: var(--bg);
    color: var(--text);
    display: grid;
    grid-template-rows: auto 1fr;
    grid-template-columns: 300px 1fr;
    height: 100vh;
}}
.topbar {{
    grid-column: 1 / -1;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 24px;
}}
.topbar h1 {{ font-size: 16px; font-weight: 600; }}
.stat {{
    display: flex;
    flex-direction: column;
    align-items: center;
}}
.stat-value {{ font-size: 20px; font-weight: 700; }}
.stat-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; }}
.stat-value.green {{ color: var(--accent-green); }}
.stat-value.accent {{ color: var(--accent); }}
.theme-toggle {{
    margin-left: auto;
    background: var(--border);
    border: none;
    color: var(--text);
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
}}
.sidebar {{
    background: var(--bg-sidebar);
    overflow-y: auto;
    border-right: 1px solid var(--border);
    padding: 8px;
}}
.task-item {{
    padding: 10px;
    border-radius: 6px;
    cursor: pointer;
    margin-bottom: 4px;
    border: 1px solid transparent;
}}
.task-item:hover {{ border-color: var(--border); }}
.task-item.active {{ background: var(--bg-card); border-color: var(--accent); }}
.task-name {{ font-size: 13px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.task-meta {{ font-size: 11px; color: var(--text-muted); margin-top: 2px; }}
.main {{
    overflow-y: auto;
    padding: 16px;
}}
.frame-timeline {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}
.frame-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}}
.frame-header {{
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    cursor: pointer;
    user-select: none;
}}
.frame-header:hover {{ background: var(--bg-sidebar); }}
.badge {{
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 3px;
    text-transform: uppercase;
}}
.badge.keyframe {{ background: var(--accent); color: white; }}
.badge.skipped {{ background: var(--border); color: var(--text-muted); }}
.frame-meta-inline {{ font-size: 12px; color: var(--text-muted); }}
.frame-detail {{
    padding: 14px;
    border-top: 1px solid var(--border);
    display: none;
}}
.frame-detail.open {{ display: block; }}
.frame-detail img {{
    max-width: 100%;
    border-radius: 4px;
    margin-bottom: 8px;
}}
.frame-props {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px;
    font-size: 12px;
}}
.prop-label {{ color: var(--text-muted); }}
.prop-value {{ font-weight: 500; }}
.action-repr {{
    margin-top: 8px;
    padding: 6px 10px;
    background: var(--bg-sidebar);
    border-radius: 4px;
    font-size: 12px;
}}
.image-compare {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 8px;
}}
.image-compare img {{ width: 100%; border-radius: 4px; }}
.image-label {{ font-size: 11px; color: var(--text-muted); text-align: center; margin-top: 4px; }}
.no-tasks {{ padding: 40px; text-align: center; color: var(--text-muted); }}
</style>
</head>
<body>

<div class="topbar">
    <h1>Peeklet Results Explorer</h1>
    <div class="stat">
        <span class="stat-value">{summary['total_frames']}</span>
        <span class="stat-label">Total Frames</span>
    </div>
    <div class="stat">
        <span class="stat-value accent">{summary['total_keyframes']}</span>
        <span class="stat-label">Keyframes</span>
    </div>
    <div class="stat">
        <span class="stat-value green">{total_skip_rate:.0%}</span>
        <span class="stat-label">Skip Rate</span>
    </div>
    <div class="stat">
        <span class="stat-value">{summary['total_time_ms']:.0f}ms</span>
        <span class="stat-label">Processing</span>
    </div>
    <button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
</div>

<div class="sidebar" id="sidebar"></div>
<div class="main" id="main">
    <div class="no-tasks">Select a task from the sidebar</div>
</div>

<script>
const TASKS = {tasks_json};
let activeTask = null;

function toggleTheme() {{
    document.documentElement.toggleAttribute('data-theme');
    if (document.documentElement.hasAttribute('data-theme')) {{
        document.documentElement.setAttribute('data-theme', 'light');
    }} else {{
        document.documentElement.removeAttribute('data-theme');
    }}
}}

function renderSidebar() {{
    const sidebar = document.getElementById('sidebar');
    sidebar.innerHTML = TASKS.map((t, i) => `
        <div class="task-item ${{activeTask === i ? 'active' : ''}}" onclick="selectTask(${{i}})">
            <div class="task-name" title="${{t.confirmed_task}}">${{t.confirmed_task || t.task_id}}</div>
            <div class="task-meta">
                ${{t.website}} &middot; ${{t.num_keyframes}}/${{t.num_frames}} keyframes &middot; ${{Math.round(t.skip_rate * 100)}}% skip
            </div>
        </div>
    `).join('');
}}

function selectTask(index) {{
    activeTask = index;
    renderSidebar();
    renderMain(TASKS[index]);
}}

function renderMain(task) {{
    const main = document.getElementById('main');
    if (!task.frames) {{
        main.innerHTML = '<div class="no-tasks">No frame data available</div>';
        return;
    }}
    main.innerHTML = `
        <h2 style="margin-bottom:12px;font-size:15px;">${{task.confirmed_task || task.task_id}}</h2>
        <div class="frame-timeline">
            ${{task.frames.map((f, i) => renderFrame(f, i, task)).join('')}}
        </div>
    `;
}}

function renderFrame(frame, index, task) {{
    const isKF = frame.is_keyframe;
    const openByDefault = isKF;
    const ssimStr = frame.ssim_score !== null ? frame.ssim_score.toFixed(4) : 'N/A (hash match)';
    const actionStr = frame.action ? frame.action.target_action_reprs || '' : '';
    const opStr = frame.action && frame.action.operation ? frame.action.operation.op || '' : '';

    let imageSection = '';
    if (frame.source_image_path) {{
        if (isKF && frame.keyframe_image_path) {{
            imageSection = `
                <div class="image-compare">
                    <div><img src="${{frame.source_image_path}}" loading="lazy"><div class="image-label">Source Screenshot</div></div>
                    <div><img src="${{frame.keyframe_image_path}}" loading="lazy"><div class="image-label">Saved Keyframe</div></div>
                </div>
            `;
        }} else {{
            imageSection = `<img src="${{frame.source_image_path}}" loading="lazy">`;
        }}
    }}

    return `
        <div class="frame-card">
            <div class="frame-header" onclick="this.nextElementSibling.classList.toggle('open')">
                <span class="badge ${{isKF ? 'keyframe' : 'skipped'}}">${{isKF ? 'Keyframe' : 'Skipped'}}</span>
                <strong>${{frame.frame_id}}</strong>
                <span class="frame-meta-inline">
                    SSIM: ${{ssimStr}}
                    ${{opStr ? '&middot; ' + opStr : ''}}
                </span>
            </div>
            <div class="frame-detail ${{openByDefault ? 'open' : ''}}">
                ${{imageSection}}
                <div class="frame-props">
                    <div><span class="prop-label">Hash:</span> <span class="prop-value">${{frame.perceptual_hash}}</span></div>
                    <div><span class="prop-label">SSIM:</span> <span class="prop-value">${{ssimStr}}</span></div>
                    <div><span class="prop-label">Change %:</span> <span class="prop-value">${{frame.changed_pct !== null ? frame.changed_pct.toFixed(1) + '%' : 'N/A'}}</span></div>
                    <div><span class="prop-label">Mask Regions:</span> <span class="prop-value">${{frame.adaptive_mask ? frame.adaptive_mask.length : 0}}</span></div>
                </div>
                ${{actionStr ? `<div class="action-repr"><strong>Action:</strong> ${{actionStr}}</div>` : ''}}
            </div>
        </div>
    `;
}}

renderSidebar();
if (TASKS.length > 0) selectTask(0);
</script>

</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Peeklet HTML results report")
    parser.add_argument("results_dir", nargs="?", default="results", help="Results directory")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "report.html"
    images_dir = output_path.parent / "images"

    print(f"Loading results from {results_dir}...")
    summary = load_results(results_dir, images_dir)
    print(f"Generating report for {len(summary['tasks'])} tasks...")
    html = generate_html(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Report saved to {output_path}")
    print(f"Images copied to {images_dir}")


if __name__ == "__main__":
    main()
