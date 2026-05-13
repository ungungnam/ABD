"""Plot elapsed time vs task-normalized cumulative episode accumulation.

For each source (a directory containing episodes.jsonl), episodes are grouped
by task. Per task, y = cumsum(success) / total_successes_for_that_task, a step
function in [0,1] over elapsed time. Curves are resampled onto a shared time
grid and averaged across tasks within the source, so the result is
task-agnostic.

Usage:
    python plot_accumulation.py \
        --runs ABD:/path/to/run_abd Naive:/path/to/run_naive \
        --out /path/to/accumulation.html
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def load_episodes(run_dir: Path) -> list[dict]:
    path = run_dir / "episodes.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def task_normalized_curve(episodes: list[dict], grid: np.ndarray) -> np.ndarray:
    """Average per-task normalized cumulative-success curves onto `grid`."""
    by_task: dict[str, list[dict]] = {}
    for ep in episodes:
        by_task.setdefault(ep["task_name"], []).append(ep)

    per_task_curves = []
    for task_eps in by_task.values():
        task_eps = sorted(task_eps, key=lambda e: e["timestamp"])
        t0 = task_eps[0]["timestamp"]
        times = np.array([e["timestamp"] - t0 for e in task_eps])
        succ = np.array([1.0 if e["success"] else 0.0 for e in task_eps])
        cum = np.cumsum(succ)
        total = cum[-1] if cum[-1] > 0 else 1.0
        norm = cum / total
        # step function: value at grid point t = last norm with times <= t
        idx = np.searchsorted(times, grid, side="right") - 1
        curve = np.where(idx >= 0, norm[np.clip(idx, 0, len(norm) - 1)], 0.0)
        per_task_curves.append(curve)

    return np.mean(np.stack(per_task_curves, axis=0), axis=0)


def parse_runs(args: list[str]) -> list[tuple[str, Path]]:
    out = []
    for a in args:
        if ":" in a:
            label, path = a.split(":", 1)
        else:
            path = a
            label = Path(a).name
        out.append((label, Path(path)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="LABEL:DIR or DIR (DIR contains episodes.jsonl)")
    ap.add_argument("--out", required=True, help="Output .html (or .png)")
    ap.add_argument("--time-unit", choices=["s", "min", "h"], default="s")
    ap.add_argument("--grid-points", type=int, default=400)
    args = ap.parse_args()

    runs = parse_runs(args.runs)
    sources = [(label, load_episodes(d)) for label, d in runs]

    # Shared elapsed-time grid across all sources, based on max duration.
    max_elapsed = 0.0
    for _, eps in sources:
        for task_eps in {}.fromkeys([e["task_name"] for e in eps]):
            ts = [e["timestamp"] for e in eps if e["task_name"] == task_eps]
            if ts:
                max_elapsed = max(max_elapsed, max(ts) - min(ts))
    if max_elapsed == 0:
        max_elapsed = 1.0
    grid = np.linspace(0, max_elapsed, args.grid_points)

    unit_div = {"s": 1.0, "min": 60.0, "h": 3600.0}[args.time_unit]
    x = grid / unit_div

    fig = go.Figure()
    for label, eps in sources:
        y = task_normalized_curve(eps, grid)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))

    fig.update_layout(
        title="Task-normalized episode accumulation",
        xaxis_title=f"Elapsed time ({args.time_unit})",
        yaxis_title="Mean cumulative valid fraction (per task)",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".html":
        fig.write_html(out)
        pdf = out.with_suffix(".pdf")
        fig.write_image(pdf)
        print(f"wrote {out} and {pdf}")
    else:
        fig.write_image(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
