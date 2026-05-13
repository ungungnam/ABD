"""Plot episode index vs per-episode score(s) from one or more sources.

Score fields are addressed by dotted paths into the episodes.jsonl records
(e.g. ``abd_features.f_succ`` or ``validation_details.effective_success_rate``).
Pass any number of ``--score NAME=PATH`` flags to plot that many panels.

Usage:
    python plot_episode_scores.py \
        --runs ABD:/path/to/run_abd Naive:/path/to/run_naive \
        --score validity=abd_features.f_succ \
        --score feasibility=abd_features.f_reach \
        --rolling 5 \
        --out /path/to/scores.html
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_episodes(run_dir: Path) -> list[dict]:
    with (run_dir / "episodes.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def get_dotted(d: dict, path: str):
    cur = d
    for k in path.split("."):
        if cur is None or k not in cur:
            return None
        cur = cur[k]
    return cur


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) == 0:
        return y
    w = min(window, len(y))
    kernel = np.ones(w) / w
    pad = w - 1
    padded = np.concatenate([np.full(pad, y[0]), y])
    return np.convolve(padded, kernel, mode="valid")


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


def parse_scores(args: list[str]) -> list[tuple[str, str]]:
    out = []
    for a in args:
        if "=" not in a:
            raise ValueError(f"--score must be NAME=PATH, got {a!r}")
        name, path = a.split("=", 1)
        out.append((name, path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="LABEL:DIR or DIR (DIR contains episodes.jsonl)")
    ap.add_argument("--score", action="append", required=True,
                    help="NAME=DOTTED.PATH (repeatable)")
    ap.add_argument("--rolling", type=int, default=1,
                    help="Rolling mean window (1 = raw only)")
    ap.add_argument("--combined", action="store_true",
                    help="Plot all scores in a single panel (different dash per score)")
    ap.add_argument("--out", required=True, help="Output .html (or .png)")
    args = ap.parse_args()

    runs = parse_runs(args.runs)
    scores = parse_scores(args.score)
    sources = [(label, load_episodes(d)) for label, d in runs]

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    dashes = ["solid", "dash", "dot", "dashdot"]

    if args.combined:
        fig = go.Figure()
        for j, (name, path) in enumerate(scores):
            dash = dashes[j % len(dashes)]
            for i, (label, eps) in enumerate(sources):
                eps_sorted = sorted(eps, key=lambda e: e.get("episode_idx", 0))
                xs = [e.get("episode_idx", k) for k, e in enumerate(eps_sorted)]
                ys_raw = [get_dotted(e, path) for e in eps_sorted]
                xy = [(x, y) for x, y in zip(xs, ys_raw) if y is not None]
                if not xy:
                    continue
                x = np.array([p[0] for p in xy])
                y = np.array([float(p[1]) for p in xy])
                color = palette[i % len(palette)]
                trace_name = f"{label} · {name}"
                line_y = rolling_mean(y, args.rolling) if args.rolling > 1 else y
                fig.add_trace(go.Scatter(
                    x=x, y=line_y, mode="lines", name=trace_name,
                    line=dict(color=color, width=2, dash=dash),
                ))
                if args.rolling > 1:
                    fig.add_trace(go.Scatter(
                        x=x, y=y, mode="markers", name=trace_name + " (raw)",
                        marker=dict(color=color, size=4, opacity=0.3,
                                    symbol="circle" if dash == "solid" else "x"),
                        showlegend=False,
                    ))
        fig.update_layout(
            template="plotly_white",
            title="Per-episode scores",
            xaxis_title="Episode index",
            yaxis_title="Score",
        )
    else:
        fig = make_subplots(rows=len(scores), cols=1, shared_xaxes=True,
                            subplot_titles=[name for name, _ in scores])
        for row, (name, path) in enumerate(scores, start=1):
            for i, (label, eps) in enumerate(sources):
                eps_sorted = sorted(eps, key=lambda e: e.get("episode_idx", 0))
                xs = [e.get("episode_idx", k) for k, e in enumerate(eps_sorted)]
                ys_raw = [get_dotted(e, path) for e in eps_sorted]
                xy = [(x, y) for x, y in zip(xs, ys_raw) if y is not None]
                if not xy:
                    continue
                x = np.array([p[0] for p in xy])
                y = np.array([float(p[1]) for p in xy])
                color = palette[i % len(palette)]
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode="markers", name=f"{label} (raw)",
                               marker=dict(color=color, size=5, opacity=0.4),
                               legendgroup=label, showlegend=(row == 1)),
                    row=row, col=1,
                )
                if args.rolling > 1:
                    fig.add_trace(
                        go.Scatter(x=x, y=rolling_mean(y, args.rolling),
                                   mode="lines", name=f"{label} (rolling)",
                                   line=dict(color=color, width=2),
                                   legendgroup=label, showlegend=False),
                        row=row, col=1,
                    )
            fig.update_yaxes(title_text=name, row=row, col=1)

        fig.update_xaxes(title_text="Episode index", row=len(scores), col=1)
        fig.update_layout(template="plotly_white",
                          height=300 * len(scores) + 100,
                          title="Per-episode scores")

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
