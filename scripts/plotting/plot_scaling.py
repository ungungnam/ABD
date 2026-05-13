"""Plot dataset size vs success rate from one or more scaling experiments.

Each source is either:
  - a ``scaling_results.json`` file (a list of point dicts), or
  - a directory containing ``scaling_results.json`` or ``frac_*/train_summary.json``.

Each point needs ``dataset_size`` and ``final_success_rate`` (configurable).

Usage:
    python plot_scaling.py \
        --runs ABD:/path/to/scaling_test/ABD Naive:/path/to/scaling_test/Naive \
        --out /path/to/scaling.html
"""

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go


def load_points(src: Path, size_key: str, success_key: str) -> list[tuple[float, float]]:
    if src.is_file():
        records = json.loads(src.read_text())
    else:
        results = src / "scaling_results.json"
        if results.exists():
            records = json.loads(results.read_text())
        else:
            records = []
            for sub in sorted(src.glob("frac_*/train_summary.json")):
                records.append(json.loads(sub.read_text()))

    points = []
    for r in records:
        if size_key in r and success_key in r:
            points.append((float(r[size_key]), float(r[success_key])))
    points.sort(key=lambda p: p[0])
    return points


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
                    help="LABEL:PATH (PATH = scaling_test/<policy> dir or scaling_results.json)")
    ap.add_argument("--size-key", default="dataset_size")
    ap.add_argument("--success-key", default="final_success_rate")
    ap.add_argument("--log-x", action="store_true")
    ap.add_argument("--out", required=True, help="Output .html (or .png)")
    args = ap.parse_args()

    fig = go.Figure()
    for label, path in parse_runs(args.runs):
        pts = load_points(path, args.size_key, args.success_key)
        if not pts:
            print(f"warn: no points for {label} ({path})")
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=label))

    fig.update_layout(
        title="Dataset size vs success rate",
        xaxis_title="Dataset size (episodes)",
        yaxis_title="Success rate",
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(type="log" if args.log_x else "linear"),
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
