"""Post-experiment analysis script.

Usage:
    python scripts/analyze_results.py <log_dir1> <log_dir2> ... [--output <output_dir>]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401, E402

from metrics.metrics_analyzer import compare_policies, load_summary


def main():
    parser = argparse.ArgumentParser(description="Compare ABD experiment results")
    parser.add_argument("log_dirs", nargs="+", help="Log directories to compare")
    parser.add_argument("--output", default="outputs/analysis", help="Output directory for plots")
    args = parser.parse_args()

    # Infer policy name from directory name
    policy_dirs = {}
    for d in args.log_dirs:
        name = os.path.basename(d).split("_")[0]  # e.g., "abd" from "abd_20260325_120000"
        policy_dirs[name] = d

        # Print summary if available
        summary_path = os.path.join(d, "summary.json")
        if os.path.exists(summary_path):
            summary = load_summary(summary_path)
            print(f"[{name}] {summary}")

    compare_policies(policy_dirs, args.output)


if __name__ == "__main__":
    main()
