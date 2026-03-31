"""Post-hoc metrics analysis and plotting.

Loads logs from multiple experiment runs and generates comparison plots.
"""

import json
import os
from typing import Dict, List

import numpy as np

from metrics.metrics_logger import EpisodeRecord, MetricsLogger


def load_run(episodes_path: str) -> List[EpisodeRecord]:
    """Load a single run's episode records."""
    return MetricsLogger.load(episodes_path)


def load_summary(summary_path: str) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def compare_policies(log_dirs: Dict[str, str], output_dir: str):
    """Compare multiple policy runs and generate plots.

    Args:
        log_dirs: {policy_name: log_directory_path}
        output_dir: Directory to save plots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[metrics_analyzer] matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_runs = {}
    for name, log_dir in log_dirs.items():
        ep_path = os.path.join(log_dir, "episodes.jsonl")
        if os.path.exists(ep_path):
            all_runs[name] = load_run(ep_path)

    if not all_runs:
        print("[metrics_analyzer] No runs found.")
        return

    # --- Plot 1: Cumulative success over episodes ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, episodes in all_runs.items():
        cum_success = np.cumsum([int(e.success) for e in episodes])
        ax.plot(cum_success, label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Valid Trajectories")
    ax.set_title("Cumulative Valid Trajectories Over Episodes")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "cumulative_success.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Moving-window success rate ---
    fig, ax = plt.subplots(figsize=(10, 6))
    window = 10
    for name, episodes in all_runs.items():
        successes = [int(e.success) for e in episodes]
        rates = []
        for i in range(len(successes)):
            start = max(0, i - window + 1)
            rates.append(np.mean(successes[start:i + 1]))
        ax.plot(rates, label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success Rate (window={window})")
    ax.set_title("Success Rate Over Episodes")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "success_rate.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: State deviation over time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, episodes in all_runs.items():
        devs = []
        for e in episodes:
            if e.abd_features and "f_dev" in e.abd_features:
                devs.append(e.abd_features["f_dev"])
            else:
                devs.append(float("nan"))
        ax.plot(devs, label=name, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("State Deviation (f_dev)")
    ax.set_title("State Deviation Over Time")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "state_deviation.png"), dpi=150)
    plt.close(fig)

    # --- Plot 4: Interventions bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(all_runs.keys())
    interventions = [
        sum(1 for e in all_runs[n] if e.human_reset) for n in names
    ]
    ax.bar(names, interventions)
    ax.set_ylabel("Total Human Resets")
    ax.set_title("Human Interventions Per Policy")
    fig.savefig(os.path.join(output_dir, "interventions.png"), dpi=150)
    plt.close(fig)

    print(f"[metrics_analyzer] Plots saved to {output_dir}")
