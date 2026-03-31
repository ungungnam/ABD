"""Dataset-size scaling experiment.

Trains BC at different dataset fractions to measure:
  dataset size vs BC performance

This is the key downstream metric from Section 14.4 of the spec.

Usage:
    python scripts/train_scaling.py --policy_method ABD
    python scripts/train_scaling.py --policy_method ABD --dataset_root outputs/dataset/abd_collection
    python scripts/train_scaling.py --dummy  # test with dummy data
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401, E402

from train.bc_trainer import BCTrainer


FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def run_scaling_experiment(
    policy_method: str,
    dataset_root: str = None,
    dummy: bool = True,
    dummy_episodes: int = 50,
    num_epochs: int = 30,
    batch_size: int = 16,
    seed: int = 42,
    output_dir: str = "outputs/scaling",
):
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"\n{'='*70}")
    print(f"  Scaling Experiment: {policy_method}")
    print(f"  Fractions: {FRACTIONS}")
    print(f"  Dataset: {'dummy' if dummy else dataset_root}")
    print(f"{'='*70}\n")

    for frac in FRACTIONS:
        print(f"\n>>> Training with {frac:.0%} of data <<<\n")

        log_dir = os.path.join(output_dir, policy_method, f"frac_{frac:.2f}")

        trainer = BCTrainer(
            dataset_root=dataset_root,
            dummy=dummy,
            dummy_episodes=dummy_episodes,
            dummy_episode_length=30,
            dataset_fraction=frac,
            num_epochs=num_epochs,
            batch_size=batch_size,
            log_dir=log_dir,
            experiment_name=f"scaling_{policy_method}_{frac:.2f}",
            policy_method=policy_method,
            seed=seed,
        )

        summary = trainer.train()
        summary["fraction"] = frac
        results.append(summary)

    # ── Save combined results ──
    results_path = os.path.join(output_dir, policy_method, "scaling_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary table ──
    print(f"\n{'='*70}")
    print(f"  Scaling Results: {policy_method}")
    print(f"{'='*70}")
    print(f"  {'Fraction':>10} | {'Dataset Size':>12} | {'Best Val Loss':>14} | {'Success Rate':>13}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*13}")
    for r in results:
        print(f"  {r['fraction']:>10.0%} | {r['dataset_size']:>12} | "
              f"{r['best_val_loss']:>14.6f} | {r['final_success_rate']:>13.4f}")
    print(f"\n  Results saved to: {results_path}")

    # ── Plot scaling curve ──
    _plot_scaling(results, output_dir, policy_method)

    return results


def _plot_scaling(results, output_dir, policy_method):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not available, skipping plot]")
        return

    fracs = [r["fraction"] for r in results]
    sizes = [r["dataset_size"] for r in results]
    losses = [r["best_val_loss"] for r in results]
    success = [r["final_success_rate"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fracs, losses, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Dataset Fraction")
    ax1.set_ylabel("Best Validation Loss")
    ax1.set_title(f"Scaling: Val Loss ({policy_method})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(fracs, success, "s-", linewidth=2, markersize=8, color="green")
    ax2.set_xlabel("Dataset Fraction")
    ax2.set_ylabel("Estimated Success Rate")
    ax2.set_title(f"Scaling: Success Rate ({policy_method})")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, policy_method, "scaling_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="BC Scaling Experiment")
    parser.add_argument("--policy_method", type=str, default="ABD",
                        help="Name of the reset policy that collected the data")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Path to collected dataset")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy data for testing")
    parser.add_argument("--dummy_episodes", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/scaling")
    args = parser.parse_args()

    run_scaling_experiment(
        policy_method=args.policy_method,
        dataset_root=args.dataset_root,
        dummy=args.dummy or (args.dataset_root is None),
        dummy_episodes=args.dummy_episodes,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
