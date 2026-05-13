"""Generate dummy episode + scaling data for four baselines to smoke test plots."""

import json
import random
from pathlib import Path

import numpy as np

BASELINES = {
    "ABD":      dict(success_p=0.85, feas_mean=0.90, ep_rate=2.0, scale_top=0.92),
    "Periodic": dict(success_p=0.70, feas_mean=0.75, ep_rate=1.6, scale_top=0.80),
    "NoReset":  dict(success_p=0.60, feas_mean=0.70, ep_rate=1.4, scale_top=0.72),
    "Naive":    dict(success_p=0.45, feas_mean=0.55, ep_rate=1.0, scale_top=0.60),
}
TASKS = ["banana_to_pan", "stack_cups", "open_drawer", "pick_block"]
EPISODES_PER_TASK = 25
FRACS = [0.05, 0.1, 0.25, 0.5, 1.0]
BASE_DATASET = 200


def make_episodes(out_dir: Path, label: str, cfg: dict, seed: int):
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000.0
    rows = []
    ep_idx = 0
    for task in TASKS:
        t = t0 + rng.uniform(0, 5)
        for _ in range(EPISODES_PER_TASK):
            success = bool(rng.random() < cfg["success_p"])
            feas = float(np.clip(rng.normal(cfg["feas_mean"], 0.10), 0, 1))
            valid = float(np.clip(rng.normal(0.85 if success else 0.30, 0.12), 0, 1))
            rows.append({
                "episode_idx": ep_idx,
                "task_name": task,
                "timestamp": t,
                "success": success,
                "policy_method": label,
                "abd_features": {
                    "f_succ": 1.0 if success else 0.0,
                    "f_vis":  float(np.clip(rng.normal(0.9, 0.05), 0, 1)),
                    "f_reach": feas,
                    "f_rec":  float(np.clip(rng.normal(0.9, 0.05), 0, 1)),
                    "f_dev":  float(np.clip(rng.normal(0.1, 0.05), 0, 1)),
                    "f_fail": 0.0 if success else 1.0,
                },
                "validation_details": {"effective_success_rate": valid},
            })
            ep_idx += 1
            t += rng.exponential(1.0 / cfg["ep_rate"])

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_scaling(out_dir: Path, label: str, cfg: dict, seed: int):
    rng = np.random.default_rng(seed + 100)
    points = []
    for frac in FRACS:
        size = int(BASE_DATASET * frac)
        # Saturating curve: top * (1 - exp(-k * frac))
        sr = cfg["scale_top"] * (1 - np.exp(-3.0 * frac)) + rng.normal(0, 0.02)
        points.append({
            "dataset_size": size,
            "final_success_rate": float(np.clip(sr, 0, 1)),
            "fraction": frac,
            "policy_method": label,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaling_results.json").write_text(json.dumps(points, indent=2))


def main():
    random.seed(0)
    root = Path("outputs/dummy_smoke")
    for i, (label, cfg) in enumerate(BASELINES.items()):
        make_episodes(root / "collection" / label, label, cfg, seed=i)
        make_scaling(root / "scaling" / label, label, cfg, seed=i)
    print(f"wrote dummy data under {root}/")


if __name__ == "__main__":
    main()
