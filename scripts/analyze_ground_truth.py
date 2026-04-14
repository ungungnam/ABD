"""Ground truth reset label analysis for the most recent (or specified) run.

Ground truth labels:
    1 = policy decision matched human judgement (correct)
    2 = policy decision did NOT match human judgement (incorrect)
    3 = ambiguous

Usage:
    python scripts/analyze_ground_truth.py [data_root] [run_id]

    data_root  defaults to /home/gpuadmin/Desktop/data
    run_id     defaults to the most recent run_* directory
"""

import json
import sys
from pathlib import Path


DATA_ROOT_DEFAULT = Path("/home/gpuadmin/Desktop/data")

LABEL_DESC = {
    1: "Correct   (policy ↔ human agree)",
    2: "Incorrect (policy ↔ human disagree)",
    3: "Ambiguous",
}


def load_episodes(run_dir: Path) -> list[dict]:
    """Load all meta.json files from a run directory."""
    episodes = []
    for meta_path in sorted(run_dir.rglob("meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["_meta_path"] = str(meta_path)
        episodes.append(meta)
    return episodes


def latest_run(data_root: Path) -> Path:
    runs = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )
    if not runs:
        raise FileNotFoundError(f"No run_* directories found in {data_root}")
    return runs[-1]


def fmt_ep(ep: dict) -> str:
    run_id  = ep.get("run_id", "?")
    ep_idx  = ep.get("episode_idx", "?")
    task    = ep.get("task_name", ep.get("task", "?"))
    direc   = ep.get("task_direction", "?")
    success = ep.get("success", "?")
    decision = ep.get("policy_decision", "?")
    score   = None
    if ep.get("checklist_eval"):
        score = ep["checklist_eval"].get("score")
    score_s = f"score={score:.3f}" if score is not None else "score=n/a"
    return (
        f"  ep{ep_idx:>3}  [{run_id}]  task={task}  dir={direc}  "
        f"success={success}  decision={decision}  {score_s}"
    )


def main():
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_ROOT_DEFAULT

    if len(sys.argv) > 2:
        run_dir = data_root / f"run_{sys.argv[2]}"
    else:
        run_dir = latest_run(data_root)

    print(f"Run: {run_dir.name}")
    print()

    episodes = load_episodes(run_dir)
    labeled = [e for e in episodes if e.get("ground_truth_reset") is not None]
    unlabeled = len(episodes) - len(labeled)

    # Bucket by label
    buckets: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for ep in labeled:
        label = ep["ground_truth_reset"]
        buckets.setdefault(label, []).append(ep)

    # Summary counts
    print(f"{'='*60}")
    print(f"  Total episodes : {len(episodes)}")
    print(f"  Labeled        : {len(labeled)}")
    print(f"  Unlabeled      : {unlabeled}")
    print(f"{'='*60}")
    print()
    for label in (1, 2, 3):
        eps = buckets.get(label, [])
        print(f"  Label {label} — {LABEL_DESC[label]}: {len(eps)} episodes")
    print()

    # Detailed breakdown for labels 2 and 3
    for label in (2, 3):
        eps = buckets.get(label, [])
        if not eps:
            continue
        print(f"{'='*60}")
        print(f"  Label {label} — {LABEL_DESC[label]} ({len(eps)} episodes)")
        print(f"{'='*60}")
        for ep in sorted(eps, key=lambda e: e.get("episode_idx", 0)):
            print(fmt_ep(ep))
            # Show checklist items if available
            checklist = ep.get("checklist_eval")
            if checklist and checklist.get("items"):
                no_items = [
                    it for it in checklist["items"] if it.get("answer") == "no"
                ]
                if no_items:
                    print(f"    Checklist NO items:")
                    for it in no_items:
                        print(f"      [NO] (w={it['weight']:.2f}) {it['question']}")
        print()


if __name__ == "__main__":
    main()
