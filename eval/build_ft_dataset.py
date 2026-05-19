"""Build a LLaVA-format train/val set to LoRA fine-tune AHA on reset detection.

Source: the reset-VQA dataset (``manifest.jsonl``), restricted to the
ABD-collected episodes (``collection_policy == "ABD"``).

Each training example is one episode:
    image  : the front-view last frame (the AHA server consumes one image)
    human  : "<image>\\n" + a reset-decision question (task interpolated)
    gpt    : "Yes" / "No"  -- the binary ground-truth reset decision

Split / balancing:
    * random sample-level 85/15 train/val split (seeded)
    * train is balanced toward ~50/50 by oversampling the reset-needed
      (minority) class up to ``--max-oversample`` x, then undersampling the
      majority to match -- otherwise AHA collapses to constant "no"
    * val keeps the natural (~14% positive) distribution for honest metrics

Outputs (to ``--out-root``):
    train.json          LLaVA conversation list (balanced)
    val.json            LLaVA conversation list (natural distribution)
    val_sample_ids.json sample ids in val -- used to score the eval pipeline
    split_info.json     counts / config provenance

The images are referenced relative to the reset-VQA root, so training is run
with ``--image_folder /data/abd/eval/reset_vqa``.

Usage:
    python eval/build_ft_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# The reset-decision question. {task} is filled per sample. Kept identical to
# what the fine-tuned model will be prompted with at eval time.
QUESTION = (
    "The robot's task is: \"{task}\". "
    "Does a human need to intervene and reset the environment before the robot "
    "can attempt this task again? Answer with a single word: Yes or No."
)


def _split_random(subset: list, val_frac: float, rng) -> tuple:
    """Random sample-level split (episodes assigned independently)."""
    subset = list(subset)
    rng.shuffle(subset)
    n_val = int(round(len(subset) * val_frac))
    return subset[n_val:], subset[:n_val]


def _split_by_run(subset: list, val_frac: float, rng) -> tuple:
    """Run-level split: whole collection runs go entirely to train or val.

    Stratified by task_family so val covers every family. Within a family,
    runs are accepted into val (greedily, shuffled) until ~val_frac of that
    family's episodes are held out. No run is shared across the split, so
    visually near-identical frames from one run cannot leak between sides.
    """
    families: dict = {}
    for r in subset:
        fam = r.get("task_family")
        run = f"{fam}/{r.get('run_id')}"
        families.setdefault(fam, {}).setdefault(run, []).append(r)

    train, val = [], []
    for fam, runs in sorted(families.items(), key=lambda kv: str(kv[0])):
        run_ids = list(runs)
        rng.shuffle(run_ids)
        fam_total = sum(len(runs[r]) for r in run_ids)
        target = val_frac * fam_total
        acc = 0
        for run in run_ids:
            eps = runs[run]
            if acc < target:
                val += eps
                acc += len(eps)
            else:
                train += eps
    return train, val


def _conversation_entry(sample: dict) -> dict:
    """One LLaVA-format example from a reset-VQA manifest row."""
    task = sample.get("language_task") or sample.get("task_name") or "the task"
    answer = "Yes" if sample["gt_reset"] else "No"
    front = sample.get("images", {}).get("front")
    return {
        "id": sample["sample_id"],
        "image": front,                       # relative to --image_folder
        "conversations": [
            {"from": "human", "value": "<image>\n" + QUESTION.format(task=task)},
            {"from": "gpt", "value": answer},
        ],
        # provenance (ignored by the trainer, handy for debugging)
        "_gt_reset": sample["gt_reset"],
        "_task_name": sample.get("task_name"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset-vqa-root", type=Path,
                    default=Path("/data/abd/eval/reset_vqa"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("/data/abd/eval/aha_ft"))
    ap.add_argument("--policy", default="ABD",
                    help="collection_policy to use as the fine-tune source")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--split-mode", choices=["random", "run"], default="random",
                    help="random: per-episode split; run: whole runs to one "
                         "side (no cross-run leakage)")
    ap.add_argument("--max-oversample", type=int, default=4,
                    help="cap on how many times a minority example is repeated")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = args.reset_vqa_root / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]

    # --- source subset: ABD-collected episodes with a front-view frame ----- #
    subset = [
        r for r in rows
        if r.get("collection_policy") == args.policy
        and r.get("images", {}).get("front")
    ]
    print(f"{args.policy}-collected episodes with a front frame: {len(subset)}")

    # --- train/val split --------------------------------------------------- #
    rng = random.Random(args.seed)
    if args.split_mode == "run":
        train_rows, val_rows = _split_by_run(subset, args.val_frac, rng)
    else:
        train_rows, val_rows = _split_random(subset, args.val_frac, rng)
    print(f"split-mode={args.split_mode}: "
          f"train={len(train_rows)}  val={len(val_rows)}")

    # --- balance the train set toward ~50/50 ------------------------------- #
    pos = [r for r in train_rows if r["gt_reset"]]
    neg = [r for r in train_rows if not r["gt_reset"]]
    if not pos or not neg:
        raise RuntimeError("train split has only one class; cannot balance")

    # Oversample the minority (positives) up to the cap; the per-class target
    # count is the oversampled positive count, and the majority (negatives) is
    # then resampled to exactly match it -> a 50/50 train set.
    factor = min(args.max_oversample, max(1, round(len(neg) / len(pos))))
    pos_bal = pos * factor
    target = len(pos_bal)
    neg_bal = (neg * (target // len(neg) + 1))[:target]
    balanced = pos_bal + neg_bal
    rng.shuffle(balanced)

    # --- write outputs ----------------------------------------------------- #
    args.out_root.mkdir(parents=True, exist_ok=True)
    train_json = [_conversation_entry(r) for r in balanced]
    val_json = [_conversation_entry(r) for r in val_rows]

    (args.out_root / "train.json").write_text(json.dumps(train_json, indent=1))
    (args.out_root / "val.json").write_text(json.dumps(val_json, indent=1))
    (args.out_root / "val_sample_ids.json").write_text(
        json.dumps([r["sample_id"] for r in val_rows], indent=1)
    )

    info = {
        "source_policy": args.policy,
        "reset_vqa_root": str(args.reset_vqa_root),
        "image_folder": str(args.reset_vqa_root),
        "question": QUESTION,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "max_oversample": args.max_oversample,
        "counts": {
            "subset_total": len(subset),
            "train_episodes_raw": len(train_rows),
            "train_raw_pos": len(pos),
            "train_raw_neg": len(neg),
            "oversample_factor": factor,
            "train_examples_balanced": len(train_json),
            "train_balanced_pos": len(pos_bal),
            "train_balanced_neg": len(neg_bal),
            "val_episodes": len(val_rows),
            "val_pos": sum(1 for r in val_rows if r["gt_reset"]),
            "val_neg": sum(1 for r in val_rows if not r["gt_reset"]),
        },
    }
    (args.out_root / "split_info.json").write_text(json.dumps(info, indent=2))

    print(json.dumps(info["counts"], indent=2))
    print(f"\nWrote train/val to {args.out_root}")


if __name__ == "__main__":
    main()
