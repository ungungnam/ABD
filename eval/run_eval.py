"""CLI entry point for the reset-decision VQA evaluation.

Examples
--------
List every registered method::

    python eval/run_eval.py --list

Smoke-test with the trivial baselines (no VLM server needed)::

    python eval/run_eval.py --methods always_reset never_reset random

Evaluate ABD's method against the baselines (needs a VLM server up)::

    python eval/run_eval.py \
        --methods abd_checklist single_vqa zero_shot_vlm code_as_monitor \
        --method-config eval/configs/methods.example.json

Per-method keyword arguments are supplied via ``--method-config``: a JSON file
mapping method name -> kwargs dict, forwarded to the method constructor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --- make both `eval` (this package) and ABD's `src` importable ------------- #
# Note: we deliberately do NOT pull in PaPA's path here -- PaPA's repo root
# contains its own top-level `eval.py`, which would shadow this `eval` package.
# The reset-decision method adapters only need ABD's own `src/` modules.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, _SRC)        # ABD modules: policy / task / vlm_client / ...
sys.path.insert(0, _REPO_ROOT)  # the `eval` package itself (highest priority)

from eval.methods import available_methods  # noqa: E402
from eval.runner import run_suite  # noqa: E402

DEFAULT_DATASET = "/data/abd/eval/reset_vqa"
DEFAULT_RESULTS = "/result/reset_vqa_eval"


def _load_dotenv() -> None:
    """Load ``KEY=VALUE`` lines from the repo ``.env`` into the environment.

    Lets VLM backends pick up credentials (e.g. ``OPENAI_API_KEY``) without the
    key being passed on the command line. Existing env vars are not overridden.
    """
    env_path = os.path.join(_REPO_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _stratified_subset(dataset, n: int, seed: int = 0) -> set:
    """Pick ~``n`` sample ids spread evenly across tasks, balanced on gt_reset."""
    import random

    rng = random.Random(seed)
    by_task: dict = {}
    for s in dataset.samples:
        by_task.setdefault(s.task_name, []).append(s)

    per = max(2, n // max(1, len(by_task)))
    chosen: list = []
    for _task, items in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        pos = [s for s in items if s.gt_reset]
        neg = [s for s in items if not s.gt_reset]
        rng.shuffle(pos)
        rng.shuffle(neg)
        half = per // 2
        chosen += pos[:half] + neg[: per - half]
    return {s.sample_id for s in chosen}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--methods", nargs="+",
                    default=["always_reset", "never_reset", "random"],
                    help="method names to evaluate (default: trivial baselines)")
    ap.add_argument("--method-config", type=Path, default=None,
                    help="JSON file: {method_name: {kwarg: value, ...}}")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N samples (debug)")
    ap.add_argument("--stratified", type=int, default=0, metavar="N",
                    help="evaluate a ~N-sample subset spread evenly across "
                         "tasks and balanced on gt_reset (overrides --limit)")
    ap.add_argument("--sample-ids", type=Path, default=None,
                    help="JSON file with a list of sample_ids to evaluate "
                         "(e.g. a held-out val split; overrides --stratified)")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for --stratified subset selection")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore cached predictions and re-predict everything")
    ap.add_argument("--list", action="store_true",
                    help="list registered methods and exit")
    args = ap.parse_args()

    if args.list:
        print("Registered reset-decision methods:")
        for name in available_methods():
            print(f"  - {name}")
        return

    _load_dotenv()

    method_cfg: dict = {}
    if args.method_config:
        method_cfg = json.loads(args.method_config.read_text())

    specs = [{"name": m, "kwargs": method_cfg.get(m, {})} for m in args.methods]

    sample_ids = None
    if args.sample_ids:
        sample_ids = set(json.loads(args.sample_ids.read_text()))
        print(f"Restricting to {len(sample_ids)} sample ids from "
              f"{args.sample_ids}")
    elif args.stratified:
        from eval.dataset import ResetVQADataset
        ds = ResetVQADataset(args.dataset_root)
        sample_ids = _stratified_subset(ds, args.stratified, seed=args.seed)
        print(f"Stratified subset: {len(sample_ids)} samples "
              f"(seed={args.seed})")

    run_suite(args.dataset_root, specs, args.results_dir,
              resume=not args.no_resume, limit=args.limit,
              sample_ids=sample_ids)


if __name__ == "__main__":
    main()
