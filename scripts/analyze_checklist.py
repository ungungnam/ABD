"""Checklist score analysis across all collected episodes.

Usage:
    python scripts/analyze_checklist.py [data_root]

data_root defaults to /home/gpuadmin/Desktop/data
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_checklist_items(data_root: Path):
    """Collect all checklist item answers from every meta.json."""
    # stats[task][question] = {"yes": int, "no": int, "weight": float, "no_episodes": [id, ...]}
    stats = defaultdict(lambda: defaultdict(lambda: {"yes": 0, "no": 0, "weight": 0.0, "no_episodes": []}))
    episode_count = defaultdict(int)
    score_sums = defaultdict(float)

    for meta_path in sorted(data_root.rglob("meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)

        checklist = meta.get("checklist_eval")
        if not checklist or not checklist.get("items"):
            continue

        task = checklist.get("task_name", meta.get("task_name", "unknown"))
        run_id = meta.get("run_id", "?")
        ep_idx = meta.get("episode_idx", "?")
        ep_label = f"{run_id}-ep{ep_idx}"
        episode_count[task] += 1
        score_sums[task] += checklist.get("score", 0.0)

        for item in checklist["items"]:
            q = item["question"]
            ans = item.get("answer", "no").lower()
            stats[task][q]["weight"] = item["weight"]
            if ans == "yes":
                stats[task][q]["yes"] += 1
            else:
                stats[task][q]["no"] += 1
                stats[task][q]["no_episodes"].append((run_id, ep_idx, ep_label))

    return stats, episode_count, score_sums


def print_report(stats, episode_count, score_sums):
    for task, questions in sorted(stats.items()):
        n = episode_count[task]
        avg_score = score_sums[task] / n if n else 0.0
        print(f"\n{'='*70}")
        print(f"  Task: {task}  |  episodes: {n}  |  avg score: {avg_score:.3f}")
        print(f"{'='*70}")

        rows = []
        for q, counts in questions.items():
            total = counts["yes"] + counts["no"]
            no_rate = counts["no"] / total if total else 0.0
            rows.append((no_rate, q, counts, total))
        rows.sort(reverse=True)

        print(f"  {'NO%':>5}  {'NO':>4}  {'YES':>4}  {'w':>4}  Question")
        print(f"  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*40}")
        for no_rate, q, counts, total in rows:
            print(f"  {no_rate:>4.0%}  {counts['no']:>4}  {counts['yes']:>4}"
                  f"  {counts['weight']:>4.2f}  {q}")
            if counts["no_episodes"]:
                sorted_eps = sorted(counts["no_episodes"], key=lambda x: (x[0], x[1]))
                ids = ", ".join(e[2] for e in sorted_eps)
                print(f"         {'':37}  NO episodes: {ids}")
        print()

        most_no = rows[0] if rows else None
        most_yes = rows[-1] if rows else None
        if most_no:
            print(f"  ▶ 가장 많이 걸리는 질문 ({most_no[0]:.0%} NO):")
            print(f"    \"{most_no[1]}\"")
        if most_yes and most_yes[1] != most_no[1]:
            print(f"  ▶ 가장 적게 걸리는 질문 ({most_yes[0]:.0%} NO):")
            print(f"    \"{most_yes[1]}\"")


def main():
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/gpuadmin/Desktop/data")
    if not data_root.exists():
        print(f"[ERROR] data_root not found: {data_root}")
        sys.exit(1)

    print(f"Scanning: {data_root}")
    stats, episode_count, score_sums = load_checklist_items(data_root)

    if not stats:
        print("checklist_eval이 있는 에피소드가 없습니다.")
        sys.exit(0)

    print_report(stats, episode_count, score_sums)


if __name__ == "__main__":
    main()
