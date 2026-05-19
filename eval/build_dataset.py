"""Build the reset-decision VQA dataset from ABD raw collection episodes.

Each VQA sample is one collected episode:

    input  : the LAST-FRAME observation of the episode, as three camera views
             (front / wrist / table) extracted from the episode videos
    label  : a binary reset decision  ->  gt_reset  (True = a human reset is
             needed before the robot can continue)
    meta   : the full episode `meta.json` plus derived fields, kept verbatim so
             downstream analysis can slice by task / policy / failure type / etc.

Ground-truth derivation
-----------------------
Every raw episode carries a human-assigned `ground_truth_reset` label that is a
confusion-matrix cell relative to the *collecting* policy's decision:

    1 = TP  (reset needed,      policy decided reset)
    2 = TN  (no reset needed,   policy decided no-reset)
    3 = FP  (reset NOT needed,  policy decided reset)
    4 = FN  (reset needed,      policy decided no-reset)

The objective, policy-independent binary ground truth is therefore:

    gt_reset = ground_truth_reset in {1, 4}

(Verified empirically against `human_reset` / `success`: labels 1 & 4 are
exactly the episodes where a reset was genuinely required.)

Usage
-----
    python eval/build_dataset.py \
        --raw-root /data/abd/raw \
        --out-root /data/abd/eval/reset_vqa \
        --workers 16

Re-running is safe: samples whose images already exist are skipped unless
`--overwrite` is given. The manifest is always rewritten from scratch.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CAMERA_VIEWS = ("front", "wrist", "table")

# ground_truth_reset confusion-matrix label -> (short name, gt_reset bool)
GT_LABEL_TABLE = {
    1: ("TP", True),
    2: ("TN", False),
    3: ("FP", False),
    4: ("FN", True),
}


def _slug(text: str) -> str:
    """Filesystem-safe slug (keeps alnum / dash / underscore)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(text)).strip("-")


@dataclass
class EpisodeRecord:
    """One discovered raw episode awaiting frame extraction."""

    episode_dir: Path
    task_family: str          # raw/<task_family>/...   e.g. pick_and_place
    collection_policy: str    # raw/.../<policy>/...     e.g. ABD / Periodic
    meta: dict = field(default_factory=dict)


def discover_episodes(raw_root: Path) -> list[EpisodeRecord]:
    """Find every `episode_*/meta.json` under raw/<family>/<policy>/run_*/."""
    records: list[EpisodeRecord] = []
    for meta_path in sorted(raw_root.glob("*/*/run_*/**/meta.json")):
        parts = meta_path.relative_to(raw_root).parts
        # parts: <family>/<policy>/run_*/{success,failure}/<variant>/episode_*/meta.json
        task_family, collection_policy = parts[0], parts[1]
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [skip] unreadable meta: {meta_path} ({exc})", file=sys.stderr)
            continue
        records.append(
            EpisodeRecord(
                episode_dir=meta_path.parent,
                task_family=task_family,
                collection_policy=collection_policy,
                meta=meta,
            )
        )
    return records


def sample_id_for(rec: EpisodeRecord) -> str:
    """Stable, unique, filesystem-safe id for one episode."""
    m = rec.meta
    run_id = m.get("run_id") or rec.episode_dir.parents[2].name
    episode_id = m.get("episode_id") or "noid"
    ep_idx = m.get("episode_idx", m.get("episode_index", 0))
    return _slug(f"{rec.collection_policy}__{run_id}__ep{int(ep_idx):05d}__{episode_id}")


def _extract_last_frame(video_path: Path, out_path: Path) -> bool:
    """Extract the final frame of `video_path` to `out_path` (JPEG).

    Uses ffmpeg `-sseof` to seek to the tail of the file and `-update 1` so the
    last decoded frame overwrites the output -- robust across codecs and does
    not require knowing the exact frame count.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-sseof", "-1.0", "-i", str(video_path),
        "-update", "1", "-frames:v", "1", "-q:v", "2",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # Fallback: decode the whole stream, keep the last frame.
        cmd_full = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
            "-update", "1", "-q:v", "2", str(out_path),
        ]
        try:
            subprocess.run(cmd_full, check=True, capture_output=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f"  [warn] frame extraction failed: {video_path} ({exc})",
                  file=sys.stderr)
            return False
    return out_path.exists() and out_path.stat().st_size > 0


def build_sample(rec: EpisodeRecord, out_root: Path, overwrite: bool) -> Optional[dict]:
    """Extract frames + assemble the manifest entry for one episode.

    Returns the manifest dict, or None if the episode is unusable (no GT label
    or no extractable frames).
    """
    meta = rec.meta
    raw_label = meta.get("ground_truth_reset")
    if raw_label not in GT_LABEL_TABLE:
        return None  # unlabeled / out-of-range -> not part of the eval set

    gt_label_name, gt_reset = GT_LABEL_TABLE[raw_label]
    sid = sample_id_for(rec)
    img_dir = out_root / "images" / sid
    img_dir.mkdir(parents=True, exist_ok=True)

    images: dict[str, str] = {}
    for view in CAMERA_VIEWS:
        video = rec.episode_dir / "videos" / f"{view}.mp4"
        if not video.exists():
            continue
        out_jpg = img_dir / f"{view}.jpg"
        if out_jpg.exists() and not overwrite:
            images[view] = str(out_jpg.relative_to(out_root))
            continue
        if _extract_last_frame(video, out_jpg):
            images[view] = str(out_jpg.relative_to(out_root))

    if not images:
        print(f"  [warn] no frames for {sid}; dropped", file=sys.stderr)
        return None

    return {
        "sample_id": sid,
        "task_family": rec.task_family,
        "task_name": meta.get("task_name"),
        "task_direction": meta.get("task_direction"),
        "language_task": meta.get("task"),
        "collection_policy": rec.collection_policy,
        "run_id": meta.get("run_id"),
        "episode_id": meta.get("episode_id"),
        "episode_idx": meta.get("episode_idx", meta.get("episode_index")),
        # --- the VQA target ---
        "gt_reset": gt_reset,
        "gt_label": raw_label,
        "gt_label_name": gt_label_name,
        # --- input images (paths relative to out_root) ---
        "images": images,
        # --- convenience surface fields (also present inside metadata) ---
        "num_frames": meta.get("num_frames"),
        "success": meta.get("success"),
        "failure_type": meta.get("failure_type"),
        "policy_decision": meta.get("policy_decision"),
        "human_reset": meta.get("human_reset"),
        # --- full episode metadata, verbatim ---
        "metadata": meta,
        "source_episode_dir": str(rec.episode_dir),
    }


def _worker(args) -> Optional[dict]:
    rec, out_root, overwrite = args
    return build_sample(rec, out_root, overwrite)


def summarize(manifest: list[dict]) -> dict:
    """Aggregate dataset statistics for dataset_info.json."""
    def _count(key):
        out: dict = {}
        for s in manifest:
            out[s.get(key)] = out.get(s.get(key), 0) + 1
        return out

    n = len(manifest)
    n_pos = sum(1 for s in manifest if s["gt_reset"])
    return {
        "total_samples": n,
        "positives_reset_needed": n_pos,
        "negatives_no_reset": n - n_pos,
        "positive_rate": round(n_pos / n, 4) if n else 0.0,
        "by_gt_label": _count("gt_label_name"),
        "by_task_name": _count("task_name"),
        "by_task_family": _count("task_family"),
        "by_collection_policy": _count("collection_policy"),
        "camera_views": list(CAMERA_VIEWS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, default=Path("/data/abd/raw"))
    ap.add_argument("--out-root", type=Path, default=Path("/data/abd/eval/reset_vqa"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract frames even if the JPEG already exists")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N episodes (debug)")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.raw_root} ...")
    records = discover_episodes(args.raw_root)
    if args.limit:
        records = records[: args.limit]
    print(f"Discovered {len(records)} raw episodes.")

    manifest: list[dict] = []
    payloads = [(r, args.out_root, args.overwrite) for r in records]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, p) for p in payloads]
        for fut in as_completed(futures):
            done += 1
            entry = fut.result()
            if entry is not None:
                manifest.append(entry)
            if done % 250 == 0 or done == len(futures):
                print(f"  processed {done}/{len(futures)}  (kept {len(manifest)})")

    manifest.sort(key=lambda s: s["sample_id"])

    manifest_path = args.out_root / "manifest.jsonl"
    with manifest_path.open("w") as fh:
        for entry in manifest:
            fh.write(json.dumps(entry) + "\n")

    info = summarize(manifest)
    (args.out_root / "dataset_info.json").write_text(json.dumps(info, indent=2))

    print(f"\nWrote {len(manifest)} samples -> {manifest_path}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
