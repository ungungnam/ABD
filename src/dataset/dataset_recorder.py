"""Simple episode-based dataset recorder.

Directory layout:
  <root>/run_<run_id>/
      run_meta.json              – run-level summary (written by finalize())
      success/<task_name>/episode_XXXXXX/
      failure/<task_name>/episode_XXXXXX/

Each episode directory contains:
  meta.json        – full episode record (timing, policy, checklist, reset, etc.)
  actions.npy      – (N, 7) float32: pose_6d + gripper
  states.npy       – (N, 7) float32: robot state
  videos/<cam>.mp4 – per-camera video
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

VIDEO_FPS = 10


class DatasetRecorder:
    def __init__(
        self,
        config,
        run_id: str,
        policy_method: str = "",
        task_subdir: str = "",
        policy_subdir: str = "",
        policy_params: dict = None,
    ):
        base = Path(config.root)
        if task_subdir:
            base = base / task_subdir
        if policy_subdir:
            base = base / policy_subdir
        self.root = base / f"run_{run_id}"
        self.run_id = run_id
        self.policy_method = policy_method
        self.policy_params = dict(policy_params) if policy_params else {}
        if self.root.exists():
            raise FileExistsError(
                f"[DatasetRecorder] Target run directory already exists: {self.root}. "
                f"Refusing to write into it (would mix episodes from multiple runs). "
                f"Check _make_run_id collision logic."
            )
        self.root.mkdir(parents=True, exist_ok=False)
        self._buffer: list[dict] = []
        self._episode_counts: dict[tuple, int] = {}
        self._episode_summaries: list[dict] = []
        self._start_time = time.time()
        log.info(f"[DatasetRecorder] root={self.root}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_frame(self, frame: dict) -> None:
        """Buffer one waypoint frame."""
        self._buffer.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                             for k, v in frame.items()})

    def save_episode(self, task_name: str, success: bool,
                     record: dict = None, language_task: str = None) -> Path:
        """Persist buffered frames + episode metadata.

        Args:
            task_name: task identifier used for directory naming.
            success:   whether the episode succeeded.
            record:    full EpisodeRecord dict (asdict(record)) — all fields
                       are written verbatim to meta.json.

        With an empty buffer (e.g. trajectory generation never produced an
        executable plan, or the operator fast-forwarded the rest of a
        periodic period) we still log the episode summary so run_meta.json
        reflects intervention/skip events; only the video/state arrays are
        skipped.
        """
        empty_buffer = not self._buffer
        if empty_buffer:
            log.info(
                "[DatasetRecorder] save_episode called with empty buffer — "
                "recording metadata only, skipping video/state write."
            )

        outcome = "success" if success else "failure"
        ep_idx = self._next_episode_index(outcome, task_name)
        ep_dir = self.root / outcome / task_name / f"episode_{ep_idx:06d}"
        if not empty_buffer:
            ep_dir.mkdir(parents=True, exist_ok=True)

        if not empty_buffer:
            # --- videos ---
            img_keys = [k for k in self._buffer[0] if k.startswith("observation.images")]
            for key in img_keys:
                cam_name = key.split(".")[-1]
                vid_dir = ep_dir / "videos"
                vid_dir.mkdir(parents=True, exist_ok=True)
                vid_path = str(vid_dir / f"{cam_name}.mp4")

                first = self._buffer[0][key]
                h, w = first.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(vid_path, fourcc, VIDEO_FPS, (w, h))
                for frame in self._buffer:
                    arr = frame[key]
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                writer.release()

            # --- actions / states ---
            actions = np.stack([f["action"] for f in self._buffer]).astype(np.float32)
            states  = np.stack([f["observation.state"] for f in self._buffer]).astype(np.float32)
            np.save(ep_dir / "actions.npy", actions)
            np.save(ep_dir / "states.npy",  states)

        # --- meta.json : base fields + full episode record ---
        num_frames = len(self._buffer)
        meta = {
            "episode_index": ep_idx,
            "task": language_task or (self._buffer[0].get("task", "") if not empty_buffer else ""),
            "num_frames": num_frames,
            "saved_at": time.time(),
        }
        if record is not None:
            # Merge full record; convert numpy arrays to lists for JSON
            for k, v in record.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
                elif isinstance(v, dict):
                    meta[k] = _sanitize(v)
                else:
                    meta[k] = v

        if not empty_buffer:
            with open(ep_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2, default=str)

        self._episode_counts[(outcome, task_name)] = ep_idx + 1
        self._episode_summaries.append({
            "episode_index": ep_idx,
            "episode_id": record.get("episode_id", "") if record else "",
            "outcome": outcome,
            "task_name": task_name,
            "task_direction": record.get("task_direction", "") if record else "",
            "num_frames": num_frames,
            "success": success,
            "policy_decision": record.get("policy_decision", "") if record else "",
            "failure_type": record.get("failure_type", "") if record else "",
            "human_reset": record.get("human_reset", False) if record else False,
            "ground_truth_reset": record.get("ground_truth_reset", None) if record else None,
            "generation_time": record.get("generation_time", 0.0) if record else 0.0,
            "execution_time": record.get("execution_time", 0.0) if record else 0.0,
            "user_skipped": record.get("user_skipped", False) if record else False,
            "skipped_episodes": record.get("skipped_episodes", 0) if record else 0,
            "skipped_time_s": record.get("skipped_time_s", 0.0) if record else 0.0,
            "episode_duration": record.get("episode_duration", None) if record else None,
            "saved_at": meta["saved_at"],
        })

        self._buffer = []
        log.info(f"[DatasetRecorder] Saved {outcome}/{task_name}/episode_{ep_idx:06d} "
                 f"({num_frames} frames"
                 f"{', metadata-only' if empty_buffer else ''})")
        return ep_dir if not empty_buffer else None

    def clear_episode_buffer(self) -> None:
        """Discard buffered frames without saving."""
        self._buffer = []

    def finalize(self) -> None:
        """Write run_meta.json with run-level summary."""
        total = len(self._episode_summaries)
        n_success = sum(1 for e in self._episode_summaries if e["outcome"] == "success")
        n_failure = total - n_success

        # per-task breakdown
        task_counts: dict = {}
        for e in self._episode_summaries:
            key = e["task_name"]
            if key not in task_counts:
                task_counts[key] = {"success": 0, "failure": 0,
                                    "total_frames": 0, "resets": 0}
            task_counts[key][e["outcome"]] += 1
            task_counts[key]["total_frames"] += e["num_frames"]
            if e["human_reset"]:
                task_counts[key]["resets"] += 1

        end_time = time.time()
        total_user_skips = sum(1 for e in self._episode_summaries if e.get("user_skipped"))
        total_skipped_episodes = sum(int(e.get("skipped_episodes", 0)) for e in self._episode_summaries)
        total_skipped_time_s = sum(float(e.get("skipped_time_s", 0.0)) for e in self._episode_summaries)
        durations = [
            float(e["episode_duration"])
            for e in self._episode_summaries
            if e.get("episode_duration") is not None
        ]
        avg_episode_duration_s = (
            round(sum(durations) / len(durations), 2) if durations else 0.0
        )
        run_meta = {
            "run_id": self.run_id,
            "policy_method": self.policy_method,
            "policy_params": self.policy_params,
            "period": self.policy_params.get("period"),
            "start_time": self._start_time,
            "end_time": end_time,
            "duration_s": end_time - self._start_time,
            "total_episodes": total,
            "total_success": n_success,
            "total_failure": n_failure,
            "success_rate": n_success / total if total > 0 else 0.0,
            "avg_episode_duration_s": avg_episode_duration_s,
            "total_human_resets": sum(1 for e in self._episode_summaries if e["human_reset"]),
            "total_user_skips": total_user_skips,
            "total_skipped_episodes": total_skipped_episodes,
            "total_skipped_time_s": round(total_skipped_time_s, 1),
            "reset_cycle_gt": _reset_cycle_stats(self._episode_summaries),
            "per_task": task_counts,
            "episodes": self._episode_summaries,
        }
        path = self.root / "run_meta.json"
        with open(path, "w") as f:
            json.dump(run_meta, f, indent=2, default=str)
        log.info(f"[DatasetRecorder] run_meta.json → {path}")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _next_episode_index(self, outcome: str, task_name: str) -> int:
        key = (outcome, task_name)
        if key not in self._episode_counts:
            task_dir = self.root / outcome / task_name
            count = len(sorted(task_dir.glob("episode_*"))) if task_dir.exists() else 0
            self._episode_counts[key] = count
        return self._episode_counts[key]


def _sanitize(obj):
    """Recursively convert numpy arrays to lists for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# Human GT label values that mean "reset was actually needed":
#   1 = True Positive  (reset needed,    policy: reset)
#   4 = False Negative (reset needed,    policy: no reset)
_GT_RESET_POSITIVE = (1, 4)


def _reset_cycle_stats(summaries: list[dict]) -> dict:
    """Compute human-GT reset interval stats over labeled episodes in order.

    Sequence is restricted to episodes that carry a ground_truth_reset label
    (terminal/labeling steps). Each label is mapped to a boolean
    ``positive = label in {1, 4}`` (= human says reset was needed).

    Two metrics are produced:
      * ``episodes_to_first_reset`` — 1-indexed position of the first positive
        in the labeled sequence (= how many labeled episodes elapsed before the
        first human-confirmed reset). ``None`` if no positive ever occurs.
      * ``avg_no_reset_to_reset_gap`` — mean of per-cycle gaps. For each
        positive, find the first negative after it; from that negative, find
        the next positive, and record ``gap = next_pos_idx - first_neg_idx``.
        Cycles with no intervening negative, or that never see another
        positive, are skipped. ``None`` if no complete cycle exists.
    """
    seq = [
        e["ground_truth_reset"]
        for e in summaries
        if e.get("ground_truth_reset") is not None
    ]
    pos = [g in _GT_RESET_POSITIVE for g in seq]
    n = len(pos)

    episodes_to_first_reset: int | None
    try:
        episodes_to_first_reset = pos.index(True) + 1
    except ValueError:
        episodes_to_first_reset = None

    gaps: list[int] = []
    i = 0
    while i < n:
        if not pos[i]:
            i += 1
            continue
        # at a positive: find first negative after it
        j = i + 1
        while j < n and pos[j]:
            j += 1
        if j >= n:
            break
        # from that negative, find the next positive
        k = j + 1
        while k < n and not pos[k]:
            k += 1
        if k >= n:
            break
        gaps.append(k - j)
        i = k

    avg_gap = sum(gaps) / len(gaps) if gaps else None
    return {
        "labeled_episodes": n,
        "episodes_to_first_reset": episodes_to_first_reset,
        "no_reset_to_reset_gaps": gaps,
        "avg_no_reset_to_reset_gap": round(avg_gap, 2) if avg_gap is not None else None,
    }
