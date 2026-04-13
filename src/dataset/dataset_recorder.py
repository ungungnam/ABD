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
    def __init__(self, config, run_id: str):
        self.root = Path(config.root) / f"run_{run_id}"
        self.run_id = run_id
        self.root.mkdir(parents=True, exist_ok=True)
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
                     record: dict = None) -> Path:
        """Persist buffered frames + episode metadata.

        Args:
            task_name: task identifier used for directory naming.
            success:   whether the episode succeeded.
            record:    full EpisodeRecord dict (asdict(record)) — all fields
                       are written verbatim to meta.json.
        """
        if not self._buffer:
            log.warning("[DatasetRecorder] save_episode called with empty buffer.")
            return None

        outcome = "success" if success else "failure"
        ep_idx = self._next_episode_index(outcome, task_name)
        ep_dir = self.root / outcome / task_name / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

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
        # episode_index: dataset-local counter (per outcome/task).
        # task_name, success come from EpisodeRecord below to avoid duplication.
        meta = {
            "episode_index": ep_idx,
            "task": self._buffer[0].get("task", ""),   # language task string
            "num_frames": len(self._buffer),
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

        with open(ep_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        self._episode_counts[(outcome, task_name)] = ep_idx + 1
        self._episode_summaries.append({
            "episode_index": ep_idx,
            "episode_id": record.get("episode_id", "") if record else "",
            "outcome": outcome,
            "task_name": task_name,
            "task_direction": record.get("task_direction", "") if record else "",
            "num_frames": len(self._buffer),
            "success": success,
            "policy_decision": record.get("policy_decision", "") if record else "",
            "failure_type": record.get("failure_type", "") if record else "",
            "human_reset": record.get("human_reset", False) if record else False,
            "generation_time": record.get("generation_time", 0.0) if record else 0.0,
            "execution_time": record.get("execution_time", 0.0) if record else 0.0,
            "saved_at": meta["saved_at"],
        })

        self._buffer = []
        log.info(f"[DatasetRecorder] Saved {outcome}/{task_name}/episode_{ep_idx:06d} "
                 f"({meta['num_frames']} frames)")
        return ep_dir

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
        run_meta = {
            "run_id": self.run_id,
            "start_time": self._start_time,
            "end_time": end_time,
            "duration_s": end_time - self._start_time,
            "total_episodes": total,
            "total_success": n_success,
            "total_failure": n_failure,
            "success_rate": n_success / total if total > 0 else 0.0,
            "total_human_resets": sum(1 for e in self._episode_summaries if e["human_reset"]),
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
