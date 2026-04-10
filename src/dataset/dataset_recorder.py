"""Simple episode-based dataset recorder.

Directory layout:
  <root>/run_<run_id>/
      success/<task_name>/episode_XXXXXX/
      failure/<task_name>/episode_XXXXXX/

Each episode directory contains:
  meta.json        – task, success, timestamps, frame count
  actions.npy      – (N, 7) float32: pose_6d + gripper
  states.npy       – (N, 7) float32: robot state
  images/<cam>/    – 000000.png … per frame
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class DatasetRecorder:
    def __init__(self, config, run_id: str):
        self.root = Path(config.root) / f"run_{run_id}"
        self.root.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        # (outcome, task_name) -> episode count
        self._episode_counts: dict[tuple, int] = {}
        log.info(f"[DatasetRecorder] root={self.root}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_frame(self, frame: dict) -> None:
        """Buffer one waypoint frame."""
        self._buffer.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                             for k, v in frame.items()})

    def save_episode(self, task_name: str, success: bool) -> Path:
        """Persist buffered frames.

        Saved under success/<task_name>/ or failure/<task_name>/
        depending on the success flag.
        """
        if not self._buffer:
            log.warning("[DatasetRecorder] save_episode called with empty buffer.")
            return None

        outcome = "success" if success else "failure"
        ep_idx = self._next_episode_index(outcome, task_name)
        ep_dir = self.root / outcome / task_name / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # --- images ---
        img_keys = [k for k in self._buffer[0] if k.startswith("observation.images")]
        for key in img_keys:
            cam_name = key.split(".")[-1]
            cam_dir = ep_dir / "images" / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(self._buffer):
                arr = frame[key]
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(cam_dir / f"{i:06d}.png")

        # --- actions / states ---
        actions = np.stack([f["action"] for f in self._buffer]).astype(np.float32)
        states  = np.stack([f["observation.state"] for f in self._buffer]).astype(np.float32)
        np.save(ep_dir / "actions.npy", actions)
        np.save(ep_dir / "states.npy",  states)

        # --- meta ---
        meta = {
            "episode_index": ep_idx,
            "task_name": task_name,
            "task": self._buffer[0].get("task", ""),
            "success": success,
            "num_frames": len(self._buffer),
            "saved_at": time.time(),
        }
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._episode_counts[(outcome, task_name)] = ep_idx + 1
        self._buffer = []
        log.info(f"[DatasetRecorder] Saved {outcome}/{task_name}/episode_{ep_idx:06d} ({meta['num_frames']} frames)")
        return ep_dir

    def clear_episode_buffer(self) -> None:
        """Discard buffered frames without saving."""
        self._buffer = []

    def finalize(self) -> None:
        """No-op."""
        pass

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
