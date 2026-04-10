"""Simple episode-based dataset recorder.

Each successful episode is saved as:
  <root>/episode_XXXXXX/
      meta.json          – task, timestamps, frame count
      actions.npy        – (N, 7) float32: pose_6d + gripper
      states.npy         – (N, 7) float32: same as action (robot state)
      images/<cam>/      – 000000.png … per frame

Only called save_episode() on success; failed episodes are discarded via
clear_episode_buffer().
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class DatasetRecorder:
    def __init__(self, config):
        self.root = Path(config.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._episode_count: int = self._count_existing_episodes()
        log.info(
            f"[DatasetRecorder] root={self.root}, "
            f"existing episodes={self._episode_count}"
        )

    # ------------------------------------------------------------------ #
    # Public API (same interface as before)
    # ------------------------------------------------------------------ #

    def add_frame(self, frame: dict) -> None:
        """Buffer one waypoint frame. Called per waypoint during execution."""
        self._buffer.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                             for k, v in frame.items()})

    def save_episode(self) -> Path:
        """Persist buffered frames as a new episode directory. Returns path."""
        if not self._buffer:
            log.warning("[DatasetRecorder] save_episode called with empty buffer.")
            return None

        ep_idx = self._episode_count
        ep_dir = self.root / f"episode_{ep_idx:06d}"
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
            "task": self._buffer[0].get("task", ""),
            "num_frames": len(self._buffer),
            "saved_at": time.time(),
        }
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._episode_count += 1
        self._buffer = []
        log.info(f"[DatasetRecorder] Saved episode {ep_idx} → {ep_dir}")
        return ep_dir

    def clear_episode_buffer(self) -> None:
        """Discard buffered frames for a failed episode."""
        self._buffer = []

    def finalize(self) -> None:
        """No-op: nothing to flush."""
        pass

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _count_existing_episodes(self) -> int:
        return len(sorted(self.root.glob("episode_*")))
