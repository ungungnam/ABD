"""Dataset manifest tracker for ABD experiments.

Tracks the mapping between pipeline episode indices and the
success-only LeRobot dataset episode indices. Every episode
(success or failure) gets a row; only successes get a
dataset_episode_idx.
"""

import json
import os
from typing import List, Optional


class DatasetManifest:
    """Tracks mapping between pipeline episodes and dataset episodes."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.entries: List[dict] = []
        self._success_counter: int = 0

    def log_episode(
        self,
        episode_idx: int,
        episode_id: str,
        run_id: str,
        task_name: str,
        success: bool,
        policy_method: str,
        failure_type: str = "",
    ) -> Optional[int]:
        """Record an episode in the manifest.

        Returns:
            dataset_episode_idx if success, else None.
        """
        dataset_episode_idx = None
        if success:
            dataset_episode_idx = self._success_counter
            self._success_counter += 1

        entry = {
            "episode_idx": episode_idx,
            "episode_id": episode_id,
            "run_id": run_id,
            "task_name": task_name,
            "success": success,
            "policy_method": policy_method,
            "failure_type": failure_type,
            "dataset_episode_idx": dataset_episode_idx,
        }
        self.entries.append(entry)
        return dataset_episode_idx

    def save(self, path: str = None):
        """Write manifest to a JSON-lines file."""
        if path is None:
            path = os.path.join(self.log_dir, "dataset_manifest.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")

    @staticmethod
    def load(path: str) -> List[dict]:
        """Load manifest entries from a JSON-lines file."""
        entries = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        return entries
