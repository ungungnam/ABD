"""Metrics logger for ABD data collection experiments.

Logs per-episode records and computes runtime + stability metrics.
"""

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class EpisodeRecord:
    """Single episode record for logging."""
    episode_idx: int = 0
    task_name: str = ""
    success: bool = False
    policy_decision: str = ""       # "next", "retry", "reset"
    human_reset: bool = False
    generation_time: float = 0.0
    execution_time: float = 0.0
    validation_method: str = ""
    validation_details: dict = field(default_factory=dict)
    episode_id: str = ""                           # UUID per episode
    run_id: str = ""                               # shared across all episodes in a run
    policy_method: str = ""                        # "NoReset"/"Periodic"/"Naive"/"ABD"
    task_direction: str = ""                       # "forward" / "reverse"
    fail_count: int = 0                            # consecutive failures at decision time
    failure_type: str = ""                         # from FailureClassifier
    generation_success: bool = True                # False if trajectory gen failed
    dataset_episode_idx: Optional[int] = None      # index in success-only dataset
    checklist_eval: Optional[dict] = None          # VLMChecklist result: score + per-item breakdown
    # --- timing (all durations in seconds) ---
    episode_duration: Optional[float] = None       # total episode time (includes human reset wait if any)
    reset_prompt_to_confirm: Optional[float] = None    # time human took to confirm reset


@dataclass
class InterventionRecord:
    """Record for human interventions and emergency events."""
    timestamp: float = 0.0
    run_id: str = ""
    episode_idx: int = 0
    episode_id: str = ""
    intervention_type: str = ""  # "policy_reset" / "collision" / "overheat" / "crash"
    trigger: str = ""            # what triggered it
    details: dict = field(default_factory=dict)


class MetricsLogger:
    """Collects per-episode metrics and computes aggregate statistics."""

    def __init__(self, log_dir: str, run_id: str = "", policy_method: str = ""):
        self.log_dir = log_dir
        self.run_id = run_id
        self.policy_method = policy_method
        self.episodes: List[EpisodeRecord] = []
        self.interventions: List[InterventionRecord] = []
        self.start_time: Optional[float] = None
    def start_run(self):
        self.start_time = time.time()
        os.makedirs(self.log_dir, exist_ok=True)

    def log_episode(self, record: EpisodeRecord):
        self.episodes.append(record)

    def log_intervention(self, record: InterventionRecord):
        self.interventions.append(record)

    def save_run_config(self, config_dict: dict):
        """Save run configuration snapshot."""
        config_dict["run_id"] = self.run_id
        config_dict["policy_method"] = self.policy_method
        config_dict["start_time"] = self.start_time
        path = os.path.join(self.log_dir, "run_config.json")
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    # ---------- Runtime metrics ----------

    def valid_trajectories_per_hour(self) -> float:
        if self.start_time is None or not self.episodes:
            return 0.0
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        if elapsed_hours == 0:
            return 0.0
        n_valid = sum(1 for e in self.episodes if e.success)
        return n_valid / elapsed_hours

    def cumulative_valid_trajectories(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    def interventions_per_hour(self) -> float:
        if self.start_time is None or not self.episodes:
            return 0.0
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        if elapsed_hours == 0:
            return 0.0
        n_resets = sum(1 for e in self.episodes if e.human_reset)
        return n_resets / elapsed_hours

    def uptime_ratio(self) -> float:
        """Fraction of time spent in autonomous execution (not waiting for human)."""
        if not self.episodes:
            return 1.0
        total_exec_time = sum(e.generation_time + e.execution_time for e in self.episodes)
        total_wall_time = (time.time() - self.start_time) if self.start_time else 1.0
        if total_wall_time == 0:
            return 1.0
        return min(1.0, total_exec_time / total_wall_time)

    # ---------- Stability metrics ----------

    def success_rate_over_episodes(self, window: int = 10) -> List[float]:
        """Moving-window success rate."""
        rates = []
        for i in range(len(self.episodes)):
            start = max(0, i - window + 1)
            window_episodes = self.episodes[start:i + 1]
            rate = sum(1 for e in window_episodes if e.success) / len(window_episodes)
            rates.append(rate)
        return rates

    def cumulative_success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    # ---------- Experiment-specific aggregates ----------

    def interventions_per_valid_episodes(self, n: int = 100) -> float:
        """Human interventions per N valid (successful) episodes. (E3)"""
        n_resets = sum(1 for e in self.episodes if e.human_reset)
        n_valid = sum(1 for e in self.episodes if e.success)
        if n_valid == 0:
            return 0.0
        return n_resets / n_valid * n

    def failure_type_distribution(self) -> dict:
        """Count of each failure_type across all episodes. (E4)"""
        counts = Counter(
            e.failure_type for e in self.episodes
            if not e.success and e.failure_type
        )
        return dict(counts)

    def total_intervention_time(self) -> float:
        """Sum of all human reset durations in seconds. (E3)"""
        return sum(
            e.reset_prompt_to_confirm for e in self.episodes
            if e.reset_prompt_to_confirm is not None
        )

    # ---------- I/O ----------

    def save(self, path: str = None):
        """Save all episode records to a JSON-lines file."""
        if path is None:
            path = os.path.join(self.log_dir, "episodes.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            for record in self.episodes:
                f.write(json.dumps(asdict(record)) + "\n")

        # Save intervention log
        intervention_path = os.path.join(self.log_dir, "intervention_log.jsonl")
        with open(intervention_path, "w") as f:
            for record in self.interventions:
                f.write(json.dumps(asdict(record)) + "\n")

        # Save summary
        summary_path = os.path.join(self.log_dir, "summary.json")
        summary = {
            "run_id": self.run_id,
            "policy_method": self.policy_method,
            "total_episodes": len(self.episodes),
            "total_successes": self.cumulative_valid_trajectories(),
            "success_rate": self.cumulative_success_rate(),
            "interventions": sum(1 for e in self.episodes if e.human_reset),
            "valid_per_hour": self.valid_trajectories_per_hour(),
            "interventions_per_hour": self.interventions_per_hour(),
            "interventions_per_100_valid": self.interventions_per_valid_episodes(100),
            "total_intervention_time_s": self.total_intervention_time(),
            "failure_type_distribution": self.failure_type_distribution(),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    @classmethod
    def load(cls, path: str) -> List[EpisodeRecord]:
        """Load episode records from a JSON-lines file."""
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line.strip())
                    records.append(EpisodeRecord(**d))
        return records

    @staticmethod
    def load_interventions(path: str) -> List[InterventionRecord]:
        """Load intervention records from a JSON-lines file."""
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line.strip())
                    records.append(InterventionRecord(**d))
        return records
