"""Dummy ABD feature extractor for testing.

Produces synthetic 6D feature vectors correlated with
validation results and failure state.
"""

import numpy as np

from task.task_family import TaskDefinition
from validator.base_validator import ValidationResult


class DummyFeatureExtractor:
    """Generates realistic-looking ABD features without perception.

    Features degrade naturally with consecutive failures and
    improve after resets, simulating real state drift.
    """

    def __init__(self, max_fail_count: int = 5, seed: int = 42):
        self.max_fail_count = max_fail_count
        self.rng = np.random.RandomState(seed)
        self._drift = 0.0  # cumulative state drift

    def calibrate(self, task: TaskDefinition):
        """Reset drift after human intervention."""
        self._drift = 0.0

    def extract(self, task: TaskDefinition, validation_result: ValidationResult,
                fail_count: int) -> dict:
        noise = self.rng.randn(6) * 0.05

        f_succ = float(validation_result.success)

        # Visibility: usually good, degrades slightly with drift
        f_vis = np.clip(1.0 - self._drift * 0.3 + noise[1], 0.0, 1.0)

        # Reachability: usually good, degrades with drift
        f_reach = np.clip(1.0 - self._drift * 0.2 + noise[2], 0.0, 1.0)

        # Recoverability
        f_rec = np.clip(f_vis * f_reach + noise[3] * 0.1, 0.0, 1.0)

        # State deviation: grows with failures
        f_dev = np.clip(self._drift + noise[4] * 0.05, 0.0, 1.0)

        # Failure count
        f_fail = min(1.0, fail_count / max(self.max_fail_count, 1))

        # Update drift
        if not validation_result.success:
            self._drift = min(1.0, self._drift + 0.1 + self.rng.rand() * 0.05)
        else:
            self._drift = max(0.0, self._drift - 0.02)

        vector = np.array([f_succ, f_vis, f_reach, f_rec, f_dev, f_fail])

        return {
            "f_succ": f_succ,
            "f_vis": float(f_vis),
            "f_reach": float(f_reach),
            "f_rec": float(f_rec),
            "f_dev": float(f_dev),
            "f_fail": float(f_fail),
            "vector": vector,
        }
