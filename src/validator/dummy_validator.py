"""Dummy task success validator for testing.

Returns probabilistic success/failure without real perception.
Success rate degrades as the environment drifts from canonical state.
"""

import numpy as np

from task.task_family import TaskDefinition
from validator.base_validator import BaseTaskValidator, ValidationResult


class DummyValidator(BaseTaskValidator):
    """Simulates task validation with configurable success probability.

    Success probability degrades with consecutive failures (simulating
    state drift making the task harder).
    """

    def __init__(self, base_success_rate: float = 0.7, seed: int = 42):
        self.base_success_rate = base_success_rate
        self.rng = np.random.RandomState(seed)
        self._consecutive_failures = 0

    def validate(self, task: TaskDefinition, post_obs: dict, env) -> ValidationResult:
        # Degrade success rate with drift (simulated by consecutive failures)
        drift_penalty = min(0.4, self._consecutive_failures * 0.08)
        effective_rate = max(0.1, self.base_success_rate - drift_penalty)

        success = self.rng.rand() < effective_rate

        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        xy_dist = self.rng.exponential(0.03) if success else self.rng.exponential(0.08)

        return ValidationResult(
            success=success,
            confidence=0.8 if success else 0.4,
            method="dummy_geometric",
            details={
                "xy_distance": float(xy_dist),
                "threshold": 0.05,
                "effective_success_rate": effective_rate,
            },
        )

    def reset_state(self):
        """Call after human reset to restore success probability."""
        self._consecutive_failures = 0
