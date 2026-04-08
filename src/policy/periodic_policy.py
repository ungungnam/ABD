"""Periodic reset baseline policy (Section 12.2).

Requests human reset every N episodes.
"""

from validator.base_validator import ValidationResult
from policy.base_policy import BaseResetPolicy


class PeriodicPolicy(BaseResetPolicy):

    def __init__(self, period: int = 10):
        self.period = period
        self._episode_since_reset = 0

    def decide(self, validation: ValidationResult, fail_count: int,
               episode_idx: int, features: dict = None,
               task=None, observation=None) -> str:
        self._episode_since_reset += 1

        if self._episode_since_reset >= self.period:
            self._episode_since_reset = 0
            return "reset"

        return "next" if validation.success else "retry"
