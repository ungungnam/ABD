"""Periodic reset baseline policy — resets every N episodes."""

from validator.base_validator import ValidationResult
from policy.base_policy import BaseResetPolicy


class PeriodicPolicy(BaseResetPolicy):

    def __init__(self, period: int = 10):
        self.period = period
        self._episodes_since_reset = 0

    def needs_reset(self, validation: ValidationResult, fail_count: int,
                    episode_idx: int, task=None, observation=None,
                    detection_info: dict = None) -> bool:
        self._episodes_since_reset += 1
        if self._episodes_since_reset >= self.period:
            self._episodes_since_reset = 0
            return True
        return False
