"""Naive reset baseline policy — resets on any failure."""

from validator.base_validator import ValidationResult
from policy.base_policy import BaseResetPolicy


class NaivePolicy(BaseResetPolicy):

    def needs_reset(self, validation: ValidationResult, fail_count: int,
                    episode_idx: int, task=None, observation=None,
                    detection_info: dict = None) -> bool:
        if validation is None:
            return True
        return not validation.success
