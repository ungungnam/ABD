"""No-reset baseline policy — never requests a human reset."""

from validator.base_validator import ValidationResult
from policy.base_policy import BaseResetPolicy


class NoResetPolicy(BaseResetPolicy):

    def needs_reset(self, validation: ValidationResult, fail_count: int,
                    episode_idx: int, task=None, observation=None,
                    detection_info: dict = None) -> bool:
        return False
