"""Naive reset detection baseline policy (Section 12.3).

Resets on first failure, or if target becomes invisible / unreachable.
"""

from validator.base_validator import ValidationResult
from policy.base_policy import BaseResetPolicy


class NaivePolicy(BaseResetPolicy):

    def decide(self, validation: ValidationResult, fail_count: int,
               episode_idx: int, features: dict = None) -> str:
        # Reset on any failure
        if not validation.success:
            return "reset"

        # Reset if target invisible or unreachable (when features available)
        if features is not None:
            if features.get("f_vis", 1.0) < 0.5:
                return "reset"
            if features.get("f_reach", 1.0) < 0.5:
                return "reset"

        return "next"
