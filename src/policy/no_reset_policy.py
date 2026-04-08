"""No-reset baseline policy (Section 12.1).

Never requests human reset. On failure, retries the current task.
"""

from validator.base_validator import ValidationResult
from policy.base_policy import BaseResetPolicy


class NoResetPolicy(BaseResetPolicy):

    def decide(self, validation: ValidationResult, fail_count: int,
               episode_idx: int, features: dict = None,
               task=None, observation=None) -> str:
        return "next" if validation.success else "retry"
