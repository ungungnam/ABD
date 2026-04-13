from abc import ABC, abstractmethod

from validator.base_validator import ValidationResult


class BaseResetPolicy(ABC):
    """Abstract interface for reset policies.

    Each policy answers one question: does the environment need a human reset
    before the next episode?  The runner applies the decision matrix:

        success=True,  needs_reset=False  →  next  (advance direction)
        success=True,  needs_reset=True   →  reset + reverse
        success=False, needs_reset=False  →  retry
        success=False, needs_reset=True   →  reset + forward
    """

    @abstractmethod
    def needs_reset(self, validation: ValidationResult, fail_count: int,
                    episode_idx: int, task=None, observation=None) -> bool:
        """Return True if the environment needs a human reset."""
        ...
