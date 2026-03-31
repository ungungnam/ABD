from abc import ABC, abstractmethod

from validator.base_validator import ValidationResult


class BaseResetPolicy(ABC):
    """Abstract interface for reset policies.

    Each policy decides whether to: 'next', 'retry', or 'reset'
    after an episode completes.
    """

    @abstractmethod
    def decide(self, validation: ValidationResult, fail_count: int,
               episode_idx: int, features: dict = None) -> str:
        """Decide the next control action.

        Args:
            validation: Task success validation result.
            fail_count: Consecutive failure count.
            episode_idx: Current episode index.
            features: ABD feature dict (may be None for non-ABD policies).

        Returns:
            One of: 'next', 'retry', 'reset'.
        """
        ...
