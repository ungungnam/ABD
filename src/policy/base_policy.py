from abc import ABC, abstractmethod

from validator.base_validator import ValidationResult


class BaseResetPolicy(ABC):
    """Abstract interface for reset policies.

    Each policy decides whether to: 'next', 'retry', or 'reset'
    after an episode completes.
    """

    @abstractmethod
    def decide(self, validation: ValidationResult, fail_count: int,
               episode_idx: int, features: dict = None,
               task=None, observation=None) -> str:
        """Decide the next control action.

        Args:
            validation: Task success validation result.
            fail_count: Consecutive failure count.
            episode_idx: Current episode index.
            features: ABD feature dict (may be None for non-ABD policies).
            task: Current TaskDefinition (used by policies that need
                task name / language description, e.g. VLMChecklistPolicy).
            observation: Post-execution observation dict (used by policies
                that re-query a VLM at decision time).

        Returns:
            One of: 'next', 'retry', 'reset'.
        """
        ...
