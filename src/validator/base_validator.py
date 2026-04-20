from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from task.task_family import TaskDefinition


@dataclass
class ValidationResult:
    """Result of task success validation."""
    success: bool
    confidence: float = 0.0          # 0.0 to 1.0
    method: str = "unknown"          # "geometric", "vlm", "perception_failed", etc.
    details: dict = field(default_factory=dict)
    needs_reset: bool = False        # True → scene must be reset before next attempt


class BaseTaskValidator(ABC):
    """Abstract interface for task success validation."""

    @abstractmethod
    def validate(self, task: TaskDefinition, post_obs: dict, env) -> ValidationResult:
        """Determine whether the task was successfully completed.

        Args:
            task: The task that was attempted.
            post_obs: Observation after execution.
            env: The environment (for querying additional state).

        Returns:
            ValidationResult with success flag, confidence, and details.
        """
        ...
