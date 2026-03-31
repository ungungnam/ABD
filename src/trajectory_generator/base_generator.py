from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from task.task_family import TaskDefinition


@dataclass
class GenerationResult:
    """Result of trajectory generation."""
    trajectory: Optional[List[np.ndarray]]   # list of 4x4 SE(3) waypoints, or None on failure
    events: List[dict] = field(default_factory=list)  # [{"at": idx, "cmd": "CLOSE"}, ...]
    metadata: dict = field(default_factory=dict)


class BaseTrajectoryGenerator(ABC):
    """Abstract interface for trajectory generation."""

    @abstractmethod
    def generate(self, task: TaskDefinition, env) -> GenerationResult:
        """Generate a trajectory for the given task.

        Returns GenerationResult with trajectory, events, and metadata.
        trajectory is None if generation fails.
        """
        ...
