from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskDefinition:
    """Single atomic task in the collection loop."""
    name: str                          # e.g., "banana_to_pan"
    language_task: str                 # e.g., "pick the banana and put it in the pan"
    task_type: str                     # e.g., "pick_place"
    canonical_state: dict = field(default_factory=dict)
    # canonical_state example: {"object": "banana", "location": "plate"}


@dataclass
class ReversibleTaskPair:
    """A pair of tasks that undo each other (A->B / B->A)."""
    forward: TaskDefinition
    reverse: TaskDefinition


class TaskScheduler:
    """Alternates forward/reverse within a ReversibleTaskPair.

    After a successful forward task, advances to reverse (and vice versa),
    so the environment returns to near-canonical state after each pair.
    """

    def __init__(self, task_pair: ReversibleTaskPair):
        self._pair = task_pair
        self._is_forward = True
        self._total_advances = 0

    def current_task(self) -> TaskDefinition:
        return self._pair.forward if self._is_forward else self._pair.reverse

    def advance(self):
        """Toggle forward <-> reverse after a successful episode."""
        self._is_forward = not self._is_forward
        self._total_advances += 1

    def reset_to_forward(self):
        """After human reset, restart from forward direction."""
        self._is_forward = True

    @property
    def is_forward(self) -> bool:
        return self._is_forward

    @property
    def total_advances(self) -> int:
        return self._total_advances
