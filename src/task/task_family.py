from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class TaskDefinition:
    """Single atomic task in the collection loop."""
    name: str                          # e.g., "banana_to_pan"
    language_task: str                 # e.g., "pick the banana and put it in the pan"
    task_type: str                     # "pick_place" | "stack_cups"
    canonical_state: dict = field(default_factory=dict)
    # canonical_state example: {"object": "banana", "location": "plate", "target": "pan"}
    # stack_cups only:
    pick_tag_id: Optional[int] = None      # AprilTag ID on the cup to pick
    place_tag_id: Optional[int] = None     # AprilTag ID on the target cup / reference cup
    place_xy_offset: Optional[list] = None # non-None → unstack mode (use cached position)
    validation_question: Optional[str] = None  # Override VLM validation question for terminal steps
    stack_step: Optional[str] = None       # "forward_1" | "forward_2" | "reverse_1" | "reverse_2"
    # Intermediate tag check (stack_cups only): run before execution of this step
    intermediate_check_tag_id: Optional[int] = None   # tag to detect
    intermediate_check_skip_if_visible: bool = True    # True → skip if tag visible; False → skip if tag hidden


@dataclass
class ReversibleTaskPair:
    """A pair of tasks that undo each other (A->B / B->A)."""
    forward: TaskDefinition
    reverse: TaskDefinition


@dataclass
class TaskSequence:
    """Ordered list of tasks executed in a cycle (forward steps then reverse steps)."""
    tasks: List[TaskDefinition]
    n_forward: int  # first n_forward tasks are "forward"; the rest are "reverse"


class TaskScheduler:
    """Advances through a task sequence one step at a time.

    Accepts either a ReversibleTaskPair (2-step) or a TaskSequence (N-step).
    After a successful episode advance() moves to the next task; after the last
    task the cycle wraps back to the first.
    reset_to_forward() always returns to step 0.
    """

    def __init__(self, task_source: Union[ReversibleTaskPair, TaskSequence]):
        if isinstance(task_source, ReversibleTaskPair):
            self._tasks = [task_source.forward, task_source.reverse]
            self._n_forward = 1
        else:
            self._tasks = task_source.tasks
            self._n_forward = task_source.n_forward
        self._idx = 0
        self._total_advances = 0

    def current_task(self) -> TaskDefinition:
        return self._tasks[self._idx]

    def next_task(self) -> TaskDefinition:
        """Task that will run after the next advance()."""
        return self._tasks[(self._idx + 1) % len(self._tasks)]

    def advance(self):
        self._idx = (self._idx + 1) % len(self._tasks)
        self._total_advances += 1

    def reset_to_forward(self):
        self._idx = 0

    def reset_to_phase_start(self):
        """On retry: go back to the first step of the current phase.
        Forward phase (idx < n_forward) → step 0.
        Reverse phase (idx >= n_forward) → step n_forward.
        """
        if self._idx < self._n_forward:
            self._idx = 0
        else:
            self._idx = self._n_forward

    @property
    def is_forward(self) -> bool:
        return self._idx < self._n_forward

    @property
    def is_intermediate_forward(self) -> bool:
        """True if this is a forward step that is NOT the last forward step.
        Intermediate steps auto-advance without VLM validation."""
        return self._idx < self._n_forward - 1

    @property
    def is_terminal_step(self) -> bool:
        """True only for the last forward step and the last reverse step.
        Only terminal steps trigger VLM validation and ground truth labeling."""
        return self._idx == self._n_forward - 1 or self._idx == len(self._tasks) - 1

    @property
    def total_advances(self) -> int:
        return self._total_advances
