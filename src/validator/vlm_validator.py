"""VLM-based task success validator (fallback).

Queries VLM with a yes/no question about task completion.
Used when geometric validation fails (perception issues).
"""

import logging

from task.task_family import TaskDefinition
from validator.base_validator import BaseTaskValidator, ValidationResult
from vlm_client.vqa_client import VQAClient

log = logging.getLogger(__name__)


class VLMValidator(BaseTaskValidator):
    """VLM VQA-based task success validator."""

    def __init__(self, vqa_client: VQAClient):
        self.vqa_client = vqa_client

    def validate(self, task: TaskDefinition, post_obs: dict, env) -> ValidationResult:
        # Use task-level override if provided (e.g. terminal steps with multi-cup questions)
        if task.validation_question:
            question = task.validation_question
        elif task.task_type == "stack_cups":
            pick_cup  = task.canonical_state.get("pick", "purple")
            place_cup = task.canonical_state.get("place", "blue")
            if task.place_xy_offset:  # unstack
                question = (
                    f"Is the {pick_cup} cup on the table and NOT stacked on top of the {place_cup} cup?"
                )
            else:  # stack
                question = f"Is the {pick_cup} cup stacked on top of the {place_cup} cup (directly or indirectly)?"
        else:
            obj_name    = task.canonical_state["object"]
            target_name = task.canonical_state["target"]
            question = f"Is the {obj_name} in or on the {target_name}?"

        success = self.vqa_client.ask_yes_no(post_obs, question)

        return ValidationResult(
            success=success,
            confidence=0.6,
            method="vlm",
            details={"question": question},
        )


