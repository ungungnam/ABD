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
        obj_name = task.canonical_state["object"]
        target_name = task.canonical_state["target"]

        question = f"Is the {obj_name} in or on the {target_name}?"
        success = self.vqa_client.ask_yes_no(post_obs, question)

        return ValidationResult(
            success=success,
            confidence=0.6,
            method="vlm",
            details={"question": question},
        )


