"""VLM-based task success validator (fallback).

Queries VLM with a yes/no question about task completion.
Used when geometric validation fails (perception issues).
"""

from task.task_family import TaskDefinition
from validator.base_validator import BaseTaskValidator, ValidationResult
from vlm_client.vqa_client import VQAClient


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


class CombinedValidator(BaseTaskValidator):
    """Tries geometric validation first, falls back to VLM on perception failure."""

    def __init__(self, geometric_validator, vlm_validator):
        self.geometric = geometric_validator
        self.vlm = vlm_validator

    def validate(self, task: TaskDefinition, post_obs: dict, env) -> ValidationResult:
        result = self.geometric.validate(task, post_obs, env)

        if result.method in ("perception_failed", "reconstruction_failed"):
            # Fall back to VLM-based validation
            vlm_result = self.vlm.validate(task, post_obs, env)
            vlm_result.details["fallback_from"] = result.method
            vlm_result.details["geometric_reason"] = result.details.get("reason", "")
            return vlm_result

        return result
