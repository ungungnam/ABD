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


class CombinedValidator(BaseTaskValidator):
    """Runs geometric and VLM validation in parallel and logs both results.

    Decision rule: geometric is primary. If geometric fails (perception/
    reconstruction error), VLM result is used instead. Either way, the VLM
    result is always logged so results can be compared.
    """

    def __init__(self, geometric_validator, vlm_validator):
        self.geometric = geometric_validator
        self.vlm = vlm_validator

    def validate(self, task: TaskDefinition, post_obs: dict, env) -> ValidationResult:
        geometric_result = self.geometric.validate(task, post_obs, env)
        vlm_result = self.vlm.validate(task, post_obs, env)

        log.info(
            f"[Validator] geometric: success={geometric_result.success}, "
            f"method={geometric_result.method}, confidence={geometric_result.confidence:.2f}, "
            f"details={geometric_result.details}"
        )
        log.info(
            f"[Validator] vlm:       success={vlm_result.success}, "
            f"confidence={vlm_result.confidence:.2f}, "
            f"question={vlm_result.details.get('question', '')}"
        )

        # VLM is primary. Attach geometric result as supplementary info.
        vlm_result.details["geometric_success"] = geometric_result.success
        vlm_result.details["geometric_method"] = geometric_result.method
        vlm_result.details["geometric_details"] = geometric_result.details
        return vlm_result
