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
                    "Are any of the three cups still stacked on top of another cup?"
                )
            else:  # stack
                question = f"Are all three cups fully stacked together?"
        elif task.task_type == "open_drawer":
            drawer_state = task.canonical_state.get("drawer", "open")
            if drawer_state == "open":
                question = "Is the drawer open?"
            else:
                question = "Is the drawer closed?"
        else:
            obj_name    = task.canonical_state["object"]
            target_name = task.canonical_state["target"]
            # pan-type containers use "in", flat surfaces (plate/tray) use "on"
            _in_targets = {"pan", "bowl", "pot", "box", "container", "basket"}
            prep = "in" if target_name.lower() in _in_targets else "on"
            question = f"Is the {obj_name} placed {prep} the {target_name}?"

        log.info(f"[VLMValidator] task={task.name} | query: {question}")
        raw = self.vqa_client.ask_yes_no(post_obs, question)

        # Unstack queries use negative framing ("Are any cups still stacked?")
        # so the success condition is the opposite: No → success, Yes → failure.
        invert = (
            task.task_type == "stack_cups"
            and task.stack_step.startswith("reverse")
        )
        success = (not raw) if invert else raw
        log.info(f"[VLMValidator] raw={raw} invert={invert} → success={success}")

        # stack_cups terminal step: if VQA says task failed, the scene is no longer
        # in its initial state (cups partially stacked/unstacked) and needs human reset.
        needs_reset = (
            not success
            and task.task_type == "stack_cups"
            and task.stack_step in ("forward_2", "reverse_2")
        )

        return ValidationResult(
            success=success,
            confidence=0.6,
            method="vlm",
            details={"question": question},
            needs_reset=needs_reset,
        )


