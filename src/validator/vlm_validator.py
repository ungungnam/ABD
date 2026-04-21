"""VLM-based task success validator (fallback).

Queries VLM with a yes/no question about task completion.
Used when geometric validation fails (perception issues).

stack_cups: two-query approach
  forward: both yes  → success (purple on blue AND pink on purple)
  reverse: both no   → success (neither cup is stacked)
"""

import logging

from task.task_family import TaskDefinition
from validator.base_validator import BaseTaskValidator, ValidationResult
from vlm_client.vqa_client import VQAClient

log = logging.getLogger(__name__)

_STACK_Q1 = "Is the purple cup stacked on top of the blue cup?"
_STACK_Q2 = "Is the pink cup stacked on top of the purple cup?"

_PREAMBLE_STACK_CUPS = (
    "The provided images are different views of the same scene at the same time. "
    "They show the same three cups from different camera angles. "
    "Use all images together to judge the spatial relationship between the cups. "
    "If one view is ambiguous because of occlusion or perspective, rely more on the clearer external views."
)
_PREAMBLE_OPEN_DRAWER = (
    "The provided images are different views of the same scene at the same time. "
    "They show a drawer from different camera angles. "
    "Use all images together to judge the state of the drawer. "
    "If one view is ambiguous because of occlusion or perspective, rely more on the clearer external views."
)
_PREAMBLE_PICK_PLACE = (
    "The provided images are different views of the same scene at the same time. "
    "They show the same objects from different camera angles. "
    "Use all images together to judge the position and placement of the object. "
    "If one view is ambiguous because of occlusion or perspective, rely more on the clearer external views."
)


class VLMValidator(BaseTaskValidator):
    """VLM VQA-based task success validator."""

    def __init__(self, vqa_client: VQAClient):
        self.vqa_client = vqa_client

    def validate(self, task: TaskDefinition, post_obs: dict, env) -> ValidationResult:
        # stack_cups: two-query approach (wrist cam 포함)
        if task.task_type == "stack_cups":
            return self._validate_stack_cups(task, post_obs)

        # Other tasks: single yes/no query
        if task.validation_question:
            question = task.validation_question
        elif task.task_type == "open_drawer":
            drawer_state = task.canonical_state.get("drawer", "open")
            question = "Is the drawer open?" if drawer_state == "open" else "Is the drawer closed?"
        else:
            obj_name    = task.canonical_state["object"]
            target_name = task.canonical_state["target"]
            _in_targets = {"pan", "bowl", "pot", "box", "container", "basket"}
            prep = "in" if target_name.lower() in _in_targets else "on"
            question = f"Is the {obj_name} placed {prep} the {target_name}?"

        if task.task_type == "open_drawer":
            preamble = _PREAMBLE_OPEN_DRAWER
        else:
            preamble = _PREAMBLE_PICK_PLACE

        log.info(f"[VLMValidator] task={task.name} | query: {question}")
        raw = self.vqa_client.ask_yes_no(post_obs, question, preamble=preamble)
        log.info(f"[VLMValidator] raw={raw} → success={raw}")

        return ValidationResult(
            success=raw,
            confidence=0.6,
            method="vlm",
            details={"question": question},
            needs_reset=False,
        )

    def _validate_stack_cups(self, task: TaskDefinition, post_obs: dict) -> ValidationResult:
        """Two-query success detection for stack_cups.

        forward: purple on blue (Q1=yes) AND pink on purple (Q2=yes) → success
        reverse: purple NOT on blue (Q1=no) AND pink NOT on purple (Q2=no) → success
        """
        r1 = self.vqa_client.ask_yes_no(post_obs, _STACK_Q1, preamble=_PREAMBLE_STACK_CUPS)
        r2 = self.vqa_client.ask_yes_no(post_obs, _STACK_Q2, preamble=_PREAMBLE_STACK_CUPS)

        is_reverse = task.stack_step.startswith("reverse")
        if is_reverse:
            success = (not r1) and (not r2)
        else:
            success = r1 and r2

        log.info(
            f"[VLMValidator] task={task.name} | "
            f"Q1={_STACK_Q1!r} → {r1} | "
            f"Q2={_STACK_Q2!r} → {r2} | "
            f"reverse={is_reverse} → success={success}"
        )

        needs_reset = (
            not success
            and task.stack_step in ("forward_2", "reverse_2")
        )

        return ValidationResult(
            success=success,
            confidence=0.7,
            method="vlm",
            details={"q1": _STACK_Q1, "r1": r1, "q2": _STACK_Q2, "r2": r2},
            needs_reset=needs_reset,
        )
