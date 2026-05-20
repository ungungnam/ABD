"""Forward-only ABD ablation policy.

This variant assumes the previous task succeeded and evaluates only the paired
next task's feasibility checklist. It does not use Phase 1 success detection.

"policy=abd_forward_only"
"""

from typing import List, Optional, Tuple

from policy.vlm_checklist_policy import VLMChecklistPolicy, _PAIRED_CHECKLIST


class ForwardOnlyPolicy(VLMChecklistPolicy):
    """Forward-only = next task feasibility only."""

    def _evaluate(
        self, checklist: dict, task, observation: dict, detection_info: dict = None
    ) -> Tuple[float, List[dict], Optional[bool]]:
        if task.task_type == "open_drawer":
            preamble = _drawer_preamble()
            obs = {k: v for k, v in observation.items() if "front" not in k}
        else:
            preamble = _default_preamble()
            obs = observation

        info = detection_info or {}
        positions = info.get("__object_positions__") or {}
        bounds = info.get("__workspace_bounds__") or {}
        det_status = {k: v for k, v in info.items() if not k.startswith("__")}

        detection_block = self._build_context_block(det_status, positions, bounds)

        success = True
        current_name = self._canonical_checklist_name(task)
        next_name = _PAIRED_CHECKLIST.get(current_name, current_name)

        if next_name == current_name:
            p2_checklist = checklist
        else:
            p2_checklist = self._load_by_name(next_name)

        items_p2: List[dict] = []
        answers_p2: dict = {}
        if p2_checklist:
            items_p2 = p2_checklist.get("next_task", [])
            if items_p2:
                answers_p2 = self._ask_vlm(
                    items_p2, preamble, detection_block, obs, task
                )

        score, per_item = self._compute_score([], {}, items_p2, answers_p2)
        for item in per_item:
            item["phase"] = "forward"
        return score, per_item, success


def _default_preamble() -> str:
    return (
        "The provided images are different views of the same scene at the same time. "
        "Use all images together to evaluate the checklist items. "
        "If one view is ambiguous because of occlusion or perspective, rely more on "
        "the clearer external views."
    )


def _drawer_preamble() -> str:
    return (
        "The provided images are two views of the same scene at the same time: "
        "one from a wrist-mounted camera on the robot arm, and one from a "
        "table-level camera. Use both images together to evaluate the checklist "
        "items. If one view is ambiguous because of occlusion or perspective, "
        "rely more on the clearer view."
    )
