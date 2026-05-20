"""Backward-only ABD ablation policy.

This variant uses only success detection for the previous task. If the previous
task succeeded, collection advances. If it failed, the policy requests reset
without allowing retry.

"policy=abd_backward_only"
"""

from typing import List, Optional, Tuple

from policy.vlm_checklist_policy import VLMChecklistPolicy


class BackwardOnlyPolicy(VLMChecklistPolicy):
    """Backward-only = previous task success detection only."""

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

        success_items = [
            item for item in checklist["current_task"]
            if item.get("success_item", False)
        ]
        if not success_items:
            success_items = checklist["current_task"]

        answers = self._ask_vlm(success_items, preamble, detection_block, obs, task)

        success_answers = [
            answers.get(str(item["id"]), "no") == item.get("success_answer", "yes")
            for item in success_items
        ]
        success = all(success_answers) if success_answers else None

        score = 1.0 if success else 0.0
        per_item = []
        for item in success_items:
            ans = answers.get(str(item["id"]), "no")
            expected = item.get("success_answer", "yes")
            per_item.append({
                "id": item["id"],
                "question": item["question"],
                "weight": item["weight"],
                "answer": ans,
                "contrib": 1.0 if ans == expected else 0.0,
                "success_item": bool(item.get("success_item", False)),
                "success_answer": expected,
                "phase": "backward_success",
            })
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
