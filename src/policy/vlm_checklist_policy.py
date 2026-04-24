"""VLM checklist reset policy.

Generates a yes/no checklist (with weights) for each task by prompting a VLM
once, then evaluates the checklist against post-execution observations to
decide whether the environment needs a human reset.

    needs_reset = (score < tau_reset)

where score = Σ(weight_i × yes_i) / Σ(weight_i).

Per-task checklists are persisted to ``config/checklists/<task_name>.json``
so users can hand-edit weights and question wording between runs.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from policy.base_policy import BaseResetPolicy
from policy.checklist_prompts import META_PROMPT, EVAL_PROMPT
from validator.base_validator import ValidationResult
from vlm_client.vqa_client import VQAClient

_CHECKLIST_PREAMBLE = (
    "The provided images are different views of the same scene at the same time. "
    "Use all images together to evaluate the checklist items. "
    "If one view is ambiguous because of occlusion or perspective, rely more on the clearer external views."
)
_CHECKLIST_PREAMBLE_DRAWER = (
    "The provided images are two views of the same scene at the same time: "
    "one from a wrist-mounted camera on the robot arm, and one from a table-level camera. "
    "Use both images together to evaluate the checklist items. "
    "If one view is ambiguous because of occlusion or perspective, rely more on the clearer view."
)

log = logging.getLogger(__name__)


class VLMChecklistPolicy(BaseResetPolicy):
    """Reset policy that asks a VLM to fill out a per-task checklist.

    needs_reset = (score < tau_reset)
    where score = Σ(weight_i × yes_i) / Σ(weight_i)
    """

    def __init__(
        self,
        vqa_client: VQAClient,
        checklist_dir: str,
        tau_reset: float = 0.9,
    ):
        self.vqa_client = vqa_client
        self.checklist_dir = Path(checklist_dir)
        self.checklist_dir.mkdir(parents=True, exist_ok=True)
        self.tau_reset = tau_reset
        self._checklists: dict = {}  # task_name -> loaded checklist dict
        self.last_eval: Optional[dict] = None
        self.last_success: Optional[bool] = None  # set by needs_reset(); None if no success_item defined

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def needs_reset(
        self,
        validation: ValidationResult,
        fail_count: int,
        episode_idx: int,
        task=None,
        observation=None,
        detection_info: dict = None,
    ) -> bool:
        if task is None or observation is None:
            self.last_eval = None
            self.last_success = None
            return False

        checklist = self._load_or_generate(task)
        score, per_item, success = self._evaluate(
            checklist, task, observation, detection_info=detection_info
        )
        self.last_eval = {
            "task_name": task.name,
            "score": score,
            "items": per_item,
        }
        self.last_success = success  # None when no success_item defined in checklist

        reset = score < self.tau_reset

        log.info(
            f"[VLMChecklistPolicy] task={task.name} | score={score:.3f} | tau={self.tau_reset} | "
            f"needs_reset={reset} | success={success}"
        )
        for item in per_item:
            marker = "★" if item.get("success_item") else " "
            log.info(f"  {marker}[{item['answer'].upper():3s}] (w={item['weight']:.2f}) {item['question']}")

        return reset

    # ------------------------------------------------------------------ #
    # Checklist generation / loading
    # ------------------------------------------------------------------ #

    def _checklist_path(self, task_name: str) -> Path:
        return self.checklist_dir / f"{task_name}.json"

    def _canonical_checklist_name(self, task) -> str:
        """Return the canonical checklist file stem for this task.

        stack_cups steps share two checklists regardless of their individual
        task.name (which varies by cup colour):
          forward steps (stack_step starts with "forward") → stack_cups_forward
          reverse steps (stack_step starts with "reverse") → stack_cups_reverse
        All other tasks use task.name directly.
        """
        if task.task_type == "stack_cups":
            is_forward = task.stack_step.startswith("forward")
            return "stack_cups_forward" if is_forward else "stack_cups_reverse"
        return task.name

    def _load_or_generate(self, task) -> dict:
        name = self._canonical_checklist_name(task)

        if name in self._checklists:
            return self._checklists[name]

        path = self._checklist_path(name)
        if path.exists():
            with open(path) as f:
                checklist = json.load(f)
            log.info(f"[VLMChecklistPolicy] Loaded checklist from {path}")
        else:
            log.info(
                f"[VLMChecklistPolicy] No checklist for '{name}' at {path}; "
                f"generating via VLM."
            )
            checklist = self._generate_checklist(task)
            with open(path, "w") as f:
                json.dump(checklist, f, indent=2)
            log.info(f"[VLMChecklistPolicy] Wrote new checklist to {path}")

        self._validate_checklist_schema(checklist, name)
        self._checklists[name] = checklist
        return checklist

    def _generate_checklist(self, task) -> dict:
        prompt = META_PROMPT.format(
            task_name=task.name,
            task_description=task.language_task,
        )
        # Text-only generation: no images needed at this stage.
        raw = self.vqa_client.ask_text(None, prompt)
        if not raw:
            raise RuntimeError(
                f"VLM returned empty response while generating checklist for "
                f"'{task.name}'. Check the VLM server."
            )
        try:
            return _extract_json_object(raw)
        except ValueError as e:
            raise RuntimeError(
                f"Failed to parse checklist JSON for '{task.name}': {e}\n"
                f"Raw VLM response:\n{raw}"
            )

    @staticmethod
    def _validate_checklist_schema(checklist: dict, task_name: str) -> None:
        if "items" not in checklist or not isinstance(checklist["items"], list):
            raise ValueError(
                f"Checklist for '{task_name}' missing 'items' list."
            )
        if not checklist["items"]:
            raise ValueError(f"Checklist for '{task_name}' has no items.")
        for i, item in enumerate(checklist["items"]):
            for key in ("id", "question", "weight"):
                if key not in item:
                    raise ValueError(
                        f"Checklist '{task_name}' item {i} missing '{key}'."
                    )

    # ------------------------------------------------------------------ #
    # Checklist evaluation
    # ------------------------------------------------------------------ #

    def _evaluate(
        self, checklist: dict, task, observation: dict, detection_info: dict = None
    ) -> Tuple[float, List[dict], Optional[bool]]:
        """Evaluate checklist against observation.

        Returns:
            score:          weighted reset score
            per_item:       per-item details
            success:        True/False from success_item answers; None if none defined
            reset_override: True  → mixed group answers → force reset
                            False → all-wrong group answers → force retry (no reset)
                            None  → no group constraint, use score threshold
        """
        items = checklist["items"]
        items_block = "\n".join(
            f"{item['id']}. {item['question']}" for item in items
        )
        if task.task_type == "open_drawer":
            preamble = _CHECKLIST_PREAMBLE_DRAWER
            obs = {k: v for k, v in observation.items() if "front" not in k}
        else:
            preamble = _CHECKLIST_PREAMBLE
            obs = observation

        detection_block = ""
        if detection_info:
            det_lines = " / ".join(
                f"{obj}: {status}" for obj, status in detection_info.items()
            )
            detection_block = f"\nObject detection status: {det_lines}\n"

        prompt = preamble + detection_block + "\n" + EVAL_PROMPT.format(
            task_description=task.language_task,
            items_block=items_block,
        )
        if detection_block:
            log.info(f"[VLMChecklistPolicy] task={task.name} | detection:{detection_block.strip()}")
        raw = self.vqa_client.ask_text(obs, prompt)

        answers = self._parse_answers(raw, items)

        # Determine which reset_groups have mixed (inconsistent) answers.
        # Mixed group items score 0 to drive score below tau_reset.
        group_answers: dict = {}
        for item in items:
            gid = item.get("reset_group")
            if gid is not None:
                ans = answers.get(str(item["id"]), "no")
                group_answers.setdefault(gid, set()).add(ans)
        mixed_groups = {gid for gid, ans_set in group_answers.items() if len(ans_set) > 1}

        per_item: List[dict] = []
        weighted_sum = 0.0
        weight_total = 0.0
        success_answers: List[bool] = []

        for item in items:
            ans = answers.get(str(item["id"]), "no")
            yes = ans == "yes"
            w = float(item["weight"])
            is_success_item = bool(item.get("success_item", False))
            success_answer = item.get("success_answer", "yes")

            if is_success_item:
                success_answers.append(ans == success_answer)

            in_mixed_group = item.get("reset_group") in mixed_groups
            in_reset_group = item.get("reset_group") is not None
            if in_reset_group:
                # Consistency check: consistent answers (all-yes or all-no) → 1
                # Mixed answers → 0 (force reset via low score)
                # Use abs(w) so negative-weight items still contribute positively
                # when answers are consistent.
                weighted_sum += abs(w) * (0.0 if in_mixed_group else 1.0)
            else:
                weighted_sum += w * (1.0 if yes else 0.0)
            weight_total += abs(w)

            per_item.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "weight": item["weight"],
                    "answer": ans,
                    "success_item": is_success_item,
                    "success_answer": success_answer,
                }
            )

        score = weighted_sum / weight_total if weight_total > 0 else 0.0
        success = all(success_answers) if success_answers else None

        return score, per_item, success

    @staticmethod
    def _parse_answers(raw: str, items: List[dict]) -> dict:
        """Parse VLM response into {id_str: 'yes'|'no'}.

        Tries strict JSON first, then a permissive ``"id": "yes"|"no"`` regex
        fallback so that minor formatting drift (extra prose, code fences)
        doesn't lose the entire evaluation.
        """
        if not raw:
            log.warning(
                "[VLMChecklistPolicy] Empty VLM response during evaluation; "
                "treating all items as 'no'."
            )
            return {}

        try:
            obj = _extract_json_object(raw)
            return {
                str(k): str(v).strip().lower()
                for k, v in obj.items()
                if str(v).strip().lower() in ("yes", "no")
            }
        except ValueError:
            pass

        # Fallback: permissive regex per item id.
        out = {}
        for item in items:
            pat = rf'["\']?{re.escape(str(item["id"]))}["\']?\s*:\s*["\']?(yes|no)["\']?(?!\d)'
            m = re.search(pat, raw, re.IGNORECASE)
            if m:
                out[str(item["id"])] = m.group(1).lower()
        if not out:
            log.warning(
                f"[VLMChecklistPolicy] Could not parse any answers from VLM "
                f"response; treating all items as 'no'. Raw response:\n{raw}"
            )
        return out


def _extract_json_object(text: str) -> dict:
    """Extract the first top-level JSON object from a string.

    Tolerates ```json fences and surrounding prose by scanning for the first
    balanced ``{...}`` block.
    """
    text = text.strip()

    # Strip common code fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first balanced {...}
    start = text.find("{")
    if start == -1:
        raise ValueError("no '{' found in text")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError as e:
                    raise ValueError(f"balanced block was not valid JSON: {e}")
    raise ValueError("unbalanced braces")
