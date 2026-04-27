"""VLM checklist reset policy.

Each task checklist has three sections:
  - common:       environment/safety items always evaluated
  - current_task: items that determine whether the current task succeeded
  - next_task:    items that verify the environment is ready to START this task
                  (used as phase 2 when this task will run next)

Evaluation is two-phase:
  Phase 1 — always runs:
      Evaluate common + current_task from the current task's checklist.
      Determine success from all current_task answers.
  Phase 2 — always runs, source depends on Phase 1 result:
      success  → evaluate next_task from the PAIRED task's checklist
                 (environment should be ready for the paired task)
      failure  → evaluate next_task from the CURRENT task's checklist
                 (environment should be ready to retry the same task)

    score = Σ(weight_i × yes_i) / Σ(weight_i)  over all evaluated items
    needs_reset = (score < tau_reset)

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

# Maps each canonical checklist name to the paired next task's checklist name.
# After task A succeeds, task_specific items from _PAIRED[A] are evaluated to
# verify the environment is ready for the next task.
_PAIRED_CHECKLIST: dict = {
    "banana_to_pan":       "banana_to_plate",
    "banana_to_plate":     "banana_to_pan",
    "open_drawer":         "close_drawer",
    "close_drawer":        "open_drawer",
    "stack_cups_forward":  "stack_cups_reverse",
    "stack_cups_reverse":  "stack_cups_forward",
}

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
        self._checklists: dict = {}  # canonical_name -> loaded checklist dict
        self.last_eval: Optional[dict] = None
        self.last_success: Optional[bool] = None  # set by needs_reset(); None if no current_task items

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
        self.last_success = success

        reset = score < self.tau_reset

        log.info(
            f"[VLMChecklistPolicy] task={task.name} | score={score:.3f} | tau={self.tau_reset} | "
            f"needs_reset={reset} | success={success}"
        )
        for item in per_item:
            marker = "★" if item.get("success_item") else " "
            phase = f"[{item.get('phase', 'current')}]"
            log.info(
                f"  {marker}{phase}[{item['answer'].upper():3s}] (w={item['weight']:.2f}) {item['question']}"
            )

        return reset

    # ------------------------------------------------------------------ #
    # Checklist loading
    # ------------------------------------------------------------------ #

    def _checklist_path(self, task_name: str) -> Path:
        return self.checklist_dir / f"{task_name}.json"

    def _canonical_checklist_name(self, task) -> str:
        """Return the canonical checklist file stem for this task."""
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

    def _load_by_name(self, name: str) -> Optional[dict]:
        """Load checklist by canonical name without VLM generation."""
        if name in self._checklists:
            return self._checklists[name]
        path = self._checklist_path(name)
        if not path.exists():
            log.warning(
                f"[VLMChecklistPolicy] Paired checklist '{name}' not found at {path}; "
                f"skipping phase 2."
            )
            return None
        with open(path) as f:
            checklist = json.load(f)
        self._validate_checklist_schema(checklist, name)
        self._checklists[name] = checklist
        return checklist

    def _generate_checklist(self, task) -> dict:
        prompt = META_PROMPT.format(
            task_name=task.name,
            task_description=task.language_task,
        )
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
        for section in ("common", "current_task", "next_task"):
            if section not in checklist or not isinstance(checklist[section], list):
                raise ValueError(
                    f"Checklist '{task_name}' missing '{section}' list."
                )
        for section in ("common", "current_task", "next_task"):
            for i, item in enumerate(checklist[section]):
                for key in ("id", "question", "weight"):
                    if key not in item:
                        raise ValueError(
                            f"Checklist '{task_name}' {section}[{i}] missing '{key}'."
                        )

    # ------------------------------------------------------------------ #
    # Checklist evaluation
    # ------------------------------------------------------------------ #

    def _evaluate(
        self, checklist: dict, task, observation: dict, detection_info: dict = None
    ) -> Tuple[float, List[dict], Optional[bool]]:
        """Two-phase checklist evaluation.

        Phase 1: common + current_task from current task's checklist.
                 Determines success/failure of the current task.
        Phase 2: next_task items — source depends on Phase 1 result:
                 success → next_task from PAIRED task's checklist
                 failure → next_task from CURRENT task's checklist (retry)
        Returns (score, per_item, success).
        """
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
            log.info(f"[VLMChecklistPolicy] task={task.name} | detection:{detection_block.strip()}")

        # ---- Phase 1: common + current_task ---- #
        items_p1 = checklist["common"] + checklist["current_task"]
        answers_p1 = self._ask_vlm(items_p1, preamble, detection_block, obs, task)

        success_answers = [
            answers_p1.get(str(item["id"]), "no") == item.get("success_answer", "yes")
            for item in checklist["current_task"]
            if item.get("success_item", False)
        ]
        success = all(success_answers) if success_answers else None

        # ---- Phase 2: next_task ---- #
        # success  → next_task from paired task's checklist
        # failure  → next_task from current task's checklist (same task will retry)
        current_name = self._canonical_checklist_name(task)
        if success:
            next_name = _PAIRED_CHECKLIST.get(current_name, current_name)
        else:
            next_name = current_name

        if next_name == current_name:
            p2_checklist = checklist
        else:
            p2_checklist = self._load_by_name(next_name)

        items_p2: List[dict] = []
        answers_p2: dict = {}
        if p2_checklist:
            items_p2 = p2_checklist.get("next_task", [])
            if items_p2:
                log.info(
                    f"[VLMChecklistPolicy] Phase 2 (next={'paired:' + next_name if success else 'retry:' + next_name})"
                    f" | {len(items_p2)} items"
                )
                answers_p2 = self._ask_vlm(items_p2, preamble, detection_block, obs, task)

        # ---- Score ---- #
        score, per_item = self._compute_score(items_p1, answers_p1, items_p2, answers_p2)
        return score, per_item, success

    def _ask_vlm(
        self,
        items: List[dict],
        preamble: str,
        detection_block: str,
        obs: dict,
        task,
    ) -> dict:
        items_block = "\n".join(
            f"{item['id']}. {item['question']}" for item in items
        )
        prompt = preamble + detection_block + "\n" + EVAL_PROMPT.format(
            task_description=task.language_task,
            items_block=items_block,
        )
        raw = self.vqa_client.ask_text(obs, prompt)
        return self._parse_answers(raw, items)

    @staticmethod
    def _compute_score(
        items_p1: List[dict],
        answers_p1: dict,
        items_p2: List[dict],
        answers_p2: dict,
    ) -> Tuple[float, List[dict]]:
        # Detect mixed reset_groups (only within phase 1).
        group_answers: dict = {}
        for item in items_p1:
            gid = item.get("reset_group")
            if gid is not None:
                ans = answers_p1.get(str(item["id"]), "no")
                group_answers.setdefault(gid, set()).add(ans)
        mixed_groups = {gid for gid, ans_set in group_answers.items() if len(ans_set) > 1}

        weighted_sum = 0.0
        weight_total = 0.0
        per_item: List[dict] = []

        for phase_tag, items, answers in [("current", items_p1, answers_p1), ("next", items_p2, answers_p2)]:
            for item in items:
                ans = answers.get(str(item["id"]), "no")
                yes = ans == "yes"
                w = float(item["weight"])
                is_success_item = bool(item.get("success_item", False))
                success_answer = item.get("success_answer", "yes")

                in_mixed_group = item.get("reset_group") in mixed_groups
                in_reset_group = item.get("reset_group") is not None

                if in_reset_group:
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
                        "phase": phase_tag,
                    }
                )

        score = weighted_sum / weight_total if weight_total > 0 else 0.0
        return score, per_item

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
