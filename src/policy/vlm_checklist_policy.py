"""VLM checklist reset policy.

Generates a yes/no checklist (with weights) for each task by prompting a VLM
once, then evaluates the checklist against post-execution observations to
decide next / retry / reset. Per-task checklists are persisted to
``config/checklists/<task_name>.json`` so users can hand-edit weights and
question wording between runs.
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

log = logging.getLogger(__name__)


class VLMChecklistPolicy(BaseResetPolicy):
    """Reset policy that asks a VLM to fill out a per-task checklist.

    Decision rule (weighted normalized score):
        score = sum(weight_i * 1[answer_i == "yes"]) / sum(weight_i)
        score >= tau_reset            -> "next"
        tau_retry <= score < tau_reset -> "retry"
        score < tau_retry             -> "reset"

    A failed escalation (fail_count + 1 >= max_fail_count) is forced to "reset"
    regardless of score, mirroring the behaviour of the existing ABD policy.
    """

    def __init__(
        self,
        vqa_client: VQAClient,
        checklist_dir: str,
        tau_retry: float = 0.5,
        tau_reset: float = 0.9,
        max_fail_count: int = 5,
    ):
        self.vqa_client = vqa_client
        self.checklist_dir = Path(checklist_dir)
        self.checklist_dir.mkdir(parents=True, exist_ok=True)
        self.tau_retry = tau_retry
        self.tau_reset = tau_reset
        self.max_fail_count = max_fail_count
        self._checklists: dict = {}  # task_name -> loaded checklist dict
        self.last_eval: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def decide(
        self,
        validation: ValidationResult,
        fail_count: int,
        episode_idx: int,
        features: dict = None,
        task=None,
        observation=None,
    ) -> str:
        # Fallback for callers that didn't supply task/observation
        # (e.g., dummy paths or generation-failure branches).
        if task is None or observation is None:
            self.last_eval = None
            return "next" if validation.success else "retry"

        checklist = self._load_or_generate(task)
        score, per_item = self._evaluate(checklist, task, observation)
        self.last_eval = {
            "task_name": task.name,
            "score": score,
            "items": per_item,
        }

        if score >= self.tau_reset:
            return "next"
        if fail_count + 1 >= self.max_fail_count:
            return "reset"
        if score >= self.tau_retry:
            return "retry"
        return "reset"

    # ------------------------------------------------------------------ #
    # Checklist generation / loading
    # ------------------------------------------------------------------ #

    def _checklist_path(self, task_name: str) -> Path:
        return self.checklist_dir / f"{task_name}.json"

    def _load_or_generate(self, task) -> dict:
        if task.name in self._checklists:
            return self._checklists[task.name]

        path = self._checklist_path(task.name)
        if path.exists():
            with open(path) as f:
                checklist = json.load(f)
            log.info(f"[VLMChecklistPolicy] Loaded checklist from {path}")
        else:
            log.info(
                f"[VLMChecklistPolicy] No checklist for '{task.name}' at {path}; "
                f"generating via VLM."
            )
            checklist = self._generate_checklist(task)
            with open(path, "w") as f:
                json.dump(checklist, f, indent=2)
            log.info(f"[VLMChecklistPolicy] Wrote new checklist to {path}")

        self._validate_checklist_schema(checklist, task.name)
        self._checklists[task.name] = checklist
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
        self, checklist: dict, task, observation: dict
    ) -> Tuple[float, List[dict]]:
        items = checklist["items"]
        items_block = "\n".join(
            f"{item['id']}. {item['question']}" for item in items
        )
        prompt = EVAL_PROMPT.format(
            task_description=task.language_task,
            items_block=items_block,
        )
        raw = self.vqa_client.ask_text(observation, prompt)

        answers = self._parse_answers(raw, items)

        per_item: List[dict] = []
        weighted_sum = 0.0
        weight_total = 0.0
        for item in items:
            ans = answers.get(str(item["id"]), "no")
            yes = ans == "yes"
            w = float(item["weight"])
            weighted_sum += w * (1.0 if yes else 0.0)
            weight_total += w
            per_item.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "weight": item["weight"],
                    "answer": ans,
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
            pat = rf'["\']?{re.escape(str(item["id"]))}["\']?\s*:\s*["\']?(yes|no)["\']?'
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
