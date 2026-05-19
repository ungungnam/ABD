"""CaM stage 3 -- the reset-decision method that ties the pipeline together.

For each sample:

  1. fetch the task's :class:`ConstraintSet` and :class:`MonitorProgram`
     (generated once per task, then cached);
  2. ground the constraint elements from the last-frame image -- a VLM answers
     every element attribute question, yielding a ``scene`` dict
     (this replaces the paper's RGB-D painter + tracker, which need depth/video
     the VQA dataset does not have);
  3. run the generated monitor code on that scene.

``ready == False`` (a constraint is violated) ==> a reset is needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..base import ResetMethod, ResetPrediction, register
from .._vlm import build_vqa_client, observation_from_sample
from .constraint_generator import ConstraintGenerator
from .code_generator import MonitorCodeGenerator

_FAMILY_TO_TASK_TYPE = {
    "pick_and_place": "pick-and-place manipulation",
    "open_drawer": "articulated drawer manipulation",
    "stack_cups": "cup stacking manipulation",
}

# --------------------------------------------------------------------------- #
# Element-grounding prompt
# --------------------------------------------------------------------------- #
GROUNDING_PROMPT = """\
You are the SCENE GROUNDING module of a Code-as-Monitor system.

The images are different camera views of a robot workspace at the END of an \
episode. The robot will next attempt: "{task_description}".

For each constraint element below, answer every yes/no question by looking at \
the images.

ELEMENTS AND QUESTIONS:
{questions_block}

Return ONLY a JSON object mapping each element id to an object of its answers, \
each answer strictly true or false:

{schema_example}

No prose, JSON only.
"""


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in grounding response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced braces in grounding response")


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "yes", "y", "1")


@register("code_as_monitor")
class CodeAsMonitorMethod(ResetMethod):
    """Code-as-Monitor adapted to single-frame reset-decision VQA.

    Args:
        backend: VLM backend name (``gpt`` / ``aha`` / ``qwen``).
        cache_dir: directory for per-task constraint/monitor artefacts;
            generation is skipped when an artefact already exists.
        **vlm_params: forwarded to the backend (model, url, timeout, ...).
    """

    def __init__(self, backend: str = "gpt", cache_dir: str | None = None,
                 **vlm_params):
        self.backend = backend
        self.cache_dir = cache_dir
        self.vlm_params = vlm_params
        self.client = None
        self.constraint_gen = None
        self.code_gen = None
        # per-task artefacts: task_name -> (ConstraintSet, MonitorProgram)
        self._task_artefacts: dict = {}

    def setup(self) -> None:
        self.client = build_vqa_client(self.backend, **self.vlm_params)
        cdir = Path(self.cache_dir) if self.cache_dir else None
        self.constraint_gen = ConstraintGenerator(
            self.client, cache_dir=cdir / "constraints" if cdir else None
        )
        self.code_gen = MonitorCodeGenerator(
            self.client, cache_dir=cdir / "monitors" if cdir else None
        )

    # ------------------------------------------------------------------ #
    def _artefacts(self, sample):
        """Get-or-build the (ConstraintSet, MonitorProgram) for a task.

        The first sample seen for a task supplies a representative observation,
        so image-conditioned VLM backends can answer the (otherwise text-only)
        constraint- and code-generation prompts.
        """
        task = sample.task_name or "task"
        if task not in self._task_artefacts:
            images = observation_from_sample(sample)
            cset = self.constraint_gen.generate(
                task_name=task,
                task_description=sample.language_task or task,
                task_type=_FAMILY_TO_TASK_TYPE.get(
                    sample.task_family, "manipulation"
                ),
                images=images,
            )
            prog = self.code_gen.generate(cset, images=images)
            self._task_artefacts[task] = (cset, prog)
        return self._task_artefacts[task]

    def _ground_scene(self, sample, cset) -> dict:
        """VLM-ground every element attribute from the last-frame images."""
        questions = []
        for el in cset.elements:
            for attr in el.get("attributes", []):
                questions.append(
                    f'  {el["id"]}.{attr["name"]}: {attr.get("question", "")}'
                )
        schema = {
            el["id"]: {a["name"]: True for a in el.get("attributes", [])}
            for el in cset.elements
        }
        prompt = GROUNDING_PROMPT.format(
            task_description=sample.language_task or sample.task_name,
            questions_block="\n".join(questions),
            schema_example=json.dumps(schema),
        )
        raw = self.client.ask(observation_from_sample(sample), prompt) or ""
        obj = _extract_json_object(raw)

        scene: dict = {}
        for el in cset.elements:
            el_id = el["id"]
            answers = obj.get(el_id, {}) or {}
            scene[el_id] = {
                a["name"]: _coerce_bool(answers.get(a["name"], False))
                for a in el.get("attributes", [])
            }
        return scene

    # ------------------------------------------------------------------ #
    def predict(self, sample) -> ResetPrediction:
        if not sample.image_paths:
            return ResetPrediction(reset=None, error="no images for sample")

        cset, prog = self._artefacts(sample)
        scene = self._ground_scene(sample, cset)
        result = prog.run(scene)

        raw = {
            "subgoal": cset.subgoal,
            "scene": scene,
            "monitor_reason": result["reason"],
            "monitor_error": result["error"],
            "monitor_is_fallback": prog.is_fallback,
        }
        if result["ready"] is None:
            return ResetPrediction(
                reset=None, raw=raw,
                error=f"monitor failed: {result['error']}",
            )
        # ready == workspace OK; reset is needed when NOT ready.
        return ResetPrediction(reset=not result["ready"], raw=raw)
