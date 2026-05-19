"""Baseline: zero-shot VLM reset decision.

A single, minimal prompt to a raw VLM with no checklist, no reference images
and no ABD-specific prompt engineering. Comparing this against
``abd_checklist`` / ``single_vqa`` isolates how much ABD's scaffolding adds
over simply asking a capable VLM the question directly.
"""

from __future__ import annotations

import re

from .base import ResetMethod, ResetPrediction, register
from ._vlm import build_vqa_client, observation_from_sample

_PROMPT = (
    "You are monitoring a robot manipulation workcell between episodes.\n"
    "The robot will next attempt this task: \"{task}\".\n"
    "The images are different camera views of the workspace at the end of the "
    "previous episode.\n"
    "Decide whether a human MUST physically intervene to reset the workspace "
    "(e.g. an object is out of the robot's reach, fell off the table, is stuck, "
    "or the scene cannot support the next attempt) before the robot can "
    "continue on its own.\n"
    "Answer strictly with a single word: 'yes' (reset needed) or 'no'."
)


def _parse_yes_no(text: str) -> bool | None:
    m = re.search(r"\b(yes|no)\b", (text or "").strip().lower())
    if not m:
        return None
    return m.group(1) == "yes"


@register("zero_shot_vlm")
class ZeroShotVLMMethod(ResetMethod):
    """Single direct yes/no VLM query, no scaffolding.

    Args:
        backend: VLM backend name (``gpt`` / ``aha`` / ``qwen``).
        prompt: optional override for the question template (``{task}`` slot).
        **vlm_params: forwarded to the backend (model, url, timeout, ...).
    """

    def __init__(self, backend: str = "gpt", prompt: str = "", **vlm_params):
        self.backend = backend
        self.prompt_template = prompt or _PROMPT
        self.vlm_params = vlm_params
        self.client = None

    def setup(self) -> None:
        self.client = build_vqa_client(self.backend, **self.vlm_params)

    def predict(self, sample) -> ResetPrediction:
        observation = observation_from_sample(sample)
        if not observation:
            return ResetPrediction(reset=None, error="no images for sample")

        prompt = self.prompt_template.format(
            task=sample.language_task or sample.task_name or "the task"
        )
        raw = self.client.ask(observation, prompt) or ""
        answer = _parse_yes_no(raw)
        if answer is None:
            return ResetPrediction(
                reset=None, raw=raw, error="could not parse yes/no from VLM"
            )
        return ResetPrediction(reset=answer, raw={"answer": answer, "text": raw})
