"""Baseline: ABD's Single-VQA reset policy.

``policy.single_vqa_policy.SingleVQAPolicy`` asks the VLM exactly one yes/no
question and resets on "yes". It is the most direct learned-method comparison
point for the checklist: same backend, no checklist scaffolding.

Prompt note
-----------
The policy's production question (`single_vqa_policy._QUESTION`) is written for
the *online collection* setting -- it tells the VLM to weigh detection status,
object positions, workspace bounds and a canonical reference image. None of
that context exists in this image-only VQA dataset (``detection_info`` is
``None``, no reference image), so asking the VLM to consider it just adds
noise. This adapter therefore drives the policy with a simple, self-contained
question that interpolates the task description per sample. Override it with
the ``question`` kwarg (a template with a ``{task_description}`` slot).
"""

from __future__ import annotations

from .base import ResetMethod, ResetPrediction, register
from ._vlm import build_vqa_client, observation_from_sample, task_stub_from_sample

# Simple, image-only question. `{task_description}` is filled per sample.
_DEFAULT_QUESTION = (
    "The robot's task is: \"{task_description}\". "
    "Does a human have to intervene to reset the environment before the robot "
    "can attempt this task again? "
    "Answer with a single word: 'yes' or 'no'."
)


@register("single_vqa")
class SingleVQAMethod(ResetMethod):
    """ABD Single-VQA policy adapted to single-frame VQA evaluation.

    Args:
        backend: VLM backend name (``gpt`` / ``aha`` / ``qwen``).
        question: question template; must contain a ``{task_description}``
            placeholder. Defaults to a simple image-only question.
        reference_dir: optional directory of ``{key}_initial.*`` reference
            images; left empty for the image-only VQA setting.
        **vlm_params: forwarded to the backend (model, url, timeout, ...).
    """

    def __init__(
        self,
        backend: str = "gpt",
        question: str = "",
        reference_dir: str = "",
        **vlm_params,
    ):
        self.backend = backend
        self.question_template = question or _DEFAULT_QUESTION
        self.reference_dir = reference_dir
        self.vlm_params = vlm_params
        self.policy = None

    def setup(self) -> None:
        from policy.single_vqa_policy import SingleVQAPolicy

        client = build_vqa_client(self.backend, **self.vlm_params)
        # The per-sample question is assigned in predict(); pass a placeholder.
        self.policy = SingleVQAPolicy(
            vqa_client=client,
            question=self.question_template,
            reference_dir=self.reference_dir,
        )

    def predict(self, sample) -> ResetPrediction:
        task = task_stub_from_sample(sample)
        observation = observation_from_sample(sample)
        if not observation:
            return ResetPrediction(reset=None, error="no images for sample")

        # Interpolate the task description into the question for this sample.
        self.policy.question = self.question_template.format(
            task_description=sample.language_task or sample.task_name or "the task"
        )
        reset = self.policy.needs_reset(
            validation=None,
            fail_count=int(sample.metadata.get("fail_count", 0) or 0),
            episode_idx=int(sample.metadata.get("episode_idx", 0) or 0),
            task=task,
            observation=observation,
            detection_info=None,
        )
        return ResetPrediction(reset=bool(reset), raw=self.policy.last_eval)
