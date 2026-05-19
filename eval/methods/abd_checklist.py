"""ABD's own method -- the VLM-checklist reset policy -- as an eval method.

This is the "our method" entry of the benchmark. It reuses
``policy.vlm_checklist_policy.VLMChecklistPolicy`` unchanged: the policy fills
out a per-task checklist via the VLM and triggers a reset when the weighted
score falls below ``tau_reset``.

The collection-time policy can also consume a ``detection_info`` dict (object
poses, workspace bounds). The VQA dataset is image-only by construction, so
``detection_info`` is left as ``None`` here -- the policy already treats that
as "no extra context" and falls back to image-only checklist evaluation.
"""

from __future__ import annotations

from pathlib import Path

from .base import ResetMethod, ResetPrediction, register
from ._vlm import build_vqa_client, observation_from_sample, task_stub_from_sample

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKLIST_DIR = _REPO_ROOT / "config" / "checklists"


@register("abd_checklist")
class ABDChecklistMethod(ResetMethod):
    """ABD VLM-checklist policy adapted to single-frame VQA evaluation.

    Args:
        backend: VLM backend name (``gpt`` / ``aha`` / ``qwen``).
        checklist_dir: directory of per-task checklist JSON files.
        tau_reset: reset threshold; ``needs_reset = score < tau_reset``.
        version: checklist file version suffix ("" -> v1, "_v2" -> v2).
        **vlm_params: forwarded to the backend (model, url, timeout, ...).
    """

    def __init__(
        self,
        backend: str = "gpt",
        checklist_dir: str | None = None,
        tau_reset: float = 0.9,
        version: str = "",
        **vlm_params,
    ):
        self.backend = backend
        self.checklist_dir = str(checklist_dir or _DEFAULT_CHECKLIST_DIR)
        self.tau_reset = float(tau_reset)
        self.version = version
        self.vlm_params = vlm_params
        self.policy = None

    def setup(self) -> None:
        from policy.vlm_checklist_policy import VLMChecklistPolicy

        client = build_vqa_client(self.backend, **self.vlm_params)
        self.policy = VLMChecklistPolicy(
            vqa_client=client,
            checklist_dir=self.checklist_dir,
            tau_reset=self.tau_reset,
            use_cache=True,
            version=self.version,
        )

    def predict(self, sample) -> ResetPrediction:
        task = task_stub_from_sample(sample)
        observation = observation_from_sample(sample)
        if not observation:
            return ResetPrediction(reset=None, error="no images for sample")

        fail_count = int(sample.metadata.get("fail_count", 0) or 0)
        episode_idx = int(sample.metadata.get("episode_idx", 0) or 0)

        reset = self.policy.needs_reset(
            validation=None,
            fail_count=fail_count,
            episode_idx=episode_idx,
            task=task,
            observation=observation,
            detection_info=None,
        )
        return ResetPrediction(reset=bool(reset), raw=self.policy.last_eval)
