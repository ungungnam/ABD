"""Shared helpers for VLM-backed reset-decision methods.

ABD's production code builds a VLM backend from a Hydra config
(``vlm_client.factory.build_vlm_backend``). The eval suite avoids the Hydra
dependency and constructs the same backends directly from a plain spec, so a
method can be configured from a JSON file passed to ``run_eval.py``.

It also builds a lightweight ``TaskDefinition`` stub from a dataset sample so
the in-repo policies (which expect a real task object) can be reused unchanged.
"""

from __future__ import annotations

from typing import Optional

# ABD task_family -> TaskDefinition.task_type expected by the policies.
_FAMILY_TO_TASK_TYPE = {
    "pick_and_place": "pick_place",
    "open_drawer": "open_drawer",
    "stack_cups": "stack_cups",
}


def build_vqa_client(backend: str = "gpt", **params):
    """Construct a :class:`VQAClient` for the requested backend.

    Args:
        backend: one of ``gpt`` / ``aha`` / ``qwen``.
        **params: backend-specific options --
            gpt:  model, max_tokens, api_key_env
            aha:  url, timeout
            qwen: url, timeout
    """
    from vlm_client.vqa_client import VQAClient

    backend = backend.lower()
    if backend == "gpt":
        from vlm_client.backends.gpt import GPTBackend
        be = GPTBackend(
            model=params.get("model", "gpt-4.1"),
            max_tokens=params.get("max_tokens", 1024),
            api_key_env=params.get("api_key_env", "OPENAI_API_KEY"),
        )
    elif backend == "aha":
        from vlm_client.backends.aha import AHABackend
        be = AHABackend(
            url=params.get("url", "http://localhost:9878/inference"),
            timeout=params.get("timeout", 60),
        )
    elif backend == "qwen":
        from vlm_client.backends.qwen import QwenBackend
        be = QwenBackend(
            url=params.get("url", "http://localhost:9879/inference"),
            timeout=params.get("timeout", 60),
        )
    else:
        raise ValueError(f"Unknown VLM backend '{backend}' (gpt|aha|qwen).")
    return VQAClient(be)


def task_stub_from_sample(sample):
    """Build a minimal ``TaskDefinition`` for the in-repo reset policies.

    Only the fields the policies actually read are populated:
    ``name`` / ``language_task`` / ``task_type`` / ``stack_step``.
    """
    from task.task_family import TaskDefinition

    task_type = _FAMILY_TO_TASK_TYPE.get(sample.task_family, "pick_place")
    stack_step: Optional[str] = None
    if task_type == "stack_cups":
        # checklist policy only inspects stack_step.startswith("forward")
        stack_step = "forward_1" if (sample.task_direction == "forward"
                                     or "forward" in (sample.task_name or "")) \
            else "reverse_1"

    return TaskDefinition(
        name=sample.task_name or "task",
        language_task=sample.language_task or sample.task_name or "the task",
        task_type=task_type,
        stack_step=stack_step,
    )


def observation_from_sample(sample, views: Optional[list] = None) -> dict:
    """Return the ``{view: ndarray}`` observation dict the policies expect."""
    return sample.load_images(views)
