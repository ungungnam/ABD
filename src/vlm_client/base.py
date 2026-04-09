"""Backend interface for ABD's VLM clients.

A ``VLMBackend`` is a thin adapter that converts a ``VLMRequest`` (image dict +
prompt + intent) into a string response. Each backend owns its own image
encoding and prompt envelope, since GPT, AHA (LLaVA-style) and Qwen3-VL all
expect different input shapes. Higher-level callers (``VQAClient``,
``VLMPlanner``, ``VLMValidator``, ``VLMChecklistPolicy``) only see this
interface and don't care which model is wired in.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np


TaskKind = Literal[
    "plan",            # VLMPlanner: pick/place/action extraction
    "checklist_gen",   # VLMChecklistPolicy: generate checklist (text only)
    "checklist_eval",  # VLMChecklistPolicy: score post-execution images
    "vqa",             # Generic free-form VQA prompt with images
    "yes_no",          # Single yes/no question with images
]


@dataclass
class VLMRequest:
    """One unit of work for a VLM backend.

    Attributes:
        prompt: The fully-formatted user prompt (already includes task
            description, checklist items, etc.). Backends may wrap this in
            their own system/instruction template but must not re-template it.
        images: Optional dict of camera-name -> HxWx3 uint8 RGB array.
            ``None`` is valid and means "text-only" (e.g. ``checklist_gen``).
        task_kind: What the caller is trying to do; backends use this to pick
            decoding parameters (max_tokens, temperature) or skip image
            handling for text-only kinds.
    """

    prompt: str
    images: Optional[Dict[str, np.ndarray]] = None
    task_kind: TaskKind = "vqa"
    extra: dict = field(default_factory=dict)


class VLMBackend(ABC):
    """Abstract VLM backend. One concrete subclass per model family."""

    name: str = "base"

    @abstractmethod
    def generate(self, request: VLMRequest) -> str:
        """Run inference and return the raw response string.

        Implementations should return ``""`` on a recoverable error rather
        than raise, so that the existing parsers in ``VQAClient`` and
        ``VLMChecklistPolicy`` can apply their fallbacks.
        """
        raise NotImplementedError
