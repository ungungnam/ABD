"""Visual question answering client.

Thin facade over a ``VLMBackend``. The public surface
(``ask`` / ``ask_text`` / ``ask_yes_no``) is unchanged so existing
callers in ``VLMValidator`` and ``VLMChecklistPolicy`` keep working
regardless of which backend is wired in via Hydra config.
"""

from typing import Optional

from vlm_client.base import VLMBackend, VLMRequest


class VQAClient:
    """Visual Question Answering client backed by a pluggable VLM."""

    def __init__(self, backend: VLMBackend):
        self.backend = backend

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", "unknown")

    def ask(self, observation: dict, question: str) -> str:
        """Send a VQA-style question with observation images.

        Args:
            observation: dict of {"camera_rgb": np.ndarray(H,W,3), ...}
            question: The question to ask.

        Returns:
            The raw response string from the backend.
        """
        return self.backend.generate(
            VLMRequest(prompt=question, images=observation, task_kind="vqa")
        )

    def ask_text(self, observation: Optional[dict], prompt: str) -> str:
        """Send a free-form prompt.

        Used for prompts that don't fit the yes/no VQA pattern (e.g.
        checklist generation, structured JSON outputs). ``observation``
        may be ``None`` for text-only queries such as checklist generation
        from a task description alone.
        """
        task_kind = "checklist_eval" if observation else "checklist_gen"
        return self.backend.generate(
            VLMRequest(prompt=prompt, images=observation, task_kind=task_kind)
        )

    def ask_yes_no(self, observation: dict, question: str) -> bool:
        """Ask a yes/no question and parse the boolean answer."""
        if not question.strip().endswith("?"):
            question = question.strip() + "?"
        full_question = f"{question} Answer with only 'yes' or 'no'."
        response = self.backend.generate(
            VLMRequest(
                prompt=full_question,
                images=observation,
                task_kind="yes_no",
            )
        )
        return self._parse_yes_no(response)

    @staticmethod
    def _parse_yes_no(response: str) -> bool:
        response_lower = response.strip().lower()
        if response_lower.startswith("yes"):
            return True
        if response_lower.startswith("no"):
            return False
        # Fallback: check if "yes" appears more than "no"
        return response_lower.count("yes") > response_lower.count("no")
