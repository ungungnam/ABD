"""VQA client — sends yes/no questions to the VLM server.

Reuses PaPA's VLM server at localhost:9876 with a VQA-style prompt.
"""

import base64
import io
import requests

import numpy as np


class VQAClient:
    """Visual Question Answering client using the VLM server."""

    def __init__(self, url: str = "http://localhost:9876/inference"):
        self.url = url

    def ask(self, observation: dict, question: str) -> str:
        """Send a VQA question with observation images to the VLM server.

        Args:
            observation: dict of {"camera_rgb": np.ndarray(H,W,3), ...}
            question: The question to ask (e.g., "Is the banana on the pan?")

        Returns:
            The VLM response string.
        """
        payload = self._build_vqa_payload(observation, question)
        return self._post(payload)

    def ask_text(self, observation: dict | None, prompt: str) -> str:
        """Send a free-form prompt to the VLM server.

        Used for prompts that don't fit the yes/no VQA pattern (e.g.,
        checklist generation, structured JSON outputs). The observation
        may be None for text-only queries such as checklist generation
        from a task description alone.

        Returns:
            The raw VLM response string (caller is responsible for parsing).
        """
        payload = {
            "images": self._encode_observation(observation) if observation else {},
            "task": prompt,
            "mode": "vqa",
        }
        return self._post(payload)

    def ask_yes_no(self, observation: dict, question: str) -> bool:
        """Ask a yes/no question and parse the boolean answer."""
        if not question.strip().endswith("?"):
            question = question.strip() + "?"
        full_question = f"{question} Answer with only 'yes' or 'no'."
        response = self.ask(observation, full_question)
        return self._parse_yes_no(response)

    def _post(self, payload: dict) -> str:
        try:
            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"[VQAClient] Error querying VLM server: {e}")
            return ""

    def _build_vqa_payload(self, observation: dict, question: str) -> dict:
        """Build VLM server payload with images + question."""
        return {
            "images": self._encode_observation(observation),
            "task": question,
            "mode": "vqa",
        }

    def _encode_observation(self, observation: dict) -> dict:
        """Encode all RGB image arrays in an observation dict to base64 JPEGs."""
        images = {}
        for k, v in observation.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                cam_name = k.replace("_rgb", "")
                images[cam_name] = self._encode_image(v)
        return images

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        """Encode numpy RGB image to base64 JPEG string."""
        from PIL import Image
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _parse_yes_no(response: str) -> bool:
        response_lower = response.strip().lower()
        if response_lower.startswith("yes"):
            return True
        if response_lower.startswith("no"):
            return False
        # Fallback: check if "yes" appears more than "no"
        return response_lower.count("yes") > response_lower.count("no")
