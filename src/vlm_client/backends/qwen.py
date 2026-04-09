"""HTTP client for ABD's Qwen3-VL inference server (``scripts/run_qwen_server.py``).

Symmetric to ``AHABackend``: encode observations to base64 JPEG, ship the
unified payload to a local server that holds the model weights, and
return the raw text response. Qwen's chat-template wrapping happens on
the server side because it depends on the loaded ``AutoProcessor``.
"""

import logging

import requests

from vlm_client.base import VLMBackend, VLMRequest
from vlm_client.encoding import encode_observation_b64

log = logging.getLogger(__name__)


class QwenBackend(VLMBackend):
    name = "qwen"

    def __init__(
        self,
        url: str = "http://localhost:9879/inference",
        timeout: float = 60.0,
    ):
        self.url = url
        self.timeout = timeout

    def generate(self, request: VLMRequest) -> str:
        payload = {
            "images": encode_observation_b64(request.images),
            "prompt": request.prompt,
            "task_kind": request.task_kind,
        }
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            log.error(f"[QwenBackend] Request to {self.url} failed: {e}")
            return ""
