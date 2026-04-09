"""HTTP client for ABD's AHA inference server (``scripts/run_aha_server.py``).

AHA is a LoRA fine-tune of RoboPoint/Vicuna and ships with a LLaVA-style
chat conversation format (``<image>\\n{question}``). The heavy model
weights live in the ``aha`` conda env, so we keep them out of ABD's data
collection process by talking to a local FastAPI server. The server
accepts the unified ``{images, prompt, task_kind}`` payload that all
ABD-side backends use; the LLaVA-specific prompt envelope is constructed
on the server, not here, because it depends on tokenizer state.
"""

import logging

import requests

from vlm_client.base import VLMBackend, VLMRequest
from vlm_client.encoding import encode_observation_b64

log = logging.getLogger(__name__)


class AHABackend(VLMBackend):
    name = "aha"

    def __init__(
        self,
        url: str = "http://localhost:9878/inference",
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
            log.error(f"[AHABackend] Request to {self.url} failed: {e}")
            return ""
