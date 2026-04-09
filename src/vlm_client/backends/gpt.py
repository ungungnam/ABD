"""OpenAI hosted GPT backend (gpt-4.1 / gpt-4o family).

Uses the ``openai`` SDK's ``responses`` API with ``input_text`` +
``input_image`` blocks (mirroring the format PaPA's ``run_vlm_server.py``
already uses for its ``USE_GPT`` path). The API key is read from the
environment variable named in the config (default ``OPENAI_API_KEY``).
"""

import logging
import os
from typing import Optional

from vlm_client.base import VLMBackend, VLMRequest
from vlm_client.encoding import encode_observation_b64

log = logging.getLogger(__name__)


class GPTBackend(VLMBackend):
    name = "gpt"

    def __init__(
        self,
        model: str = "gpt-4.1",
        max_tokens: int = 1024,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.model = model
        self.max_tokens = max_tokens
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"GPTBackend: env var '{api_key_env}' is not set. "
                f"Export it before running with vlm.backend=gpt."
            )

        # Imported lazily so that ABD installs without ``openai`` still
        # work for the AHA/Qwen backends.
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    def generate(self, request: VLMRequest) -> str:
        encoded = encode_observation_b64(request.images)
        content = [{"type": "input_text", "text": request.prompt}]
        for cam_name, b64 in encoded.items():
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                }
            )

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": content}],
                max_output_tokens=self.max_tokens,
            )
            return response.output_text or ""
        except Exception as e:
            log.error(f"[GPTBackend] OpenAI request failed: {e}")
            return ""
