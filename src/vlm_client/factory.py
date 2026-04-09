"""Factory for building a ``VLMBackend`` from Hydra config.

The active backend is selected by ``config.vlm.backend`` (one of
``gpt``, ``aha``, ``qwen``). Each backend reads its own subset of fields
from the same ``config.vlm`` group, so swapping backends is just a matter
of pointing the Hydra ``vlm`` default at a different group file.
"""

from typing import TYPE_CHECKING

from vlm_client.base import VLMBackend

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_vlm_backend(config: "DictConfig") -> VLMBackend:
    """Construct the VLM backend selected by ``config.vlm.backend``."""
    vlm_cfg = config.vlm
    backend_name = str(vlm_cfg.backend).lower()

    if backend_name == "gpt":
        from vlm_client.backends.gpt import GPTBackend
        return GPTBackend(
            model=vlm_cfg.get("model", "gpt-4.1"),
            max_tokens=vlm_cfg.get("max_tokens", 1024),
            api_key_env=vlm_cfg.get("api_key_env", "OPENAI_API_KEY"),
        )

    if backend_name == "aha":
        from vlm_client.backends.aha import AHABackend
        return AHABackend(
            url=vlm_cfg.get("url", "http://localhost:9878/inference"),
            timeout=vlm_cfg.get("timeout", 60),
        )

    if backend_name == "qwen":
        from vlm_client.backends.qwen import QwenBackend
        return QwenBackend(
            url=vlm_cfg.get("url", "http://localhost:9879/inference"),
            timeout=vlm_cfg.get("timeout", 60),
        )

    raise ValueError(
        f"Unknown VLM backend '{backend_name}'. "
        f"Expected one of: gpt, aha, qwen."
    )
