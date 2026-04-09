"""Shared image encoding helpers for VLM backends.

GPT, AHA and Qwen backends all need base64-JPEG encoded images at some
point, so we centralize the conversion here. The previous helper lived on
``VQAClient`` and was duplicated by every caller that needed encoding.
"""

import base64
import io
from typing import Dict, Optional

import numpy as np
from PIL import Image


def encode_image_b64(rgb: np.ndarray, fmt: str = "JPEG") -> str:
    """Encode an HxWx3 uint8 RGB array as a base64 string."""
    pil = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_observation_b64(
    observation: Optional[Dict[str, np.ndarray]],
) -> Dict[str, str]:
    """Encode every HxWx3 array in an observation dict to base64 JPEG.

    Keys ending in ``_rgb`` are stripped to a clean camera name (so that
    ``front_rgb`` becomes ``front``). Non-image entries are silently
    skipped, which keeps callers from having to filter the dict first.
    """
    if not observation:
        return {}

    out: Dict[str, str] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            cam_name = key.replace("_rgb", "")
            out[cam_name] = encode_image_b64(value)
    return out
