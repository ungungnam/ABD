from typing import Dict, Any, List
from PIL import Image

import numpy as np


def build_vlm_input(observation: Dict[str, np.ndarray], task: str) -> Dict[str, Any]:
    vlm_input_image = _build_vlm_input_image(observation)
    vlm_input_text = _build_vlm_input_text(task)

    vlm_input = {**vlm_input_image, **vlm_input_text}

    return vlm_input

def _build_vlm_input_image(observation: Dict[str, np.ndarray]) -> Dict[str, List]:
    vlm_input_image = {}
    for key, image in observation.items():
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize((256, 256), Image.BILINEAR)

        image = np.array(image, dtype=np.uint8)
        image = image.reshape(-1).tolist()

        vlm_input_image[key] = image
    return vlm_input_image

def _build_vlm_input_text(task: str) -> Dict[str, str]:
    return {"task": task}