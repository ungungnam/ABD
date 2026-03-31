from typing import Dict, List


import numpy as np

def build_perception_input(observation: Dict[str, np.ndarray], reference_object: str) -> Dict:
    perception_input_image = _build_perception_input_image(observation)
    perception_input_text = _build_perception_input_plan(reference_object)

    perception_input = {**perception_input_image, **perception_input_text}

    return perception_input


def _build_perception_input_image(observation: Dict[str, np.ndarray]) -> List[List]:
    perception_input_image = []
    for _, value in observation.items():
        # Convert numpy arrays to flattened list
        if isinstance(value, np.ndarray):
            perception_input_image.append(value.reshape(-1).tolist())
        else:
            perception_input_image.append(value)
    return {
        "image": perception_input_image
    }

def _build_perception_input_plan(reference_object: str):
    return {
        "text": reference_object
    }