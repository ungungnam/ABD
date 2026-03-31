from typing import Any, Dict
import requests
from perception.perception_input import build_perception_input


class PerceptionAgent:
    def __init__(self, config, cameras):
        self.config = config
        self.url = "http://localhost:9877/inference"
        self.cameras = cameras

    def query(
        self,
        observation,
        reference_object
    ) -> Dict[str, Any]:
        perception_input = build_perception_input(
            observation= observation,
            reference_object= reference_object
        )

        response = requests.post(self.url, json=perception_input).json()

        responses_result = {}
        camera_names = list(self.cameras.keys())

        for idx, cam_name in enumerate(camera_names):
            if self._validate_response(response[idx]):
                responses_result[cam_name] = response[idx]

        responses_result_is_valid = (len(responses_result) >= 2)


        return {
            "responses_result": responses_result,
            "responses_result_is_valid": responses_result_is_valid
        }

    def _validate_response(self, response):
        if len(response['mask']) > 0:
            is_valid = True
        else:
            is_valid = False
        return is_valid
