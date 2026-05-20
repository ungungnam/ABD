import logging
from typing import Any, Dict
import requests
from perception.perception_input import build_perception_input

log = logging.getLogger(__name__)

# Hard cap on a single perception HTTP round-trip.  Without this the request
# can block forever if the perception server hangs, locking up the whole
# trajectory generator and preventing reset escalation.
_REQUEST_TIMEOUT_S = 10.0


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

        try:
            response = requests.post(
                self.url, json=perception_input, timeout=_REQUEST_TIMEOUT_S
            ).json()
        except (requests.Timeout, requests.ConnectionError, ValueError) as e:
            log.warning(
                f"[PerceptionAgent] {type(e).__name__} after {_REQUEST_TIMEOUT_S:.0f}s "
                f"— treating as failed perception (object='{reference_object}')."
            )
            return {
                "responses_result": {},
                "responses_result_is_valid": False,
            }

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
