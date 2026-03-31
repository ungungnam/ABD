"""ABD Feature Extractor.

Computes the 6-dimensional feature vector f_k:
  [f_succ, f_vis, f_reach, f_rec, f_dev, f_fail]

Each feature is in [0, 1].
"""

import numpy as np

from perception.perception_agent import PerceptionAgent
from motion_planner.motion_planner import MotionPlanner
from task.task_family import TaskDefinition
from validator.base_validator import ValidationResult


class ABDFeatureExtractor:
    """Extracts the ABD feature vector from the current state."""

    def __init__(self, config, perception_agent: PerceptionAgent,
                 motion_planner: MotionPlanner, env, vqa_client=None):
        self.perception_agent = perception_agent
        self.motion_planner = motion_planner
        self.env = env
        self.vqa_client = vqa_client

        # Workspace bounds
        wb = config.env.workspace_bounds
        self.workspace_bounds = {
            "x_min": wb.x_min, "x_max": wb.x_max,
            "y_min": wb.y_min, "y_max": wb.y_max,
            "z_min": wb.z_min, "z_max": wb.z_max,
        }

        self.max_fail_count = config.policy.max_fail_count
        self.max_deviation = 0.2  # meters, for normalization

        # Canonical positions: calibrated at start / after human reset
        self.canonical_positions = {}

    def calibrate(self, task: TaskDefinition):
        """Record canonical 3D positions for the task-relevant objects.

        Call at system start and after each human reset.
        """
        obs = self.env.get_observation()

        # Calibrate positions for both the object and target location
        for obj_name in [task.canonical_state.get("object"),
                         task.canonical_state.get("location"),
                         task.canonical_state.get("target")]:
            if obj_name is None:
                continue
            perc = self.perception_agent.query(obs, obj_name)
            if perc["responses_result_is_valid"]:
                pts = self.motion_planner.get_reference_object_points(
                    perc["responses_result"]
                )
                if pts is not None and len(pts) > 0:
                    self.canonical_positions[obj_name] = pts.mean(axis=0)

    def extract(self, task: TaskDefinition, validation_result: ValidationResult,
                fail_count: int) -> dict:
        """Compute the ABD feature vector.

        Args:
            task: Current task definition.
            validation_result: Result from task success validation.
            fail_count: Number of consecutive failures on this task.

        Returns:
            dict with individual features and the combined 6D vector.
        """
        obs = self.env.get_observation()
        target_obj = task.canonical_state["object"]

        # ---- f_succ: task success ----
        f_succ = float(validation_result.success)

        # ---- f_vis: target visibility ----
        perc = self.perception_agent.query(obs, target_obj)
        f_vis = 1.0 if perc["responses_result_is_valid"] else 0.0

        # ---- f_reach: target reachability (within workspace bounds) ----
        f_reach = self._compute_reachability(perc)

        # ---- f_rec: recoverability (minimal impl: visible AND reachable) ----
        f_rec = f_vis * f_reach

        # ---- f_dev: state deviation from canonical ----
        f_dev = self._compute_deviation(perc, target_obj)

        # ---- f_fail: normalized consecutive failure count ----
        f_fail = min(1.0, fail_count / max(self.max_fail_count, 1))

        feature_vector = np.array([f_succ, f_vis, f_reach, f_rec, f_dev, f_fail])

        return {
            "f_succ": f_succ,
            "f_vis": f_vis,
            "f_reach": f_reach,
            "f_rec": f_rec,
            "f_dev": f_dev,
            "f_fail": f_fail,
            "vector": feature_vector,
        }

    def _compute_reachability(self, perception: dict) -> float:
        """Check if the detected object is within the robot's workspace bounds."""
        if not perception["responses_result_is_valid"]:
            return 0.0

        pts = self.motion_planner.get_reference_object_points(
            perception["responses_result"]
        )
        if pts is None or len(pts) == 0:
            return 0.0

        center = pts.mean(axis=0)
        wb = self.workspace_bounds
        in_bounds = (
            wb["x_min"] <= center[0] <= wb["x_max"]
            and wb["y_min"] <= center[1] <= wb["y_max"]
            and wb["z_min"] <= center[2] <= wb["z_max"]
        )
        return 1.0 if in_bounds else 0.0

    def _compute_deviation(self, perception: dict, target_obj: str) -> float:
        """Compute normalized deviation from canonical state."""
        if target_obj not in self.canonical_positions:
            return 0.0
        if not perception["responses_result_is_valid"]:
            return 1.0  # cannot observe => assume max deviation

        pts = self.motion_planner.get_reference_object_points(
            perception["responses_result"]
        )
        if pts is None or len(pts) == 0:
            return 1.0

        current_center = pts.mean(axis=0)
        canonical_center = self.canonical_positions[target_obj]
        distance = float(np.linalg.norm(current_center - canonical_center))
        return min(1.0, distance / self.max_deviation)
