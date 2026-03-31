"""Geometric task success validator.

Uses PaPA's perception pipeline to detect the manipulated object and
target location in 3D, then checks if the object is near the target.
"""

import numpy as np

from perception.perception_agent import PerceptionAgent
from motion_planner.motion_planner import MotionPlanner
from task.task_family import TaskDefinition
from validator.base_validator import BaseTaskValidator, ValidationResult


class GeometricValidator(BaseTaskValidator):
    """Vision-based geometric task success validator.

    Primary validator: detects object and target via perception,
    reconstructs 3D positions, checks proximity.
    """

    def __init__(self, perception_agent: PerceptionAgent,
                 motion_planner: MotionPlanner, env, config):
        self.perception_agent = perception_agent
        self.motion_planner = motion_planner
        self.env = env
        self.distance_threshold = getattr(config, "distance_threshold", 0.05)

    def validate(self, task: TaskDefinition, post_obs: dict, env) -> ValidationResult:
        obj_name = task.canonical_state["object"]
        target_name = task.canonical_state["target"]

        # 1. Detect the manipulated object
        obs = env.get_observation()
        obj_perception = self.perception_agent.query(obs, obj_name)
        if not obj_perception["responses_result_is_valid"]:
            return ValidationResult(
                success=False, confidence=0.0, method="perception_failed",
                details={"reason": f"cannot detect object '{obj_name}'"},
            )

        # 2. Detect the target location
        target_perception = self.perception_agent.query(obs, target_name)
        if not target_perception["responses_result_is_valid"]:
            return ValidationResult(
                success=False, confidence=0.0, method="perception_failed",
                details={"reason": f"cannot detect target '{target_name}'"},
            )

        # 3. Extract 3D positions via multi-view reconstruction
        obj_points = self.motion_planner.get_reference_object_points(
            obj_perception["responses_result"]
        )
        target_points = self.motion_planner.get_reference_object_points(
            target_perception["responses_result"]
        )

        if obj_points is None or len(obj_points) == 0:
            return ValidationResult(
                success=False, confidence=0.0, method="reconstruction_failed",
                details={"reason": "object 3D reconstruction failed"},
            )
        if target_points is None or len(target_points) == 0:
            return ValidationResult(
                success=False, confidence=0.0, method="reconstruction_failed",
                details={"reason": "target 3D reconstruction failed"},
            )

        # 4. Check XY proximity (object should be above/near target)
        obj_center = obj_points.mean(axis=0)
        target_center = target_points.mean(axis=0)
        xy_distance = float(np.linalg.norm(obj_center[:2] - target_center[:2]))

        success = xy_distance < self.distance_threshold
        return ValidationResult(
            success=success,
            confidence=0.8 if success else 0.3,
            method="geometric",
            details={
                "xy_distance": xy_distance,
                "threshold": self.distance_threshold,
                "obj_pos": obj_center.tolist(),
                "target_pos": target_center.tolist(),
            },
        )
