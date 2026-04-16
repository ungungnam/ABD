"""PaPA-based trajectory generator.

Wraps the full PaPA pipeline: VLM Planner -> Perception Agent -> Motion Planner.
Adapted from PaPA/src/papa/papa.py.
"""

import time
from typing import Optional

from vlm.vlm_planner import VLMPlanner
from vlm_client.base import VLMBackend
from perception.perception_agent import PerceptionAgent
from motion_planner.motion_planner import MotionPlanner

from task.task_family import TaskDefinition
from trajectory_generator.base_generator import BaseTrajectoryGenerator, GenerationResult


class PaPATrajectoryGenerator(BaseTrajectoryGenerator):
    """Trajectory generator using PaPA's VLM -> Perception -> Motion pipeline."""

    def __init__(self, config, env, vlm_backend: VLMBackend):
        self.vlm_planner = VLMPlanner(vlm_backend)
        self.perception_agent = PerceptionAgent(
            config.perception, cameras=env.get_cameras()
        )
        self.motion_planner = MotionPlanner(
            config.motion_planner,
            env=env,
            robot=env.get_robot(),
            cameras=env.get_cameras(),
        )
        self.max_vlm_retries = 10
        self.max_perception_retries = 10

    def generate(self, task: TaskDefinition, env) -> GenerationResult:
        if task.task_type == "stack_cups":
            return self._generate_stack_cups(task, env)
        return self._generate_pick_place(task, env)

    def _generate_pick_place(self, task: TaskDefinition, env) -> GenerationResult:
        start_time = time.time()
        metadata = {"generation_failed": False, "elapsed_time": 0.0}

        try:
            # 1. VLM query — retry until valid pick/place detected
            pick_object, place_object, action = None, None, None
            for _ in range(self.max_vlm_retries):
                pick_object, place_object, action = self.vlm_planner.query(
                    observation=env.get_observation(),
                    task=task.language_task,
                )
                if pick_object is not None and place_object is not None:
                    break

            if pick_object is None or place_object is None:
                metadata["generation_failed"] = True
                metadata["reason"] = "vlm_failed"
                metadata["elapsed_time"] = time.time() - start_time
                return GenerationResult(trajectory=None, metadata=metadata)

            # 2. Perception for pick object
            pick_perception = self._query_perception_with_retry(env, pick_object)
            if not pick_perception["responses_result_is_valid"]:
                metadata["generation_failed"] = True
                metadata["reason"] = "pick_perception_failed"
                metadata["elapsed_time"] = time.time() - start_time
                return GenerationResult(trajectory=None, metadata=metadata)

            # 3. Perception for place object
            place_perception = self._query_perception_with_retry(env, place_object)
            if not place_perception["responses_result_is_valid"]:
                metadata["generation_failed"] = True
                metadata["reason"] = "place_perception_failed"
                metadata["elapsed_time"] = time.time() - start_time
                return GenerationResult(trajectory=None, metadata=metadata)

            # 4. Motion planning
            trajectory, events, key_poses = self.motion_planner.query(
                vlm_action=action,
                pick_perception=pick_perception,
                place_perception=place_perception,
            )

            metadata.update({
                "pick_object": pick_object,
                "place_object": place_object,
                "pick_perception_valid": True,
                "place_perception_valid": True,
                "elapsed_time": time.time() - start_time,
            })

            return GenerationResult(
                trajectory=trajectory, events=events, metadata=metadata, key_poses=key_poses
            )

        except Exception as e:
            metadata["generation_failed"] = True
            metadata["reason"] = f"exception: {e}"
            metadata["elapsed_time"] = time.time() - start_time
            return GenerationResult(trajectory=None, metadata=metadata)

    def _generate_stack_cups(self, task: TaskDefinition, env) -> GenerationResult:
        start_time = time.time()
        metadata = {"generation_failed": False, "elapsed_time": 0.0}

        try:
            trajectory, events, key_poses = self.motion_planner.query(
                task_type="stack_cups",
                pick_tag_id=task.pick_tag_id,
                place_tag_id=task.place_tag_id,
                place_xy_offset=task.place_xy_offset,
                stack_step=task.stack_step,
            )

            if trajectory is None:
                metadata["generation_failed"] = True
                metadata["reason"] = "apriltag_not_found"
                metadata["elapsed_time"] = time.time() - start_time
                return GenerationResult(trajectory=None, metadata=metadata)

            metadata.update({
                "pick_tag_id": task.pick_tag_id,
                "place_tag_id": task.place_tag_id,
                "elapsed_time": time.time() - start_time,
            })

            return GenerationResult(
                trajectory=trajectory, events=events, metadata=metadata, key_poses=key_poses
            )

        except Exception as e:
            metadata["generation_failed"] = True
            metadata["reason"] = f"exception: {e}"
            metadata["elapsed_time"] = time.time() - start_time
            return GenerationResult(trajectory=None, metadata=metadata)

    def _query_perception_with_retry(self, env, reference_object: str) -> dict:
        """Query perception agent with retries (up to max_perception_retries)."""
        perception = None
        for _ in range(self.max_perception_retries):
            perception = self.perception_agent.query(
                observation=env.get_observation(),
                reference_object=reference_object,
            )
            if perception["responses_result_is_valid"]:
                break
        return perception

    def get_perception_agent(self) -> PerceptionAgent:
        """Expose perception agent for use by validator / ABD."""
        return self.perception_agent

    def get_motion_planner(self) -> MotionPlanner:
        """Expose motion planner for use by validator / ABD (3D reconstruction)."""
        return self.motion_planner
