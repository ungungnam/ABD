"""Dummy trajectory generator for testing.

Produces synthetic SE(3) waypoint trajectories without
calling VLM, perception, or motion planner.
"""

import numpy as np

from task.task_family import TaskDefinition
from trajectory_generator.base_generator import BaseTrajectoryGenerator, GenerationResult


class DummyTrajectoryGenerator(BaseTrajectoryGenerator):
    """Generates fake pick-place trajectories for pipeline testing.

    Simulates occasional generation failures to test error paths.
    """

    def __init__(self, failure_rate: float = 0.1, num_waypoints: int = 15, seed: int = 42):
        self.failure_rate = failure_rate
        self.num_waypoints = num_waypoints
        self.rng = np.random.RandomState(seed)

    def generate(self, task: TaskDefinition, env) -> GenerationResult:
        # Simulate occasional generation failure
        if self.rng.rand() < self.failure_rate:
            return GenerationResult(
                trajectory=None,
                metadata={"generation_failed": True, "reason": "dummy_vlm_failed", "elapsed_time": 0.5},
            )

        # Generate a smooth pick-place trajectory
        trajectory = []
        start_pos = np.array([0.25, 0.0, 0.30])
        pick_pos = np.array([0.25, 0.05, 0.10])
        place_pos = np.array([0.25, -0.05, 0.10])
        end_pos = np.array([0.25, 0.0, 0.30])

        keypoints = [start_pos, pick_pos, pick_pos, place_pos, end_pos]
        n_per_segment = max(1, self.num_waypoints // (len(keypoints) - 1))

        for i in range(len(keypoints) - 1):
            for t in np.linspace(0, 1, n_per_segment, endpoint=(i == len(keypoints) - 2)):
                pos = keypoints[i] * (1 - t) + keypoints[i + 1] * t
                wp = np.eye(4)
                wp[:3, 3] = pos + self.rng.randn(3) * 0.002
                trajectory.append(wp)

        # Events: close gripper at pick, open at place
        pick_idx = n_per_segment + 1
        place_idx = 3 * n_per_segment + 1
        events = [
            {"at": min(pick_idx, len(trajectory)), "cmd": "CLOSE"},
            {"at": min(place_idx, len(trajectory)), "cmd": "OPEN"},
        ]

        return GenerationResult(
            trajectory=trajectory,
            events=events,
            metadata={
                "generation_failed": False,
                "pick_object": "banana",
                "place_object": "pan",
                "elapsed_time": 0.3 + self.rng.rand() * 0.5,
            },
        )
