"""Trajectory executor — adapted from PaPA's apply_action().

Executes a generated trajectory on the robot and records frames
to a LeRobot dataset.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from env.base_env import ABDBaseEnv

log = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of trajectory execution."""
    final_obs: dict = field(default_factory=dict)
    num_waypoints: int = 0
    elapsed_time: float = 0.0
    completed: bool = True


class TrajectoryExecutor:
    """Execute a trajectory on the robot and optionally record frames."""

    def __init__(self, env: ABDBaseEnv, dataset_recorder=None):
        self.env = env
        self.dataset_recorder = dataset_recorder

    def execute(self, trajectory, events, language_task: str) -> ExecutionResult:
        """Execute trajectory waypoints and return execution metadata.

        Adapted from PaPA's PaPA.apply_action() (PaPA/src/papa/papa.py lines 124-151).
        """
        start_time = time.time()

        # Build event map: waypoint_index -> list of commands
        event_map = {}
        for e in (events or []):
            event_map.setdefault(int(e["at"]), []).append(e["cmd"])

        obs = self.env.get_observation()

        for i, waypoint in enumerate(trajectory):
            idx = i + 1
            cmds = event_map.get(idx, [])
            event = cmds[0] if cmds else None

            log.debug(f"[WP {idx:02d}] raw xyz=({waypoint[0,3]:.4f}, {waypoint[1,3]:.4f}, {waypoint[2,3]:.4f}) event={event}")

            pose_6d, gripper = self.env.waypoint_event_to_pose_6d_gripper(
                waypoint, event
            )
            log.debug(f"pose_6d xyz(um)=({pose_6d[0]}, {pose_6d[1]}, {pose_6d[2]}) gripper={gripper}")

            if self.dataset_recorder is not None:
                frame = self._make_frame(obs, pose_6d, gripper, language_task)
                self.dataset_recorder.add_frame(frame)

            obs, reward, done = self.env.step(pose_6d, gripper)

        return ExecutionResult(
            final_obs=obs,
            num_waypoints=len(trajectory),
            elapsed_time=time.time() - start_time,
            completed=True,
        )

    def _make_frame(self, obs: dict, pose_6d, gripper, language_task: str) -> dict:
        """Build a LeRobot dataset frame.

        Adapted from PaPA's PaPA._make_lerobot_dataset_frame().
        """
        frame = {}
        for k, v in obs.items():
            k_clean = k.replace("_rgb", "")
            frame[f"observation.images.rgb.{k_clean}"] = v

        action = [int(x) for x in pose_6d] + [int(gripper)]
        action = np.asarray(action, dtype=np.float32)
        frame["observation.state"] = action
        frame["action"] = action
        frame["task"] = language_task
        return frame
