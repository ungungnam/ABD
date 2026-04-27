"""Trajectory executor — adapted from PaPA's apply_action().

Executes a generated trajectory on the robot and records frames
to a LeRobot dataset.
"""

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from env.base_env import ABDBaseEnv
from utils.transform_utils import rot_about_axis

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IK-fallback tuning constants
# ---------------------------------------------------------------------------
_IK_POS_THRESHOLD_M  = 0.05   # position error that triggers fallback
_IK_SETTLE_S         = 0.10   # extra settle time inside each fallback attempt
_IK_ANGLE_STEPS_DEG  = [0, 2, 4, 6, 8, 10]
_IK_AXES = [
    np.array([1., 0., 0.]),
    np.array([-1., 0., 0.]),
    np.array([0., 1., 0.]),
    np.array([0., -1., 0.]),
    np.array([0., 0., 1.]),
    np.array([0., 0., -1.]),
]


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

    def execute(self, trajectory, events, language_task: str,
                key_poses=None, use_ik_fallback: bool = True) -> ExecutionResult:
        """Execute trajectory waypoints and return execution metadata.

        Adapted from PaPA's PaPA.apply_action() (PaPA/src/papa/papa.py lines 124-151).

        key_poses: list of 4x4 SE(3) matrices (pre_pick, pick, pre_place, place).
            IK fallback is applied only to waypoints within _IK_KEY_POSE_RADIUS_M of
            any key pose.  Pass None (default) to apply fallback to every waypoint.
        use_ik_fallback: set False to skip IK fallback entirely for all waypoints.
        """
        start_time = time.time()

        event_map = {}
        for e in (events or []):
            event_map.setdefault(int(e["at"]), []).append(e["cmd"])

        obs = self.env.get_observation()

        ik_indices = self._key_indices(trajectory, key_poses) if key_poses is not None else None
        log.debug(f"[IK Fallback] key waypoint indices: {sorted(ik_indices) if ik_indices else 'all'}")

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

            if use_ik_fallback and (ik_indices is None or i in ik_indices):
                obs, reward, done = self._step_with_ik_fallback(waypoint, pose_6d, gripper)
            else:
               obs, reward, done = self.env.step(pose_6d, gripper)

        return ExecutionResult(
            final_obs=obs,
            num_waypoints=len(trajectory),
            elapsed_time=time.time() - start_time,
            completed=True,
        )

    # ------------------------------------------------------------------
    # IK fallback — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key_indices(trajectory, key_poses) -> set:
        """Return the set of trajectory indices closest to each key pose (one per pose)."""
        indices = set()
        traj_pts = np.stack([wp[:3, 3] for wp in trajectory])  # (N, 3)
        for kp in key_poses:
            dists = np.linalg.norm(traj_pts - kp[:3, 3], axis=1)
            indices.add(int(np.argmin(dists)))
        return indices

    def _pos_error(self, robot, target_T: np.ndarray) -> Optional[float]:
        """Return translation error (m) between robot's current pose and target_T.

        Returns None if the robot state cannot be read (e.g. dummy env).
        """
        try:
            actual_T = robot.get_end_pose(gripper_depth=True)
            return float(np.linalg.norm(actual_T[:3, 3] - target_T[:3, 3]))
        except Exception:
            return None

    def _rotation_candidates(self, target_R: np.ndarray) -> Iterator[np.ndarray]:
        """Yield perturbed rotation matrices around target_R.

        Samples by composing axis-angle rotations onto target_R — equivalent
        to quaternion multiplication — so it is free from the ±180° Euler
        discontinuity.  Candidates are ordered from smallest to largest
        perturbation angle.

        For each angle step we try:
          1. 단일 축 (6개)
          2. 두 축의 대각 방향 (최대 12개, 반대 축 쌍은 합이 0이므로 건너뜀)
          3. 세 축의 대각 방향 (최대 20개, 마찬가지로 영벡터 건너뜀)
        """
        for angle_deg in _IK_ANGLE_STEPS_DEG:
            angle_rad = np.deg2rad(angle_deg)

            # 1-axis
            for axis in _IK_AXES:
                yield rot_about_axis(axis, angle_rad) @ target_R

            # 2-axis diagonal
            for ax1, ax2 in itertools.combinations(_IK_AXES, 2):
                combined = ax1 + ax2
                norm = np.linalg.norm(combined)
                if norm > 1e-6:
                    yield rot_about_axis(combined / norm, angle_rad) @ target_R

            # 3-axis diagonal
            for ax1, ax2, ax3 in itertools.combinations(_IK_AXES, 3):
                combined = ax1 + ax2 + ax3
                norm = np.linalg.norm(combined)
                if norm > 1e-6:
                    yield rot_about_axis(combined / norm, angle_rad) @ target_R

    # ------------------------------------------------------------------
    # IK fallback — main
    # ------------------------------------------------------------------

    def _step_with_ik_fallback(self, waypoint: np.ndarray, pose_6d, gripper):
        """Execute one waypoint; retry with rotation perturbations on IK failure.

        1. Send the original command.
        2. Compare actual vs target position via _pos_error().
        3. If error > _IK_POS_THRESHOLD_M, iterate over rotation candidates
           from _rotation_candidates(), keeping (x, y, z) fixed.
        4. Return (obs, reward, done) from the first successful attempt, or
           from the last attempt if none succeed.

        Silently skipped on environments without a robot that exposes
        get_end_pose() (e.g. DummyEnv).
        """
        obs, reward, done = self.env.step(pose_6d, gripper)

        try:
            robot = self.env.get_robot()
            if not hasattr(robot, "get_end_pose"):
                return obs, reward, done
        except Exception:
            return obs, reward, done

        time.sleep(_IK_SETTLE_S)
        err = self._pos_error(robot, waypoint)
        if err is None or err <= _IK_POS_THRESHOLD_M:
            return obs, reward, done

        log.warning(
            f"[IK Fallback] pos_err={err * 100:.1f} cm > "
            f"{_IK_POS_THRESHOLD_M * 100:.0f} cm — searching nearby rotations"
        )

        last_obs, last_reward, last_done = obs, reward, done

        for R_candidate in self._rotation_candidates(waypoint[:3, :3]):
            T_try = waypoint.copy()
            T_try[:3, :3] = R_candidate

            try:
                pose_6d_try, _ = self.env.waypoint_event_to_pose_6d_gripper(T_try, None)
            except Exception:
                continue

            last_obs, last_reward, last_done = self.env.step(pose_6d_try, gripper)
            time.sleep(_IK_SETTLE_S)

            new_err = self._pos_error(robot, waypoint)
            if new_err is not None and new_err < _IK_POS_THRESHOLD_M:
                log.info(f"[IK Fallback] Solved: err={new_err * 100:.1f} cm")
                return last_obs, last_reward, last_done

        log.warning("[IK Fallback] No rotation found within search budget; proceeding")
        return last_obs, last_reward, last_done

    # ------------------------------------------------------------------

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
