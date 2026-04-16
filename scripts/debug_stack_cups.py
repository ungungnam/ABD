"""Stack Cups 디버깅 스크립트.

VSCode에서 F5로 실행 (launch.json: "Debug: stack cups" 항목 참고).
각 단계마다 중단점(breakpoint)을 걸어 확인할 수 있도록 구성.

STEP 1. AprilTag 감지  — 카메라별 tag 4/5 world pose 확인
STEP 2. Grasp pose     — pick / place 좌표 계산
STEP 3. Trajectory     — waypoint 수 및 구간별 구조 확인
STEP 4. Execute        — EXECUTE = True 일 때만 실제 로봇 실행
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401

import numpy as np
import hydra
from omegaconf import DictConfig

from env.base_env import build_env
from utils.camera_utils import detect_single_tag_world_pose
from motion_planner.motion_planner import MotionPlanner
from executor.trajectory_executor import TrajectoryExecutor

# ── 설정 ──────────────────────────────────────────────────────────────
EXECUTE      = False   # True로 바꾸면 실제 로봇 실행
PICK_TAG_ID  = 5       # 보라색 컵
PLACE_TAG_ID = 4       # 파란색 컵
IS_UNSTACK   = False   # True = unstack (보라컵을 파란컵 옆에 내려놓기)
UNSTACK_XY_OFFSET = [0.2, 0.0]   # unstack 시 파란컵 옆 오프셋 (m)
# ─────────────────────────────────────────────────────────────────────


def fmt(T):
    if T is None:
        return "None"
    p = T[:3, 3]
    return f"x={p[0]:.4f}  y={p[1]:.4f}  z={p[2]:.4f}"


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    env     = build_env(cfg)
    cameras = env.get_cameras()
    robot   = env.get_robot()
    mp      = MotionPlanner(cfg.motion_planner, env=env, robot=robot, cameras=cameras)

    # ── STEP 1: AprilTag 감지 ────────────────────────────────────────
    print("\n[STEP 1] AprilTag Detection")
    tag_poses = {}
    for tag_id, cup in [(PICK_TAG_ID, "purple"), (PLACE_TAG_ID, "blue")]:
        for cam_name, camera in cameras.items():
            T_wt = detect_single_tag_world_pose(camera, tag_id, tag_size=0.04)
            found = T_wt is not None
            print(f"  tag {tag_id} ({cup:6s})  cam={cam_name:8s}  {'OK  ' if found else 'MISS'}  {fmt(T_wt)}")
            if found and tag_id not in tag_poses:
                tag_poses[tag_id] = T_wt   # 처음 감지된 카메라 결과 사용

    print(f"\n  감지된 tag: {list(tag_poses.keys())}")
    # ← 중단점: tag_poses 확인

    # ── STEP 2: Grasp Pose 계산 ──────────────────────────────────────
    print("\n[STEP 2] Grasp Pose Computation")

    place_xy = UNSTACK_XY_OFFSET if IS_UNSTACK else None
    place_z  = mp._GRASP_BELOW_TAG if IS_UNSTACK else -mp._STACK_OFFSET  # unstack은 테이블 레벨

    pick_pose  = mp._get_grasp_pose_from_apriltag(
        PICK_TAG_ID,  z_offset=-mp._GRASP_BELOW_TAG, hover_offset=0.03)
    place_pose = mp._get_grasp_pose_from_apriltag(
        PLACE_TAG_ID, z_offset=-place_z,              hover_offset=0.03, xy_offset=place_xy)

    if pick_pose:
        print(f"  pick  pre_T : {fmt(pick_pose['pre_T_wg'])}")
        print(f"  pick  T_wg  : {fmt(pick_pose['T_wg'])}")
    else:
        print(f"  [FAIL] pick pose — tag {PICK_TAG_ID} not found")

    if place_pose:
        print(f"  place pre_T : {fmt(place_pose['pre_T_wg'])}")
        print(f"  place T_wg  : {fmt(place_pose['T_wg'])}")
    else:
        print(f"  [FAIL] place pose — tag {PLACE_TAG_ID} not found")
    # ← 중단점: pick_pose, place_pose 확인

    if pick_pose is None or place_pose is None:
        print("\n[ABORT] grasp pose 계산 실패")
        return

    # ── STEP 3: Trajectory ───────────────────────────────────────────
    print("\n[STEP 3] Trajectory Preview")
    traj, events = mp.generate_trajectory(
        vlm_action=None,
        pick_grasp_pose=pick_pose,
        place_grasp_pose=place_pose,
    )

    print(f"  waypoints: {len(traj)}   events: {events}")

    seg_labels = [
        "cur → pre_pick",
        "pre_pick → pick (hover+descend)",
        "pick hold + CLOSE",
        "pick → home",
        "home → pre_place",
        "pre_place → place (hover+descend)",
        "place hold + OPEN",
        "place → home",
    ]
    prev = 0
    for i, e in enumerate(events):
        bp = e["at"]
        print(f"    [{prev:3d}–{bp:3d}] {bp-prev+1:3d} wp  cmd={e['cmd']}")
        prev = bp + 1
    print(f"    [{prev:3d}–{len(traj)-1:3d}] {len(traj)-prev:3d} wp  → home")

    # CLOSE / OPEN waypoint 좌표 확인
    for e in events:
        wp = traj[e["at"]]
        print(f"  {e['cmd']:5s} @ wp {e['at']:3d}  {fmt(wp)}")
    # ← 중단점: traj, events 확인

    # ── STEP 4: Execute ──────────────────────────────────────────────
    if not EXECUTE:
        print("\n[STEP 4] Execute 생략 (EXECUTE=False)")
        return

    print("\n[STEP 4] Execute — 로봇 실행")
    task_str = ("unstack purple from blue" if IS_UNSTACK
                else "stack purple cup on blue cup")
    input("  준비되면 Enter 키를 누르세요...")

    executor = TrajectoryExecutor(env=env, dataset_recorder=None)
    result   = executor.execute(traj, events, language_task=task_str)
    print(f"  완료: {result.num_waypoints} waypoints  {result.elapsed_time:.1f}s")


if __name__ == "__main__":
    main()
