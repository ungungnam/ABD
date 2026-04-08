"""Debug full ABDRealEnv initialization: robot + cameras + AprilTag calibration.

This is the most expensive debug script — it runs camera calibration.
Run only after debug_robot.py and debug_camera.py pass.

Usage:
    python scripts/debug_env.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401

import numpy as np
from omegaconf import OmegaConf


def main():
    print("Initializing ABDRealEnv (robot + cameras + AprilTag calibration)...")
    print("This may take ~30s for camera calibration.\n")

    try:
        from env.real_env import ABDRealEnv
    except ImportError as e:
        print(f"[FAIL] Cannot import ABDRealEnv: {e}")
        sys.exit(1)

    try:
        cfg = OmegaConf.create({})
        env = ABDRealEnv(cfg)
        print("[OK] ABDRealEnv initialized.")
    except Exception as e:
        print(f"[FAIL] ABDRealEnv init error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print(f"\nCameras: {list(env.get_cameras().keys())}")

    print("\nCapturing observation...")
    try:
        obs = env.get_observation()
        for k, v in obs.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        print("[OK] Observation captured.")
    except Exception as e:
        print(f"[FAIL] get_observation() error: {e}")
        sys.exit(1)

    print("\nCalling go_to_init_pose()...")
    try:
        env.go_to_init_pose()
        print("[OK] Robot moved to init pose.")
    except Exception as e:
        print(f"[FAIL] go_to_init_pose() error: {e}")
        sys.exit(1)

    T_we = env.robot.get_end_pose(gripper_depth=False)
    print(f"\nEnd-effector pose after init:\n{np.round(T_we, 4)}")
    print("\n[OK] Full environment check passed.")


if __name__ == "__main__":
    main()
