"""Debug Piper robot arm connectivity and basic motion.

Usage:
    python scripts/debug_robot.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401

import numpy as np
from types import SimpleNamespace


def main():
    print("Testing Piper robot connection...")

    try:
        from robot.piper import Piper
    except ImportError as e:
        print(f"[FAIL] Cannot import Piper: {e}")
        print("       Ensure piper_sdk is installed.")
        sys.exit(1)

    try:
        robot = Piper(SimpleNamespace())
        print("  Piper object created.")

        print("  Connecting and enabling (lazy init)...")
        robot._lazy_init(set_to_zero=True)
        print("  [OK] Connected and moved to init pose.")

        joints = robot.get_joints()
        print(f"  Joints (rad): {[f'{j:.4f}' for j in joints]}")

        T_we = robot.get_end_pose(gripper_depth=False)
        print(f"  End-effector pose (T_we):\n{np.round(T_we, 4)}")

        gripper = robot.get_gripper()
        print(f"  Gripper value: {gripper}  (0=closed, 70000=open)")

        print("  Calling go_to_init_pose()...")
        robot.go_to_init_pose()
        print("[OK] Robot init pose confirmed.")

    except Exception as e:
        print(f"[FAIL] Robot error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
