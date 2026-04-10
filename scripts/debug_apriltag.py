"""Diagnose AprilTag detection failures per camera.

Captures one frame from each camera and prints exactly which filter
stage eliminates each detected tag.

Usage:
    python scripts/debug_apriltag.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401

import cv2
import numpy as np
from pupil_apriltags import Detector

CAMERAS = {
    "front":  ("FixedRealSenseCamera",  "f1120304"),
    "table":  ("FixedRealSenseCamera",  "f1371426"),
    "wrist":  ("WristRealSenseCamera",  "341222301935"),
}

BOARD_IDS = {0, 1, 2, 3}
DECISION_MARGIN_MIN = 10.0
MIN_TAGS = 2
OUTLIER_THRESH = 0.03  # metres

detector = Detector(
    families="tag36h11",
    nthreads=2,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)


def diagnose_frame(rgb, K, dist, cam_name):
    rgb_undist = cv2.undistort(rgb, K, dist)
    gray = cv2.cvtColor(rgb_undist, cv2.COLOR_RGB2GRAY)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    dets = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=0.102,
    )

    print(f"\n  Raw detections: {len(dets)} tag(s)")
    if not dets:
        print("  => FAIL at stage 1: no tags detected at all")
        print("     Check lighting, focus, and that tag36h11 family is used")
        return

    passed_id, passed_margin = [], []
    for d in dets:
        tid = int(d.tag_id)
        dm  = float(getattr(d, "decision_margin", 0.0))
        print(f"    tag_id={tid}  decision_margin={dm:.1f}", end="")

        if tid not in BOARD_IDS:
            print(f"  => DROPPED (ID {tid} not in board IDs {BOARD_IDS})")
            continue
        if dm < DECISION_MARGIN_MIN:
            print(f"  => DROPPED (margin {dm:.1f} < threshold {DECISION_MARGIN_MIN})")
            continue

        print("  => OK")
        passed_id.append(tid)
        passed_margin.append(dm)

    if len(passed_id) == 0:
        print(f"  => FAIL at stage 2/3: 0 tags passed ID+margin filters")
        if dets:
            max_dm = max(float(getattr(d, "decision_margin", 0.0)) for d in dets)
            print(f"     Best decision_margin seen: {max_dm:.1f}  (threshold={DECISION_MARGIN_MIN})")
            print(f"     Try lowering decision_margin_min in estimate_T_ct_from_apriltag()")
        return

    if len(passed_id) < MIN_TAGS:
        print(f"  => FAIL at stage 4: only {len(passed_id)} tag(s) passed, need min_tags={MIN_TAGS}")
        print(f"     Consider lowering min_tags=1 if only one tag is in view")
        return

    print(f"  => {len(passed_id)} tag(s) passed filters — calibration should succeed")


def main():
    try:
        from camera.realsense import FixedRealSenseCamera, WristRealSenseCamera
    except ImportError as e:
        print(f"[FAIL] Cannot import camera module: {e}")
        sys.exit(1)

    for name, (cls_name, serial) in CAMERAS.items():
        print(f"\n{'='*50}")
        print(f"Camera: {name}  (SN: {serial})")
        try:
            if cls_name == "WristRealSenseCamera":
                cam = WristRealSenseCamera(serial, mounted_on=None)
            else:
                cam = FixedRealSenseCamera(serial)
            cam._lazy_init()

            rgb = cam.get_rgb()
            save_path = f"/tmp/debug_apriltag_{name}.jpg"
            from PIL import Image
            Image.fromarray(rgb).save(save_path)
            print(f"  Frame saved → {save_path}")

            diagnose_frame(rgb, cam.K, cam.dist, name)

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print("Done. Open /tmp/debug_apriltag_*.jpg to visually inspect frames.")


if __name__ == "__main__":
    main()
