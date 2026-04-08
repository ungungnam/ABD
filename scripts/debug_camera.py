"""Debug RealSense camera connectivity and image capture.

Usage:
    python scripts/debug_camera.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401

import numpy as np

# Serial numbers from PaPA/src/envs/real_env.py
CAMERAS = {
    "front":  ("FixedRealSenseCamera",  "f1120304"),
    "table":  ("FixedRealSenseCamera",  "f1371426"),
    "wrist":  ("WristRealSenseCamera",  "341222301935"),
}


def main():
    try:
        from camera.realsense import FixedRealSenseCamera, WristRealSenseCamera
    except ImportError as e:
        print(f"[FAIL] Cannot import RealSense camera: {e}")
        print("       Ensure pyrealsense2 is installed.")
        sys.exit(1)

    all_ok = True
    for name, (cls_name, serial) in CAMERAS.items():
        print(f"\nTesting {name} camera (SN: {serial})...")
        try:
            if cls_name == "WristRealSenseCamera":
                cam = WristRealSenseCamera(serial, mounted_on=None)
            else:
                cam = FixedRealSenseCamera(serial)

            cam._lazy_init()
            print(f"  Intrinsics K:\n{np.round(cam.K, 2)}")

            rgb = cam.get_rgb()
            print(f"  RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
            print(f"  Pixel range: [{rgb.min()}, {rgb.max()}]")

            save_path = f"/tmp/debug_camera_{name}.jpg"
            try:
                from PIL import Image
                Image.fromarray(rgb).save(save_path)
                print(f"  Saved frame to {save_path}")
            except Exception:
                pass

            print(f"  [OK] {name} camera working.")

        except Exception as e:
            print(f"  [FAIL] {name} camera error: {e}")
            all_ok = False

    print()
    if all_ok:
        print("[OK] All cameras operational.")
    else:
        print("[WARN] Some cameras failed. Check USB connections and serial numbers.")
        sys.exit(1)


if __name__ == "__main__":
    main()
