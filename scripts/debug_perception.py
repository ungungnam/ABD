"""Debug perception server connectivity at localhost:9877.

Usage:
    python scripts/debug_perception.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401

import base64
import io

import numpy as np
import requests
from PIL import Image


PERCEPTION_URL = "http://localhost:9877/inference"


def encode_image(rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    print(f"Testing perception server at {PERCEPTION_URL}")

    fake_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    payload = {
        "images": {"front": encode_image(fake_rgb)},
        "reference_object": "banana",
    }

    try:
        response = requests.post(PERCEPTION_URL, json=payload, timeout=10)
        print(f"  Status: {response.status_code}")
        text = response.text[:500]
        print(f"  Response: {text}")

        if response.status_code == 200:
            data = response.json()
            print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            # Check validity field common in PaPA perception responses
            if isinstance(data, dict):
                valid = data.get("responses_result_is_valid", "N/A")
                print(f"  responses_result_is_valid: {valid}")
            print("[OK] Perception server is reachable and responding.")
        else:
            print(f"[WARN] Server returned non-200 status: {response.status_code}")

    except requests.ConnectionError:
        print(f"[FAIL] Cannot connect to perception server at {PERCEPTION_URL}")
        print("       Make sure the perception server is running before proceeding.")
        sys.exit(1)
    except requests.Timeout:
        print(f"[FAIL] Perception server timed out after 10s.")
        sys.exit(1)


if __name__ == "__main__":
    main()
