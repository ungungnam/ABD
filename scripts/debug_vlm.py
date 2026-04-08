"""Debug VLM server connectivity at localhost:9876.

Usage:
    python scripts/debug_vlm.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401

import base64
import io
import json

import numpy as np
import requests
from PIL import Image


VLM_URL = "http://localhost:9876/inference"


def encode_image(rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    print(f"Testing VLM server at {VLM_URL}")

    # Build a fake observation with a random RGB image
    fake_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    payload = {
        "images": {"front": encode_image(fake_rgb)},
        "task": "pick the banana and put it in the pan",
        "mode": "plan",
    }

    try:
        response = requests.post(VLM_URL, json=payload, timeout=10)
        print(f"  Status: {response.status_code}")
        text = response.text[:500]
        print(f"  Response: {text}")

        if response.status_code == 200:
            data = response.json()
            print(f"  Keys: {list(data.keys())}")
            print("[OK] VLM server is reachable and responding.")
        else:
            print(f"[WARN] Server returned non-200 status: {response.status_code}")

    except requests.ConnectionError:
        print(f"[FAIL] Cannot connect to VLM server at {VLM_URL}")
        print("       Make sure the VLM server is running before proceeding.")
        sys.exit(1)
    except requests.Timeout:
        print(f"[FAIL] VLM server timed out after 10s.")
        sys.exit(1)


if __name__ == "__main__":
    main()
