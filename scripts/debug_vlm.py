"""Debug the active VLM backend.

Builds a backend via ``vlm_client.factory.build_vlm_backend`` and sends a
canned planning-style prompt with a fake observation. Use ``--backend`` to
override the Hydra default and pick a specific backend without editing
``config/config.yaml``.

Usage:
    python scripts/debug_vlm.py                  # uses config/config.yaml default
    python scripts/debug_vlm.py --backend gpt
    python scripts/debug_vlm.py --backend aha
    python scripts/debug_vlm.py --backend qwen
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401

import numpy as np
from omegaconf import OmegaConf

from vlm_client.base import VLMRequest
from vlm_client.factory import build_vlm_backend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend",
        choices=["gpt", "aha", "qwen"],
        default=None,
        help="Override the active VLM backend (defaults to config/config.yaml).",
    )
    p.add_argument(
        "--task",
        default="pick the banana and put it in the pan",
        help="Task description sent as the prompt.",
    )
    return p.parse_args()


def load_vlm_config(backend_override: str | None):
    """Load just the ``vlm`` Hydra group, optionally overriding which one."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if backend_override is None:
        # Read the default group from config/config.yaml
        cfg = OmegaConf.load(os.path.join(repo_root, "config", "config.yaml"))
        defaults = cfg.get("defaults", [])
        backend_name = "gpt"
        for entry in defaults:
            if isinstance(entry, dict) and "vlm" in entry:
                backend_name = entry["vlm"]
                break
    else:
        backend_name = backend_override

    vlm_cfg = OmegaConf.load(
        os.path.join(repo_root, "config", "vlm", f"{backend_name}.yaml")
    )
    return OmegaConf.create({"vlm": vlm_cfg})


def main():
    args = parse_args()
    cfg = load_vlm_config(args.backend)
    print(f"Building backend: {cfg.vlm.backend}")

    backend = build_vlm_backend(cfg)

    fake_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    request = VLMRequest(
        prompt=args.task,
        images={"front_rgb": fake_rgb},
        task_kind="plan",
    )

    response = backend.generate(request)
    print(f"--- {backend.name} response ---")
    print(response[:1000] if response else "[empty]")
    if not response:
        print("[FAIL] Backend returned empty response.")
        sys.exit(1)
    print("[OK] Backend reachable.")


if __name__ == "__main__":
    main()
