"""Dataset loader for ABD-collected LeRobot data.

Supports:
  - Loading from episodes.jsonl (collection logs) or LeRobot parquet datasets
  - Dummy data generation for testing the training pipeline without real data
  - Subset sampling for dataset-size scaling experiments
"""

import json
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class ABDDataset(Dataset):
    """Dataset for BC training from ABD-collected episodes.

    Each sample contains:
      - images: dict of camera_name -> (3, H, W) uint8 tensor
      - state: (7,) float32 tensor [x,y,z,rx,ry,rz,gripper]
      - action: (7,) float32 tensor
      - task: str
    """

    def __init__(self, dataset_root: str, camera_names: list[str] = None,
                 image_size: tuple[int, int] = (128, 128)):
        self.dataset_root = dataset_root
        self.camera_names = camera_names or ["wrist", "front", "table"]
        self.image_size = image_size

        self.frames = []
        self._load_dataset()

    def _load_dataset(self):
        """Load frames from the dataset directory.

        Tries LeRobot parquet first, falls back to raw numpy files.
        """
        parquet_path = os.path.join(self.dataset_root, "data")
        if os.path.isdir(parquet_path):
            self._load_lerobot(parquet_path)
        else:
            # Try loading from episode directories
            ep_dir = os.path.join(self.dataset_root, "episodes")
            if os.path.isdir(ep_dir):
                self._load_episode_dirs(ep_dir)

    def _load_lerobot(self, data_path: str):
        """Load from LeRobot parquet format."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(data_path)
            df = table.to_pandas()

            for idx in range(len(df)):
                row = df.iloc[idx]
                frame = {
                    "state": np.array(row.get("observation.state", np.zeros(7)), dtype=np.float32),
                    "action": np.array(row.get("action", np.zeros(7)), dtype=np.float32),
                    "task": str(row.get("task", "")),
                    "images": {},
                }
                for cam in self.camera_names:
                    key = f"observation.images.rgb.{cam}"
                    if key in row:
                        frame["images"][cam] = np.array(row[key], dtype=np.uint8)
                self.frames.append(frame)
        except Exception:
            pass

    def _load_episode_dirs(self, ep_dir: str):
        """Load from per-episode numpy directories."""
        for ep_name in sorted(os.listdir(ep_dir)):
            ep_path = os.path.join(ep_dir, ep_name)
            if not os.path.isdir(ep_path):
                continue
            states_path = os.path.join(ep_path, "states.npy")
            actions_path = os.path.join(ep_path, "actions.npy")
            if not os.path.exists(states_path) or not os.path.exists(actions_path):
                continue
            states = np.load(states_path)
            actions = np.load(actions_path)
            for t in range(len(states)):
                frame = {
                    "state": states[t].astype(np.float32),
                    "action": actions[t].astype(np.float32),
                    "task": ep_name,
                    "images": {},
                }
                for cam in self.camera_names:
                    img_path = os.path.join(ep_path, f"{cam}_{t:04d}.npy")
                    if os.path.exists(img_path):
                        frame["images"][cam] = np.load(img_path)
                self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        frame = self.frames[idx]
        images = {}
        for cam in self.camera_names:
            if cam in frame["images"] and frame["images"][cam] is not None:
                img = frame["images"][cam]
                img = torch.from_numpy(img).float() / 255.0
                if img.ndim == 2:
                    img = img.unsqueeze(0).expand(3, -1, -1)
                elif img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                img = F.interpolate(img.unsqueeze(0), size=self.image_size,
                                    mode="bilinear", align_corners=False).squeeze(0)
            else:
                img = torch.zeros(3, *self.image_size)
            images[cam] = img

        return {
            "images": images,
            "state": torch.from_numpy(frame["state"]),
            "action": torch.from_numpy(frame["action"]),
            "task": frame["task"],
        }


class DummyABDDataset(Dataset):
    """Generates synthetic data for testing the training pipeline.

    Produces random images, states, and actions that mimic the real data shape.
    """

    def __init__(
        self,
        num_episodes: int = 20,
        episode_length: int = 30,
        camera_names: list[str] = None,
        image_size: tuple[int, int] = (128, 128),
        state_dim: int = 7,
        action_dim: int = 7,
        seed: int = 42,
    ):
        self.camera_names = camera_names or ["wrist", "front", "table"]
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        rng = np.random.RandomState(seed)
        self.frames = []

        tasks = ["banana_to_pan", "banana_to_plate"]
        for ep in range(num_episodes):
            task = tasks[ep % 2]
            # Generate a smooth trajectory per episode
            base_state = rng.randn(state_dim).astype(np.float32) * 0.1
            for t in range(episode_length):
                alpha = t / max(episode_length - 1, 1)
                state = base_state + alpha * rng.randn(state_dim).astype(np.float32) * 0.01
                action = state + rng.randn(action_dim).astype(np.float32) * 0.005

                images = {}
                for cam in self.camera_names:
                    img = rng.randint(0, 256, size=(3, *image_size), dtype=np.uint8)
                    images[cam] = torch.from_numpy(img).float() / 255.0

                self.frames.append({
                    "images": images,
                    "state": torch.from_numpy(state),
                    "action": torch.from_numpy(action),
                    "task": task,
                })

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        return self.frames[idx]


def create_subset(dataset: Dataset, fraction: float, seed: int = 42) -> Subset:
    """Create a random subset of the dataset for scaling experiments."""
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    return Subset(dataset, indices)


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for ABD dataset batches."""
    camera_names = list(batch[0]["images"].keys())
    images = {cam: torch.stack([b["images"][cam] for b in batch]) for cam in camera_names}
    states = torch.stack([b["state"] for b in batch])
    actions = torch.stack([b["action"] for b in batch])
    tasks = [b["task"] for b in batch]
    return {"images": images, "state": states, "action": actions, "task": tasks}


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )


import torch.nn.functional as F  # noqa: E402 (used in __getitem__)
