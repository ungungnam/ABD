"""Loader for the reset-decision VQA dataset produced by ``build_dataset.py``.

The dataset on disk is a directory::

    <root>/
        manifest.jsonl        one JSON object per sample
        dataset_info.json     aggregate statistics
        images/<sample_id>/{front,wrist,table}.jpg

A :class:`ResetVQASample` is the unit a reset-decision method consumes: it
exposes the last-frame camera views (lazily decoded) plus all episode metadata,
and carries the binary ground-truth ``gt_reset``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class ResetVQASample:
    """One VQA sample: last-frame observation + reset-decision ground truth."""

    sample_id: str
    root: Path                       # dataset root, for resolving image paths
    image_paths: dict                # view -> path relative to root
    gt_reset: bool                   # ground-truth binary reset decision
    gt_label: int                    # raw confusion-matrix label (1..4)
    gt_label_name: str               # TP / TN / FP / FN
    task_name: Optional[str]
    task_direction: Optional[str]
    language_task: Optional[str]
    task_family: Optional[str]
    collection_policy: Optional[str]
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Image access
    # ------------------------------------------------------------------ #
    def available_views(self) -> list:
        """Camera views that have an extracted frame for this sample."""
        return list(self.image_paths.keys())

    def image_path(self, view: str) -> Path:
        """Absolute path to one view's JPEG."""
        return self.root / self.image_paths[view]

    @cached_property
    def images(self) -> dict:
        """All available views as ``{view: HxWx3 uint8 RGB ndarray}`` (cached)."""
        import numpy as np
        from PIL import Image

        out = {}
        for view, rel in self.image_paths.items():
            path = self.root / rel
            if path.exists():
                out[view] = np.asarray(Image.open(path).convert("RGB"))
        return out

    def load_images(self, views: Optional[list] = None) -> dict:
        """Return a ``{view: ndarray}`` dict, optionally restricted to ``views``."""
        imgs = self.images
        if views is None:
            return dict(imgs)
        return {v: imgs[v] for v in views if v in imgs}

    @classmethod
    def from_manifest_entry(cls, entry: dict, root: Path) -> "ResetVQASample":
        return cls(
            sample_id=entry["sample_id"],
            root=root,
            image_paths=entry.get("images", {}),
            gt_reset=bool(entry["gt_reset"]),
            gt_label=int(entry["gt_label"]),
            gt_label_name=entry["gt_label_name"],
            task_name=entry.get("task_name"),
            task_direction=entry.get("task_direction"),
            language_task=entry.get("language_task"),
            task_family=entry.get("task_family"),
            collection_policy=entry.get("collection_policy"),
            metadata=entry.get("metadata", {}),
        )


class ResetVQADataset:
    """In-memory index over a built reset-VQA dataset directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(
                f"No manifest at {manifest}. Run eval/build_dataset.py first."
            )
        self.samples: list = [
            ResetVQASample.from_manifest_entry(json.loads(line), self.root)
            for line in manifest.read_text().splitlines()
            if line.strip()
        ]
        info_path = self.root / "dataset_info.json"
        self.info: dict = json.loads(info_path.read_text()) if info_path.exists() else {}

    # ---- container protocol ------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[ResetVQASample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> ResetVQASample:
        return self.samples[idx]

    # ---- filtering helpers -------------------------------------------- #
    def filter(self, **constraints) -> list:
        """Return samples whose attributes match every ``key=value`` constraint.

        Example::

            ds.filter(task_family="open_drawer", collection_policy="ABD")
        """
        out = []
        for s in self.samples:
            if all(getattr(s, k, None) == v for k, v in constraints.items()):
                out.append(s)
        return out

    def subset(self, sample_ids: set) -> list:
        return [s for s in self.samples if s.sample_id in sample_ids]
