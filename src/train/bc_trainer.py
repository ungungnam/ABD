"""Behavioral Cloning trainer.

Trains a BC policy on ABD-collected data and evaluates via:
  - Validation MSE (per-dimension action error)
  - Optional rollout evaluation (BC success rate)

Supports:
  - Dataset-size scaling experiments
  - Multi-policy comparison (train on data collected by different reset policies)
  - Comprehensive logging via TrainLogger
"""

import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from train.bc_policy import BCPolicy
from train.dataset_loader import (
    ABDDataset, DummyABDDataset, build_dataloader, collate_fn, create_subset,
)
from train.train_logger import TrainLogger


class BCTrainer:
    """Behavioral Cloning trainer with comprehensive logging."""

    def __init__(
        self,
        # Data
        dataset: Dataset = None,
        dataset_root: str = None,
        dummy: bool = False,
        dummy_episodes: int = 20,
        dummy_episode_length: int = 30,
        # Model
        num_cameras: int = 3,
        camera_names: list[str] = None,
        image_size: tuple[int, int] = (128, 128),
        image_feature_dim: int = 128,
        state_dim: int = 7,
        action_dim: int = 7,
        hidden_dim: int = 256,
        share_encoder: bool = True,
        # Training
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        num_epochs: int = 50,
        val_fraction: float = 0.2,
        grad_clip: float = 1.0,
        # Logging
        log_dir: str = "outputs/train",
        experiment_name: str = "bc_train",
        # Eval
        eval_every: int = 10,
        # Device
        device: str = None,
        seed: int = 42,
        # Context
        policy_method: str = "",
        dataset_fraction: float = 1.0,
    ):
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.grad_clip = grad_clip
        self.eval_every = eval_every
        self.policy_method = policy_method
        self.dataset_fraction = dataset_fraction
        self.camera_names = camera_names or ["wrist", "front", "table"]
        self.action_dim = action_dim

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Dataset ──
        if dataset is not None:
            self.dataset = dataset
        elif dummy:
            self.dataset = DummyABDDataset(
                num_episodes=dummy_episodes,
                episode_length=dummy_episode_length,
                camera_names=self.camera_names,
                image_size=image_size,
                state_dim=state_dim,
                action_dim=action_dim,
                seed=seed,
            )
        elif dataset_root:
            self.dataset = ABDDataset(
                dataset_root=dataset_root,
                camera_names=self.camera_names,
                image_size=image_size,
            )
        else:
            raise ValueError("Provide dataset, dataset_root, or set dummy=True")

        # Subset for scaling experiments
        if dataset_fraction < 1.0:
            self.dataset = create_subset(self.dataset, dataset_fraction, seed=seed)

        self.dataset_size = len(self.dataset)

        # Train/val split
        n_val = max(1, int(len(self.dataset) * val_fraction))
        n_train = len(self.dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )

        self.train_loader = build_dataloader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
        )
        self.val_loader = build_dataloader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
        )

        # ── Model ──
        self.policy = BCPolicy(
            num_cameras=num_cameras,
            image_feature_dim=image_feature_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            share_encoder=share_encoder,
        ).to(self.device)

        # ── Optimizer ──
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs,
        )

        # ── Loss ──
        self.loss_fn = nn.MSELoss(reduction="none")

        # ── Logger ──
        self.logger = TrainLogger(log_dir=log_dir, experiment_name=experiment_name)

    def train(self) -> dict:
        """Run full training loop. Returns training summary dict."""
        self.logger.start()

        print(f"  Dataset size: {self.dataset_size} "
              f"(fraction={self.dataset_fraction:.0%})")
        print(f"  Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")
        print(f"  Device: {self.device}")
        print(f"  Policy params: {sum(p.numel() for p in self.policy.parameters()):,}")
        print()

        global_step = 0

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # ── Train ──
            train_loss, train_per_dim, global_step = self._train_epoch(
                epoch, global_step
            )

            # ── Validate ──
            val_loss, val_per_dim = self._validate_epoch()

            epoch_time = time.time() - epoch_start

            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_action_mse_per_dim=train_per_dim,
                val_action_mse_per_dim=val_per_dim,
                epoch_time=epoch_time,
            )

            self.scheduler.step()

            # ── Eval (dummy rollout) ──
            if epoch % self.eval_every == 0 or epoch == self.num_epochs:
                eval_result = self._evaluate(epoch)
                self.logger.log_eval(
                    epoch=epoch,
                    success_rate=eval_result["success_rate"],
                    num_trials=eval_result["num_trials"],
                    avg_return=eval_result["avg_return"],
                    details=eval_result,
                )

            # Save best checkpoint
            if val_loss <= self.logger.best_val_loss:
                self._save_checkpoint(epoch, val_loss, is_best=True)

        # Final save
        self._save_checkpoint(self.num_epochs, val_loss, is_best=False)

        summary = self.logger.finish(
            dataset_size=self.dataset_size,
            policy_method=self.policy_method,
            config={
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "lr": self.optimizer.defaults["lr"],
                "dataset_fraction": self.dataset_fraction,
                "seed": self.seed,
            },
        )

        # Write BC eval log entry (E5: Dataset Utility and Scaling)
        bc_eval_entry = {
            "policy_method": self.policy_method,
            "dataset_size": self.dataset_size,
            "dataset_fraction": self.dataset_fraction,
            "best_val_loss": float(self.logger.best_val_loss),
            "final_success_rate": getattr(summary, "final_success_rate", None),
            "seed": self.seed,
            "timestamp": time.time(),
        }
        bc_eval_path = os.path.join(self.logger.log_dir, "bc_eval_log.jsonl")
        with open(bc_eval_path, "a") as f:
            f.write(json.dumps(bc_eval_entry) + "\n")

        return summary.__dict__

    def _train_epoch(self, epoch: int, global_step: int):
        """Single training epoch."""
        self.policy.train()
        total_loss = 0.0
        per_dim_loss = np.zeros(self.action_dim)
        n_batches = 0

        for batch in self.train_loader:
            images_list = [
                batch["images"][cam].to(self.device) for cam in self.camera_names
            ]
            state = batch["state"].to(self.device)
            action_gt = batch["action"].to(self.device)

            pred = self.policy(images_list, state)
            loss_per_dim = self.loss_fn(pred, action_gt).mean(dim=0)
            loss = loss_per_dim.mean()

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.grad_clip
            )

            self.optimizer.step()

            global_step += 1
            total_loss += loss.item()
            per_dim_loss += loss_per_dim.detach().cpu().numpy()
            n_batches += 1

            # Log every step
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log_step(
                step=global_step, epoch=epoch,
                loss=loss.item(), grad_norm=float(grad_norm), lr=lr,
            )

        avg_loss = total_loss / max(n_batches, 1)
        avg_per_dim = (per_dim_loss / max(n_batches, 1)).tolist()
        return avg_loss, avg_per_dim, global_step

    @torch.no_grad()
    def _validate_epoch(self):
        """Validation pass."""
        self.policy.eval()
        total_loss = 0.0
        per_dim_loss = np.zeros(self.action_dim)
        n_batches = 0

        for batch in self.val_loader:
            images_list = [
                batch["images"][cam].to(self.device) for cam in self.camera_names
            ]
            state = batch["state"].to(self.device)
            action_gt = batch["action"].to(self.device)

            pred = self.policy(images_list, state)
            loss_per_dim = self.loss_fn(pred, action_gt).mean(dim=0)
            loss = loss_per_dim.mean()

            total_loss += loss.item()
            per_dim_loss += loss_per_dim.cpu().numpy()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_per_dim = (per_dim_loss / max(n_batches, 1)).tolist()
        return avg_loss, avg_per_dim

    def _evaluate(self, epoch: int) -> dict:
        """Dummy evaluation — simulates rollout success rate.

        In a real setup, this would deploy the policy on the robot/sim
        and measure task success. For now, we proxy it via val loss:
        lower val loss -> higher estimated success rate.
        """
        _, val_per_dim = self._validate_epoch()
        val_mse = np.mean(val_per_dim)

        # Proxy: map val MSE to a success rate estimate
        # Lower MSE = higher success. This is a rough heuristic.
        estimated_sr = max(0.0, min(1.0, 1.0 - val_mse * 10.0))

        return {
            "success_rate": estimated_sr,
            "num_trials": 20,  # dummy
            "avg_return": estimated_sr * 100.0,
            "val_mse": float(val_mse),
            "val_per_dim": val_per_dim,
        }

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        ckpt_dir = os.path.join(self.logger.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        if is_best:
            path = os.path.join(ckpt_dir, "best.pt")
            torch.save(state, path)

        if epoch == self.num_epochs:
            path = os.path.join(ckpt_dir, "last.pt")
            torch.save(state, path)
