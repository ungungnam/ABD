"""Training logger for BC experiments.

Logs:
  - Per-step: loss, gradient norm, learning rate
  - Per-epoch: train loss, val loss, action MSE per dimension
  - Per-eval: BC success rate (when evaluation is available)
  - Scaling: dataset size vs final performance

Outputs:
  - JSON lines log (train_log.jsonl)
  - Summary JSON (train_summary.json)
  - CSV for easy pandas loading (train_metrics.csv)
  - Console progress with rich formatting
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class StepRecord:
    """Per-optimization-step record."""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    timestamp: float = 0.0


@dataclass
class EpochRecord:
    """Per-epoch aggregate record."""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_action_mse: float = 0.0
    val_action_mse: float = 0.0
    train_action_mse_per_dim: list = field(default_factory=list)
    val_action_mse_per_dim: list = field(default_factory=list)
    best_val_loss: float = float("inf")
    epoch_time: float = 0.0
    timestamp: float = 0.0


@dataclass
class EvalRecord:
    """Evaluation record (BC success rate on rollouts)."""
    epoch: int = 0
    success_rate: float = 0.0
    num_trials: int = 0
    avg_return: float = 0.0
    details: dict = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class TrainSummary:
    """Final training summary."""
    total_epochs: int = 0
    total_steps: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    final_success_rate: float = 0.0
    dataset_size: int = 0
    training_time_sec: float = 0.0
    policy_method: str = ""
    config: dict = field(default_factory=dict)


class TrainLogger:
    """Comprehensive training logger for BC experiments."""

    def __init__(self, log_dir: str, experiment_name: str = "bc_train"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        self.step_records: list[StepRecord] = []
        self.epoch_records: list[EpochRecord] = []
        self.eval_records: list[EvalRecord] = []

        self.start_time: Optional[float] = None
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Running averages for console output
        self._running_loss = defaultdict(list)

        # File handles
        self._step_log_path = os.path.join(log_dir, "step_log.jsonl")
        self._epoch_log_path = os.path.join(log_dir, "epoch_log.jsonl")
        self._eval_log_path = os.path.join(log_dir, "eval_log.jsonl")

    def start(self):
        self.start_time = time.time()
        print(f"\n{'='*70}")
        print(f"  BC Training — {self.experiment_name}")
        print(f"  Log directory: {self.log_dir}")
        print(f"{'='*70}\n")

    # ────────────────── Step-level logging ──────────────────

    def log_step(self, step: int, epoch: int, loss: float,
                 grad_norm: float = 0.0, lr: float = 0.0):
        record = StepRecord(
            step=step, epoch=epoch, loss=loss,
            grad_norm=grad_norm, lr=lr, timestamp=time.time(),
        )
        self.step_records.append(record)
        self._running_loss[epoch].append(loss)

        # Append to file incrementally
        with open(self._step_log_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    # ────────────────── Epoch-level logging ──────────────────

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_action_mse_per_dim: list = None,
                  val_action_mse_per_dim: list = None,
                  epoch_time: float = 0.0):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_action_mse=train_loss,
            val_action_mse=val_loss,
            train_action_mse_per_dim=train_action_mse_per_dim or [],
            val_action_mse_per_dim=val_action_mse_per_dim or [],
            best_val_loss=self.best_val_loss,
            epoch_time=epoch_time,
            timestamp=time.time(),
        )
        self.epoch_records.append(record)

        with open(self._epoch_log_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        # Console
        marker = " *" if is_best else ""
        print(f"  Epoch {epoch:4d} | "
              f"train_loss={train_loss:.6f} | "
              f"val_loss={val_loss:.6f} | "
              f"best={self.best_val_loss:.6f} (ep {self.best_epoch}){marker} | "
              f"time={epoch_time:.1f}s")

        if val_action_mse_per_dim:
            dims = ["  x", "  y", "  z", " rx", " ry", " rz", "grip"]
            parts = [f"{dims[i]}={v:.4f}" for i, v in enumerate(val_action_mse_per_dim)]
            print(f"           per-dim MSE: {' | '.join(parts)}")

    # ────────────────── Eval-level logging ──────────────────

    def log_eval(self, epoch: int, success_rate: float, num_trials: int = 0,
                 avg_return: float = 0.0, details: dict = None):
        record = EvalRecord(
            epoch=epoch, success_rate=success_rate,
            num_trials=num_trials, avg_return=avg_return,
            details=details or {}, timestamp=time.time(),
        )
        self.eval_records.append(record)

        with open(self._eval_log_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        print(f"  >>> Eval @ epoch {epoch}: success_rate={success_rate:.1%} "
              f"({num_trials} trials)")

    # ────────────────── Summary + save ──────────────────

    def finish(self, dataset_size: int = 0, policy_method: str = "",
               config: dict = None) -> TrainSummary:
        training_time = time.time() - self.start_time if self.start_time else 0.0

        final_eval_sr = self.eval_records[-1].success_rate if self.eval_records else 0.0

        summary = TrainSummary(
            total_epochs=len(self.epoch_records),
            total_steps=len(self.step_records),
            best_val_loss=self.best_val_loss,
            best_epoch=self.best_epoch,
            final_train_loss=self.epoch_records[-1].train_loss if self.epoch_records else 0.0,
            final_val_loss=self.epoch_records[-1].val_loss if self.epoch_records else 0.0,
            final_success_rate=final_eval_sr,
            dataset_size=dataset_size,
            training_time_sec=training_time,
            policy_method=policy_method,
            config=config or {},
        )

        # Save summary
        summary_path = os.path.join(self.log_dir, "train_summary.json")
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)

        # Save CSV for easy analysis
        self._save_csv()

        print(f"\n{'='*70}")
        print(f"  Training Complete")
        print(f"  Total epochs: {summary.total_epochs}")
        print(f"  Total steps:  {summary.total_steps}")
        print(f"  Best val loss: {summary.best_val_loss:.6f} (epoch {summary.best_epoch})")
        print(f"  Final eval success rate: {final_eval_sr:.1%}")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Dataset size: {dataset_size}")
        print(f"  Logs saved to: {self.log_dir}")
        print(f"{'='*70}\n")

        return summary

    def _save_csv(self):
        """Save epoch records as CSV for easy pandas loading."""
        csv_path = os.path.join(self.log_dir, "train_metrics.csv")
        if not self.epoch_records:
            return

        keys = ["epoch", "train_loss", "val_loss", "best_val_loss", "epoch_time"]
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for rec in self.epoch_records:
                vals = [str(getattr(rec, k, "")) for k in keys]
                f.write(",".join(vals) + "\n")
