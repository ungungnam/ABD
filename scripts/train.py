"""BC Training — Hydra entry point.

Trains a behavioral cloning policy on ABD-collected data.

Usage:
    # Dummy data (test pipeline)
    python scripts/train.py train=bc train.dummy=true

    # Real collected data
    python scripts/train.py train=bc train.dummy=false \
        train.dataset_root=outputs/dataset/abd_collection

    # Scaling experiment (25% of data)
    python scripts/train.py train=bc train.dataset_fraction=0.25

    # Compare policies: train on data from each reset policy
    python scripts/train.py train=bc \
        train.dataset_root=outputs/dataset/abd_no_reset \
        policy.method=NoReset
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401, E402

import hydra
from omegaconf import DictConfig, OmegaConf

from train.bc_trainer import BCTrainer


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train_cfg = cfg.train

    log_dir = os.path.join(
        cfg.log_dir if hasattr(cfg, "log_dir") else "outputs/train",
        f"bc_{getattr(cfg.policy, 'method', 'unknown')}",
    )

    trainer = BCTrainer(
        # Data
        dataset_root=train_cfg.get("dataset_root", None),
        dummy=train_cfg.get("dummy", True),
        dummy_episodes=train_cfg.get("dummy_episodes", 20),
        dummy_episode_length=train_cfg.get("dummy_episode_length", 30),
        dataset_fraction=train_cfg.get("dataset_fraction", 1.0),
        # Model
        num_cameras=train_cfg.get("num_cameras", 3),
        camera_names=list(train_cfg.get("camera_names", ["wrist", "front", "table"])),
        image_size=tuple(train_cfg.get("image_size", [128, 128])),
        image_feature_dim=train_cfg.get("image_feature_dim", 128),
        state_dim=train_cfg.get("state_dim", 7),
        action_dim=train_cfg.get("action_dim", 7),
        hidden_dim=train_cfg.get("hidden_dim", 256),
        share_encoder=train_cfg.get("share_encoder", True),
        # Training
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        batch_size=train_cfg.get("batch_size", 32),
        num_epochs=train_cfg.get("num_epochs", 50),
        val_fraction=train_cfg.get("val_fraction", 0.2),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        eval_every=train_cfg.get("eval_every", 10),
        # Logging
        log_dir=log_dir,
        experiment_name=f"bc_{cfg.policy.method}",
        # Context
        policy_method=cfg.policy.method,
        seed=train_cfg.get("seed", 42),
    )

    summary = trainer.train()

    # Print final result for easy parsing
    print(f"\nRESULT: policy={cfg.policy.method} "
          f"dataset_size={summary['dataset_size']} "
          f"best_val_loss={summary['best_val_loss']:.6f} "
          f"success_rate={summary['final_success_rate']:.4f}")


if __name__ == "__main__":
    main()
