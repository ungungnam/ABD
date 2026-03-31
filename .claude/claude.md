# ABD — Autonomy Boundary Detection

## Project Overview

ABD is a continuous robotic data collection system that autonomously executes pick-place tasks, detects when autonomous execution should stop (via risk-based ABD), and requests human reset only when needed. Builds on the PaPA framework for VLM planning, perception, and motion planning.

## Environment Setup

- **Conda env**: `sushi_papa` — always use `conda run -n sushi_papa python ...` for all Python/pip commands
- **Python path**: `src/bootstrap.py` adds ABD `src/` and PaPA `src/` to sys.path
- **Config**: Hydra-based, root at `config/config.yaml`

## Key Commands

```bash
# Data collection (dummy mode, no hardware)
conda run -n sushi_papa python scripts/test_collection.py --all_policies --episodes 30
conda run -n sushi_papa python scripts/main.py env=dummy policy=abd max_episodes=20

# Data collection (real robot)
conda run -n sushi_papa python scripts/main.py env=real policy=abd max_episodes=100

# BC training (dummy data)
conda run -n sushi_papa python scripts/train.py --dummy --num_epochs 15

# Scaling experiment
conda run -n sushi_papa python scripts/train_scaling.py --dummy --num_epochs 10
```

## Architecture

```
src/
  bootstrap.py              # sys.path setup (ABD + PaPA)
  task/                     # TaskDefinition, ReversibleTaskPair, TaskScheduler
  env/                      # ABDBaseEnv + real/rlbench/dummy implementations
  trajectory_generator/     # PaPA pipeline wrapper (VLM -> Perception -> Motion)
  executor/                 # Trajectory execution + LeRobot frame recording
  validator/                # Geometric (3D proximity) + VLM VQA fallback
  abd/                      # Feature extractor (6D vector) + risk scorer
  policy/                   # 4 reset policies: NoReset, Periodic, Naive, ABD
  metrics/                  # Per-episode logging + post-hoc analysis
  runner/                   # CollectionRunner (main loop) + HumanResetInterface
  train/                    # BC policy, dataset loader, trainer, logger
  vlm_client/               # VQA client for VLM server
  vlm/, perception/, motion_planner/, robot/, camera/, dataset/, utils/
                            # Copied from PaPA (some adapted, some as-is)
```

## File Modification Rules

- Freely modify anything under `/home/uhnam/workspaces/ABD/`
- **Never modify PaPA** (`/home/uhnam/workspaces/PaPA/`) — copy into ABD's `src/` and modify locally
- PaPA code can be imported read-only via bootstrap.py sys.path

## Config System (Hydra)

Defaults in `config/config.yaml`:
- `env`: `real` | `rlbench` | `dummy`
- `policy`: `abd` | `no_reset` | `periodic` | `naive`
- `task`: `pick_place`
- `train`: `bc`

Override via CLI: `python scripts/main.py env=dummy policy=naive max_episodes=50`

## ABD Core Concepts

- **6D feature vector**: `[f_succ, f_vis, f_reach, f_rec, f_dev, f_fail]` — all in [0, 1]
- **Risk score**: weighted sum with inverted "good" features: `r = w^T * [1-f_succ, 1-f_vis, 1-f_reach, 1-f_rec, f_dev, f_fail]`
- **Two thresholds**: `r < tau_retry` -> next, `tau_retry <= r < tau_reset` -> retry, `r >= tau_reset` -> reset
- **Reversible task pairs**: forward (A->B) / reverse (B->A) alternation to maintain canonical state

## Output Locations

- Collection logs: `outputs/test_collection/<policy>/episodes.jsonl`, `summary.json`
- Training logs: `outputs/train/*/step_log.jsonl`, `epoch_log.jsonl`, `train_summary.json`
- Scaling results: `outputs/scaling/*/scaling_results.json`, `scaling_curve.png`
- Checkpoints: `outputs/train/*/checkpoints/best.pt`, `last.pt`
