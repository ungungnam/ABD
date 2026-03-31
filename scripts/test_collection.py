"""Test the full data collection pipeline with dummy components.

Usage:
    python scripts/test_collection.py
    python scripts/test_collection.py --policy ABD
    python scripts/test_collection.py --all_policies
    python scripts/test_collection.py --episodes 30 --policy Periodic
"""

import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401, E402

from task.task_family import TaskScheduler
from task.task_registry import build_task_pair_from_config
from env.dummy_env import DummyEnv
from trajectory_generator.dummy_generator import DummyTrajectoryGenerator
from executor.trajectory_executor import TrajectoryExecutor
from validator.dummy_validator import DummyValidator
from abd.dummy_feature_extractor import DummyFeatureExtractor
from abd.risk_scorer import RiskScorer
from abd.abd_module import ABDModule
from policy.no_reset_policy import NoResetPolicy
from policy.periodic_policy import PeriodicPolicy
from policy.naive_policy import NaivePolicy
from policy.abd_policy import ABDPolicy
from metrics.metrics_logger import MetricsLogger, EpisodeRecord, InterventionRecord
from metrics.failure_classifier import FailureClassifier
from metrics.dataset_manifest import DatasetManifest


POLICIES = {
    "NoReset": lambda rs: NoResetPolicy(),
    "Periodic": lambda rs: PeriodicPolicy(period=10),
    "Naive": lambda rs: NaivePolicy(),
    "ABD": lambda rs: ABDPolicy(risk_scorer=rs),
}


class DummyTaskConfig:
    """Minimal config stub matching the Hydra task config shape."""
    object = "banana"
    location_a = "plate"
    location_b = "pan"
    forward_task = "pick the banana and put it in the pan"
    reverse_task = "pick the banana and put it in the plate"


def run_collection(policy_name: str, max_episodes: int = 20, max_retries: int = 3,
                   seed: int = 42, output_dir: str = "outputs/test_collection"):
    """Run a full collection loop with dummy components."""
    log_dir = os.path.join(output_dir, policy_name)
    os.makedirs(log_dir, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]

    # Build components
    task_pair = build_task_pair_from_config(DummyTaskConfig())
    scheduler = TaskScheduler(task_pair)
    env = DummyEnv(seed=seed)
    generator = DummyTrajectoryGenerator(failure_rate=0.1, seed=seed)
    executor = TrajectoryExecutor(env)
    validator = DummyValidator(base_success_rate=0.7, seed=seed + 1)
    feature_extractor = DummyFeatureExtractor(max_fail_count=5, seed=seed + 2)
    risk_scorer = RiskScorer(
        weights=[0.25, 0.15, 0.15, 0.10, 0.15, 0.20],
        tau_retry=0.3, tau_reset=0.7,
    )
    abd_module = ABDModule(feature_extractor, risk_scorer)
    policy = POLICIES[policy_name](risk_scorer)
    metrics = MetricsLogger(log_dir, run_id=run_id, policy_method=policy_name)
    manifest = DatasetManifest(log_dir)

    # Run
    metrics.start_run()
    metrics.save_run_config({
        "policy_method": policy_name,
        "max_episodes": max_episodes,
        "max_retries": max_retries,
        "seed": seed,
        "env": "dummy",
    })
    feature_extractor.calibrate(scheduler.current_task())
    fail_count = 0
    human_resets = 0

    print(f"\n{'='*70}")
    print(f"  Collection Test: policy={policy_name}, episodes={max_episodes}, run_id={run_id}")
    print(f"{'='*70}")

    for ep in range(max_episodes):
        task = scheduler.current_task()
        episode_id = str(uuid.uuid4())[:8]
        task_direction = "forward" if scheduler.is_forward else "reverse"
        direction = "FWD" if scheduler.is_forward else "REV"

        # Generate trajectory
        gen_result = generator.generate(task, env)

        if gen_result.trajectory is None:
            print(f"  Ep {ep+1:3d} | {task.name:20s} [{direction}] | GEN FAILED → auto-reset")
            # Log generation-failure episode
            failure_type = FailureClassifier.classify(
                validation=None, features=None,
                fail_count=fail_count, max_retries=max_retries,
                generation_success=False,
            )
            reset_request_time = time.time()
            reset_confirm_time = time.time()  # dummy: instant confirm
            intervention_duration = reset_confirm_time - reset_request_time

            record = EpisodeRecord(
                episode_idx=ep,
                episode_id=episode_id,
                run_id=run_id,
                policy_method=policy_name,
                task_name=task.name,
                task_direction=task_direction,
                timestamp=time.time(),
                success=False,
                policy_decision="reset",
                human_reset=True,
                generation_success=False,
                generation_time=gen_result.metadata.get("elapsed_time", 0.0),
                fail_count=fail_count,
                failure_type=failure_type,
                reset_request_time=reset_request_time,
                reset_confirm_time=reset_confirm_time,
                intervention_duration=intervention_duration,
            )
            metrics.log_episode(record)
            metrics.log_intervention(InterventionRecord(
                timestamp=reset_request_time,
                run_id=run_id,
                episode_idx=ep,
                episode_id=episode_id,
                intervention_type="generation_failure_reset",
                trigger="trajectory_generation_failed",
            ))
            manifest.log_episode(
                episode_idx=ep, episode_id=episode_id, run_id=run_id,
                task_name=task.name, success=False,
                policy_method=policy_name, failure_type=failure_type,
            )

            human_resets += 1
            scheduler.reset_to_forward()
            feature_extractor.calibrate(scheduler.current_task())
            validator.reset_state()
            fail_count = 0
            continue

        # Execute
        exec_result = executor.execute(gen_result.trajectory, gen_result.events, task.language_task)

        # Validate
        validation = validator.validate(task, exec_result.final_obs, env)

        # ABD features + policy decision
        features = feature_extractor.extract(task, validation, fail_count)
        risk = risk_scorer.compute_risk(features["vector"])
        decision = policy.decide(validation, fail_count, ep, features)

        # Classify failure
        failure_type = ""
        if not validation.success:
            failure_type = FailureClassifier.classify(
                validation=validation, features=features,
                fail_count=fail_count, max_retries=max_retries,
            )

        status = "OK" if validation.success else "FAIL"
        print(f"  Ep {ep+1:3d} | {task.name:20s} [{direction}] | {status:4s} | "
              f"risk={risk:.3f} | → {decision}"
              + (f" [{failure_type}]" if failure_type else ""))

        # Dataset manifest
        dataset_episode_idx = None
        if validation.success:
            dataset_episode_idx = manifest.log_episode(
                episode_idx=ep, episode_id=episode_id, run_id=run_id,
                task_name=task.name, success=True, policy_method=policy_name,
            )
        else:
            manifest.log_episode(
                episode_idx=ep, episode_id=episode_id, run_id=run_id,
                task_name=task.name, success=False,
                policy_method=policy_name, failure_type=failure_type,
            )

        # Build record
        record = EpisodeRecord(
            episode_idx=ep,
            episode_id=episode_id,
            run_id=run_id,
            policy_method=policy_name,
            task_name=task.name,
            task_direction=task_direction,
            timestamp=time.time(),
            success=validation.success,
            policy_decision=decision,
            human_reset=False,
            abd_features=features,
            risk_score=risk,
            generation_time=gen_result.metadata.get("elapsed_time", 0.0),
            execution_time=exec_result.elapsed_time,
            validation_method=validation.method,
            validation_details=validation.details,
            generation_success=True,
            fail_count=fail_count,
            failure_type=failure_type,
            dataset_episode_idx=dataset_episode_idx,
        )

        # Act on decision
        if decision == "next":
            if validation.success:
                scheduler.advance()
            fail_count = 0
        elif decision == "retry":
            fail_count += 1
            if fail_count >= max_retries:
                print(f"         Max retries → escalate to reset")
                decision = "reset"
                record.policy_decision = "reset"
                record.human_reset = True
                record.failure_type = "retry_limit"

        if decision == "reset":
            record.human_reset = True
            reset_request_time = time.time()
            reset_confirm_time = time.time()  # dummy: instant confirm
            record.reset_request_time = reset_request_time
            record.reset_confirm_time = reset_confirm_time
            record.intervention_duration = reset_confirm_time - reset_request_time
            metrics.log_intervention(InterventionRecord(
                timestamp=reset_request_time,
                run_id=run_id,
                episode_idx=ep,
                episode_id=episode_id,
                intervention_type="policy_reset",
                trigger=f"policy={policy_name}, decision={decision}",
            ))
            human_resets += 1
            scheduler.reset_to_forward()
            feature_extractor.calibrate(scheduler.current_task())
            validator.reset_state()
            fail_count = 0

        metrics.log_episode(record)

    # Save & summarize
    metrics.save()
    manifest.save()

    sr = metrics.cumulative_success_rate()
    valid = metrics.cumulative_valid_trajectories()
    print(f"\n  --- {policy_name} Summary ---")
    print(f"  Run ID:         {run_id}")
    print(f"  Episodes:       {len(metrics.episodes)}")
    print(f"  Valid trajs:    {valid}")
    print(f"  Success rate:   {sr:.1%}")
    print(f"  Human resets:   {human_resets}")
    print(f"  Failure types:  {metrics.failure_type_distribution()}")
    print(f"  Logs saved to:  {log_dir}")

    return {
        "policy": policy_name,
        "run_id": run_id,
        "episodes": len(metrics.episodes),
        "valid_trajectories": valid,
        "success_rate": sr,
        "human_resets": human_resets,
        "failure_types": metrics.failure_type_distribution(),
    }


def main():
    parser = argparse.ArgumentParser(description="Test ABD collection pipeline")
    parser.add_argument("--policy", type=str, default="ABD", choices=list(POLICIES.keys()))
    parser.add_argument("--all_policies", action="store_true", help="Run all 4 policies")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/test_collection")
    args = parser.parse_args()

    policies_to_run = list(POLICIES.keys()) if args.all_policies else [args.policy]
    all_results = []

    for p in policies_to_run:
        result = run_collection(
            policy_name=p,
            max_episodes=args.episodes,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  Comparison Summary")
        print(f"{'='*70}")
        print(f"  {'Policy':>12s} | {'Episodes':>8s} | {'Valid':>5s} | {'Success':>8s} | {'Resets':>6s}")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}")
        for r in all_results:
            print(f"  {r['policy']:>12s} | {r['episodes']:>8d} | {r['valid_trajectories']:>5d} | "
                  f"{r['success_rate']:>8.1%} | {r['human_resets']:>6d}")

        summary_path = os.path.join(args.output_dir, "comparison.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Comparison saved to: {summary_path}")


if __name__ == "__main__":
    main()
