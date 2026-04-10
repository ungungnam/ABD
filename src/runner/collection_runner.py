"""Main ABD data collection runner.

Implements the full system loop from Section 11 of the spec:
  observation -> generate trajectory -> execute -> validate -> ABD -> control action
"""

import time
import uuid
import logging

from omegaconf import DictConfig, OmegaConf

from env.base_env import ABDBaseEnv
from task.task_family import TaskScheduler
from task.task_registry import build_task_pair_from_config
from trajectory_generator.papa_generator import PaPATrajectoryGenerator
from executor.trajectory_executor import TrajectoryExecutor
from validator.base_validator import ValidationResult
from validator.geometric_validator import GeometricValidator
from validator.vlm_validator import VLMValidator, CombinedValidator
from vlm_client.vqa_client import VQAClient
from vlm_client.factory import build_vlm_backend
from abd.feature_extractor import ABDFeatureExtractor
from abd.risk_scorer import RiskScorer
from abd.abd_module import ABDModule
from policy.base_policy import BaseResetPolicy
from policy.no_reset_policy import NoResetPolicy
from policy.periodic_policy import PeriodicPolicy
from policy.naive_policy import NaivePolicy
from policy.abd_policy import ABDPolicy
from policy.vlm_checklist_policy import VLMChecklistPolicy
from metrics.metrics_logger import MetricsLogger, EpisodeRecord, InterventionRecord
from metrics.failure_classifier import FailureClassifier
from metrics.dataset_manifest import DatasetManifest
from runner.human_reset_interface import HumanResetInterface
from dataset.dataset_recorder import DatasetRecorder

log = logging.getLogger(__name__)


def build_env(config: DictConfig) -> ABDBaseEnv:
    """Build the appropriate environment from config."""
    if config.env.name == "real":
        from env.real_env import ABDRealEnv
        return ABDRealEnv(config.env)
    elif config.env.name == "rlbench":
        from env.rlbench_env import ABDRLBenchEnv
        return ABDRLBenchEnv(config.env)
    elif config.env.name == "dummy":
        from env.dummy_env import DummyEnv
        return DummyEnv(config.env, seed=getattr(config, "seed", 42))
    else:
        raise ValueError(f"Unknown environment: {config.env.name}")


def build_policy(
    config: DictConfig,
    risk_scorer: RiskScorer = None,
    vqa_client: VQAClient = None,
) -> BaseResetPolicy:
    """Build the reset policy from config."""
    method = config.policy.method

    if method == "NoReset":
        return NoResetPolicy()
    elif method == "Periodic":
        return PeriodicPolicy(period=config.policy.period)
    elif method == "Naive":
        return NaivePolicy()
    elif method == "ABD":
        return ABDPolicy(risk_scorer=risk_scorer)
    elif method == "VLMChecklist":
        if vqa_client is None:
            raise ValueError(
                "VLMChecklist policy requires a VQA client; this is not "
                "available in the dummy environment."
            )
        return VLMChecklistPolicy(
            vqa_client=vqa_client,
            checklist_dir=config.policy.checklist_dir,
            tau_reset=config.policy.tau_reset,
        )
    else:
        raise ValueError(f"Unknown policy method: {method}")


class CollectionRunner:
    """Main data collection loop with ABD.

    Orchestrates: trajectory generation -> execution -> validation -> ABD -> control.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.max_episodes = config.max_episodes
        self.max_retries = config.max_retries
        self.is_dummy = config.env.name == "dummy"

        # Environment
        self.env = build_env(config)

        # Task scheduling
        task_pair = build_task_pair_from_config(config.task)
        self.task_scheduler = TaskScheduler(task_pair)

        # Dataset recorder
        self.dataset_recorder = None
        if not self.is_dummy and hasattr(config, "dataset_recorder"):
            try:
                self.dataset_recorder = DatasetRecorder(config.dataset_recorder)
            except Exception as e:
                log.warning(f"Dataset recorder init failed: {e}. Recording disabled.")

        # Executor
        self.executor = TrajectoryExecutor(self.env, self.dataset_recorder)

        # Risk scorer (shared by ABD module and ABD policy)
        risk_scorer = RiskScorer(
            weights=list(config.policy.weights) if hasattr(config.policy, "weights") else [0.25, 0.15, 0.15, 0.10, 0.15, 0.20],
            tau_retry=getattr(config.policy, "tau_retry", 0.3),
            tau_reset=getattr(config.policy, "tau_reset", 0.7),
        )

        if self.is_dummy:
            self._init_dummy(config, risk_scorer)
        else:
            self._init_real(config, risk_scorer)

        # ABD module
        self.abd_module = ABDModule(self.feature_extractor, risk_scorer)

        # Policy
        self.policy = build_policy(config, risk_scorer, vqa_client=self.vqa_client)

        # checklist_observer is only available in real mode
        if self.is_dummy:
            self.checklist_observer = None

        # Run identity
        self.run_id = str(uuid.uuid4())[:8]
        self.policy_method = config.policy.method

        # Metrics
        self.metrics = MetricsLogger(
            config.log_dir, run_id=self.run_id, policy_method=self.policy_method,
        )

        # Dataset manifest
        self.manifest = DatasetManifest(config.log_dir)

        # Human interface
        self.human_interface = HumanResetInterface()

    def _init_real(self, config, risk_scorer):
        """Initialize real-hardware components (PaPA pipeline)."""
        # Build the VLM backend once and share it across the planner,
        # the VQA client, the validator, and the policy.
        self.vlm_backend = build_vlm_backend(config)
        log.info(f"[CollectionRunner] VLM backend: {self.vlm_backend.name}")

        self.generator = PaPATrajectoryGenerator(config, self.env, self.vlm_backend)

        self.vqa_client = VQAClient(backend=self.vlm_backend)
        geometric = GeometricValidator(
            perception_agent=self.generator.get_perception_agent(),
            motion_planner=self.generator.get_motion_planner(),
            env=self.env,
            config=config,
        )
        vlm_val = VLMValidator(self.vqa_client)
        self.validator = CombinedValidator(geometric, vlm_val)

        # VLMChecklistPolicy observer: always runs alongside the main policy
        # for logging/comparison, regardless of which policy is configured.
        self.checklist_observer = VLMChecklistPolicy(
            vqa_client=self.vqa_client,
            checklist_dir=config.policy.get("checklist_dir", "config/checklists"),
            tau_reset=config.policy.get("tau_reset", 0.9),
        )

        self.feature_extractor = ABDFeatureExtractor(
            config=config,
            perception_agent=self.generator.get_perception_agent(),
            motion_planner=self.generator.get_motion_planner(),
            env=self.env,
            vqa_client=self.vqa_client,
        )

    def _init_dummy(self, config, risk_scorer):
        """Initialize dummy components for testing without hardware."""
        from trajectory_generator.dummy_generator import DummyTrajectoryGenerator
        from validator.dummy_validator import DummyValidator
        from abd.dummy_feature_extractor import DummyFeatureExtractor

        self.vqa_client = None
        seed = getattr(config, "seed", 42)
        self.generator = DummyTrajectoryGenerator(
            failure_rate=0.1, num_waypoints=15, seed=seed,
        )
        self.validator = DummyValidator(base_success_rate=0.7, seed=seed + 1)
        self.feature_extractor = DummyFeatureExtractor(
            max_fail_count=getattr(config.policy, "max_fail_count", 5),
            seed=seed + 2,
        )

    def run(self):
        """Execute the full data collection loop."""
        self.metrics.start_run()

        # Save run config snapshot
        try:
            config_dict = OmegaConf.to_container(self.config, resolve=True)
        except Exception:
            config_dict = {"raw": str(self.config)}
        self.metrics.save_run_config(config_dict)

        # Calibrate ABD canonical state
        self.abd_module.calibrate(self.task_scheduler.current_task())

        if not self.is_dummy:
            input("\n[Calibration complete] Press Enter to start data collection...")

        fail_count = 0

        for ep in range(self.max_episodes):
            task = self.task_scheduler.current_task()
            episode_id = str(uuid.uuid4())[:8]
            task_direction = "forward" if self.task_scheduler.is_forward else "reverse"

            log.info(f"Episode {ep+1}/{self.max_episodes} | Task: {task.name} | "
                     f"Direction: {task_direction}")
            print(f"\n--- Episode {ep+1}/{self.max_episodes}: {task.name} ---")

            # Move robot to init pose
            self.env.go_to_init_pose()

            # Module A: Generate trajectory
            gen_result = self.generator.generate(task, self.env)

            if gen_result.trajectory is None:
                print(f"  Trajectory generation failed: {gen_result.metadata.get('reason', 'unknown')}")
                # Log the generation-failure episode before reset
                failure_type = FailureClassifier.classify(
                    validation=None, features=None,
                    fail_count=fail_count, max_retries=self.max_retries,
                    generation_success=False,
                )
                reset_request_time = time.time()
                should_continue, reset_confirm_time = self._handle_reset(task)
                intervention_duration = reset_confirm_time - reset_request_time

                record = EpisodeRecord(
                    episode_idx=ep,
                    episode_id=episode_id,
                    run_id=self.run_id,
                    policy_method=self.policy_method,
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
                self.metrics.log_episode(record)
                self.metrics.log_intervention(InterventionRecord(
                    timestamp=reset_request_time,
                    run_id=self.run_id,
                    episode_idx=ep,
                    episode_id=episode_id,
                    intervention_type="generation_failure_reset",
                    trigger="trajectory_generation_failed",
                ))
                self.manifest.log_episode(
                    episode_idx=ep, episode_id=episode_id, run_id=self.run_id,
                    task_name=task.name, success=False,
                    policy_method=self.policy_method, failure_type=failure_type,
                )
                fail_count = 0
                if not should_continue:
                    break
                continue

            # Module B: Execute trajectory
            exec_result = self.executor.execute(
                gen_result.trajectory, gen_result.events, task.language_task
            )

            # Module C: Validate task success
            validation = self.validator.validate(task, exec_result.final_obs, self.env)
            print(f"  Validation: success={validation.success}, "
                  f"method={validation.method}, confidence={validation.confidence:.2f}")

            # Module D: ABD features + policy decision
            features = self.feature_extractor.extract(task, validation, fail_count)
            decision = self.policy.decide(
                validation, fail_count, ep, features,
                task=task, observation=exec_result.final_obs,
            )
            policy_details = getattr(self.policy, "last_eval", None)
            risk_score = self.abd_module.risk_scorer.compute_risk(features["vector"])
            print(f"  ABD: risk={risk_score:.3f}, decision={decision}")

            # VLMChecklistPolicy observer — always runs; overrides decision to "reset" if needed
            if self.checklist_observer is not None:
                checklist_decision = self.checklist_observer.decide(
                    validation, fail_count, ep,
                    task=task, observation=exec_result.final_obs,
                )
                checklist_eval = self.checklist_observer.last_eval
                if checklist_eval:
                    score = checklist_eval.get("score", 0.0)
                    items = checklist_eval.get("items", [])
                    log.info(
                        f"[ChecklistObserver] decision={checklist_decision}, score={score:.3f}"
                    )
                    for item in items:
                        log.info(
                            f"  [{item['answer'].upper()}] (w={item['weight']}) {item['question']}"
                        )
                print(f"  Checklist: score={checklist_eval['score']:.3f}, decision={checklist_decision}" if checklist_eval else "  Checklist: eval unavailable")

                if checklist_decision == "reset" and decision != "reset":
                    log.info(f"[ChecklistObserver] Overriding decision '{decision}' → 'reset'")
                    decision = "reset"
                elif checklist_decision == "retry" and decision == "next":
                    log.info(f"[ChecklistObserver] Overriding decision 'next' → 'retry'")
                    decision = "retry"

            # Classify failure type (only for failed episodes)
            failure_type = ""
            if not validation.success:
                failure_type = FailureClassifier.classify(
                    validation=validation, features=features,
                    fail_count=fail_count, max_retries=self.max_retries,
                )

            # Store episode data + manifest
            dataset_episode_idx = None
            if validation.success and self.dataset_recorder is not None:
                self.dataset_recorder.save_episode()
                dataset_episode_idx = self.manifest.log_episode(
                    episode_idx=ep, episode_id=episode_id, run_id=self.run_id,
                    task_name=task.name, success=True,
                    policy_method=self.policy_method,
                )
            else:
                if self.dataset_recorder is not None:
                    self.dataset_recorder.clear_episode_buffer()
                self.manifest.log_episode(
                    episode_idx=ep, episode_id=episode_id, run_id=self.run_id,
                    task_name=task.name, success=False,
                    policy_method=self.policy_method, failure_type=failure_type,
                )

            # Build episode record
            record = EpisodeRecord(
                episode_idx=ep,
                episode_id=episode_id,
                run_id=self.run_id,
                policy_method=self.policy_method,
                task_name=task.name,
                task_direction=task_direction,
                timestamp=time.time(),
                success=validation.success,
                policy_decision=decision,
                human_reset=False,
                abd_features=features,
                risk_score=risk_score,
                generation_time=gen_result.metadata.get("elapsed_time", 0.0),
                execution_time=exec_result.elapsed_time,
                validation_method=validation.method,
                validation_details=validation.details,
                generation_success=True,
                fail_count=fail_count,
                failure_type=failure_type,
                dataset_episode_idx=dataset_episode_idx,
                policy_details=policy_details,
            )

            # Act on decision
            if decision == "next":
                self.task_scheduler.advance()
                fail_count = 0

            elif decision == "retry":
                fail_count += 1
                if fail_count >= self.max_retries:
                    print(f"  Max retries ({self.max_retries}) reached, escalating to reset.")
                    decision = "reset"
                    record.failure_type = "retry_limit"

            if decision == "reset":
                record.human_reset = True
                reset_request_time = time.time()
                should_continue, reset_confirm_time = self._handle_reset(task)
                record.reset_request_time = reset_request_time
                record.reset_confirm_time = reset_confirm_time
                record.intervention_duration = reset_confirm_time - reset_request_time
                self.metrics.log_intervention(InterventionRecord(
                    timestamp=reset_request_time,
                    run_id=self.run_id,
                    episode_idx=ep,
                    episode_id=episode_id,
                    intervention_type="policy_reset",
                    trigger=f"policy={self.policy_method}, decision={decision}",
                ))
                fail_count = 0
                if not should_continue:
                    self.metrics.log_episode(record)
                    break

            self.metrics.log_episode(record)

        # Finalize
        self.metrics.save()
        self.manifest.save()
        if self.dataset_recorder is not None:
            self.dataset_recorder.finalize()

        print(f"\n{'='*60}")
        print(f"Collection complete: {self.metrics.cumulative_valid_trajectories()} "
              f"valid trajectories in {len(self.metrics.episodes)} episodes")
        print(f"Success rate: {self.metrics.cumulative_success_rate():.1%}")
        print(f"Human resets: {sum(1 for e in self.metrics.episodes if e.human_reset)}")
        print(f"Failure types: {self.metrics.failure_type_distribution()}")
        print(f"Logs saved to: {self.config.log_dir}")
        print(f"{'='*60}")

    def _handle_reset(self, task):
        """Request human reset and recalibrate.

        Returns:
            (should_continue, confirm_time): should_continue is False if
            user aborts; confirm_time is the timestamp when reset was confirmed.
        """
        if self.is_dummy:
            print("  [Dummy] Auto-confirming human reset.")
            should_continue = True
            confirm_time = time.time()
        else:
            should_continue, confirm_time = self.human_interface.request_reset(task)

        if should_continue:
            self.task_scheduler.reset_to_forward()
            self.abd_module.calibrate(self.task_scheduler.current_task())
            # Reset dummy validator state if applicable
            if hasattr(self.validator, "reset_state"):
                self.validator.reset_state()
        return should_continue, confirm_time
