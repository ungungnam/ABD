"""Main ABD data collection runner.

Implements the full system loop from Section 11 of the spec:
  observation -> generate trajectory -> execute -> validate -> ABD -> control action
"""

import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from omegaconf import DictConfig, OmegaConf

from env.base_env import ABDBaseEnv
from task.task_family import TaskScheduler
from task.task_registry import build_task_pair_from_config
from trajectory_generator.papa_generator import PaPATrajectoryGenerator
from executor.trajectory_executor import TrajectoryExecutor
from validator.base_validator import ValidationResult
from validator.vlm_validator import VLMValidator
from vlm_client.vqa_client import VQAClient
from vlm_client.factory import build_vlm_backend
from policy.base_policy import BaseResetPolicy
from policy.no_reset_policy import NoResetPolicy
from policy.periodic_policy import PeriodicPolicy
from policy.naive_policy import NaivePolicy
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

        # Run identity (must be set before DatasetRecorder)
        self.run_id = self._make_run_id(config)
        self.policy_method = config.policy.name

        # Environment
        self.env = build_env(config)

        # Task scheduling
        task_pair = build_task_pair_from_config(config.task)
        self.task_scheduler = TaskScheduler(task_pair)

        # Dataset recorder
        self.dataset_recorder = None
        if not self.is_dummy and hasattr(config, "dataset_recorder"):
            try:
                self.dataset_recorder = DatasetRecorder(config.dataset_recorder, run_id=self.run_id)
            except Exception as e:
                log.warning(f"Dataset recorder init failed: {e}. Recording disabled.")

        # Executor
        self.executor = TrajectoryExecutor(self.env, self.dataset_recorder)

        if self.is_dummy:
            self._init_dummy(config)
        else:
            self._init_real(config)

        # Policy
        self.policy = build_policy(config, vqa_client=self.vqa_client)

        # Metrics
        self.metrics = MetricsLogger(
            config.log_dir, run_id=self.run_id, policy_method=self.policy_method,
        )

        # Dataset manifest
        self.manifest = DatasetManifest(config.log_dir)

        # Human interface
        self.human_interface = HumanResetInterface()

    def _init_real(self, config):
        """Initialize real-hardware components (PaPA pipeline)."""
        self.vlm_backend = build_vlm_backend(config)
        log.info(f"[CollectionRunner] VLM backend: {self.vlm_backend.name}")

        self.generator = PaPATrajectoryGenerator(config, self.env, self.vlm_backend)

        self.vqa_client = VQAClient(backend=self.vlm_backend)
        self.validator = VLMValidator(self.vqa_client)


    def _init_dummy(self, config):
        """Initialize dummy components for testing without hardware."""
        from trajectory_generator.dummy_generator import DummyTrajectoryGenerator
        from validator.dummy_validator import DummyValidator

        self.vqa_client = None
        seed = getattr(config, "seed", 42)
        self.generator = DummyTrajectoryGenerator(
            failure_rate=0.1, num_waypoints=15, seed=seed,
        )
        self.validator = DummyValidator(base_success_rate=0.7, seed=seed + 1)

    def run(self):
        """Execute the full data collection loop."""
        self.metrics.start_run()

        # Save run config snapshot
        try:
            config_dict = OmegaConf.to_container(self.config, resolve=True)
        except Exception:
            config_dict = {"raw": str(self.config)}
        self.metrics.save_run_config(config_dict)

        if not self.is_dummy:
            input("\n[Calibration complete] Press Enter to start data collection...")

        fail_count = 0

        for ep in range(self.max_episodes):
            task = self.task_scheduler.current_task()
            episode_id = str(uuid.uuid4())[:8]
            task_direction = "forward" if self.task_scheduler.is_forward else "reverse"
            episode_start_time = time.time()

            log.info(f"Episode {ep+1}/{self.max_episodes} | Task: {task.name} | "
                     f"Direction: {task_direction}")

            try:
                # Move robot to init pose
                self.env.go_to_init_pose()

                # Module A: Generate trajectory
                gen_result = self.generator.generate(task, self.env)

                if gen_result.trajectory is None:
                    log.info(f"Trajectory generation failed: {gen_result.metadata.get('reason', 'unknown')}")
                    failure_type = FailureClassifier.classify(
                        validation=None, features=None,
                        fail_count=fail_count, max_retries=self.max_retries,
                        generation_success=False,
                    )
                    reset_decided_at = time.time()
                    should_continue, reset_confirm_time = self._handle_reset(task)

                    record = EpisodeRecord(
                        episode_idx=ep,
                        episode_id=episode_id,
                        run_id=self.run_id,
                        policy_method=self.policy_method,
                        task_name=task.name,
                        task_direction=task_direction,
                        success=False,
                        policy_decision="reset",
                        human_reset=True,
                        generation_success=False,
                        generation_time=gen_result.metadata.get("elapsed_time", 0.0),
                        fail_count=fail_count,
                        failure_type=failure_type,
                        episode_duration=reset_confirm_time - episode_start_time,
                        reset_prompt_to_confirm=reset_confirm_time - reset_decided_at,
                    )
                    self.metrics.log_episode(record)
                    self.metrics.log_intervention(InterventionRecord(
                        timestamp=reset_decided_at,
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
                log.info(f"Validation: success={validation.success}, "
                        f"method={validation.method}, confidence={validation.confidence:.2f}")

                # Module D: Policy decision (2×2 matrix)
                #   success  reset  → action
                #   True     False  → next   (advance direction)
                #   True     True   → reset  + advance
                #   False    False  → retry
                #   False    True   → reset  + same direction
                #
                # 체크리스트 평가 대상:
                #   success=True  → 다음 task (반대 방향) 기준으로 환경 준비 여부 판단
                #   success=False → 현재 task (같은 방향) 기준으로 재시도 가능 여부 판단
                eval_task = self.task_scheduler.next_task() if validation.success else task
                reset_needed = self.policy.needs_reset(
                    validation, fail_count, ep,
                    task=eval_task, observation=exec_result.final_obs,
                )
                checklist_eval = getattr(self.policy, "last_eval", None)

                success = validation.success

                if success and not reset_needed:
                    decision = "next"
                elif success and reset_needed:
                    decision = "reset"
                elif not success and not reset_needed:
                    decision = "retry"
                else:  # not success and reset_needed
                    decision = "reset"

                score = checklist_eval.get("score") if checklist_eval else None
                score_str = f"{score:.3f}" if score is not None else "n/a"
                print(f"  Policy: success={success}, score={score_str}, tau={self.policy.tau_reset if hasattr(self.policy, 'tau_reset') else '?'}, needs_reset={reset_needed} → {decision}")

                # Classify failure type
                failure_type = ""
                if not success:
                    failure_type = FailureClassifier.classify(
                        validation=validation, features=None,
                        fail_count=fail_count, max_retries=self.max_retries,
                    )

                # Manifest
                dataset_episode_idx = None
                if success:
                    dataset_episode_idx = self.manifest.log_episode(
                        episode_idx=ep, episode_id=episode_id, run_id=self.run_id,
                        task_name=task.name, success=True,
                        policy_method=self.policy_method,
                    )
                else:
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
                    success=success,
                    policy_decision=decision,
                    human_reset=False,
                    generation_time=gen_result.metadata.get("elapsed_time", 0.0),
                    execution_time=exec_result.elapsed_time,
                    validation_method=validation.method,
                    validation_details=validation.details,
                    generation_success=True,
                    fail_count=fail_count,
                    failure_type=failure_type,
                    dataset_episode_idx=dataset_episode_idx,
                    checklist_eval=checklist_eval,
                    episode_duration=time.time() - episode_start_time,
                )

                # Act on decision
                if decision == "next":
                    self.task_scheduler.advance()
                    fail_count = 0

                elif decision == "retry":
                    fail_count += 1
                    if fail_count >= self.max_retries:
                        log.warning(f"Max retries ({self.max_retries}) reached, escalating to reset.")
                        decision = "reset"
                        record.policy_decision = "reset"
                        record.failure_type = "retry_limit"

                if decision == "reset":
                    record.human_reset = True
                    reset_decided_at = time.time()
                    should_continue, reset_confirm_time = self._handle_reset(task)
                    record.episode_duration = reset_confirm_time - episode_start_time
                    record.reset_prompt_to_confirm = reset_confirm_time - reset_decided_at
                    self.metrics.log_intervention(InterventionRecord(
                        timestamp=reset_decided_at,
                        run_id=self.run_id,
                        episode_idx=ep,
                        episode_id=episode_id,
                        intervention_type="policy_reset",
                        trigger=f"policy={self.policy_method}",
                    ))
                    fail_count = 0
                    if not should_continue:
                        if not self.is_dummy:
                            record.ground_truth_reset = self.human_interface.request_ground_truth_label()
                        self.metrics.log_episode(record)
                        if self.dataset_recorder is not None:
                            self.dataset_recorder.save_episode(
                                task_name=task.name, success=success,
                                record=asdict(record),
                            )
                        break

                # Ground truth label (after reset if any, before saving)
                if not self.is_dummy:
                    record.ground_truth_reset = self.human_interface.request_ground_truth_label()

                # Save episode data (after record is fully populated incl. reset times)
                if self.dataset_recorder is not None:
                    self.dataset_recorder.save_episode(
                        task_name=task.name, success=validation.success,
                        record=asdict(record),
                    )

                self.metrics.log_episode(record)

            except KeyboardInterrupt:
                print("\n\n[비상 정지] Ctrl+C 입력됨. 현재 에피소드를 중단하고 run을 종료합니다.")
                log.warning(f"[EmergencyStop] Episode {ep} interrupted by user (Ctrl+C).")

                # 버퍼에 프레임이 있으면 버림 (중단된 에피소드는 저장하지 않음)
                if self.dataset_recorder is not None:
                    self.dataset_recorder.clear_episode_buffer()

                # 에피소드 실패로 기록 (human_reset=False: reset 아님)
                stop_time = time.time()
                self.manifest.log_episode(
                    episode_idx=ep, episode_id=episode_id, run_id=self.run_id,
                    task_name=task.name, success=False,
                    policy_method=self.policy_method, failure_type="emergency_stop",
                )
                self.metrics.log_episode(EpisodeRecord(
                    episode_idx=ep,
                    episode_id=episode_id,
                    run_id=self.run_id,
                    policy_method=self.policy_method,
                    task_name=task.name,
                    task_direction=task_direction,
                    success=False,
                    policy_decision="aborted",
                    human_reset=False,
                    failure_type="emergency_stop",
                    fail_count=fail_count,
                    episode_duration=stop_time - episode_start_time,
                ))
                break

        # Finalize — 정상 종료와 비상 정지 모두 여기서 저장
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

    def _make_run_id(self, config) -> str:
        """Generate a run ID of the form YYYYMMDD_N (e.g. 20260413_1).

        Scans the dataset root for existing run_YYYYMMDD_* directories and
        picks the next available counter for today's date.
        """
        today = datetime.now().strftime("%Y%m%d")
        try:
            root = Path(config.dataset_recorder.root)
            existing = [
                d.name for d in root.iterdir()
                if d.is_dir() and d.name.startswith(f"run_{today}_")
            ] if root.exists() else []
            counters = []
            for name in existing:
                suffix = name[len(f"run_{today}_"):]
                if suffix.isdigit():
                    counters.append(int(suffix))
            n = max(counters) + 1 if counters else 0
        except Exception:
            n = 1
        return f"{today}_{n}"

    def _handle_reset(self, task):
        """Request human reset and recalibrate.

        Reset 후에는 항상 forward로 돌아간다.

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
            if hasattr(self.validator, "reset_state"):
                self.validator.reset_state()
        return should_continue, confirm_time
