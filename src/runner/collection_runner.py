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
from task.task_registry import build_task_pair_from_config, build_stack_cups_task_from_config, build_open_drawer_task_from_config
from trajectory_generator.papa_generator import PaPATrajectoryGenerator
from trajectory_generator.base_generator import GenerationResult
from executor.trajectory_executor import TrajectoryExecutor, ExecutionResult
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
        if config.task.family == "stack_cups":
            task_pair = build_stack_cups_task_from_config(config.task)
        elif config.task.family == "open_drawer":
            task_pair = build_open_drawer_task_from_config(config.task)
        else:
            task_pair = build_task_pair_from_config(config.task)
        self.task_scheduler = TaskScheduler(task_pair)

        # Dataset recorder
        self.dataset_recorder = None
        if self.is_dummy:
            log.info("[DatasetRecorder] Disabled (dummy env).")
        elif not hasattr(config, "dataset_recorder"):
            log.warning("[DatasetRecorder] 'dataset_recorder' key missing from config — recording disabled.")
        else:
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

        # stack_cups only: choose starting phase (single keypress, no Enter needed)
        # if self.config.task.family == "stack_cups":
        #     import sys, tty, termios
        #     print("\n[DEBUG] 1: stack (forward) / 2: unstack (reverse) ", end="", flush=True)
        #     while True:
        #         fd = sys.stdin.fileno()
        #         old = termios.tcgetattr(fd)
        #         try:
        #             tty.setraw(fd)
        #             mode = sys.stdin.read(1)
        #         finally:
        #             termios.tcsetattr(fd, termios.TCSADRAIN, old)
        #         if mode in ("1", "2"):
        #             print(mode)
        #             break

        #     if mode == "2":
        #         self.task_scheduler._idx = self.task_scheduler._n_forward

        fail_count = 0
        ep = 0
        episode_id = str(uuid.uuid4())[:8]
        episode_start_time = time.time()

        while ep < self.max_episodes:
            task = self.task_scheduler.current_task()

            # stack_cups: forward_1+forward_2 share one episode_id;
            # reverse_1+reverse_2 share another.  A new episode starts only
            # at the first step of each phase (idx==0 or idx==n_forward).
            # All other task types treat every step as its own episode.
            is_phase_start = (
                task.task_type != "stack_cups"
                or self.task_scheduler._idx == 0
                or self.task_scheduler._idx == self.task_scheduler._n_forward
            )
            if is_phase_start:
                episode_id = str(uuid.uuid4())[:8]
                episode_start_time = time.time()

            task_direction = "forward" if self.task_scheduler.is_forward else "reverse"

            log.info(f"Episode {ep}/{self.max_episodes} | Step: {task.name} | "
                     f"Direction: {task_direction}")

            try:
                # Move robot to init pose
                self.env.go_to_init_pose()


                # skip_to_validation = (
                #     not self.is_dummy
                #     and task.intermediate_check_tag_id is not None
                #     and self._check_intermediate_tag(task)
                # )

                # if skip_to_validation:
                #     log.info(
                #         f"[IntermediateCheck] tag {task.intermediate_check_tag_id} check failed — "
                #         f"skipping {task.name} execution, going to validation"
                #     )
                #     gen_result  = GenerationResult(trajectory=[], events=[],
                #                                    metadata={"elapsed_time": 0.0}, key_poses=[])
                #     exec_result = ExecutionResult(final_obs=self.env.get_observation(),
                #                                   num_waypoints=0, elapsed_time=0.0, completed=False)
                # else:
                    # Module A: Generate trajectory
                gen_result = self.generator.generate(task, self.env)

                skip_execution = False
                gen_elapsed = gen_result.metadata.get("elapsed_time", 0.0)

                if gen_result.trajectory is None:
                    # stack_cups: if tag detection fails at steps AFTER forward_1
                    # (forward_2, reverse_1, reverse_2), the scene is likely in a
                    # partial state from a prior step failure.  Route to VLM validation
                    # instead of an immediate human reset.
                    _route_to_vlm = (
                        task.task_type == "stack_cups"
                        and task.stack_step != "forward_1"
                        and not self.is_dummy
                    )
                    if _route_to_vlm:
                        log.info(
                            f"[StackCups] Tag not found at step {task.stack_step} — "
                            f"routing to VLM validation (prior step may have partially failed)"
                        )
                        skip_execution = True
                        exec_result = ExecutionResult(
                            final_obs=self.env.get_observation(),
                            num_waypoints=0, elapsed_time=0.0, completed=False,
                        )
                        # fall through to Module C (validation)
                    else:
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
                            generation_time=gen_elapsed,
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
                        ep += 1
                        if not should_continue:
                            break
                        continue

                # Module B: Execute trajectory
                if not skip_execution:
                    if task.task_type == "stack_cups":
                        gripper_close = 36000
                        gripper_effort = 300
                    else:
                        gripper_close = 0
                        gripper_effort = 300

                    self.env.set_gripper_close(gripper_close)
                    self.env.set_gripper_effort(gripper_effort)
                    exec_result = self.executor.execute(
                        gen_result.trajectory, gen_result.events, task.language_task,
                        key_poses=gen_result.key_poses,
                    )

                # Module C: Validate task success
                # stack_cups: intermediate steps (forward_1, reverse_1) auto-advance
                # without validation — UNLESS execution was skipped due to tag detection
                # failure (skip_execution=True), in which case always run VLM validation.
                if task.task_type == "stack_cups" and not self.task_scheduler.is_terminal_step and not skip_execution:
                    validation = ValidationResult(
                        success=True, confidence=1.0, method="auto_advance", details={}
                    )
                    success = True
                    reset_needed = False
                    checklist_eval = None
                    decision = "next"
                else:
                    validation = self.validator.validate(task, exec_result.final_obs, self.env)
                    success = validation.success
                    checklist_reset = self.policy.needs_reset(
                        validation=validation,
                        fail_count=fail_count,
                        episode_idx=ep,
                        task=task,
                        observation=exec_result.final_obs,
                    )
                    checklist_eval = getattr(self.policy, "last_eval", None)
                    # Checklist is authoritative when it runs; fall back to
                    # validation.needs_reset only when checklist is unavailable.
                    reset_needed = checklist_reset if checklist_eval is not None else validation.needs_reset

                    if success and not reset_needed:
                        decision = "next"
                    elif not success and not reset_needed:
                        decision = "retry"
                    else:
                        decision = "reset"

                # Classify failure type
                failure_type = ""
                if not success:
                    failure_type = FailureClassifier.classify(
                        validation=validation, features=None,
                        fail_count=fail_count, max_retries=self.max_retries,
                    )

                log.info(
                    f"[Result] task={task.name} | success={success} | "
                    f"decision={decision} | reset_needed={reset_needed} | "
                    f"failure_type={failure_type or 'none'}"
                )

                # Manifest — stack_cups: only log at terminal steps under "stack_cups"
                dataset_episode_idx = None
                _save_name = (
                    f"stack_cups_{task_direction}" if task.task_type == "stack_cups"
                    else task.name
                )
                _is_terminal = self.task_scheduler.is_terminal_step
                if not (task.task_type == "stack_cups" and not _is_terminal):
                    if success:
                        dataset_episode_idx = self.manifest.log_episode(
                            episode_idx=ep, episode_id=episode_id, run_id=self.run_id,
                            task_name=_save_name, success=True,
                            policy_method=self.policy_method,
                        )
                    else:
                        self.manifest.log_episode(
                            episode_idx=ep, episode_id=episode_id, run_id=self.run_id,
                            task_name=_save_name, success=False,
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
                    generation_time=gen_elapsed,
                    execution_time=exec_result.elapsed_time,
                    validation_method=validation.method,
                    validation_details=validation.details,
                    generation_success=not skip_execution,
                    fail_count=fail_count,
                    failure_type=failure_type,
                    dataset_episode_idx=dataset_episode_idx,
                    checklist_eval=checklist_eval,
                    episode_duration=time.time() - episode_start_time,
                )

                # Capture terminal flag before advance() mutates scheduler state
                was_terminal_step = self.task_scheduler.is_terminal_step

                # Act on decision
                if decision == "next":
                    self.task_scheduler.advance()
                    if was_terminal_step:
                        fail_count = 0
                        ep += 1

                elif decision == "retry":
                    fail_count += 1
                    self.task_scheduler.reset_to_phase_start()
                    log.info(f"[Retry] reset to phase start (step={self.task_scheduler._idx}), fail_count={fail_count}")
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
                    ep += 1
                    if not should_continue:
                        if not self.is_dummy and was_terminal_step:
                            record.ground_truth_reset = self.human_interface.request_ground_truth_label()
                        self.metrics.log_episode(record)
                        if self.dataset_recorder is not None:
                            self._save_or_clear(
                                task, was_terminal_step, _save_name,
                                success, record, decision,
                            )
                        break

                # Ground truth label only at terminal steps (all cups stacked / all cups unstacked)
                if not self.is_dummy and was_terminal_step:
                    record.ground_truth_reset = self.human_interface.request_ground_truth_label()

                # Save episode data.
                # stack_cups success (auto_advance) on non-terminal step: accumulate frames,
                # save later at the terminal step.  All other outcomes save immediately so
                # that perception-error / retry trajectories are not discarded.
                if self.dataset_recorder is not None:
                    self._save_or_clear(
                        task, was_terminal_step, _save_name,
                        success, record, decision,
                    )

                # stack_cups 비터미널 auto_advance 스텝은 metrics에 기록하지 않음
                # (forward_1, reverse_1은 에피소드의 일부로 terminal 스텝에서 함께 기록)
                _is_accumulating_step = (
                    task.task_type == "stack_cups"
                    and not was_terminal_step
                    and decision == "next"
                )
                if not _is_accumulating_step:
                    self.metrics.log_episode(record)

            except Exception as exc:
                _task_name = task.name if "task" in locals() else "?"
                log.exception(f"[ERROR] Unexpected exception in episode {ep} (task={_task_name}): {exc}")
                if self.dataset_recorder is not None:
                    self.dataset_recorder.clear_episode_buffer()
                ep += 1
                continue

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

        n_gen_fail = self.metrics.cumulative_gen_failures()
        n_exec     = self.metrics.cumulative_execution_attempts()
        print(f"\n{'='*60}")
        print(f"Collection complete: {self.metrics.cumulative_valid_trajectories()} "
              f"valid trajectories in {n_exec} execution attempts "
              f"({n_gen_fail} gen failures)")
        print(f"Success rate: {self.metrics.cumulative_success_rate():.1%}")
        print(f"Human resets: {sum(1 for e in self.metrics.episodes if e.human_reset)}")
        print(f"Failure types: {self.metrics.failure_type_distribution()}")
        print(f"Logs saved to: {self.config.log_dir}")
        print(f"{'='*60}")

    def _save_or_clear(
        self,
        task,
        was_terminal_step: bool,
        save_name: str,
        success: bool,
        record,
        decision: str,
    ) -> None:
        """Save the episode buffer or clear it, based on task context and decision.

        Save policy:
        - stack_cups, decision=="next" (auto_advance), non-terminal step:
            Do NOT save — frames accumulate for the terminal step.
        - All other cases (non-stack_cups, terminal step, retry, reset, perception error):
            Save immediately so no executed trajectory is silently discarded.

        After saving on retry: clear the buffer so the next attempt starts fresh.
        The save itself clears the buffer on success/reset paths.
        """
        if self.dataset_recorder is None:
            return

        # Only skip save when auto-advancing a non-terminal stack_cups step
        accumulating = (
            task.task_type == "stack_cups"
            and not was_terminal_step
            and decision == "next"
        )
        if accumulating:
            return

        buf_size = len(self.dataset_recorder._buffer)
        log.info(
            f"[DatasetRecorder] Saving: task={save_name} success={success} "
            f"decision={decision} buffer_frames={buf_size}"
        )
        self.dataset_recorder.save_episode(
            task_name=save_name,
            success=success,
            record=asdict(record),
            language_task=(
                "stack the cups" if (task.task_type == "stack_cups"
                                     and task.stack_step.startswith("forward"))
                else "unstack the cups" if task.task_type == "stack_cups"
                else None
            ),
        )

        # On retry (not reset-escalation): clear the buffer so the next attempt
        # starts with an empty buffer, not contaminated by the failed attempt.
        if decision == "retry":
            self.dataset_recorder.clear_episode_buffer()

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
            print("  [DEBUG] Auto-confirming reset (human reset disabled).")
            should_continue = True
            confirm_time = time.time()
        else:
            should_continue, confirm_time = self.human_interface.request_reset(task)

        if should_continue:
            self.task_scheduler.reset_to_forward()
            if hasattr(self.validator, "reset_state"):
                self.validator.reset_state()
        return should_continue, confirm_time

    # def _check_intermediate_tag(self, task) -> bool:
    #     """Detect an AprilTag to verify the previous stack_cups step succeeded.

    #     Returns True if execution of the current step should be skipped
    #     (meaning the previous step failed and we go directly to validation).

    #     Logic:
    #       intermediate_check_skip_if_visible=True  → skip if tag IS   detected
    #       intermediate_check_skip_if_visible=False → skip if tag is NOT detected
    #     """
    #     from utils.camera_utils import detect_single_tag_world_pose

    #     tag_id = task.intermediate_check_tag_id
    #     skip_if_visible = task.intermediate_check_skip_if_visible

    #     visible = False
    #     for camera in self.env.get_cameras().values():
    #         if detect_single_tag_world_pose(camera, tag_id) is not None:
    #             visible = True
    #             break

    #     should_skip = visible if skip_if_visible else not visible
    #     log.info(
    #         f"[IntermediateCheck] tag={tag_id} visible={visible} "
    #         f"skip_if_visible={skip_if_visible} → skip={should_skip}"
    #     )
    #     return should_skip

    def _debug_init_unstack_cache(self):
        """DEBUG: populate tag position cache for unstack mode.

        Detects the blue cup tag, then sets hardcoded original positions:
          purple cup: blue tag XY + 5cm right (+X)
          pink   cup: blue tag XY + 5cm left  (-X)
        """
        import numpy as np
        from utils.camera_utils import detect_single_tag_world_pose, detect_tag_world_pose_stereo

        mp = self.generator.motion_planner

        # tag IDs from task definitions (forward_1: stack_a_on_b)
        forward_1 = self.task_scheduler._tasks[0]  # stack_purple_on_blue
        forward_2 = self.task_scheduler._tasks[1]  # stack_pink_on_purple
        tag_a = forward_1.pick_tag_id    # purple
        tag_b = forward_1.place_tag_id   # blue
        tag_c = forward_2.pick_tag_id    # pink

        # pink cup is on top → detect pink tag to get blue cup XY reference
        T_pink_detected = None
        for camera in mp.cameras.values():
            T_pink_detected = detect_single_tag_world_pose(camera, tag_c)
            if T_pink_detected is not None:
                break
        if T_pink_detected is None:
            T_pink_detected = detect_tag_world_pose_stereo(mp.cameras, tag_c)

        if T_pink_detected is None:
            log.warning("[DEBUG] Could not detect pink tag — unstack cache not populated!")
            return

        # Use pink tag XY as blue cup XY reference, but use table-level Z for original positions
        ref_x = T_pink_detected[0, 3]
        ref_y = T_pink_detected[1, 3]
        table_z = mp._TAG_Z + 0.01

        T_purple = np.eye(4, dtype=np.float64)
        T_purple[0, 3] = ref_x + 0.05   # purple: 5cm right (+X)
        T_purple[1, 3] = ref_y
        T_purple[2, 3] = table_z
        mp._tag_position_cache[tag_a] = T_purple

        T_pink = np.eye(4, dtype=np.float64)
        T_pink[0, 3] = ref_x - 0.05     # pink: 5cm left (-X)
        T_pink[1, 3] = ref_y
        T_pink[2, 3] = table_z
        mp._tag_position_cache[tag_c] = T_pink

        log.info(f"[DEBUG] Unstack cache set from pink tag XY ({ref_x:.3f}, {ref_y:.3f}): "
                 f"purple tag_id={tag_a} at {T_purple[:3,3]}, "
                 f"pink tag_id={tag_c} at {T_pink[:3,3]}")
