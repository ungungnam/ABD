"""Periodic reset baseline policy — resets every N episodes.

If detection_info["__user_skip__"] is True (set by the runner when the
operator presses the configured skip key), fast-forward the period counter
to trigger a reset immediately. The wall-clock time that would otherwise
have been spent running the remaining episodes is reported in the log.
"""

import logging
import time

from validator.base_validator import ValidationResult
from policy.base_policy import BaseResetPolicy

log = logging.getLogger(__name__)


class PeriodicPolicy(BaseResetPolicy):

    def __init__(self, period: int = 10):
        self.period = period
        self._episodes_since_reset = 0
        # Wall-clock tracking used to estimate the time saved when the
        # operator chooses to skip the rest of the reset period.
        self._last_call_t: float = 0.0
        self._sum_dt: float = 0.0
        self._n_dt: int = 0
        self._total_skipped_s: float = 0.0
        self._total_skipped_eps: int = 0
        # Set on each call: details of any skip that just fired (None if no skip).
        # Consumers (collection_runner) can read and persist these per episode.
        self.last_skip: dict = None

    def needs_reset(self, validation: ValidationResult, fail_count: int,
                    episode_idx: int, task=None, observation=None,
                    detection_info: dict = None) -> bool:
        now = time.time()
        if self._last_call_t > 0.0:
            self._sum_dt += now - self._last_call_t
            self._n_dt += 1
        self._last_call_t = now
        avg_dt = self._sum_dt / self._n_dt if self._n_dt > 0 else 0.0

        user_skip = bool((detection_info or {}).get("__user_skip__"))
        self.last_skip = None

        self._episodes_since_reset += 1
        if user_skip and self._episodes_since_reset < self.period:
            skipped = self.period - self._episodes_since_reset
            saved_s = avg_dt * skipped
            self._total_skipped_eps += skipped
            self._total_skipped_s += saved_s
            self.last_skip = {
                "skipped_episodes": skipped,
                "skipped_time_s": saved_s,
                "avg_dt_s": avg_dt,
                "cumulative_skipped_episodes": self._total_skipped_eps,
                "cumulative_skipped_time_s": self._total_skipped_s,
            }
            log.info(
                f"[PeriodicPolicy] User skip — fast-forwarding {skipped} "
                f"episode(s) (avg_dt={avg_dt:.1f}s, ~{saved_s:.0f}s saved). "
                f"Cumulative: {self._total_skipped_eps} eps / "
                f"{self._total_skipped_s:.0f}s saved."
            )
            self._episodes_since_reset = self.period

        if self._episodes_since_reset >= self.period:
            self._episodes_since_reset = 0
            return True
        return False
