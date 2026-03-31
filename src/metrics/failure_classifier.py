"""Failure type classifier for ABD episodes.

Maps ABD features and validation results to the failure taxonomy:
  invisible, unreachable, out_of_bounds, mis_execution,
  retry_limit, generation_failure, unknown
"""

from validator.base_validator import ValidationResult


class FailureClassifier:
    """Classify episode failures into the failure taxonomy."""

    @staticmethod
    def classify(
        validation: ValidationResult,
        features: dict,
        fail_count: int,
        max_retries: int,
        generation_success: bool = True,
    ) -> str:
        """Classify a failed episode into a failure type.

        Args:
            validation: Task validation result.
            features: ABD feature dict (f_succ, f_vis, f_reach, f_dev, ...).
            fail_count: Consecutive failure count at decision time.
            max_retries: Maximum retries before escalation.
            generation_success: Whether trajectory generation succeeded.

        Returns:
            One of: "generation_failure", "retry_limit", "invisible",
            "unreachable", "out_of_bounds", "mis_execution", "unknown".
        """
        if not generation_success:
            return "generation_failure"

        if fail_count >= max_retries:
            return "retry_limit"

        if features is None:
            return "unknown"

        # Target not detected by perception
        if features.get("f_vis", 1.0) < 0.5:
            return "invisible"

        # Detected but outside workspace bounds
        if features.get("f_reach", 1.0) < 0.5:
            # High deviation from canonical → likely knocked out of bounds
            if features.get("f_dev", 0.0) > 0.7:
                return "out_of_bounds"
            return "unreachable"

        # Visible, reachable, but task still failed
        if not validation.success:
            return "mis_execution"

        return "unknown"
