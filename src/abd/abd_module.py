"""ABD Module — high-level Autonomy Boundary Detection.

Combines feature extraction and risk scoring into a single interface.
"""

from dataclasses import dataclass, field

import numpy as np

from abd.feature_extractor import ABDFeatureExtractor
from abd.risk_scorer import RiskScorer
from task.task_family import TaskDefinition
from validator.base_validator import ValidationResult


@dataclass
class ABDDecision:
    """Full ABD decision output."""
    features: dict = field(default_factory=dict)
    risk_score: float = 0.0
    action: str = "next"  # "next", "retry", or "reset"


class ABDModule:
    """Autonomy Boundary Detection module.

    Evaluates whether autonomous execution should continue, retry,
    or request human reset based on the current system state.
    """

    def __init__(self, feature_extractor: ABDFeatureExtractor,
                 risk_scorer: RiskScorer):
        self.feature_extractor = feature_extractor
        self.risk_scorer = risk_scorer

    def evaluate(self, task: TaskDefinition, validation_result: ValidationResult,
                 fail_count: int) -> ABDDecision:
        """Run full ABD pipeline: extract features -> compute risk -> decide.

        Args:
            task: Current task.
            validation_result: Result of task success validation.
            fail_count: Consecutive failure count.

        Returns:
            ABDDecision with features, risk score, and action.
        """
        features = self.feature_extractor.extract(task, validation_result, fail_count)
        risk = self.risk_scorer.compute_risk(features["vector"])
        action = self.risk_scorer.decide(risk)

        return ABDDecision(features=features, risk_score=risk, action=action)

    def calibrate(self, task: TaskDefinition):
        """Calibrate canonical state for ABD features."""
        self.feature_extractor.calibrate(task)
