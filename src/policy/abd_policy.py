"""ABD reset policy (Section 12.4) — the proposed risk-based policy.

Uses the full ABD module (feature extraction + risk scoring) to
make next / retry / reset decisions.
"""

from abd.risk_scorer import RiskScorer
from validator.base_validator import ValidationResult
from policy.base_policy import BaseResetPolicy


class ABDPolicy(BaseResetPolicy):

    def __init__(self, risk_scorer: RiskScorer):
        self.risk_scorer = risk_scorer

    def decide(self, validation: ValidationResult, fail_count: int,
               episode_idx: int, features: dict = None) -> str:
        if features is None or "vector" not in features:
            # Fallback: if features unavailable, use simple logic
            return "next" if validation.success else "retry"

        risk = self.risk_scorer.compute_risk(features["vector"])
        return self.risk_scorer.decide(risk)
