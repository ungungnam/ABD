"""ABD Risk Scorer.

Computes risk score r_k from the feature vector f_k and decides
the control action: next / retry / reset.

r_k = w1*(1 - f_succ) + w2*(1 - f_vis) + w3*(1 - f_reach)
    + w4*(1 - f_rec)  + w5*f_dev         + w6*f_fail
"""

import numpy as np


class RiskScorer:
    """Compute ABD risk score and apply threshold-based decision rule."""

    def __init__(self, weights, tau_retry: float = 0.3, tau_reset: float = 0.7):
        """
        Args:
            weights: 6-element weight vector [w_succ, w_vis, w_reach, w_rec, w_dev, w_fail].
            tau_retry: risk threshold below which action is 'next'.
            tau_reset: risk threshold above which action is 'reset'.
                       Between tau_retry and tau_reset, action is 'retry'.
        """
        self.weights = np.array(weights, dtype=np.float64)
        assert len(self.weights) == 6
        self.tau_retry = tau_retry
        self.tau_reset = tau_reset

    def compute_risk(self, feature_vector: np.ndarray) -> float:
        """Compute weighted risk score from the 6D feature vector.

        Features [0:4] (f_succ, f_vis, f_reach, f_rec): higher = better, so invert.
        Features [4:6] (f_dev, f_fail): higher = worse, use directly.
        """
        fv = np.array(feature_vector, dtype=np.float64)
        inverted = fv.copy()
        inverted[:4] = 1.0 - inverted[:4]
        return float(self.weights @ inverted)

    def decide(self, risk: float) -> str:
        """Apply two-threshold decision rule.

        Returns:
            'next'  if risk < tau_retry
            'retry' if tau_retry <= risk < tau_reset
            'reset' if risk >= tau_reset
        """
        if risk < self.tau_retry:
            return "next"
        elif risk < self.tau_reset:
            return "retry"
        else:
            return "reset"
