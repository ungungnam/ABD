"""Trivial reset-decision baselines.

These need no VLM server and serve two purposes: they are reference points for
the real methods (any useful method must beat them on balanced accuracy / MCC),
and they double as the evaluation-pipeline smoke test.
"""

from __future__ import annotations

import random

from .base import ResetMethod, ResetPrediction, register


@register("always_reset")
class AlwaysReset(ResetMethod):
    """Always predict that a reset is needed (recall = 1, specificity = 0)."""

    def predict(self, sample) -> ResetPrediction:
        return ResetPrediction(reset=True, raw="constant:always")


@register("never_reset")
class NeverReset(ResetMethod):
    """Never predict a reset -- also the majority-class baseline (~77% acc)."""

    def predict(self, sample) -> ResetPrediction:
        return ResetPrediction(reset=False, raw="constant:never")


@register("random")
class RandomReset(ResetMethod):
    """Predict a reset with fixed probability ``p`` from a seeded RNG."""

    def __init__(self, p: float = 0.5, seed: int = 0):
        self.p = float(p)
        self.seed = int(seed)
        self._rng = random.Random(seed)

    def setup(self) -> None:
        self._rng = random.Random(self.seed)

    def predict(self, sample) -> ResetPrediction:
        draw = self._rng.random()
        return ResetPrediction(reset=draw < self.p, raw=f"random:{draw:.3f}")
