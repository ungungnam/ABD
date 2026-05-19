"""Injectable reset-decision method interface.

A reset-decision method answers the dataset's VQA question:

    given the last-frame observation of an episode, does the environment need
    a human reset before the robot can continue?

To add a new method, subclass :class:`ResetMethod`, implement :meth:`predict`,
and decorate the class with :func:`register`. The evaluation runner will then
be able to select it by name -- no other wiring is required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ResetPrediction:
    """One method's verdict on one sample.

    Attributes:
        reset: Predicted binary reset decision. ``None`` means the method
            failed on this sample (counted as an error, excluded from metrics).
        raw: Method-specific detail kept for inspection -- e.g. a checklist
            score, a VLM answer string, generated monitor code, etc.
        error: Human-readable error message when ``reset is None``.
    """

    reset: Optional[bool]
    raw: Any = None
    error: Optional[str] = None


class ResetMethod(ABC):
    """Base class for an injectable reset-decision method.

    Lifecycle::

        method.setup()                  # once, before the run
        for sample in dataset:
            method.predict(sample)       # once per sample
        method.teardown()                # once, after the run
    """

    #: Unique method name (used on the CLI and as the results sub-directory).
    name: str = "base"

    def setup(self) -> None:
        """Optional one-time initialisation (load models, open clients)."""

    @abstractmethod
    def predict(self, sample) -> ResetPrediction:
        """Return a :class:`ResetPrediction` for one :class:`ResetVQASample`."""

    def teardown(self) -> None:
        """Optional one-time cleanup (close clients, free GPU memory)."""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Callable[..., ResetMethod]] = {}


def register(name: str) -> Callable:
    """Class decorator that registers a :class:`ResetMethod` factory by name."""

    def _wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"Reset method '{name}' is already registered.")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def available_methods() -> list:
    """Names of every registered method."""
    return sorted(_REGISTRY)


def build_method(name: str, **kwargs) -> ResetMethod:
    """Instantiate a registered method by name, forwarding ``kwargs`` to it."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown reset method '{name}'. "
            f"Registered: {', '.join(available_methods()) or '(none)'}"
        )
    return _REGISTRY[name](**kwargs)
