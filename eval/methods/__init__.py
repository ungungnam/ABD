"""Reset-decision method registry.

Importing this package self-registers every method. Methods that depend on
optional pieces (ABD ``src`` modules, a VLM client, etc.) are imported
defensively: if their imports fail, the trivial baselines still work and the
failure is surfaced as a warning rather than crashing the whole run.
"""

from __future__ import annotations

import warnings

from .base import ResetMethod, ResetPrediction, available_methods, build_method, register

# Trivial baselines have no optional dependencies -- import eagerly.
from . import trivial  # noqa: F401

# Real methods depend on ABD's src/ and a VLM backend; import defensively so a
# missing optional dependency only disables that one method.
_OPTIONAL_MODULES = (
    "abd_checklist",
    "single_vqa",
    "zero_shot_vlm",
    "code_as_monitor.method",
)

for _mod in _OPTIONAL_MODULES:
    try:
        __import__(f"{__name__}.{_mod}", fromlist=["*"])
    except Exception as exc:  # noqa: BLE001 -- intentional broad guard
        warnings.warn(
            f"reset method module '{_mod}' unavailable: {exc!r}",
            RuntimeWarning,
        )

__all__ = [
    "ResetMethod",
    "ResetPrediction",
    "register",
    "build_method",
    "available_methods",
]
