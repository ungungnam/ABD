"""Code-as-Monitor (CaM) reset-decision method.

Adapts the CVPR'25 paper *Code-as-Monitor: Constraint-aware Visual Programming
for Reactive and Proactive Robotic Failure Detection* (arXiv:2412.04455) to the
single-frame reset-decision VQA task.

Pipeline (see the submodules):

    constraint_generator  VLM -> per-task spatio-temporal constraints + the
                          geometric "elements" they are defined over.
    code_generator        VLM -> an executable Python `monitor(scene)` program
                          that deterministically checks those constraints.
    method                per sample: VLM grounds the elements from the last
                          frame into a `scene` dict, then runs the monitor.

The paper's RGB-D constraint *painter* + CoTracker tracking are replaced by a
single VLM grounding step, because the VQA dataset is image-only (no depth, no
video). Everything else -- the unified constraint-satisfaction formulation and
"VLM writes code, code does the checking" -- is preserved.
"""

from .constraint_generator import ConstraintGenerator, ConstraintSet
from .code_generator import MonitorCodeGenerator, MonitorProgram

__all__ = [
    "ConstraintGenerator",
    "ConstraintSet",
    "MonitorCodeGenerator",
    "MonitorProgram",
]
