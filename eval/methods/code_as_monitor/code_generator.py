"""CaM stage 2 -- the monitor code generator.

Given a :class:`ConstraintSet`, a VLM performs *constraint-aware visual
programming*: it writes an executable Python ``monitor(scene)`` function that
deterministically checks every constraint. Once generated, the code runs
without any further VLM calls -- exactly the paper's "code does the checking"
property -- so it is generated once per task and cached.

The ``scene`` argument is the element-grounding dict produced at inference
time::

    scene = {element_id: {attribute_name: bool, ...}, ...}

and ``monitor`` returns ``(ready: bool, reason: str)`` where ``ready`` is True
when the workspace satisfies all constraints (no reset needed).

Generated code is executed in a restricted namespace (no imports, no file/OS
builtins) since it is model-authored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .constraint_generator import ConstraintSet

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
CODE_PROMPT = """\
You are the MONITOR CODE GENERATOR of a Code-as-Monitor system.

You are given the constraint set for a robot task. Write a single Python \
function that checks, deterministically, whether the workspace is READY for the \
next attempt (no human reset needed).

SUBGOAL: {subgoal}

ELEMENTS (geometric abstractions) and their observable boolean attributes:
{elements_block}

CONSTRAINTS to enforce:
{constraints_block}

The function you write will be called as:

    monitor(scene)

where `scene` is a dict mapping each element id to a dict of its attribute
booleans, e.g.:

    scene = {{ {scene_example} }}

Some attributes may be missing -- treat a missing attribute as False.

Requirements for your code:
- Define exactly one function: `def monitor(scene):`
- It must return a tuple `(ready, reason)`:
    ready  -> True if ALL constraints are satisfied, else False
    reason -> short string naming the first violated constraint (or "ok")
- Use only plain Python: dict access, boolean logic, comparisons, `all`, `any`.
- NO imports, NO file/network/OS access, NO `eval`/`exec`.
- Read attributes defensively, e.g. `scene.get("banana", {{}}).get("on_table", False)`.
- One `if not <condition>: return (False, "<constraint id>: <why>")` per constraint.

Return ONLY the Python code inside a single ```python code block, nothing else.
"""

# Builtins exposed to model-authored monitor code -- safe, side-effect-free.
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict)
    else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "range", "round", "set", "sorted",
        "str", "sum", "tuple", "zip", "isinstance", "True", "False", "None",
    )
}


@dataclass
class MonitorProgram:
    """A compiled monitor for one task: source code + callable + provenance."""

    task_name: str
    source: str
    monitor: Optional[Callable] = None
    is_fallback: bool = False
    error: str = ""
    raw_response: str = ""

    def run(self, scene: dict) -> dict:
        """Execute the monitor on a grounded scene.

        Returns ``{"ready": bool|None, "reason": str, "error": str}``.
        ``ready is None`` indicates the monitor itself raised.
        """
        if self.monitor is None:
            return {"ready": None, "reason": "", "error": self.error or
                    "no compiled monitor"}
        try:
            result = self.monitor(scene)
            if isinstance(result, tuple) and len(result) == 2:
                ready, reason = result
            else:
                ready, reason = bool(result), ""
            return {"ready": bool(ready), "reason": str(reason), "error": ""}
        except Exception as exc:  # noqa: BLE001 -- model-authored code
            return {"ready": None, "reason": "", "error":
                    f"{type(exc).__name__}: {exc}"}

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "source": self.source,
            "is_fallback": self.is_fallback,
            "error": self.error,
        }


def _extract_python_block(text: str) -> str:
    """Pull the Python source out of a ```python ...``` fenced block."""
    text = text or ""
    m = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    return code.strip()


def _compile_monitor(source: str) -> Callable:
    """Compile ``source`` in a restricted namespace and return ``monitor``.

    Raises ``ValueError`` if the source defines no ``monitor`` function or
    fails to compile.
    """
    if "exec" in source or "eval" in source or "import" in source:
        raise ValueError("generated code contains a disallowed keyword")
    namespace: dict = {"__builtins__": _SAFE_BUILTINS}
    code_obj = compile(source, "<cam_monitor>", "exec")
    exec(code_obj, namespace)  # noqa: S102 -- sandboxed, restricted builtins
    monitor = namespace.get("monitor")
    if not callable(monitor):
        raise ValueError("generated code did not define a `monitor` function")
    return monitor


def _fallback_source(cset: ConstraintSet) -> str:
    """Deterministic monitor derived directly from the constraint set.

    Heuristic used when the VLM is unavailable or its code will not compile:
    a constraint is treated as satisfied when every observable attribute of
    every element it references reads True.
    """
    lines = [
        "def monitor(scene):",
        '    """Fallback monitor: all attributes of a constraint\'s elements'
        ' must hold."""',
    ]
    attr_index = cset.attribute_index()
    for c in cset.constraints:
        cid = c.get("id", "c")
        checks = []
        for el in c.get("elements", []):
            for attr in attr_index.get(el, []):
                checks.append(
                    f'scene.get("{el}", {{}}).get("{attr}", False)'
                )
        cond = " and ".join(checks) if checks else "True"
        lines.append(f"    if not ({cond}):")
        lines.append(f'        return (False, "{cid}: constraint violated")')
    lines.append('    return (True, "ok")')
    return "\n".join(lines)


class MonitorCodeGenerator:
    """Generates (and caches) a :class:`MonitorProgram` per task via a VLM."""

    def __init__(self, vqa_client, cache_dir: Optional[str | Path] = None):
        self.vqa_client = vqa_client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict = {}

    def _cache_path(self, task_name: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{task_name}.monitor.py"

    @staticmethod
    def _scene_example(cset: ConstraintSet) -> str:
        parts = []
        for el in cset.elements[:2]:
            attrs = ", ".join(
                f'"{a["name"]}": true' for a in el.get("attributes", [])[:2]
            )
            parts.append(f'"{el["id"]}": {{{attrs}}}')
        return ", ".join(parts) or '"element_id": {"attribute": true}'

    @staticmethod
    def _elements_block(cset: ConstraintSet) -> str:
        out = []
        for el in cset.elements:
            attrs = "; ".join(
                f'{a["name"]} ({a.get("question", "")})'
                for a in el.get("attributes", [])
            )
            out.append(
                f'- {el["id"]} [{el.get("geometry", "point")}]: '
                f'{el.get("description", "")} | attributes: {attrs}'
            )
        return "\n".join(out)

    @staticmethod
    def _constraints_block(cset: ConstraintSet) -> str:
        out = []
        for c in cset.constraints:
            out.append(
                f'- {c.get("id")} [{c.get("type", "?")}]: '
                f'{c.get("description", "")} '
                f'(satisfied when: {c.get("satisfied_when", "?")})'
            )
        return "\n".join(out)

    def generate(self, cset: ConstraintSet, images=None) -> MonitorProgram:
        """Return the monitor program for a constraint set (cached).

        ``images`` is an optional representative observation, forwarded to the
        VLM so image-conditioned backends (e.g. AHA) can respond to what is
        otherwise a text-only code-generation prompt.

        On any VLM/compile failure this returns a deterministic fallback
        monitor instead of raising, so the pipeline degrades gracefully.
        """
        task_name = cset.task_name
        if task_name in self._mem:
            return self._mem[task_name]

        path = self._cache_path(task_name)
        if path and path.exists():
            source = path.read_text()
            prog = self._build_program(task_name, source, raw="", fallback=False)
            self._mem[task_name] = prog
            return prog

        prog: MonitorProgram
        try:
            prompt = CODE_PROMPT.format(
                subgoal=cset.subgoal,
                elements_block=self._elements_block(cset),
                constraints_block=self._constraints_block(cset),
                scene_example=self._scene_example(cset),
            )
            raw = self.vqa_client.ask_text(images or None, prompt)
            source = _extract_python_block(raw)
            prog = self._build_program(task_name, source, raw=raw, fallback=False)
        except Exception as exc:  # noqa: BLE001
            prog = self._fallback_program(cset, reason=str(exc))

        if prog.monitor is None and not prog.is_fallback:
            # VLM code failed to compile -> degrade to fallback.
            prog = self._fallback_program(cset, reason=prog.error)

        if path and prog.monitor is not None:
            path.write_text(prog.source)
        self._mem[task_name] = prog
        return prog

    def _build_program(
        self, task_name: str, source: str, raw: str, fallback: bool
    ) -> MonitorProgram:
        try:
            monitor = _compile_monitor(source)
            return MonitorProgram(
                task_name=task_name, source=source, monitor=monitor,
                is_fallback=fallback, raw_response=raw,
            )
        except Exception as exc:  # noqa: BLE001
            return MonitorProgram(
                task_name=task_name, source=source, monitor=None,
                is_fallback=fallback, error=f"{type(exc).__name__}: {exc}",
                raw_response=raw,
            )

    def _fallback_program(self, cset: ConstraintSet, reason: str) -> MonitorProgram:
        source = _fallback_source(cset)
        prog = self._build_program(
            cset.task_name, source, raw="", fallback=True
        )
        prog.error = f"using fallback monitor ({reason})"
        return prog
