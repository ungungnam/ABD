"""CaM stage 1 -- the constraint generator.

Given a task, a VLM produces the set of spatio-temporal constraints that must
hold for the workspace to be *ready* for the next attempt (i.e. NOT need a
human reset), together with the geometric "constraint elements" those
constraints are defined over.

Following the paper, each element abstracts a constraint-related entity (or a
part of it) into a compact geometric primitive -- a ``point``, ``line`` or
``surface`` -- discarding visual detail irrelevant to the constraint. Each
element also declares the boolean *attributes* that must be observed from the
scene; the downstream grounding step answers exactly those, and the generated
monitor code checks the constraints over them.

The result is a per-task artefact: it depends only on the task, not on any
individual episode, so it is generated once and cached to disk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
CONSTRAINT_PROMPT = """\
You are the CONSTRAINT GENERATOR of a Code-as-Monitor robotic failure-detection \
system.

A robot repeatedly performs a manipulation task in a fixed workcell. Between \
episodes a monitor must decide whether the WORKSPACE needs a human to physically \
reset it before the robot can attempt the task again.

TASK NAME : {task_name}
TASK GOAL : "{task_description}"
TASK TYPE : {task_type}

Your job: enumerate the spatio-temporal constraints that must ALL hold for the \
workspace to be ready for the next attempt WITHOUT human intervention. A \
violated constraint means a reset IS needed.

Think in terms of CONSTRAINT ELEMENTS: abstract each constraint-related entity \
(or relevant part) into one compact geometric primitive:
  - "point"   : a small/localised entity (e.g. a graspable object, a handle)
  - "line"    : an elongated entity or axis (e.g. a drawer slide, a rim edge)
  - "surface" : an extended region or support (e.g. the table, a pan opening,
                the robot's reachable workspace)

For every element also list the boolean ATTRIBUTES that an observer must read \
off a single image of the workspace to evaluate the constraints (each attribute \
is a precise yes/no question).

Return ONLY a JSON object, no prose, with this exact schema:

{{
  "subgoal": "<one sentence: the state the workspace must be in to be ready>",
  "elements": [
    {{
      "id": "<short_snake_case_id>",
      "kind": "object" | "region" | "part",
      "geometry": "point" | "line" | "surface",
      "description": "<what this element is>",
      "attributes": [
        {{"name": "<short_snake_case_attr>", "question": "<precise yes/no question>"}}
      ]
    }}
  ],
  "constraints": [
    {{
      "id": "c1",
      "type": "presence" | "reachability" | "spatial_relation" | "pose" | "clearance",
      "description": "<human-readable constraint>",
      "elements": ["<element id>", ...],
      "satisfied_when": "<boolean condition over element attributes, in words>"
    }}
  ]
}}

Rules:
- 2-5 elements, 2-6 constraints. Keep ids stable and snake_case.
- Every attribute referenced in "satisfied_when" must be declared on some element.
- Constraints describe the READY (no-reset) state; violation => reset needed.
"""


@dataclass
class ConstraintSet:
    """A per-task set of constraints + the elements they are defined over."""

    task_name: str
    subgoal: str
    elements: list = field(default_factory=list)      # list[dict]
    constraints: list = field(default_factory=list)   # list[dict]
    raw_response: str = ""

    # -- serialisation ------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "subgoal": self.subgoal,
            "elements": self.elements,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConstraintSet":
        return cls(
            task_name=d["task_name"],
            subgoal=d.get("subgoal", ""),
            elements=d.get("elements", []),
            constraints=d.get("constraints", []),
            raw_response=d.get("raw_response", ""),
        )

    # -- convenience --------------------------------------------------- #
    def attribute_index(self) -> dict:
        """``{element_id: [attribute_name, ...]}`` for grounding / codegen."""
        return {
            el["id"]: [a["name"] for a in el.get("attributes", [])]
            for el in self.elements
        }

    def validate(self) -> None:
        """Raise ``ValueError`` if the constraint set is structurally broken."""
        if not self.elements:
            raise ValueError("constraint set has no elements")
        if not self.constraints:
            raise ValueError("constraint set has no constraints")
        known_attrs = self.attribute_index()
        for el in self.elements:
            for key in ("id", "geometry"):
                if key not in el:
                    raise ValueError(f"element missing '{key}': {el}")
        for c in self.constraints:
            for ref in c.get("elements", []):
                if ref not in known_attrs:
                    raise ValueError(
                        f"constraint {c.get('id')} references unknown "
                        f"element '{ref}'"
                    )


def _extract_json_object(text: str) -> dict:
    """Extract the first balanced ``{...}`` JSON object from a VLM response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced braces in response")


class ConstraintGenerator:
    """Generates (and caches) a :class:`ConstraintSet` per task via a VLM."""

    def __init__(self, vqa_client, cache_dir: Optional[str | Path] = None):
        """
        Args:
            vqa_client: an ABD ``VQAClient`` (any backend).
            cache_dir: directory for ``<task>.constraints.json`` artefacts;
                if given, generation is skipped when the file already exists.
        """
        self.vqa_client = vqa_client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict = {}

    def _cache_path(self, task_name: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{task_name}.constraints.json"

    def generate(
        self,
        task_name: str,
        task_description: str,
        task_type: str = "manipulation",
        images: Optional[dict] = None,
    ) -> ConstraintSet:
        """Return the constraint set for a task (memory- and disk-cached).

        ``images`` is an optional representative observation of the task. It is
        passed to the VLM so that image-conditioned backends (e.g. AHA, which
        rejects text-only requests) can respond; text-capable backends ignore
        whether images are present.
        """
        if task_name in self._mem:
            return self._mem[task_name]

        path = self._cache_path(task_name)
        if path and path.exists():
            cset = ConstraintSet.from_dict(json.loads(path.read_text()))
            self._mem[task_name] = cset
            return cset

        prompt = CONSTRAINT_PROMPT.format(
            task_name=task_name,
            task_description=task_description,
            task_type=task_type,
        )
        raw = self.vqa_client.ask_text(images or None, prompt)
        if not raw:
            raise RuntimeError(
                f"constraint generator: empty VLM response for '{task_name}'"
            )
        obj = _extract_json_object(raw)
        cset = ConstraintSet(
            task_name=task_name,
            subgoal=obj.get("subgoal", ""),
            elements=obj.get("elements", []),
            constraints=obj.get("constraints", []),
            raw_response=raw,
        )
        cset.validate()

        if path:
            path.write_text(json.dumps(cset.to_dict(), indent=2))
        self._mem[task_name] = cset
        return cset
