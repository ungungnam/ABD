"""Single-VQA reset policy — asks the VLM one yes/no question per episode.

Reset is triggered when the VLM answers "yes" to the single question. Success
detection is delegated to the external validator (same path as NoReset/Periodic).

If a reference-image directory is configured, this policy injects a reference
"initial state" image alongside the current observation so the VLM can compare
the current scene with the canonical initial state of the upcoming task.
Reference image filenames follow ``{reference_dir}/{key}_initial.{jpg,png,jpeg}``
where ``key`` is the next-task identifier passed via
``detection_info["__next_task_key__"]``.

The prompt also includes — when supplied via ``detection_info`` — the per-object
detection status, current object world-frame positions, and the robot's reachable
workspace bounds, so the VLM can reason about whether intervention is required.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from policy.base_policy import BaseResetPolicy
from validator.base_validator import ValidationResult
from vlm_client.vqa_client import VQAClient

log = logging.getLogger(__name__)

_PREAMBLE_WITH_REF = (
    "The first image is the canonical INITIAL state expected before the upcoming task. "
    "The remaining images are different views of the CURRENT scene."
)
_PREAMBLE_NO_REF = (
    "The provided images are different views of the current scene."
)

_QUESTION = (
    "Considering all of the above — detection status, object positions vs the robot's "
    "reachable workspace bounds, and (if provided) the gap between the current scene "
    "and the canonical initial state — is this a situation where a human MUST "
    "intervene to reset the workspace before the robot can continue? "
    "Answer strictly with a single word: 'yes' or 'no'."
)


class SingleVQAPolicy(BaseResetPolicy):

    def __init__(
        self,
        vqa_client: VQAClient,
        question: str = "",
        reference_dir: str = "",
    ):
        self.vqa_client = vqa_client
        self.question = question or _QUESTION
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self._ref_cache: dict = {}
        self.last_eval: Optional[dict] = None

    def needs_reset(
        self,
        validation: ValidationResult,
        fail_count: int,
        episode_idx: int,
        task=None,
        observation=None,
        detection_info: dict = None,
    ) -> bool:
        if observation is None:
            self.last_eval = None
            return False

        info = detection_info or {}
        ref_key = info.get("__next_task_key__")
        positions = info.get("__object_positions__") or {}
        bounds = info.get("__workspace_bounds__") or {}
        # Public detection_info entries are everything that is not a "__sentinel__".
        det_status = {k: v for k, v in info.items() if not k.startswith("__")}

        ref_img = self._load_reference(ref_key) if ref_key else None

        if ref_img is not None:
            obs_with_ref = {"reference_initial": ref_img, **observation}
            preamble = _PREAMBLE_WITH_REF
        else:
            obs_with_ref = observation
            preamble = _PREAMBLE_NO_REF

        context = self._build_context_block(det_status, positions, bounds)
        prompt = f"{preamble}\n\n{context}\n{self.question}"

        raw = self.vqa_client.ask_text(obs_with_ref, prompt) or ""
        ans = self._parse_yes_no(raw)
        reset = ans == "yes"

        self.last_eval = {
            "task_name": getattr(task, "name", None),
            "next_task_key": ref_key,
            "reference_used": ref_img is not None,
            "detection_status": det_status,
            "object_positions": positions,
            "workspace_bounds": bounds,
            "answer": ans,
            "raw": raw,
        }
        log.info("=" * 88)
        log.info(
            f"[SingleVQAPolicy] task={getattr(task, 'name', '?')} | "
            f"next_key={ref_key} | ref={'yes' if ref_img is not None else 'no'} | "
            f"answer={ans} | needs_reset={reset}"
        )
        for line in context.strip().splitlines():
            log.info(f"  {line}")
        log.info(f"  Q: {self.question}")
        log.info(f"  A: {ans}  (raw: {raw.strip()[:120]})")
        log.info("=" * 88)
        return reset

    @staticmethod
    def _build_context_block(det_status: dict, positions: dict, bounds: dict) -> str:
        """Inline each object's last-detected world position with its detection
        status; fall back to a separate line for positions that have no matching
        detection entry."""
        def _pos_for(name: str):
            if name in positions:
                return positions[name]
            base = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
            return positions.get(base)

        lines = []
        consumed_pos_keys: set = set()
        if det_status:
            parts = []
            for k, v in det_status.items():
                is_detected = str(v).strip().lower() == "detected"
                pos = _pos_for(k) if is_detected else None
                if pos is not None:
                    parts.append(
                        f"{k}: {v} at (x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f})"
                    )
                    consumed_pos_keys.add(k)
                    base = re.sub(r"\s*\([^)]*\)\s*$", "", k).strip()
                    consumed_pos_keys.add(base)
                else:
                    parts.append(f"{k}: {v}")
            lines.append("Object detection status: " + " / ".join(parts))

        leftover = {n: p for n, p in positions.items() if n not in consumed_pos_keys}
        if leftover:
            pos_str = " / ".join(
                f"{name} at (x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f})"
                for name, p in leftover.items()
            )
            lines.append(f"Other object positions (world frame, meters): {pos_str}")
        if bounds:
            lines.append(
                "Robot reachable workspace bounds (world frame, meters): "
                f"x in [{bounds.get('x_min', '?'):.2f}, {bounds.get('x_max', '?'):.2f}], "
                f"y in [{bounds.get('y_min', '?'):.2f}, {bounds.get('y_max', '?'):.2f}]"
            )
        return "\n".join(lines) + ("\n" if lines else "")

    def _load_reference(self, key: str) -> Optional[np.ndarray]:
        if self.reference_dir is None:
            return None
        if key in self._ref_cache:
            return self._ref_cache[key]
        for ext in ("jpg", "jpeg", "png"):
            path = self.reference_dir / f"{key}_initial.{ext}"
            if path.exists():
                arr = np.array(Image.open(path).convert("RGB"))
                self._ref_cache[key] = arr
                log.info(f"[SingleVQAPolicy] Loaded reference image: {path}")
                return arr
        log.warning(
            f"[SingleVQAPolicy] No reference image for key='{key}' in "
            f"{self.reference_dir} (looked for {key}_initial.jpg/png/jpeg). "
            f"Falling back to no-reference prompt."
        )
        self._ref_cache[key] = None
        return None

    @staticmethod
    def _parse_yes_no(raw: str) -> str:
        text = raw.strip().lower()
        m = re.search(r"\b(yes|no)\b", text)
        if m:
            return m.group(1)
        log.warning(
            f"[SingleVQAPolicy] Could not parse yes/no from VLM response; "
            f"defaulting to 'no'. Raw: {raw!r}"
        )
        return "no"
