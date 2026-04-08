"""Editable prompt templates for the VLMChecklistPolicy.

Loaded from ``src/prompts/checklist_prompts.json`` so the wording can be
tweaked without touching policy logic. The policy formats META_PROMPT once
per task to generate a checklist, and EVAL_PROMPT once per episode to
score the post-execution observation against that checklist.
"""

import json
from pathlib import Path

_PROMPTS_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "checklist_prompts.json"
)

with open(_PROMPTS_PATH) as f:
    _prompts = json.load(f)


def _as_text(value) -> str:
    """Allow each prompt to be stored as either a single string or a list
    of lines (joined with newlines), so the JSON file stays human-readable.
    """
    if isinstance(value, list):
        return "\n".join(value)
    return value


_meta_template = _as_text(_prompts["meta_prompt"])
_example_context = _as_text(_prompts.get("example_context", ""))

# Inject the example block into the meta prompt before the per-task
# format() call. .replace() avoids touching other ``{...}`` placeholders.
META_PROMPT: str = _meta_template.replace("{example_context}", _example_context)
EVAL_PROMPT: str = _as_text(_prompts["eval_prompt"])
