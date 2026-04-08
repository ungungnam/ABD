"""Editable prompt templates for the VLMChecklistPolicy.

These are kept in their own module so the wording can be tweaked
without touching policy logic. The policy formats META_PROMPT once
per task to generate a checklist, and EVAL_PROMPT once per episode
to score the post-execution observation against that checklist.
"""

META_PROMPT = """\
You are designing a verification checklist for a robot manipulation task.

Task description: {task_description}

Produce a JSON checklist that another VLM will use to inspect camera images
after the robot executes the task and decide whether the task succeeded.

Rules:
- 4-8 items, each phrased as a yes/no question answerable from images alone.
- Each item has an integer "weight" 1-5 (higher = more critical to success).
- Cover: object presence, target location, physical contact/containment,
  gripper state, and absence of obvious failure modes.
- Output ONLY valid JSON, no commentary, in this exact schema:

{{
  "task_name": "{task_name}",
  "task_description": "{task_description}",
  "items": [
    {{"id": 1, "question": "<yes/no question>", "weight": <int 1-5>}}
  ]
}}
"""

EVAL_PROMPT = """\
You are verifying whether a robot completed this task: {task_description}

Inspect the camera images and answer each checklist item with strictly
"yes" or "no". Output ONLY a JSON object mapping item id (as string) to
"yes" or "no", nothing else.

Checklist:
{items_block}

Example output: {{"1": "yes", "2": "no", "3": "yes"}}
"""
