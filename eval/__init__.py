"""Reset-decision VQA evaluation suite.

Builds a VQA dataset from ABD's collected episodes (last-frame observation ->
binary reset decision) and runs an injectable set of reset-decision methods
against it so ABD's method can be benchmarked against baselines.

See ``eval/README.md`` for the full workflow.
"""
