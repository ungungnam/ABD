"""Evaluation runner: scores reset-decision methods over the reset-VQA dataset.

For each method the runner

  1. loads any cached predictions (so an interrupted run resumes),
  2. predicts the remaining samples, appending to ``predictions.jsonl``,
  3. computes overall + grouped metrics into ``metrics.json``,

and finally writes a combined ``summary.json`` / ``report.md`` across methods.

Layout under ``--results-dir``::

    results/
        <method>/predictions.jsonl
        <method>/metrics.json
        summary.json
        report.md
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, Optional

from .dataset import ResetVQADataset
from .methods import build_method
from .metrics import evaluate, format_report

# Record keys carried through to predictions.jsonl for per-group metric
# breakdowns. (gt_label_name is stored on each record too, but is not a group
# key -- grouping by a GT-derived label yields single-class, degenerate cells.)
_GROUP_KEYS = ("task_name", "task_family", "collection_policy")


def _json_safe(obj):
    """Best-effort coercion of method-specific ``raw`` payloads to JSON."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _load_cached(pred_path: Path) -> dict:
    """Return ``{sample_id: record}`` from an existing predictions file."""
    cached: dict = {}
    if not pred_path.exists():
        return cached
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cached[rec["sample_id"]] = rec
    return cached


def run_method(
    method_name: str,
    dataset: ResetVQADataset,
    results_dir: Path,
    method_kwargs: Optional[dict] = None,
    samples: Optional[Iterable] = None,
    resume: bool = True,
    verbose: bool = True,
) -> dict:
    """Evaluate one method; return its metrics dict."""
    out_dir = results_dir / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    cached = _load_cached(pred_path) if resume else {}
    samples = list(samples if samples is not None else dataset.samples)

    method = build_method(method_name, **(method_kwargs or {}))
    if verbose:
        print(f"\n[{method_name}] setup ...")
    try:
        method.setup()
    except Exception as exc:  # noqa: BLE001 -- e.g. VLM server / API key missing
        msg = f"setup failed: {type(exc).__name__}: {exc}"
        print(f"[{method_name}] {msg}", file=sys.stderr)
        records = [
            {"sample_id": s.sample_id, "gt_reset": s.gt_reset,
             "pred_reset": None, "error": msg}
            for s in samples
        ]
        result = evaluate(records, group_by=_GROUP_KEYS)
        result.update({"method": method_name, "setup_error": msg,
                        "elapsed_sec": 0.0, "predicted_this_run": 0})
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        return result

    todo = [s for s in samples if s.sample_id not in cached]
    if verbose:
        print(f"[{method_name}] {len(cached)} cached, {len(todo)} to predict.")

    t0 = time.time()
    try:
        with pred_path.open("a") as fh:
            for i, sample in enumerate(todo, 1):
                try:
                    pred = method.predict(sample)
                    rec = {
                        "sample_id": sample.sample_id,
                        "gt_reset": sample.gt_reset,
                        "pred_reset": (None if pred.reset is None
                                       else bool(pred.reset)),
                        "error": pred.error,
                        "raw": _json_safe(pred.raw),
                    }
                except Exception as exc:  # noqa: BLE001
                    rec = {
                        "sample_id": sample.sample_id,
                        "gt_reset": sample.gt_reset,
                        "pred_reset": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw": traceback.format_exc(limit=3),
                    }
                for key in _GROUP_KEYS:
                    rec[key] = getattr(sample, key, None)
                rec["gt_label_name"] = sample.gt_label_name  # kept for inspection
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                cached[sample.sample_id] = rec
                if verbose and i % 200 == 0:
                    print(f"  [{method_name}] {i}/{len(todo)}")
    finally:
        method.teardown()

    elapsed = time.time() - t0
    records = [cached[s.sample_id] for s in samples if s.sample_id in cached]
    result = evaluate(records, group_by=_GROUP_KEYS)
    result["method"] = method_name
    result["elapsed_sec"] = round(elapsed, 2)
    result["predicted_this_run"] = len(todo)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    if verbose:
        print(format_report(method_name, result))
    return result


def run_suite(
    dataset_root: str | Path,
    method_specs: list,
    results_dir: str | Path,
    resume: bool = True,
    verbose: bool = True,
    limit: int = 0,
    sample_ids: Optional[set] = None,
) -> dict:
    """Evaluate several methods and write the combined summary / report.

    ``method_specs`` is a list of either ``"name"`` strings or
    ``{"name": ..., "kwargs": {...}}`` dicts. ``sample_ids`` (if given) restricts
    the run to those samples; otherwise ``limit`` (>0) caps it to the first N.
    """
    dataset = ResetVQADataset(dataset_root)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if sample_ids is not None:
        samples = [s for s in dataset.samples if s.sample_id in sample_ids]
    elif limit:
        samples = dataset.samples[:limit]
    else:
        samples = dataset.samples
    if verbose:
        print(f"Dataset: {len(samples)}/{len(dataset)} samples from {dataset.root}")

    all_results: dict = {}
    for spec in method_specs:
        if isinstance(spec, str):
            name, kwargs = spec, {}
        else:
            name, kwargs = spec["name"], spec.get("kwargs", {})
        all_results[name] = run_method(
            name, dataset, results_dir,
            method_kwargs=kwargs, samples=samples,
            resume=resume, verbose=verbose,
        )

    summary = {
        "dataset_root": str(dataset.root),
        "n_samples": len(dataset),
        "dataset_info": dataset.info,
        "methods": all_results,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    report = ["# Reset-decision VQA evaluation", "",
              f"Dataset: `{dataset.root}`  ({len(dataset)} samples)", ""]
    report.append(_leaderboard(all_results))
    report.append("")
    for name, result in all_results.items():
        report.append("```")
        report.append(format_report(name, result))
        report.append("```")
        report.append("")
    (results_dir / "report.md").write_text("\n".join(report))

    if verbose:
        print(f"\nWrote summary -> {results_dir / 'summary.json'}")
        print(f"Wrote report  -> {results_dir / 'report.md'}")
    return summary


def _leaderboard(all_results: dict) -> str:
    """Markdown table sorted by balanced accuracy."""
    rows = []
    for name, res in all_results.items():
        o = res["overall"]
        rows.append((o["balanced_accuracy"], o["mcc"], o["f1"],
                     o["accuracy"], res["scored"], res["errors"], name))
    rows.sort(reverse=True)
    out = ["| method | bal_acc | mcc | f1 | acc | scored | errors |",
           "|---|---|---|---|---|---|---|"]
    for bal, mcc, f1, acc, scored, errors, name in rows:
        out.append(f"| {name} | {bal:.4f} | {mcc:.4f} | {f1:.4f} | "
                   f"{acc:.4f} | {scored} | {errors} |")
    return "\n".join(out)
