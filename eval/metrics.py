"""Binary-classification metrics for reset-decision evaluation.

The positive class is "reset needed" (``gt_reset == True``). Because the
dataset is imbalanced (~23% positive) we report threshold-free, imbalance-aware
metrics (balanced accuracy, MCC, F1) alongside the raw confusion matrix, plus
per-group breakdowns so a method can be inspected per task / policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Optional


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, gt: bool, pred: bool) -> None:
        if gt and pred:
            self.tp += 1
        elif gt and not pred:
            self.fn += 1
        elif not gt and pred:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics_from_confusion(cm: ConfusionMatrix) -> dict:
    """Derive the standard metric bundle from a confusion matrix."""
    tp, fp, tn, fn = cm.tp, cm.fp, cm.tn, cm.fn
    n = cm.total
    accuracy = _safe_div(tp + tn, n)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)            # TPR / sensitivity
    specificity = _safe_div(tn, tn + fp)       # TNR
    f1 = _safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2

    # Matthews correlation coefficient -- robust under class imbalance.
    mcc_den = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = _safe_div((tp * tn) - (fp * fn), mcc_den)

    return {
        "n": n,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
    }


def evaluate(
    records: Iterable[dict],
    group_by: Optional[Iterable[str]] = None,
) -> dict:
    """Compute overall + grouped metrics from prediction records.

    Each record must contain at least::

        {"gt_reset": bool, "pred_reset": bool|None, "error": str|None, ...}

    Records with ``pred_reset is None`` (method failed / errored on that
    sample) are counted as ``errors`` and excluded from the confusion matrix.

    ``group_by`` is an iterable of record keys; for each key a per-value
    metric breakdown is produced (e.g. ``["task_name", "collection_policy"]``).
    """
    overall = ConfusionMatrix()
    errors = 0
    scored = 0
    groups: dict = {key: {} for key in (group_by or [])}

    for rec in records:
        gt = bool(rec["gt_reset"])
        pred = rec.get("pred_reset")
        if pred is None:
            errors += 1
            continue
        pred = bool(pred)
        scored += 1
        overall.add(gt, pred)
        for key in groups:
            val = rec.get(key)
            groups[key].setdefault(val, ConfusionMatrix()).add(gt, pred)

    result = {
        "scored": scored,
        "errors": errors,
        "overall": metrics_from_confusion(overall),
    }
    for key, buckets in groups.items():
        result[f"by_{key}"] = {
            str(val): metrics_from_confusion(cm)
            for val, cm in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        }
    return result


def format_report(method_name: str, result: dict) -> str:
    """Render a compact human-readable metrics report."""
    o = result["overall"]
    c = o["confusion"]
    lines = [
        f"=== {method_name} ===",
        f"  scored={result['scored']}  errors={result['errors']}",
        f"  confusion: TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']}",
        f"  accuracy={o['accuracy']}  balanced_acc={o['balanced_accuracy']}  "
        f"mcc={o['mcc']}",
        f"  precision={o['precision']}  recall={o['recall']}  "
        f"specificity={o['specificity']}  f1={o['f1']}",
    ]
    for key, buckets in result.items():
        if not key.startswith("by_"):
            continue
        lines.append(f"  --- {key} ---")
        for val, m in buckets.items():
            lines.append(
                f"    {val:<24} bal_acc={m['balanced_accuracy']:.3f} "
                f"f1={m['f1']:.3f} (n={m['n']})"
            )
    return "\n".join(lines)
