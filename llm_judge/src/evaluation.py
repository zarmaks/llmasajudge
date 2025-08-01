from __future__ import annotations

"""Utility functions to compute classification metrics (precision, recall, F1,
accuracy) for the LLM‑as‑a‑Judge pipeline.

A minimal dependency‑free implementation so we avoid pulling in scikit‑learn.
We treat labels as **strings** and support arbitrary sets of classes.
"""

from collections import Counter
from typing import Dict, List, Sequence, Tuple

MetricDict = Dict[str, float]


def _confusion_counts(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> Tuple[Counter, Counter, Counter]:
    """Return per‑class counts of TP, FP, FN using Counters.

    TP[class] = predicted == class and true == class
    FP[class] = predicted == class and true != class
    FN[class] = predicted != class and true == class
    """
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    return tp, fp, fn


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def precision_recall_f1(
    y_true: Sequence[str], y_pred: Sequence[str], average: str = "macro"
) -> MetricDict:
    """Compute precision, recall, F1 and accuracy.

    Parameters
    ----------
    y_true / y_pred : list‑like
        Gold / predicted labels.
    average : "macro" | "micro" | "none"
        * macro  – unweighted mean of per‑class metrics
        * micro  – global TP / FP / FN across classes
        * none   – return per‑class metrics instead of a single score
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp, fp, fn = _confusion_counts(y_true, y_pred)
    classes = sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())))

    per_class: Dict[str, MetricDict] = {}
    for c in classes:
        prec = _safe_div(tp[c], tp[c] + fp[c])
        rec = _safe_div(tp[c], tp[c] + fn[c])
        f1 = _safe_div(2 * prec * rec, prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1}

    accuracy = sum(tp.values()) / len(y_true) if y_true else 0.0

    if average == "none":
        # attach accuracy separately
        result: MetricDict = {
            f"{cls}_{metric}": val
            for cls, md in per_class.items()
            for metric, val in md.items()
        }
        result["accuracy"] = accuracy
        return result

    if average == "macro":
        num_classes = len(classes)
        macro_p = (
            sum(d["precision"] for d in per_class.values()) / num_classes
        )
        macro_r = sum(d["recall"] for d in per_class.values()) / num_classes
        macro_f1 = sum(d["f1"] for d in per_class.values()) / num_classes
        return {
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f1,
            "accuracy": accuracy
        }

    if average == "micro":
        total_tp = sum(tp.values())  # type: ignore
        total_fp = sum(fp.values())  # type: ignore
        total_fn = sum(fn.values())  # type: ignore
        micro_p = _safe_div(total_tp, total_tp + total_fp)
        micro_r = _safe_div(total_tp, total_tp + total_fn)
        micro_f1 = (
            _safe_div(2 * micro_p * micro_r, micro_p + micro_r)
            if micro_p + micro_r else 0.0
        )
        return {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
            "accuracy": accuracy
        }

    raise ValueError("average must be 'macro', 'micro', or 'none'")


# --------------------------------------------------------------------------------------
# Pretty printing / report helpers
# --------------------------------------------------------------------------------------


def metrics_report(metrics: MetricDict, title: str = "") -> str:
    """Return a human‑readable multi‑line string of the metrics."""
    header = f"=== {title} ===\n" if title else ""
    lines: List[str] = [header]
    for k, v in metrics.items():
        lines.append(f"{k:>10s}: {v:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # simple manual test
    gold = ["Correct", "Incorrect", "Correct", "Dangerous", "Incorrect"]
    pred = ["Correct", "Incorrect", "Incorrect", "Dangerous", "Incorrect"]
    m = precision_recall_f1(gold, pred, average="macro")
    print(metrics_report(m, title="Demo metrics"))
