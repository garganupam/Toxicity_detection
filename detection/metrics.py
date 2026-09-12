"""
Evaluation metrics and comparison reporting.

Used by both train_classifier.py and test_cross_dataset.py.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_prob) -> Dict:
    """Compute full binary classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None
    return metrics


def print_comparison(baseline_metrics: Dict, augmented_metrics: Dict, title: str = "BASELINE vs AUGMENTED"):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  {'Metric':<20} {'Baseline':>15} {'Augmented':>15} {'Delta':>12}")
    print("-" * 70)

    for key in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        bv = baseline_metrics.get(key)
        av = augmented_metrics.get(key)
        if bv is not None and av is not None:
            delta = av - bv
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<20} {bv:>15.4f} {av:>15.4f} {sign}{delta:>11.4f}")
        else:
            print(f"  {key:<20} {'N/A':>15} {'N/A':>15} {'N/A':>12}")

    print("-" * 70)

    for name, m in [("Baseline", baseline_metrics), ("Augmented", augmented_metrics)]:
        cm = m.get("confusion_matrix", [])
        if cm and len(cm) == 2:
            print(f"\n  {name} Confusion Matrix:")
            print(f"                    Predicted")
            print(f"                  Non-toxic  Toxic")
            print(f"    Actual Non-toxic  {cm[0][0]:>6}  {cm[0][1]:>6}")
            print(f"    Actual Toxic      {cm[1][0]:>6}  {cm[1][1]:>6}")

    print("\n" + "=" * 70)


def save_report(report: Dict, path: str):
    """Save report dict as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved → %s", p)
