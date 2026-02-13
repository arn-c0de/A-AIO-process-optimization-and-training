"""Classification metrics computation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix as sklearn_confusion_matrix
)
from typing import Dict, List, Any, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    critical_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        class_names: List of class names

    Returns:
        Dictionary with all metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    # Macro-averaged F1
    macro_f1 = f1.mean()

    # Confusion matrix
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # Per-class metrics dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    # Compute false negative rate for critical defects
    # Critical defects are configurable to match the problem definition.
    if critical_classes is None:
        critical_classes = ['MISSING', 'MISALIGNED', 'TOMBSTONE']
    critical_fn_rates = {}

    for class_name in critical_classes:
        if class_name in class_names:
            class_idx = class_names.index(class_name)
            # False negatives: true positives that were predicted as something else
            fn_count = np.sum((y_true == class_idx) & (y_pred != class_idx))
            total_count = np.sum(y_true == class_idx)
            fn_rate = fn_count / total_count if total_count > 0 else 0.0
            critical_fn_rates[class_name] = float(fn_rate)

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'critical_fn_rates': critical_fn_rates
    }


def format_metrics(metrics: Dict[str, Any], class_names: List[str]) -> str:
    """Format metrics for console output.

    Args:
        metrics: Metrics dictionary from compute_metrics
        class_names: List of class names

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("METRICS")
    lines.append("=" * 60)
    lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"Macro F1: {metrics['macro_f1']:.4f}")
    lines.append("")
    lines.append("Per-class metrics:")
    lines.append("-" * 60)
    lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 60)

    for class_name in class_names:
        metrics_dict = metrics['per_class'][class_name]
        lines.append(
            f"{class_name:<15} "
            f"{metrics_dict['precision']:>10.4f} "
            f"{metrics_dict['recall']:>10.4f} "
            f"{metrics_dict['f1']:>10.4f} "
            f"{metrics_dict['support']:>10}"
        )

    lines.append("")
    lines.append("Confusion Matrix:")
    lines.append("-" * 60)

    cm = np.array(metrics['confusion_matrix'])
    header = "True\\Pred  " + "  ".join(f"{cn:>8}" for cn in class_names)
    lines.append(header)
    for i, class_name in enumerate(class_names):
        row = f"{class_name:<10}" + "  ".join(f"{cm[i, j]:>8}" for j in range(len(class_names)))
        lines.append(row)

    if metrics['critical_fn_rates']:
        lines.append("")
        lines.append("Critical Defect False Negative Rates:")
        lines.append("-" * 60)
        for class_name, fn_rate in metrics['critical_fn_rates'].items():
            lines.append(f"  {class_name}: {fn_rate:.4f}")

    lines.append("=" * 60)

    return "\n".join(lines)
