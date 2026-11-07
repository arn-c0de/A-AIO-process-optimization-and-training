"""Classification metrics (numpy only).

The values match scikit-learn's ``accuracy_score``, ``precision_recall_fscore_support``
(``zero_division=0``) and ``confusion_matrix`` bit for bit; ``tests/test_metrics.py`` checks it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simple_sim.classes import CRITICAL_CLASSES

DEFAULT_CRITICAL_CLASSES: tuple[str, ...] = tuple(CRITICAL_CLASSES)

SUCCESS_ACCURACY = 0.70


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` as float64 with 0 where ``den == 0`` (sklearn ``zero_division=0``)."""
    num = num.astype(np.float64)
    den = den.astype(np.float64)
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den != 0)
    return out


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """``cm[i, j]`` = samples of true class *i* predicted as *j*; labels outside ``range(n_classes)`` are ignored."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    keep = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    np.add.at(cm, (y_true[keep], y_pred[keep]), 1)
    return cm


def compute_metrics(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    class_names: list[str],
    critical_classes: Sequence[str] | None = None
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        class_names: List of class names
        critical_classes: Classes whose false-negative rate is reported
            (default: ``DEFAULT_CRITICAL_CLASSES``)

    Returns:
        Dictionary with accuracy, macro_f1, per_class, confusion_matrix, critical_fn_rates.
        Empty inputs give accuracy 0.0.
    """
    y_true = np.asarray(y_true).astype(np.int64, copy=False).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64, copy=False).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred differ in length: {y_true.size} vs {y_pred.size}")
    n = len(class_names)

    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else 0.0

    cm = confusion_matrix(y_true, y_pred, n)
    labels = np.arange(n)
    tp = ((y_true[:, None] == labels) & (y_pred[:, None] == labels)).sum(axis=0)
    pred_sum = (y_pred[:, None] == labels).sum(axis=0)
    support = (y_true[:, None] == labels).sum(axis=0)
    precision = _safe_divide(tp, pred_sum)
    recall = _safe_divide(tp, support)
    # sklearn's form: (1 + beta^2) tp / (beta^2 * true + pred), beta = 1
    f1 = _safe_divide(2.0 * tp, 1.0 * support.astype(np.float64) + pred_sum.astype(np.float64))

    # Macro-averaged F1 over classes actually present in y_true; classes with
    # support 0 would contribute F1=0 and skew model selection.
    present = support > 0
    macro_f1 = f1[present].mean() if present.any() else 0.0

    per_class_metrics = {
        class_name: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }
        for i, class_name in enumerate(class_names)
    }

    # False-negative rate of the critical classes (configurable to match the problem definition).
    if critical_classes is None:
        critical_classes = DEFAULT_CRITICAL_CLASSES
    critical_fn_rates = {}
    for class_name in critical_classes:
        if class_name in class_names:
            class_idx = class_names.index(class_name)
            fn_count = np.sum((y_true == class_idx) & (y_pred != class_idx))
            total_count = np.sum(y_true == class_idx)
            fn_rate = fn_count / total_count if total_count > 0 else 0.0
            critical_fn_rates[class_name] = float(fn_rate)

    return {
        'accuracy': accuracy,
        'macro_f1': float(macro_f1),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'critical_fn_rates': critical_fn_rates
    }


#: Bins of the expected calibration error (equal width over the confidence range (0, 1]).
ECE_BINS = 15


def calibration_metrics(
    probs: np.ndarray | Sequence[Sequence[float]],
    y_true: np.ndarray | Sequence[int],
    n_bins: int = ECE_BINS,
) -> dict[str, Any]:
    """Calibration of predicted class probabilities: ``ece``, ``nll``, ``brier`` (and ``n_bins``).

    *probs* is ``[N, C]`` (rows sum to 1), *y_true* the true class indices. ECE is the
    confidence-weighted gap between accuracy and mean confidence (top-1 probability) over *n_bins*
    equal-width bins; NLL the mean negative log probability of the true class (clipped at 1e-12);
    Brier the mean squared error between the probability vector and the one-hot label, summed over
    classes (so 0 for a perfect, 2 for a confidently wrong prediction). Empty inputs give zeros.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true).astype(np.int64, copy=False).ravel()
    if p.ndim != 2 or p.shape[0] != y.size:
        raise ValueError(f"probs must be [N, C] with N == len(y_true): {p.shape} vs {y.size}")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    n, c = p.shape
    if n == 0:
        return {'ece': 0.0, 'nll': 0.0, 'brier': 0.0, 'n_bins': int(n_bins)}
    if (y < 0).any() or (y >= c).any():
        raise ValueError("y_true contains labels outside range(C)")

    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    # Bin b covers (b/n_bins, (b+1)/n_bins]; a confidence of exactly 0 lands in bin 0.
    bins = np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())

    nll = float(-np.mean(np.log(np.clip(p[np.arange(n), y], 1e-12, None))))
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    return {'ece': float(ece), 'nll': nll, 'brier': brier, 'n_bins': int(n_bins)}


def format_calibration(cal: dict[str, Any]) -> str:
    """One line: ``Calibration: ECE 0.0123 (15 bins)  NLL 0.4567  Brier 0.2345``."""
    return (f"Calibration: ECE {cal['ece']:.4f} ({cal.get('n_bins', ECE_BINS)} bins)  "
            f"NLL {cal['nll']:.4f}  Brier {cal['brier']:.4f}")


@dataclass(frozen=True, slots=True)
class SuccessCheck:
    """Result of :func:`evaluate_success_criteria`: ``(label, passed)`` per criterion."""

    checks: tuple[tuple[str, bool], ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return all(ok for _name, ok in self.checks)


def evaluate_success_criteria(val_accuracy: float, metrics: dict[str, Any], entity: str = "class") -> SuccessCheck:
    """Built-in success criteria: val and test accuracy > 70 %, no *entity* with 0 % recall.

    Classes with zero support are ignored by the recall check.
    """
    supported = [m for m in metrics['per_class'].values() if m.get('support', 1) > 0]
    return SuccessCheck((
        ("Val accuracy > 70%", val_accuracy > SUCCESS_ACCURACY),
        ("Test accuracy > 70%", metrics['accuracy'] > SUCCESS_ACCURACY),
        (f"No {entity} with 0% recall", all(m['recall'] > 0 for m in supported)),
    ))


def format_success_criteria(check: SuccessCheck, *, ascii_only: bool = False) -> str:
    """The console ``SUCCESS CRITERIA CHECK`` block (starts with a blank line)."""
    ok_mark, fail_mark = ("", "") if ascii_only else ("✓ ", "✗ ")
    lines = ["", "=" * 60, "SUCCESS CRITERIA CHECK", "=" * 60]
    lines += [f"{ok_mark if ok else fail_mark}{'PASS' if ok else 'FAIL'}: {name}" for name, ok in check.checks]
    lines += ["", f"{ok_mark}ALL CHECKS PASSED" if check.passed else f"{fail_mark}SOME CHECKS FAILED", "=" * 60]
    return "\n".join(lines)


def format_metrics(metrics: dict[str, Any], class_names: list[str]) -> str:
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
