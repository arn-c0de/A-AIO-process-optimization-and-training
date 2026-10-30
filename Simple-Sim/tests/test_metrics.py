"""Test classification metrics and success criteria."""

import numpy as np

from simple_sim.metrics import check_success_criteria, compute_metrics


def test_macro_f1_ignores_zero_support_classes():
    # Class 2 never appears in y_true (or y_pred): it must not drag macro-F1 down.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_metrics(y_true, y_pred, ['A', 'B', 'C'], critical_classes=[])
    assert m['macro_f1'] == 1.0
    assert m['per_class']['C']['support'] == 0
    assert m['per_class']['C']['f1'] == 0.0


def test_success_criteria_ignores_zero_support(capsys):
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y, ['A', 'B', 'C'], critical_classes=[])
    assert check_success_criteria(0.9, m) is True
    assert "ALL CHECKS PASSED" in capsys.readouterr().out


def test_success_criteria_fails_on_zero_recall():
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2])
    m = compute_metrics(y_true, y_pred, ['A', 'B', 'C'], critical_classes=[])
    assert check_success_criteria(0.9, m, entity="profile", ascii_only=True) is False
