"""Classification metrics (numpy implementation) and success criteria."""

import subprocess
import sys

import numpy as np
import pytest

from simple_sim.metrics import (
    DEFAULT_CRITICAL_CLASSES,
    compute_metrics,
    evaluate_success_criteria,
    format_success_criteria,
)


def test_macro_f1_ignores_zero_support_classes():
    # Class 2 never appears in y_true (or y_pred): it must not drag macro-F1 down.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_metrics(y_true, y_pred, ['A', 'B', 'C'], critical_classes=[])
    assert m['macro_f1'] == 1.0
    assert m['per_class']['C']['support'] == 0
    assert m['per_class']['C']['f1'] == 0.0


def test_exact_values_hand_computed():
    #           true: A A A B B C      pred: A B A B C C
    m = compute_metrics([0, 0, 0, 1, 1, 2], [0, 1, 0, 1, 2, 2], ['A', 'B', 'C'], critical_classes=['A', 'C', 'X'])
    assert m['accuracy'] == 4 / 6
    assert m['confusion_matrix'] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert m['per_class']['A'] == {'precision': 1.0, 'recall': 2 / 3, 'f1': 4 / 5, 'support': 3}
    assert m['per_class']['B'] == {'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'support': 2}
    assert m['per_class']['C'] == {'precision': 0.5, 'recall': 1.0, 'f1': 2 / 3, 'support': 1}
    assert m['macro_f1'] == np.mean([4 / 5, 0.5, 2 / 3])
    assert m['critical_fn_rates'] == {'A': 1 / 3, 'C': 0.0}  # unknown class names are skipped


def test_never_predicted_class_has_zero_precision_not_nan():
    m = compute_metrics([0, 1], [0, 0], ['A', 'B'], critical_classes=[])
    assert m['per_class']['B'] == {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 1}
    assert m['per_class']['A']['precision'] == 0.5


def test_out_of_range_labels_are_ignored_by_matrix_but_count_for_accuracy():
    m = compute_metrics([0, 1, 5], [0, 1, 5], ['A', 'B'], critical_classes=[])
    assert m['accuracy'] == 1.0
    assert m['confusion_matrix'] == [[1, 0], [0, 1]]


def test_empty_and_mismatched_inputs():
    m = compute_metrics([], [], ['A'], critical_classes=[])
    assert m['accuracy'] == 0.0 and m['macro_f1'] == 0.0 and m['confusion_matrix'] == [[0]]
    with pytest.raises(ValueError, match="differ in length"):
        compute_metrics([0, 1], [0], ['A', 'B'])


def test_default_critical_classes():
    assert DEFAULT_CRITICAL_CLASSES == ('MISSING', 'MISALIGNED', 'TOMBSTONE')
    m = compute_metrics([0, 1], [0, 0], ['OK', 'MISSING'])
    assert m['critical_fn_rates'] == {'MISSING': 1.0}


def test_matches_scikit_learn_bit_for_bit():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    for n_classes, n in [(2, 10), (4, 257), (8, 1000), (5, 3)]:
        y_true = rng.integers(0, n_classes, n)
        y_pred = np.where(rng.random(n) < 0.6, y_true, rng.integers(0, n_classes, n))
        names = [f"c{i}" for i in range(n_classes)]
        m = compute_metrics(y_true, y_pred, names, critical_classes=[])
        p, r, f, s = skm.precision_recall_fscore_support(y_true, y_pred, labels=range(n_classes), zero_division=0)
        assert m['accuracy'] == float(skm.accuracy_score(y_true, y_pred))
        assert m['confusion_matrix'] == skm.confusion_matrix(y_true, y_pred, labels=range(n_classes)).tolist()
        for i, name in enumerate(names):
            assert m['per_class'][name] == {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(s[i])}
        present = s > 0
        assert m['macro_f1'] == float(f[present].mean())


def test_metrics_module_does_not_import_scikit_learn():
    code = "import sys, simple_sim.metrics; assert 'sklearn' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_success_criteria_ignores_zero_support():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y, ['A', 'B', 'C'], critical_classes=[])
    check = evaluate_success_criteria(0.9, m)
    assert check.passed is True
    assert "ALL CHECKS PASSED" in format_success_criteria(check)


def test_success_criteria_fails_on_zero_recall():
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2])
    m = compute_metrics(y_true, y_pred, ['A', 'B', 'C'], critical_classes=[])
    check = evaluate_success_criteria(0.9, m, entity="profile")
    assert check.passed is False
    assert dict(check.checks)["No profile with 0% recall"] is False


def test_success_criteria_text_block_unchanged():
    m = compute_metrics([0, 1], [0, 0], ['A', 'B'], critical_classes=[])
    check = evaluate_success_criteria(0.9, m)
    bar = "=" * 60
    assert format_success_criteria(check) == "\n".join([
        "", bar, "SUCCESS CRITERIA CHECK", bar,
        "✓ PASS: Val accuracy > 70%", "✗ FAIL: Test accuracy > 70%", "✗ FAIL: No class with 0% recall",
        "", "✗ SOME CHECKS FAILED", bar,
    ])
    assert format_success_criteria(check, ascii_only=True).splitlines()[4] == "PASS: Val accuracy > 70%"


# --------------------------------------------------------------------------- calibration

def test_calibration_metrics_hand_computed():
    from simple_sim.metrics import ECE_BINS, calibration_metrics, format_calibration

    p = np.array([[0.9, 0.1], [0.9, 0.1], [0.3, 0.7], [0.5, 0.5]])
    y = np.array([0, 1, 1, 0])
    cal = calibration_metrics(p, y)
    assert cal["n_bins"] == ECE_BINS == 15
    # bins: 0.9 -> bin 13 (acc 0.5, conf 0.9), 0.7 -> bin 10 (acc 1, conf 0.7), 0.5 -> bin 7 (acc 1, conf 0.5)
    assert cal["ece"] == pytest.approx(0.5 * 0.4 + 0.25 * 0.3 + 0.25 * 0.5)
    assert cal["nll"] == pytest.approx(-np.mean(np.log([0.9, 0.1, 0.7, 0.5])))
    assert cal["brier"] == pytest.approx(np.mean([0.02, 1.62, 0.18, 0.5]))
    assert format_calibration(cal).startswith("Calibration: ECE 0.4000 (15 bins)  NLL ")
    perfect = calibration_metrics(np.eye(3)[[0, 1, 2]], [0, 1, 2])
    assert perfect == {"ece": 0.0, "nll": pytest.approx(0.0), "brier": 0.0, "n_bins": 15}
    assert calibration_metrics(np.zeros((0, 3)), [])["ece"] == 0.0
    with pytest.raises(ValueError):
        calibration_metrics(p, [0, 1, 1])
    with pytest.raises(ValueError):
        calibration_metrics(p, [0, 2, 0, 0])
