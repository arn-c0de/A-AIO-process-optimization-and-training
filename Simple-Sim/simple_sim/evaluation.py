"""Test-set evaluation of defect and profile classifiers (plan item L2).

:func:`run_eval` evaluates a checkpoint, ensemble or bundle dir on a dataset's ``test`` split and
writes ``report_<stem>.json``; :func:`run_eval_profile` does the same for a profile classifier
(``report_profile_classifier_<stem>.json``). :func:`write_report` is the one place that writes a
report and logs the ``Report saved to: <path>`` line (``--json`` callers read ``report_path`` from
the result instead of parsing it).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simple_sim.data_loader import ProfileDataset, ROIDataset
from simple_sim.devices import resolve_device
from simple_sim.eval_utils import (
    checkpoint_eval_config,
    checkpoint_val_accuracy,
    dataset_profile,
    eval_loader,
    format_checkpoint_info,
    format_distribution,
    metrics_summary,
    resolve_bundle_checkpoint,
    run_inference,
)
from simple_sim.io_utils import atomic_write_json, now_iso
from simple_sim.manifest import check_checkpoint_profile
from simple_sim.metrics import (
    calibration_metrics,
    compute_metrics,
    evaluate_success_criteria,
    format_calibration,
    format_metrics,
    format_success_criteria,
)
from simple_sim.models import load_checkpoint_model
from simple_sim.telemetry import emit
from simple_sim.training import PROFILE_CLASSIFIER

__all__ = ["evaluate", "report_path_for", "run_eval", "run_eval_profile", "write_report"]

log = logging.getLogger(__name__)

_RULE = "=" * 60


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    critical_classes: list[str] | None = None,
    *,
    tta: bool = False,
    calibration: bool = False,
) -> tuple[dict[str, Any], int]:
    """Evaluate *model* on *loader*; return ``(metrics, num_samples)``.

    *tta* averages the softmax over flips and quarter turns (:func:`~simple_sim.eval_utils.tta_views`);
    *calibration* adds ``metrics["calibration"]`` (ECE, NLL, Brier; :func:`calibration_metrics`).
    """
    res = run_inference(model, loader, device, desc="Evaluating", event="eval_batch", tta=tta, collect_probs=calibration)
    metrics = compute_metrics(res.y_true, res.y_pred, class_names, critical_classes=critical_classes)
    if calibration:
        metrics["calibration"] = calibration_metrics(res.probs, res.y_true)
    return metrics, res.num_samples


def report_path_for(model_path: Path, *, profile_classifier: bool = False) -> Path:
    """Where the eval report of *model_path* goes: ``report_[profile_classifier_]<stem>.json`` next to it."""
    model_path = Path(model_path)
    prefix = "report_profile_classifier_" if profile_classifier else "report_"
    return model_path.parent / f"{prefix}{model_path.stem}.json"


def write_report(path: Path, report: dict[str, Any]) -> Path:
    """Atomically write *report* to *path* and log ``Report saved to: <path>``."""
    atomic_write_json(path, report)
    log.info(f"\nReport saved to: {path}")
    return Path(path)


def _load_test_split(dataset: Any, title: str) -> dict[str, int]:
    log.info(f"Test samples: {len(dataset)}")
    dist = dataset.get_class_distribution()
    log.info(format_distribution(title, dist))
    return dist


def run_eval(
    data_dir: Path,
    model_path: Path,
    device: str | torch.device = "cpu",
    *,
    allow_profile_mismatch: bool = False,
    tta: bool = False,
    calibration: bool = False,
) -> dict[str, Any]:
    """Evaluate a checkpoint, ensemble or bundle dir on the dataset's test split.

    ``report_<stem>.json`` is written next to the *requested* model path (for a bundle dir:
    next to the dir, not the resolved checkpoint inside it), which is where
    batch_predict / arena_report look it up.

    *tta* (test-time augmentation over flips and quarter turns) and *calibration* (ECE, NLL and
    Brier score under ``metrics.calibration``) are opt-in; the report has a ``tta`` key only with
    *tta*, so the default output is unchanged. A checkpoint of another component profile raises
    :class:`simple_sim.manifest.ProfileMismatchError` unless *allow_profile_mismatch*.

    Returns a summary dict (report path, metrics, success-criteria result, ...).
    """
    data_dir = Path(data_dir)
    requested_model_path = model_path = Path(model_path)
    device = resolve_device(device)

    log.info("Evaluation configuration:")
    log.info(f"  Dataset: {data_dir}")
    log.info(f"  Model: {model_path}")
    log.info(f"  Device: {device}")
    if tta:
        log.info("  Test-time augmentation: flips + quarter turns (8 views)")

    dataset_profile_id, dataset_profile_hash = dataset_profile(data_dir)
    # Multi-model bundle support: --model can be a directory containing per-profile checkpoints.
    model_path = resolve_bundle_checkpoint(
        model_path, dataset_profile_id, hint="Train this profile into the bundle first."
    )

    log.info("\nLoading model checkpoint (single model or ensemble)...")
    model, class_names, checkpoint = load_checkpoint_model(model_path, device)
    log.info(format_checkpoint_info(checkpoint))

    # GUARD: Validate component profile match
    ckpt_profile_id = check_checkpoint_profile(
        checkpoint, dataset_profile_id, dataset_profile_hash,
        action="evaluate model on",
        allow_mismatch=bool(allow_profile_mismatch),
        override_verb="Evaluating",
    )
    if ckpt_profile_id is not None:
        log.info(f"  Profile: {ckpt_profile_id}")

    emit("eval_start", dataset_dir=str(data_dir), model_path=str(model_path), device=str(device))

    log.info("\nLoading test dataset...")
    test_dataset = ROIDataset(data_dir, 'test', class_names, return_id=True)
    test_dist = _load_test_split(test_dataset, "Test class distribution")

    log.info("\n" + _RULE)
    log.info("EVALUATION ON TEST SET")
    log.info(_RULE)

    critical_classes = checkpoint_eval_config(checkpoint).get('critical_classes')
    metrics, seen = evaluate(
        model, eval_loader(test_dataset, checkpoint), device, class_names, critical_classes=critical_classes,
        tta=tta, calibration=calibration,
    )

    log.info("\n" + format_metrics(metrics, class_names))
    if calibration:
        log.info(format_calibration(metrics["calibration"]))
    log.info(f"\nSeen samples: {seen}")
    emit("eval_done", seen_samples=int(seen), accuracy=float(metrics["accuracy"]), macro_f1=float(metrics["macro_f1"]))

    resolved_model_path = str(model_path) if model_path != requested_model_path else None
    report = {
        'timestamp': now_iso(),
        'dataset_path': str(data_dir),
        'model_path': str(requested_model_path),
        'resolved_model_path': resolved_model_path,
        'split': 'test',
        'metrics': metrics,
        'test_size': len(test_dataset),
        'seen_samples': int(seen),
        'class_distribution': test_dist,
    }
    if tta:
        report['tta'] = True
    report_path = write_report(report_path_for(requested_model_path), report)

    success = evaluate_success_criteria(checkpoint_val_accuracy(checkpoint), metrics, entity="class")
    log.info(format_success_criteria(success))
    return {
        "model_path": str(requested_model_path),
        "resolved_model_path": resolved_model_path,
        "dataset_path": str(data_dir),
        "report_path": str(report_path),
        "test_size": len(test_dataset),
        "seen_samples": int(seen),
        **metrics_summary(metrics),
        **({"tta": True} if tta else {}),
        "success_criteria_passed": success.passed,
    }


def run_eval_profile(data_dir: Path, model_path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Evaluate a profile classifier on the test split and write ``report_profile_classifier_<stem>.json``."""
    data_dir = Path(data_dir)
    model_path = Path(model_path)
    device = resolve_device(device)

    log.info("Profile classifier evaluation")
    log.info(f"  Dataset: {data_dir}")
    log.info(f"  Model: {model_path}")
    log.info(f"  Device: {device}")

    log.info("\nLoading model checkpoint...")
    model, profile_names, checkpoint = load_checkpoint_model(model_path, device)
    if checkpoint.get('checkpoint_type') != PROFILE_CLASSIFIER:
        log.warning(f"checkpoint_type is '{checkpoint.get('checkpoint_type')}', expected '{PROFILE_CLASSIFIER}'")
    log.info(format_checkpoint_info(checkpoint, label="Profiles"))

    log.info("\nLoading test dataset...")
    test_dataset = ProfileDataset(data_dir, 'test', profile_names, return_id=True)
    test_dist = _load_test_split(test_dataset, "Test profile distribution")

    log.info("\n" + _RULE)
    log.info("PROFILE CLASSIFIER EVALUATION ON TEST SET")
    log.info(_RULE)

    metrics, seen = evaluate(model, eval_loader(test_dataset, checkpoint), device, profile_names)

    log.info("\n" + format_metrics(metrics, profile_names))
    log.info(f"\nSeen samples: {seen}")

    report = {
        'timestamp': now_iso(),
        'checkpoint_type': PROFILE_CLASSIFIER,
        'dataset_path': str(data_dir),
        'model_path': str(model_path),
        'profiles': profile_names,
        'metrics': metrics,
        'test_size': len(test_dataset),
        'seen_samples': int(seen),
        'profile_distribution': test_dist,
    }
    report_path = write_report(report_path_for(model_path, profile_classifier=True), report)

    success = evaluate_success_criteria(checkpoint_val_accuracy(checkpoint), metrics, entity="profile")
    log.info(format_success_criteria(success, ascii_only=True))
    return {
        "model_path": str(model_path),
        "dataset_path": str(data_dir),
        "report_path": str(report_path),
        "profiles": profile_names,
        "test_size": len(test_dataset),
        "seen_samples": int(seen),
        **metrics_summary(metrics),
        "success_criteria_passed": success.passed,
    }
