#!/usr/bin/env python3
"""Evaluate trained model on frozen test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import PROFILE_MISMATCH_HINT, add_json_flag, run_cli, with_hint
from simple_sim.data_loader import ROIDataset
from simple_sim.eval_utils import (
    checkpoint_eval_config,
    checkpoint_val_accuracy,
    dataset_profile,
    eval_loader,
    metrics_summary,
    format_checkpoint_info,
    format_distribution,
    resolve_bundle_checkpoint,
    run_inference,
)
from simple_sim.io_utils import atomic_write_json, now_iso
from simple_sim.manifest import ProfileMismatchError, check_checkpoint_profile
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
    *tta*, so the default output is unchanged.

    Returns a summary dict (report path, metrics, success-criteria result, ...).
    """
    data_dir = Path(data_dir)
    requested_model_path = model_path = Path(model_path)
    device = torch.device(device)

    print("Evaluation configuration:")
    print(f"  Dataset: {data_dir}")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    if tta:
        print("  Test-time augmentation: flips + quarter turns (8 views)")

    dataset_profile_id, dataset_profile_hash = dataset_profile(data_dir)
    # Multi-model bundle support: --model can be a directory containing per-profile checkpoints.
    model_path = resolve_bundle_checkpoint(
        model_path, dataset_profile_id, hint="Train this profile into the bundle first."
    )

    print("\nLoading model checkpoint (single model or ensemble)...")
    model, class_names, checkpoint = load_checkpoint_model(model_path, device)
    print(format_checkpoint_info(checkpoint))

    # GUARD: Validate component profile match
    ckpt_profile_id = check_checkpoint_profile(
        checkpoint, dataset_profile_id, dataset_profile_hash,
        action="evaluate model on",
        allow_mismatch=bool(allow_profile_mismatch),
        override_verb="Evaluating",
    )
    if ckpt_profile_id is not None:
        print(f"  Profile: {ckpt_profile_id}")

    emit("eval_start", dataset_dir=str(data_dir), model_path=str(model_path), device=str(device))

    print("\nLoading test dataset...")
    test_dataset = ROIDataset(data_dir, 'test', class_names, return_id=True)
    print(f"Test samples: {len(test_dataset)}")
    test_dist = test_dataset.get_class_distribution()
    print(format_distribution("Test class distribution", test_dist))

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    critical_classes = checkpoint_eval_config(checkpoint).get('critical_classes')
    metrics, seen = evaluate(
        model, eval_loader(test_dataset, checkpoint), device, class_names, critical_classes=critical_classes,
        tta=tta, calibration=calibration,
    )

    print("\n" + format_metrics(metrics, class_names))
    if calibration:
        print(format_calibration(metrics["calibration"]))
    print(f"\nSeen samples: {seen}")
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
    report_path = requested_model_path.parent / f'report_{requested_model_path.stem}.json'
    atomic_write_json(report_path, report)
    print(f"\nReport saved to: {report_path}")

    success = evaluate_success_criteria(checkpoint_val_accuracy(checkpoint), metrics, entity="class")
    print(format_success_criteria(success))
    passed = success.passed
    return {
        "model_path": str(requested_model_path),
        "resolved_model_path": resolved_model_path,
        "dataset_path": str(data_dir),
        "report_path": str(report_path),
        "test_size": len(test_dataset),
        "seen_samples": int(seen),
        **metrics_summary(metrics),
        **({"tta": True} if tta else {}),
        "success_criteria_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument(
        '--allow-profile-mismatch',
        action='store_true',
        help=(
            "DANGEROUS: allow evaluating a checkpoint on a dataset with a different component profile. "
            "This disables the hard profile_id mismatch guard and continues with a warning."
        ),
    )
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--tta', action='store_true',
                        help='Test-time augmentation: average the softmax over horizontal/vertical flips and '
                             '90-degree rotations (8 views; for cardinal-rotation-augmented datasets)')
    parser.add_argument('--calibration', action='store_true',
                        help='Also report calibration: expected calibration error (15 bins), negative '
                             'log-likelihood and Brier score (metrics.calibration in the report)')
    add_json_flag(parser)
    args = parser.parse_args()

    run_cli("eval", with_hint(lambda: run_eval(
        Path(args.data), Path(args.model), args.device, allow_profile_mismatch=args.allow_profile_mismatch,
        tta=args.tta, calibration=args.calibration,
    ), ProfileMismatchError, PROFILE_MISMATCH_HINT), args.json)


if __name__ == '__main__':
    main()
