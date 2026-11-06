#!/usr/bin/env python3
"""Evaluate trained model on frozen test set."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import add_json_flag
from simple_sim.data_loader import ROIDataset
from simple_sim.eval_utils import (
    dataset_profile,
    eval_loader,
    metrics_summary,
    print_checkpoint_info,
    print_distribution,
    resolve_bundle_checkpoint,
    run_cli,
    run_inference,
)
from simple_sim.io_utils import atomic_write_json
from simple_sim.manifest import check_checkpoint_profile
from simple_sim.metrics import check_success_criteria, compute_metrics, format_metrics
from simple_sim.models import load_checkpoint_model
from simple_sim.telemetry import emit


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    critical_classes: list[str] | None = None,
) -> tuple[dict[str, Any], int]:
    """Evaluate *model* on *loader*; return ``(metrics, num_samples)``."""
    res = run_inference(model, loader, device, desc="Evaluating", event="eval_batch")
    metrics = compute_metrics(res.y_true, res.y_pred, class_names, critical_classes=critical_classes)
    return metrics, res.num_samples


def run_eval(
    data_dir: Path,
    model_path: Path,
    device: str | torch.device = "cpu",
    *,
    allow_profile_mismatch: bool = False,
) -> dict[str, Any]:
    """Evaluate a checkpoint (or bundle dir) on the dataset's test split and write ``report_<stem>.json``.

    Returns a summary dict (report path, metrics, success-criteria result, ...).
    """
    data_dir = Path(data_dir)
    model_path = Path(model_path)
    device = torch.device(device)

    print("Evaluation configuration:")
    print(f"  Dataset: {data_dir}")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")

    dataset_profile_id, dataset_profile_hash = dataset_profile(data_dir)
    # Multi-model bundle support: --model can be a directory containing per-profile checkpoints.
    model_path = resolve_bundle_checkpoint(
        model_path, dataset_profile_id, hint="Train this profile into the bundle first."
    )

    print("\nLoading model checkpoint...")
    model, class_names, checkpoint = load_checkpoint_model(model_path, device)
    print_checkpoint_info(checkpoint)

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
    print_distribution("Test class distribution", test_dist)

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    critical_classes = checkpoint['config'].get('eval', {}).get('critical_classes')
    metrics, seen = evaluate(
        model, eval_loader(test_dataset, checkpoint), device, class_names, critical_classes=critical_classes
    )

    print("\n" + format_metrics(metrics, class_names))
    print(f"\nSeen samples: {seen}")
    emit("eval_done", seen_samples=int(seen), accuracy=float(metrics["accuracy"]), macro_f1=float(metrics["macro_f1"]))

    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset_path': str(data_dir),
        'model_path': str(model_path),
        'split': 'test',
        'metrics': metrics,
        'test_size': len(test_dataset),
        'seen_samples': int(seen),
        'class_distribution': test_dist,
    }
    report_path = model_path.parent / f'report_{model_path.stem}.json'
    atomic_write_json(report_path, report)
    print(f"\nReport saved to: {report_path}")

    passed = check_success_criteria(checkpoint['val_accuracy'], metrics, entity="class")
    return {
        "model_path": str(model_path),
        "dataset_path": str(data_dir),
        "report_path": str(report_path),
        "test_size": len(test_dataset),
        "seen_samples": int(seen),
        **metrics_summary(metrics),
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
    add_json_flag(parser)
    args = parser.parse_args()

    run_cli("eval", lambda: run_eval(
        Path(args.data), Path(args.model), args.device, allow_profile_mismatch=args.allow_profile_mismatch,
    ), args.json)


if __name__ == '__main__':
    main()
