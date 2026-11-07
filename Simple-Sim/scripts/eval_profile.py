#!/usr/bin/env python3
"""Evaluate trained profile classifier on test set.

Usage:
    .venv/bin/python scripts/eval_profile.py --data outputs/sim_data/runs/run_profile_cls --model outputs/models/profile_classifier_v1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import add_data_arg, add_json_flag, add_model_arg, run_cli
from simple_sim.devices import default_device_name, resolve_device
from simple_sim.data_loader import ProfileDataset
from simple_sim.eval_utils import checkpoint_val_accuracy, eval_loader, metrics_summary, format_checkpoint_info, format_distribution
from simple_sim.io_utils import atomic_write_json, now_iso
from simple_sim.metrics import evaluate_success_criteria, format_success_criteria, format_metrics
from simple_sim.models import load_checkpoint_model
from scripts.eval import evaluate


def run_eval_profile(data_dir: Path, model_path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Evaluate a profile classifier on the test split and write ``report_profile_classifier_<stem>.json``."""
    data_dir = Path(data_dir)
    model_path = Path(model_path)
    device = resolve_device(device)

    print("Profile classifier evaluation")
    print(f"  Dataset: {data_dir}")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")

    print("\nLoading model checkpoint...")
    model, profile_names, checkpoint = load_checkpoint_model(model_path, device)
    if checkpoint.get('checkpoint_type') != 'profile_classifier':
        print(f"WARNING: checkpoint_type is '{checkpoint.get('checkpoint_type')}', expected 'profile_classifier'")
    print(format_checkpoint_info(checkpoint, label="Profiles"))

    print("\nLoading test dataset...")
    test_dataset = ProfileDataset(data_dir, 'test', profile_names, return_id=True)
    print(f"Test samples: {len(test_dataset)}")
    test_dist = test_dataset.get_class_distribution()
    print(format_distribution("Test profile distribution", test_dist))

    print("\n" + "=" * 60)
    print("PROFILE CLASSIFIER EVALUATION ON TEST SET")
    print("=" * 60)

    metrics, seen = evaluate(model, eval_loader(test_dataset, checkpoint), device, profile_names)

    print("\n" + format_metrics(metrics, profile_names))
    print(f"\nSeen samples: {seen}")

    report = {
        'timestamp': now_iso(),
        'checkpoint_type': 'profile_classifier',
        'dataset_path': str(data_dir),
        'model_path': str(model_path),
        'profiles': profile_names,
        'metrics': metrics,
        'test_size': len(test_dataset),
        'seen_samples': int(seen),
        'profile_distribution': test_dist,
    }
    report_path = model_path.parent / f'report_profile_classifier_{model_path.stem}.json'
    atomic_write_json(report_path, report)
    print(f"\nReport saved to: {report_path}")

    success = evaluate_success_criteria(checkpoint_val_accuracy(checkpoint), metrics, entity="profile")
    print(format_success_criteria(success, ascii_only=True))
    passed = success.passed
    return {
        "model_path": str(model_path),
        "dataset_path": str(data_dir),
        "report_path": str(report_path),
        "profiles": profile_names,
        "test_size": len(test_dataset),
        "seen_samples": int(seen),
        **metrics_summary(metrics),
        "success_criteria_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate profile classifier on test set')
    add_data_arg(parser, 'Path to mixed-profile dataset directory')
    add_model_arg(parser, 'Path to trained profile classifier checkpoint')
    parser.add_argument('--device', type=str, default=default_device_name(), help='Device (cuda/cpu)')
    add_json_flag(parser)
    args = parser.parse_args()

    run_cli("eval_profile", lambda: run_eval_profile(Path(args.data), Path(args.model), args.device), args.json)


if __name__ == '__main__':
    main()
