#!/usr/bin/env python3
"""Evaluate trained profile classifier on test set.

Usage:
    .venv/bin/python scripts/eval_profile.py --data outputs/sim_data/runs/run_profile_cls --model outputs/models/profile_classifier_v1.pt
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.data_loader import ProfileDataset
from simple_sim.metrics import compute_metrics, format_metrics
from simple_sim.telemetry import emit
from simple_sim.manifest import read_dataset_manifest
from simple_sim.schema import read_jsonl, LabelRow
from scripts.eval import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate profile classifier on test set')
    parser.add_argument('--data', type=str, required=True, help='Path to mixed-profile dataset directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained profile classifier checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_path = Path(args.model)
    device = torch.device(args.device)

    print(f"Profile classifier evaluation")
    print(f"  Dataset: {data_dir}")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")

    # Load checkpoint
    print("\nLoading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)

    if checkpoint.get('checkpoint_type') != 'profile_classifier':
        print(f"WARNING: checkpoint_type is '{checkpoint.get('checkpoint_type')}', expected 'profile_classifier'")

    profile_names = checkpoint['class_names']
    num_classes = len(profile_names)

    print(f"  Profiles: {profile_names}")
    print(f"  Trained epoch: {checkpoint['epoch']}")
    print(f"  Val accuracy: {checkpoint['val_accuracy']:.4f}")
    print(f"  Val F1: {checkpoint['val_f1']:.4f}")

    # Create model
    print("\nInitializing model...")
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = ProfileDataset(data_dir, 'test', profile_names, return_id=True)
    print(f"Test samples: {len(test_dataset)}")

    test_dist = test_dataset.get_class_distribution()
    print("\nTest profile distribution:")
    for name, count in test_dist.items():
        print(f"  {name}: {count}")

    batch_size = checkpoint['config']['eval']['batch_size']
    num_workers = checkpoint['config']['eval'].get('num_workers', 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Evaluate
    print("\n" + "=" * 60)
    print("PROFILE CLASSIFIER EVALUATION ON TEST SET")
    print("=" * 60)

    metrics, seen = evaluate(model, test_loader, device, profile_names)

    print("\n" + format_metrics(metrics, profile_names))
    print(f"\nSeen samples: {seen}")

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
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
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")

    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)

    checks = []
    checks.append(("Val accuracy > 70%", checkpoint['val_accuracy'] > 0.70))
    checks.append(("Test accuracy > 70%", metrics['accuracy'] > 0.70))
    checks.append(("No profile with 0% recall", all(m['recall'] > 0 for m in metrics['per_class'].values())))

    all_passed = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {check_name}")
        all_passed = all_passed and passed

    if all_passed:
        print("\nALL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")

    print("=" * 60)


if __name__ == '__main__':
    main()
