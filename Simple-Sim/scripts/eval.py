#!/usr/bin/env python3
"""Evaluate trained model on frozen test set."""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.data_loader import ROIDataset
from simple_sim.metrics import compute_metrics, format_metrics
from simple_sim.telemetry import emit
from simple_sim.manifest import read_dataset_manifest
from simple_sim.model_bundle import bundle_checkpoint_path


def evaluate(model, loader, device, class_names, critical_classes=None):
    """Evaluate model on test set.

    Args:
        model: Neural network model
        loader: DataLoader
        device: Device (cuda/cpu)
        class_names: List of class names

    Returns:
        Metrics dictionary
    """
    model.eval()

    all_preds = []
    all_labels = []
    ema_ips = None
    num_samples = 0

    with torch.no_grad():
        progress = tqdm(loader, desc='Evaluating')
        last_img = None
        for batch in progress:
            if len(batch) == 4:
                images, labels, _, image_paths = batch
                last_img = image_paths[-1]
            elif len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            t0 = time.perf_counter()
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            dt = max(1e-9, time.perf_counter() - t0)
            bs = int(images.shape[0])
            num_samples += bs
            ips = bs / dt
            ema_ips = ips if ema_ips is None else (0.9 * ema_ips + 0.1 * ips)
            postfix = {'img/s': f'{ema_ips:.1f}'}
            if last_img is not None:
                postfix['last_img'] = str(last_img).split('/')[-1]
            progress.set_postfix(postfix)
            emit(
                "eval_batch",
                img_per_s=float(ema_ips),
                last_image_path=str(last_img) if last_img is not None else None,
                batch_size=int(images.shape[0]),
            )

            # Collect predictions
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        class_names,
        critical_classes=critical_classes
    )

    return metrics, num_samples


def main():
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

    args = parser.parse_args()

    data_dir = Path(args.data)
    model_path = Path(args.model)
    device = torch.device(args.device)

    print(f"Evaluation configuration:")
    print(f"  Dataset: {data_dir}")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")

    # GUARD: Load dataset manifest
    manifest_path = data_dir / 'dataset_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path}\n"
            f"This dataset was created before profile support.\n"
            f"Regenerate or use backfill tool:\n"
            f"  .venv/bin/python tools/backfill_manifest.py --data {data_dir}"
        )

    manifest = read_dataset_manifest(manifest_path)
    mver = int(manifest.get('manifest_version', 1) or 1)
    if mver == 1:
        dataset_profile_id = manifest['component_profile']['profile_id']
        dataset_profile_hash = manifest['component_profile']['profile_hash']
    elif mver == 2:
        dataset_profile_id = "multi"
        dataset_profile_hash = "multi"
    else:
        raise ValueError(f"Unsupported dataset manifest_version: {mver}")

    # Multi-model bundle support: --model can be a directory containing per-profile checkpoints.
    if model_path.exists() and model_path.is_dir():
        resolved = bundle_checkpoint_path(model_path, dataset_profile_id, kind="best")
        if not resolved.exists():
            raise FileNotFoundError(
                f"Multi-model bundle has no checkpoint for profile '{dataset_profile_id}':\n"
                f"  bundle: {model_path}\n"
                f"  expected: {resolved}\n"
                f"Train this profile into the bundle first."
            )
        model_path = resolved

    # Load checkpoint
    print("\nLoading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    print(f"  Classes: {class_names}")
    print(f"  Trained epoch: {checkpoint['epoch']}")
    print(f"  Val accuracy: {checkpoint['val_accuracy']:.4f}")
    print(f"  Val F1: {checkpoint['val_f1']:.4f}")

    # GUARD: Validate component profile match
    ckpt_profile = checkpoint.get('component_profile')
    if ckpt_profile:
        ckpt_profile_id = ckpt_profile.get('profile_id')
        ckpt_profile_hash = ckpt_profile.get('profile_hash')

        # HARD FAIL by default: Profile ID mismatch
        if ckpt_profile_id and ckpt_profile_id != dataset_profile_id:
            if not args.allow_profile_mismatch:
                raise ValueError(
                    f"Component profile mismatch:\n"
                    f"  Model trained on: {ckpt_profile_id}\n"
                    f"  Dataset profile:  {dataset_profile_id}\n"
                    f"Cannot evaluate model on different component type.\n\n"
                    f"To override (NOT recommended): pass --allow-profile-mismatch"
                )
            print("WARNING: Evaluating with component profile mismatch (--allow-profile-mismatch).")
            print(f"  Model trained on: {ckpt_profile_id}")
            print(f"  Dataset profile:  {dataset_profile_id}")

        # WARNING: Profile hash mismatch
        if ckpt_profile_hash and ckpt_profile_hash != dataset_profile_hash:
            print(f"WARNING: Profile hash mismatch (different tolerance/geometry)")
            print(f"  Model hash:   {ckpt_profile_hash[:72]}...")
            print(f"  Dataset hash: {dataset_profile_hash[:72]}...")

        print(f"  Profile: {ckpt_profile_id}")
    else:
        print(f"  WARNING: Model lacks profile metadata (legacy checkpoint)")

    emit("eval_start", dataset_dir=str(data_dir), model_path=str(model_path), device=str(device))

    # Create model
    print("\nInitializing model...")
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = ROIDataset(data_dir, 'test', class_names, return_id=True)
    print(f"Test samples: {len(test_dataset)}")

    test_dist = test_dataset.get_class_distribution()
    print("\nTest class distribution:")
    for class_name, count in test_dist.items():
        print(f"  {class_name}: {count}")

    # Create data loader
    batch_size = checkpoint['config']['eval']['batch_size']
    num_workers = checkpoint['config']['eval'].get('num_workers', 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)

    critical_classes = checkpoint['config'].get('eval', {}).get('critical_classes')
    metrics, seen = evaluate(model, test_loader, device, class_names, critical_classes=critical_classes)

    # Print formatted metrics
    print("\n" + format_metrics(metrics, class_names))
    print(f"\nSeen samples: {seen}")
    emit(
        "eval_done",
        seen_samples=int(seen),
        accuracy=float(metrics["accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
    )

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset_path': str(data_dir),
        'model_path': str(model_path),
        'metrics': metrics,
        'test_size': len(test_dataset),
        'seen_samples': int(seen),
        'class_distribution': test_dist
    }

    report_path = model_path.parent / f'report_{model_path.stem}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")

    # Check success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)

    checks = []
    checks.append(("Val accuracy > 70%", checkpoint['val_accuracy'] > 0.70))
    checks.append(("Test accuracy > 70%", metrics['accuracy'] > 0.70))
    checks.append(("No class with 0% recall", all(m['recall'] > 0 for m in metrics['per_class'].values())))

    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
        all_passed = all_passed and passed

    if all_passed:
        print("\n✓ ALL CHECKS PASSED")
    else:
        print("\n✗ SOME CHECKS FAILED")

    print("="*60)


if __name__ == '__main__':
    main()
