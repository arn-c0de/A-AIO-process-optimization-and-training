#!/usr/bin/env python3
"""Evaluate trained model on frozen test set."""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.data_loader import ROIDataset
from simple_sim.metrics import check_success_criteria, compute_metrics, format_metrics
from simple_sim.models import build_resnet18
from simple_sim.telemetry import emit
from simple_sim.manifest import check_checkpoint_profile, read_dataset_manifest, resolve_dataset_profile
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

    print("Evaluation configuration:")
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
    dataset_profile_id, dataset_profile_hash = resolve_dataset_profile(manifest)

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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    print(f"  Classes: {class_names}")
    print(f"  Trained epoch: {checkpoint['epoch']}")
    print(f"  Val accuracy: {checkpoint['val_accuracy']:.4f}")
    print(f"  Val F1: {checkpoint['val_f1']:.4f}")

    # GUARD: Validate component profile match
    ckpt_profile_id = check_checkpoint_profile(
        checkpoint, dataset_profile_id, dataset_profile_hash,
        action="evaluate model on",
        allow_mismatch=bool(args.allow_profile_mismatch),
        override_verb="Evaluating",
    )
    if ckpt_profile_id is not None:
        print(f"  Profile: {ckpt_profile_id}")

    emit("eval_start", dataset_dir=str(data_dir), model_path=str(model_path), device=str(device))

    # Create model
    print("\nInitializing model...")
    model = build_resnet18(num_classes)
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
        'split': 'test',
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
    check_success_criteria(checkpoint['val_accuracy'], metrics, entity="class")


if __name__ == '__main__':
    main()
