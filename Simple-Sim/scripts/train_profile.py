#!/usr/bin/env python3
"""Train ResNet18 profile (component type) classifier on mixed-profile dataset.

Usage:
    .venv/bin/python scripts/train_profile.py --data outputs/sim_data/runs/run_profile_cls --out outputs/models/profile_classifier_v1.pt
"""

import argparse
import sys
from pathlib import Path
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.data_loader import ProfileDataset
from simple_sim.metrics import compute_metrics, format_metrics
from simple_sim.telemetry import emit
from simple_sim.manifest import read_dataset_manifest, hash_file
from simple_sim.schema import read_jsonl, LabelRow
from scripts.train import train_epoch, validate


def main():
    parser = argparse.ArgumentParser(description='Train profile classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to mixed-profile dataset directory')
    parser.add_argument('--out', type=str, required=True, help='Output path for trained model (.pt)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_path = Path(args.out)
    device = torch.device(args.device)

    # Load config from dataset
    config_path = data_dir / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load manifest and get profile list
    manifest_path = data_dir / 'dataset_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    manifest = read_dataset_manifest(manifest_path)

    # Get profile names from manifest (v2) or from labels
    if manifest.get('manifest_version', 1) == 2:
        profile_names = sorted([p['profile_id'] for p in manifest['component_profiles']])
    else:
        # Fallback: scan labels for unique profile_ids
        label_rows = read_jsonl(data_dir / 'labels.jsonl', LabelRow)
        profile_names = sorted(set(r.profile_id for r in label_rows if r.profile_id))

    if len(profile_names) < 2:
        raise ValueError(f"Profile classifier requires 2+ profiles, found: {profile_names}")

    num_classes = len(profile_names)
    print(f"Profile classifier training")
    print(f"  Dataset: {data_dir}")
    print(f"  Profiles ({num_classes}): {profile_names}")
    print(f"  Device: {device}")

    # Extract training parameters
    train_config = config['train']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    lr = train_config['lr']
    weight_decay = train_config['weight_decay']
    pretrained = train_config.get('pretrained', True)
    num_workers = train_config.get('num_workers', 0)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ProfileDataset(data_dir, 'train', profile_names, return_id=True)
    val_dataset = ProfileDataset(data_dir, 'val', profile_names, return_id=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_dist = train_dataset.get_class_distribution()
    print("\nTrain profile distribution:")
    for name, count in train_dist.items():
        print(f"  {name}: {count}")

    # Validate distribution
    for name, count in train_dist.items():
        if count == 0:
            raise ValueError(f"Profile '{name}' has 0 training samples. Check dataset generation.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create model
    print("\nInitializing ResNet18...")
    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"WARNING: Failed to load pretrained weights ({e}). Falling back to random init.")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_epoch = 0

    print("\n" + "=" * 60)
    print("PROFILE CLASSIFIER TRAINING")
    print("=" * 60)
    emit("train_profile_start", dataset_dir=str(data_dir), out_model=str(output_path),
         epochs=int(epochs), profiles=profile_names)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        t_epoch = time.perf_counter()
        train_loss, train_seen = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, val_seen = validate(model, val_loader, criterion, device, profile_names)

        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['macro_f1']
        epoch_s = max(1e-9, time.perf_counter() - t_epoch)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Val Macro F1: {val_f1:.4f}")
        print(f"  Seen: train={train_seen} val={val_seen}  Epoch time: {epoch_s:.1f}s")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_accuracy': val_acc,
            'class_names': profile_names,
            'config': config,
            'checkpoint_type': 'profile_classifier',
            'dataset_manifest_hash': hash_file(manifest_path),
            'trained_on_dataset': str(data_dir),
        }

        last_path = output_path.with_name(output_path.stem + "_last.pt")
        torch.save(ckpt, last_path)

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_epoch = int(epoch + 1)
            print(f"  New best model (F1: {best_val_f1:.4f})")
            torch.save(ckpt, output_path)

    print("\n" + "=" * 60)
    print("PROFILE CLASSIFIER TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: Epoch {best_epoch}, Val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
