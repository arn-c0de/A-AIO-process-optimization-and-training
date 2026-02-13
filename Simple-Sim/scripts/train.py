#!/usr/bin/env python3
"""Train ResNet18 classifier on synthetic dataset."""

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
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.data_loader import ROIDataset
from simple_sim.metrics import compute_metrics, format_metrics
from simple_sim.telemetry import emit


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch.

    Args:
        model: Neural network model
        loader: DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)

    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    ema_ips = None
    last_seen_id = None

    progress = tqdm(loader, desc='Training', leave=False)
    last_seen_img = None
    for batch in progress:
        if len(batch) == 4:
            images, labels, sample_ids, image_paths = batch
            last_seen_id = sample_ids[-1]
            last_seen_img = image_paths[-1]
        elif len(batch) == 3:
            images, labels, sample_ids = batch
            last_seen_id = sample_ids[-1]
        else:
            images, labels = batch

        t0 = time.perf_counter()
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dt = max(1e-9, time.perf_counter() - t0)
        bs = int(images.shape[0])
        ips = bs / dt
        ema_ips = ips if ema_ips is None else (0.9 * ema_ips + 0.1 * ips)

        total_loss += loss.item()
        num_batches += 1
        num_samples += int(images.shape[0])

        postfix = {'loss': f'{loss.item():.4f}', 'img/s': f'{ema_ips:.1f}'}
        if last_seen_id is not None:
            postfix['last_id'] = str(last_seen_id).split('/')[-1]
        if last_seen_img is not None:
            postfix['last_img'] = str(last_seen_img).split('/')[-1]
        progress.set_postfix(postfix)
        # Keep this lightweight; GUIs can compute their own smoothing.
        emit(
            "train_batch",
            loss=float(loss.item()),
            img_per_s=float(ema_ips),
            last_id=str(last_seen_id) if last_seen_id is not None else None,
            last_image_path=str(last_seen_img) if last_seen_img is not None else None,
            batch_size=int(images.shape[0]),
        )

    return total_loss / max(1, num_batches), num_samples


def validate(model, loader, criterion, device, class_names, critical_classes=None):
    """Validate model.

    Args:
        model: Neural network model
        loader: DataLoader
        criterion: Loss function
        device: Device (cuda/cpu)
        class_names: List of class names

    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    ema_ips = None

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress = tqdm(loader, desc='Validation', leave=False)
        for batch in progress:
            if len(batch) == 4:
                images, labels, _, _ = batch
            elif len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            t0 = time.perf_counter()
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            dt = max(1e-9, time.perf_counter() - t0)
            bs = int(images.shape[0])
            ips = bs / dt
            ema_ips = ips if ema_ips is None else (0.9 * ema_ips + 0.1 * ips)
            progress.set_postfix({'loss': f'{loss.item():.4f}', 'img/s': f'{ema_ips:.1f}'})
            emit(
                "val_batch",
                loss=float(loss.item()),
                img_per_s=float(ema_ips),
            )

            total_loss += loss.item()
            num_batches += 1
            num_samples += int(images.shape[0])

            # Collect predictions
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Compute metrics
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        class_names,
        critical_classes=critical_classes
    )

    return avg_loss, metrics, num_samples


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--out', type=str, required=True, help='Output path for trained model')
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

    # Extract training parameters
    train_config = config['train']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    lr = train_config['lr']
    weight_decay = train_config['weight_decay']
    pretrained = train_config.get('pretrained', True)
    num_workers = train_config.get('num_workers', 0)

    # Class names (sorted for consistency)
    class_names = sorted(config['classes'].keys())
    num_classes = len(class_names)

    print(f"Training configuration:")
    print(f"  Dataset: {data_dir}")
    print(f"  Classes: {class_names}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Num workers: {num_workers}")
    print(f"  Device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    # return_id=True so we can show "last seen" sample IDs live in tqdm.
    train_dataset = ROIDataset(data_dir, 'train', class_names, return_id=True)
    val_dataset = ROIDataset(data_dir, 'val', class_names, return_id=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Print class distributions
    train_dist = train_dataset.get_class_distribution()
    print("\nTrain class distribution:")
    for class_name, count in train_dist.items():
        print(f"  {class_name}: {count}")

    # Create data loaders
    # num_workers>0 can fail in restricted environments (semaphores / shared memory perms).
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create model
    print("\nInitializing ResNet18...")
    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            # Pretrained weights may require network access on first run.
            print(f"WARNING: Failed to load pretrained weights ({e}). Falling back to random init.")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    # Modify final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    emit("train_start", dataset_dir=str(data_dir), out_model=str(output_path), epochs=int(epochs), batch_size=int(batch_size))

    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        emit("epoch_start", epoch=int(epoch + 1), epochs=int(epochs))

        # Train
        t_epoch = time.perf_counter()
        train_loss, train_seen = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        critical_classes = config.get('eval', {}).get('critical_classes')
        val_loss, val_metrics, val_seen = validate(
            model, val_loader, criterion, device, class_names, critical_classes=critical_classes
        )

        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['macro_f1']
        epoch_s = max(1e-9, time.perf_counter() - t_epoch)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Val Macro F1: {val_f1:.4f}")
        print(f"  Seen: train={train_seen} val={val_seen}  Epoch time: {epoch_s:.1f}s")
        emit(
            "epoch_end",
            epoch=int(epoch + 1),
            epochs=int(epochs),
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            val_accuracy=float(val_acc),
            val_macro_f1=float(val_f1),
            train_seen=int(train_seen),
            val_seen=int(val_seen),
            epoch_s=float(epoch_s),
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            print(f"  ✓ New best model (F1: {best_val_f1:.4f})")

            # Save checkpoint
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'class_names': class_names,
                'config': config
            }, output_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model: Epoch {best_epoch}, Val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
