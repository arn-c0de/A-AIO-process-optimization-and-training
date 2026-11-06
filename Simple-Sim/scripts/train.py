#!/usr/bin/env python3
"""Train ResNet18 classifier on synthetic dataset."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import PROFILE_MISMATCH_HINT, add_json_flag, run_cli, with_hint
from simple_sim.data_loader import ROIDataset, build_train_transform, make_loader_kwargs
from simple_sim.eval_utils import (
    ThroughputMeter,
    amp_autocast,
    format_distribution,
    require_manifest,
    run_inference,
    to_device,
    unpack_batch,
)
from simple_sim.manifest import ProfileMismatchError, check_checkpoint_profile, hash_file, read_dataset_manifest, resolve_dataset_profile
from simple_sim.metrics import compute_metrics
from simple_sim.model_bundle import bundle_checkpoint_path, upsert_bundle_meta
from simple_sim.models import build_resnet18, load_checkpoint
from simple_sim.rng import resolve_train_seed, seed_everything
from simple_sim.telemetry import emit
from simple_sim.io_utils import load_yaml



def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, int]:
    """Train for one epoch; return ``(average loss, num_samples)``."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    meter = ThroughputMeter()
    last_seen_id = None
    last_seen_img = None

    progress = tqdm(loader, desc='Training', leave=False)
    for batch in progress:
        images, labels, sample_ids, image_paths = unpack_batch(batch)
        if sample_ids is not None:
            last_seen_id = sample_ids[-1]
        if image_paths is not None:
            last_seen_img = image_paths[-1]

        t0 = time.perf_counter()
        images = to_device(images, device)
        labels = to_device(labels, device)

        # Forward/backward pass (AMP on CUDA can reduce VRAM usage).
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            if scaler is None:
                raise RuntimeError("AMP enabled but GradScaler is missing")
            with amp_autocast(True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        bs = int(images.shape[0])
        ema_ips = meter.update(bs, time.perf_counter() - t0)
        total_loss += loss.item()
        num_batches += 1
        num_samples += bs

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
            batch_size=bs,
        )

    return total_loss / max(1, num_batches), num_samples


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    critical_classes: list[str] | None = None,
    *,
    use_amp: bool = False,
) -> tuple[float, dict[str, Any], int]:
    """Validate model; return ``(average loss, metrics, num_samples)``."""
    res = run_inference(
        model, loader, device, criterion=criterion, use_amp=use_amp,
        desc='Validation', leave=False, event="val_batch",
    )
    metrics = compute_metrics(res.y_true, res.y_pred, class_names, critical_classes=critical_classes)
    return float(res.avg_loss or 0.0), metrics, res.num_samples


def load_dataset_config(data_dir: Path) -> dict[str, Any]:
    """Read ``<data_dir>/config.yaml``."""
    return load_yaml(Path(data_dir) / 'config.yaml')


def build_training(
    num_classes: int, train_config: dict[str, Any], device: torch.device, use_amp: bool
) -> tuple[nn.Module, nn.Module, optim.Optimizer, torch.amp.GradScaler]:
    """Create ``(model, criterion, optimizer, scaler)`` from the dataset's ``train`` config section.

    ``train.label_smoothing`` (default 0) smooths the training loss only; validation uses plain
    cross-entropy so val losses stay comparable between runs.
    """
    print("\nInitializing ResNet18...")
    model = build_resnet18(num_classes, pretrained=train_config.get('pretrained', True))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_config.get('label_smoothing', 0.0) or 0.0))
    optimizer = optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    return model, criterion, optimizer, scaler


LR_FLOOR = 0.02  # cosine schedule ends at this fraction of train.lr


def lr_factor(epoch: int, total_epochs: int, *, scheduler: str = "none", warmup_epochs: int = 0) -> float:
    """Learning-rate multiplier for 0-based *epoch* (``train.scheduler``: ``none`` keeps ``train.lr``).

    ``cosine``: linear warm-up over ``train.warmup_epochs`` (first epoch at ``1/warmup``), then
    cosine annealing from ``train.lr`` to ``LR_FLOOR * train.lr`` at the last epoch.
    """
    if scheduler == "none":
        return 1.0
    if scheduler != "cosine":
        raise ValueError(f"train.scheduler must be 'none' or 'cosine', got {scheduler!r}")
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / float(warmup_epochs)
    span = max(1, total_epochs - 1 - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / float(span)))
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: optim.Optimizer, train_config: dict[str, Any], total_epochs: int) -> optim.lr_scheduler.LambdaLR | None:
    """Per-epoch LR scheduler from ``train.scheduler`` / ``train.warmup_epochs``; ``None`` for ``none``."""
    name = str(train_config.get('scheduler', 'none') or 'none')
    if name == 'none':
        return None
    warmup = int(train_config.get('warmup_epochs', 0) or 0)
    return optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: lr_factor(e, total_epochs, scheduler=name, warmup_epochs=warmup))


def fit_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    class_names: list[str],
    *,
    epoch: int,
    total_epochs: int,
    critical_classes: list[str] | None = None,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    emit_events: bool = True,
) -> dict[str, Any]:
    """Train + validate one epoch (0-based *epoch*), print the summary and return its stats."""
    print(f"\nEpoch {epoch + 1}/{total_epochs}")
    if emit_events:
        emit("epoch_start", epoch=int(epoch + 1), epochs=int(total_epochs))

    t_epoch = time.perf_counter()
    train_loss, train_seen = train_epoch(
        model, train_loader, criterion, optimizer, device, use_amp=use_amp, scaler=scaler
    )
    val_loss, val_metrics, val_seen = validate(
        model, val_loader, criterion, device, class_names, critical_classes=critical_classes, use_amp=use_amp
    )
    stats = {
        "epoch": int(epoch + 1),
        "epochs": int(total_epochs),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_metrics['accuracy']),
        "val_macro_f1": float(val_metrics['macro_f1']),
        "train_seen": int(train_seen),
        "val_seen": int(val_seen),
        "epoch_s": float(max(1e-9, time.perf_counter() - t_epoch)),
    }

    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {stats['val_accuracy']:.4f}")
    print(f"  Val Macro F1: {stats['val_macro_f1']:.4f}")
    print(f"  Seen: train={train_seen} val={val_seen}  Epoch time: {stats['epoch_s']:.1f}s")
    if emit_events:
        emit("epoch_end", **stats)
    return stats


def train_model(
    data_dir: Path,
    out: Path,
    *,
    resume: Path | None = None,
    allow_profile_mismatch: bool = False,
    extra_epochs: int | None = None,
    out_mode: str = "best",
    device: str | torch.device = "cpu",
    amp: bool = True,
) -> dict[str, Any]:
    """Train a defect classifier on *data_dir* and write checkpoints to *out*.

    *out* may be a ``.pt`` path or a multi-model bundle directory (existing dir or ``*.bundle``).
    Returns a summary dict (checkpoint paths, best epoch / val metrics, epochs run).
    """
    data_dir = Path(data_dir)
    output_path_arg = Path(out)
    resume_path_arg = Path(resume) if resume else None
    device = torch.device(device)
    use_amp = bool(amp and device.type == "cuda")

    config = load_dataset_config(data_dir)

    # Reproducibility: seed Python/NumPy/PyTorch from the dataset config.
    train_seed = resolve_train_seed(config)
    seed_everything(train_seed)

    # GUARD: Load and validate dataset manifest
    manifest_path = require_manifest(data_dir)
    manifest = read_dataset_manifest(manifest_path)
    # Multi-profile (v2) datasets map to the logical profile "multi" for checkpoint metadata.
    dataset_profile_id, dataset_profile_hash = resolve_dataset_profile(manifest)
    if dataset_profile_id != "multi":
        print(f"Dataset Profile: {dataset_profile_id}")
    else:
        pids = [p.get('profile_id') for p in (manifest.get('component_profiles') or []) if isinstance(p, dict)]
        print(f"Dataset Profile: multi ({', '.join([x for x in pids if x])})")

    # "Multi-model" bundle support:
    # If --out is a directory (or ends with `.bundle`), write per-profile checkpoints into that dir.
    bundle_dir = output_path_arg if (output_path_arg.is_dir() or output_path_arg.name.endswith(".bundle")) else None

    if bundle_dir is not None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        output_path = bundle_checkpoint_path(bundle_dir, dataset_profile_id, kind="best")
        if resume_path_arg is not None and resume_path_arg.is_dir():
            resume_path = bundle_checkpoint_path(resume_path_arg, dataset_profile_id, kind="best")
        else:
            resume_path = resume_path_arg
        print(f"Multi-model bundle: {bundle_dir}")
        print(f"  Out checkpoint:   {output_path}")
        if resume_path:
            print(f"  Resume checkpoint:{resume_path}")
    else:
        output_path = output_path_arg
        resume_path = resume_path_arg

    train_config = config['train']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    loader_kwargs = make_loader_kwargs(train_config.get('num_workers', 0), device)

    # Class names (sorted for consistency)
    class_names = sorted(config['classes'].keys())

    print("Training configuration:")
    print(f"  Dataset: {data_dir}")
    print(f"  Classes: {class_names}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {train_config['lr']}")
    print(f"  Pretrained: {train_config.get('pretrained', True)}")
    print(f"  Seed: {train_seed}")
    print(f"  Device: {device}")
    print(f"  AMP: {use_amp}")
    if resume_path:
        print(f"  Resume: {resume_path}")
        print(f"  Out mode: {out_mode}")

    print("\nLoading datasets...")
    # return_id=True so we can show "last seen" sample IDs live in tqdm.
    train_augment = train_config.get('augment') or {}
    if train_augment:
        print(f"  Train augmentation: {train_augment}")
    train_dataset = ROIDataset(data_dir, 'train', class_names, return_id=True,
                               transform=build_train_transform(train_augment))
    val_dataset = ROIDataset(data_dir, 'val', class_names, return_id=True)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(format_distribution("Train class distribution", train_dataset.get_class_distribution()))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model, criterion, optimizer, scaler = build_training(len(class_names), train_config, device, use_amp)

    start_epoch = 0
    best_val_f1 = 0.0
    best_val_acc: float | None = None
    best_epoch = 0

    if resume_path:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print("\nLoading resume checkpoint...")
        checkpoint = load_checkpoint(resume_path, map_location=device)

        # Existing guard: class names
        ckpt_classes = checkpoint.get('class_names')
        if ckpt_classes and list(ckpt_classes) != list(class_names):
            raise ValueError(f"Resume checkpoint classes do not match dataset config.\n"
                             f"  checkpoint: {ckpt_classes}\n  dataset:    {class_names}")

        # GUARD: component profile validation
        check_checkpoint_profile(
            checkpoint, dataset_profile_id, dataset_profile_hash,
            action="resume training with",
            allow_mismatch=bool(allow_profile_mismatch),
            override_verb="Resuming",
            ckpt_label="Checkpoint profile",
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (KeyError, TypeError, ValueError, RuntimeError) as e:
            # Optimizer state may be missing or incompatible; continue with fresh optimizer state.
            print(f"WARNING: Could not restore optimizer state ({e}); using fresh optimizer state.")
        start_epoch = int(checkpoint.get('epoch', 0))
        # Prefer the tracked best score; older checkpoints only carry the F1 of their own epoch.
        best_val_f1 = float(checkpoint.get('best_val_f1', checkpoint.get('val_f1', 0.0)) or 0.0)
        best_epoch = int(checkpoint.get('best_epoch', checkpoint.get('epoch', 0)))
        print(f"  Resumed from epoch: {start_epoch}")
        print(f"  Resumed best val F1: {best_val_f1:.4f}")

    # Determine how many total epochs to run.
    if extra_epochs is not None:
        if extra_epochs < 0:
            raise ValueError("--extra-epochs must be >= 0")
        total_epochs = start_epoch + int(extra_epochs)
    else:
        total_epochs = int(epochs)
    total_epochs = max(total_epochs, start_epoch)

    # Where to save a "last checkpoint" so resume runs always write something.
    last_path = output_path.with_name(output_path.stem + "_last.pt")

    # LR schedule over the whole run (resume continues it at start_epoch).
    scheduler = build_scheduler(optimizer, train_config, total_epochs)
    if scheduler is not None:
        print(f"  Scheduler: {train_config.get('scheduler')} (warmup {int(train_config.get('warmup_epochs', 0) or 0)} epochs)")
        for _ in range(start_epoch):  # LambdaLR sets lr = base_lr * factor(epoch) on each step
            scheduler.step()
    if float(train_config.get('label_smoothing', 0.0) or 0.0) > 0:
        print(f"  Label smoothing: {train_config['label_smoothing']}")

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    emit("train_start", dataset_dir=str(data_dir), out_model=str(output_path), epochs=int(total_epochs), batch_size=int(batch_size))

    summary: dict[str, Any] = {
        "dataset_path": str(data_dir),
        "best_checkpoint": str(output_path),
        "last_checkpoint": str(last_path),
        "bundle_dir": str(bundle_dir) if bundle_dir is not None else None,
        "class_names": class_names,
        "start_epoch": start_epoch,
        "total_epochs": total_epochs,
    }
    if total_epochs == start_epoch:
        print(f"\nNothing to do: start_epoch={start_epoch} total_epochs={total_epochs}")
        return {**summary, "epochs_run": 0, "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                "best_val_accuracy": best_val_acc, "last_epoch": None}

    critical_classes = config.get('eval', {}).get('critical_classes')
    stats: dict[str, Any] = {}
    for epoch in range(start_epoch, total_epochs):
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        stats = fit_epoch(
            model, train_loader, val_loader, criterion, optimizer, device, class_names,
            epoch=epoch, total_epochs=total_epochs, critical_classes=critical_classes,
            use_amp=use_amp, scaler=scaler,
        )
        val_acc = stats['val_accuracy']
        val_f1 = stats['val_macro_f1']
        if scheduler is not None:
            scheduler.step()

        # Decide "best" before building the checkpoint so it carries the best-so-far score.
        if out_mode == "last":
            is_best = True
        else:
            # Always write --out at least once, even if no epoch beats the resumed/initial score.
            is_best = val_f1 > best_val_f1 or not output_path.exists()
        if is_best:
            best_val_f1 = float(val_f1)
            best_val_acc = float(val_acc)
            best_epoch = int(epoch + 1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_accuracy': val_acc,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'class_names': class_names,
            'config': config,
            # Profile metadata for validation
            'component_profile': {
                'profile_id': dataset_profile_id,
                'profile_hash': dataset_profile_hash,
            },
            'dataset_manifest_hash': hash_file(manifest_path),
            'trained_on_dataset': str(data_dir),
        }

        # Always save "last" so resume runs are observable.
        torch.save(ckpt, last_path)
        if is_best:
            if out_mode != "last":
                print(f"  ✓ New best model (F1: {best_val_f1:.4f})")
            torch.save(ckpt, output_path)

        # Bundle bookkeeping: keep a lightweight index of which profiles exist.
        if bundle_dir is not None:
            try:
                if output_path.exists():
                    upsert_bundle_meta(bundle_dir, dataset_profile_id, output_path)
            except Exception as e:  # noqa: BLE001 - index is best-effort, never fail training
                print(f"WARNING: Failed to update bundle index in {bundle_dir}: {e}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: Epoch {best_epoch}, Val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"Last checkpoint: {last_path}")
    print("=" * 60)
    return {**summary, "epochs_run": total_epochs - start_epoch, "best_epoch": best_epoch,
            "best_val_f1": best_val_f1, "best_val_accuracy": best_val_acc, "last_epoch": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description='Train ResNet18 classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--out', type=str, required=True, help='Output path for trained model')
    parser.add_argument('--resume', type=str, default=None, help='Optional: checkpoint to resume from (.pt)')
    parser.add_argument(
        '--allow-profile-mismatch',
        action='store_true',
        help=(
            "DANGEROUS: allow resuming from a checkpoint trained on a different component profile. "
            "This disables the hard profile_id mismatch guard and continues with a warning."
        ),
    )
    parser.add_argument('--extra-epochs', type=int, default=None,
                        help='Optional: train N additional epochs beyond the resumed epoch (default: use dataset config epochs)')
    parser.add_argument('--out-mode', choices=['best', 'last'], default='best',
                        help="Whether --out points to the best checkpoint ('best') or the latest epoch ('last'). Default: best")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument('--amp', dest='amp', action='store_true', help='Enable mixed precision on CUDA (default).')
    amp_group.add_argument('--no-amp', dest='amp', action='store_false', help='Disable mixed precision.')
    parser.set_defaults(amp=True)
    add_json_flag(parser)
    args = parser.parse_args()

    run_cli("train", with_hint(lambda: train_model(
        Path(args.data), Path(args.out),
        resume=Path(args.resume) if args.resume else None,
        allow_profile_mismatch=args.allow_profile_mismatch,
        extra_epochs=args.extra_epochs,
        out_mode=args.out_mode,
        device=args.device,
        amp=args.amp,
    ), ProfileMismatchError, PROFILE_MISMATCH_HINT), args.json)


if __name__ == '__main__':
    main()
