#!/usr/bin/env python3
"""Train ResNet18 profile (component type) classifier on mixed-profile dataset.

Usage:
    .venv/bin/python scripts/train_profile.py --data outputs/sim_data/runs/run_profile_cls --out outputs/models/profile_classifier_v1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import add_amp_flags, add_data_arg, add_json_flag, add_out_arg, run_cli
from simple_sim.dataset_files import LABELS_JSONL
from simple_sim.devices import default_device_name, resolve_device
from simple_sim.data_loader import ProfileDataset, make_loader_kwargs
from simple_sim.eval_utils import format_distribution, require_manifest
from simple_sim.manifest import hash_file, read_dataset_manifest
from simple_sim.rng import resolve_train_seed, seed_everything
from simple_sim.schema import LabelRow, read_jsonl
from simple_sim.telemetry import emit
from scripts.train import build_training, fit_epoch, load_dataset_config


def dataset_profile_names(data_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Sorted profile IDs of a dataset: from a v2 manifest, else scanned from labels.jsonl."""
    if manifest.get('manifest_version', 1) == 2:
        return sorted(p['profile_id'] for p in manifest['component_profiles'])
    label_rows = read_jsonl(Path(data_dir) / LABELS_JSONL, LabelRow)
    return sorted({r.profile_id for r in label_rows if r.profile_id})


def train_profile_model(
    data_dir: Path, out: Path, *, device: str | torch.device = "cpu", amp: bool = True
) -> dict[str, Any]:
    """Train a profile classifier on a mixed-profile dataset; return a summary dict."""
    data_dir = Path(data_dir)
    output_path = Path(out)
    device = resolve_device(device)
    use_amp = bool(amp and device.type == "cuda")

    config = load_dataset_config(data_dir)

    # Reproducibility: seed Python/NumPy/PyTorch from the dataset config.
    train_seed = resolve_train_seed(config)
    seed_everything(train_seed)

    manifest_path = require_manifest(data_dir)
    profile_names = dataset_profile_names(data_dir, read_dataset_manifest(manifest_path))
    if len(profile_names) < 2:
        raise ValueError(f"Profile classifier requires 2+ profiles, found: {profile_names}")

    print("Profile classifier training")
    print(f"  Dataset: {data_dir}")
    print(f"  Profiles ({len(profile_names)}): {profile_names}")
    print(f"  Device: {device}")
    print(f"  AMP: {use_amp}")
    print(f"  Seed: {train_seed}")

    train_config = config['train']
    epochs = int(train_config['epochs'])
    batch_size = train_config['batch_size']
    loader_kwargs = make_loader_kwargs(train_config.get('num_workers', 0), device)

    print("\nLoading datasets...")
    train_dataset = ProfileDataset(data_dir, 'train', profile_names, return_id=True)
    val_dataset = ProfileDataset(data_dir, 'val', profile_names, return_id=True)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_dist = train_dataset.get_class_distribution()
    print(format_distribution("Train profile distribution", train_dist))
    for name, count in train_dist.items():
        if count == 0:
            raise ValueError(f"Profile '{name}' has 0 training samples. Check dataset generation.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model, criterion, optimizer, scaler, frozen = build_training(len(profile_names), train_config, device, use_amp)

    best_val_f1 = 0.0
    best_val_acc: float | None = None
    best_epoch = 0
    last_path = output_path.with_name(output_path.stem + "_last.pt")

    print("\n" + "=" * 60)
    print("PROFILE CLASSIFIER TRAINING")
    print("=" * 60)
    emit("train_profile_start", dataset_dir=str(data_dir), out_model=str(output_path),
         epochs=epochs, profiles=profile_names)

    stats: dict[str, Any] = {}
    for epoch in range(epochs):
        stats = fit_epoch(
            model, train_loader, val_loader, criterion, optimizer, device, profile_names,
            epoch=epoch, total_epochs=epochs, use_amp=use_amp, scaler=scaler, emit_events=False, frozen=frozen,
        )
        val_f1 = stats['val_macro_f1']

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_accuracy': stats['val_accuracy'],
            'class_names': profile_names,
            'config': config,
            'checkpoint_type': 'profile_classifier',
            'dataset_manifest_hash': hash_file(manifest_path),
            'trained_on_dataset': str(data_dir),
        }
        torch.save(ckpt, last_path)

        # Always write --out at least once, even if val F1 never rises above 0.
        if val_f1 > best_val_f1 or not output_path.exists():
            best_val_f1 = float(val_f1)
            best_val_acc = float(stats['val_accuracy'])
            best_epoch = int(epoch + 1)
            print(f"  New best model (F1: {best_val_f1:.4f})")
            torch.save(ckpt, output_path)

    print("\n" + "=" * 60)
    print("PROFILE CLASSIFIER TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: Epoch {best_epoch}, Val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {output_path}")
    print("=" * 60)
    return {
        "dataset_path": str(data_dir),
        "best_checkpoint": str(output_path),
        "last_checkpoint": str(last_path),
        "profiles": profile_names,
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_accuracy": best_val_acc,
        "last_epoch": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Train profile classifier')
    add_data_arg(parser, 'Path to mixed-profile dataset directory')
    add_out_arg(parser, 'Output path for trained model (.pt)')
    parser.add_argument('--device', type=str, default=default_device_name(), help='Device (cuda/cpu)')
    add_amp_flags(parser)
    add_json_flag(parser)
    args = parser.parse_args()

    run_cli("train_profile", lambda: train_profile_model(
        Path(args.data), Path(args.out), device=args.device, amp=args.amp,
    ), args.json)


if __name__ == '__main__':
    main()
