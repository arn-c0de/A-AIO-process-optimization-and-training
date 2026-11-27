"""Checkpoints of a training run: the saved dict (:func:`make_checkpoint`), where they go
(:func:`checkpoint_targets`, :func:`last_checkpoint_path`) and resuming from one (:func:`restore_resume_state`).

Split out of :mod:`simple_sim.training`, which re-exports the public names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from simple_sim.manifest import check_checkpoint_profile, hash_file
from simple_sim.model_bundle import bundle_checkpoint_path
from simple_sim.models import load_checkpoint

__all__ = [
    "PROFILE_CLASSIFIER",
    "CheckpointMeta",
    "checkpoint_targets",
    "last_checkpoint_path",
    "make_checkpoint",
    "other_dataset",
    "restore_resume_state",
    "same_file",
]

log = logging.getLogger(__name__)

#: ``checkpoint_type`` of profile classifiers (defect classifiers carry no type key).
PROFILE_CLASSIFIER = "profile_classifier"


def last_checkpoint_path(output_path: Path) -> Path:
    """``<stem>_last.pt`` next to *output_path* (written after every epoch)."""
    output_path = Path(output_path)
    return output_path.with_name(output_path.stem + "_last.pt")


@dataclass(frozen=True)
class CheckpointMeta:
    """Per-run metadata every checkpoint of a training run carries (see :func:`make_checkpoint`).

    *profile* is the dataset's ``(profile_id, profile_hash)``
    (:func:`simple_sim.manifest.resolve_dataset_profile`; ``"multi"`` for multi-profile datasets).
    Profile classifiers pass *checkpoint_type* :data:`PROFILE_CLASSIFIER`.
    """

    class_names: list[str]
    config: dict[str, Any]
    profile: tuple[str, str]
    manifest_path: Path
    data_dir: Path
    checkpoint_type: str | None = None


def make_checkpoint(
    meta: CheckpointMeta,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    stats: dict[str, Any],
    best: tuple[float, int] | None = None,
) -> dict[str, Any]:
    """The checkpoint dict saved after 1-based *epoch*.

    *stats* is the :func:`fit_epoch` result, *best* the best-so-far ``(val_f1, epoch)`` that
    resume continues from.
    """
    ckpt: dict[str, Any] = {
        'epoch': int(epoch),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': stats['val_macro_f1'],
        'val_accuracy': stats['val_accuracy'],
    }
    if best is not None:
        ckpt['best_val_f1'], ckpt['best_epoch'] = best
    ckpt['class_names'] = list(meta.class_names)
    ckpt['config'] = meta.config
    if meta.checkpoint_type is not None:
        ckpt['checkpoint_type'] = meta.checkpoint_type
    ckpt['component_profile'] = {'profile_id': meta.profile[0], 'profile_hash': meta.profile[1]}
    ckpt['dataset_manifest_hash'] = hash_file(meta.manifest_path)
    ckpt['trained_on_dataset'] = str(meta.data_dir)
    return ckpt


def checkpoint_targets(output_path_arg: Path, resume_path_arg: Path | None,
                        dataset_profile_id: str) -> tuple[Path | None, Path, Path | None]:
    """``(bundle_dir, output_path, resume_path)``: per-profile checkpoints inside a bundle dir, else the paths as given."""
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
        log.info(f"Multi-model bundle: {bundle_dir}")
        log.info(f"  Out checkpoint:   {output_path}")
        if resume_path:
            log.info(f"  Resume checkpoint:{resume_path}")
    else:
        output_path = output_path_arg
        resume_path = resume_path_arg
    return bundle_dir, output_path, resume_path


def restore_resume_state(resume_path: Path, model: nn.Module, optimizer: optim.Optimizer, *, device: torch.device,
                          class_names: list[str], train_config: dict[str, Any], profile: tuple[str, str | None],
                          allow_profile_mismatch: bool, data_dir: Path | None = None,
                          manifest_path: Path | None = None) -> tuple[int, float, int]:
    """Load *resume_path* into *model* / *optimizer* after the class and profile guards.

    Returns ``(start_epoch, best_val_f1, best_epoch)`` of the resumed run. The best score is only
    carried over when the checkpoint was trained on this dataset (*data_dir*, same manifest): a val
    F1 measured on another val split would keep every epoch on the new data from becoming "best".
    """
    dataset_profile_id, dataset_profile_hash = profile
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    log.info("\nLoading resume checkpoint...")
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
        log.warning(f"Could not restore optimizer state ({e}); using fresh optimizer state.")
    # The optimizer state carries the (possibly decayed) lr / weight_decay of the old run;
    # the dataset config is the source of truth, also as base lr of the LR schedule.
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] = float(train_config['lr'])
        group['weight_decay'] = float(train_config['weight_decay'])
    start_epoch = int(checkpoint.get('epoch', 0))
    # Prefer the tracked best score; older checkpoints only carry the F1 of their own epoch.
    best_val_f1 = float(checkpoint.get('best_val_f1', checkpoint.get('val_f1', 0.0)) or 0.0)
    best_epoch = int(checkpoint.get('best_epoch', checkpoint.get('epoch', 0)))
    log.info(f"  Resumed from epoch: {start_epoch}")
    if other_dataset(checkpoint, data_dir, manifest_path):
        log.info(f"  Checkpoint was trained on other data ({checkpoint.get('trained_on_dataset')}); "
                 f"its best val F1 {best_val_f1:.4f} does not count for this dataset.")
        return start_epoch, 0.0, 0
    log.info(f"  Resumed best val F1: {best_val_f1:.4f}")
    return start_epoch, best_val_f1, best_epoch


def same_file(path: Path, *others: Path) -> bool:
    """True when *path* is one of *others* (resolved)."""
    try:
        target = Path(path).resolve()
        return any(Path(o).resolve() == target for o in others)
    except OSError:
        return False


def other_dataset(checkpoint: dict[str, Any], data_dir: Path | None, manifest_path: Path | None) -> bool:
    """True when *checkpoint* records another dataset (path or manifest hash) than the one being trained on.
    A checkpoint without those entries counts as the same dataset."""
    trained_on = checkpoint.get('trained_on_dataset')
    if data_dir is not None and trained_on:
        try:
            if Path(str(trained_on)).resolve() != Path(data_dir).resolve():
                return True
        except OSError:
            return True
    stored_hash = checkpoint.get('dataset_manifest_hash')
    if manifest_path is not None and stored_hash and Path(manifest_path).exists():
        return str(stored_hash) != str(hash_file(manifest_path))
    return False
