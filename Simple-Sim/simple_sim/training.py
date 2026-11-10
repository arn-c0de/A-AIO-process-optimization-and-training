"""Training of the ResNet18 defect and profile classifiers (plan item L1).

:func:`train_model` trains a defect classifier on a dataset directory, :func:`train_profile_model`
a profile (component type) classifier on a mixed-profile dataset. Both share
:func:`build_training`, :func:`fit_epoch` and :func:`make_checkpoint`, so every checkpoint carries
the same keys (including ``component_profile``; profile classifiers used to omit it, plan X4).

Progress goes to the ``simple_sim.training`` logger (the CLI shows it on stdout via
:func:`simple_sim.cli_utils.configure_cli_logging`), tqdm bars to stderr and per-batch / per-epoch
events to :mod:`simple_sim.telemetry`. Nothing here parses arguments or exits the process.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from simple_sim.data_loader import ProfileDataset, ROIDataset, build_train_transform, make_loader_kwargs
from simple_sim.dataset_files import CONFIG_YAML, LABELS_JSONL
from simple_sim.devices import resolve_device
from simple_sim.eval_utils import (
    ThroughputMeter,
    amp_autocast,
    format_distribution,
    require_manifest,
    run_inference,
    to_device,
    unpack_batch,
)
from simple_sim.io_utils import load_yaml
from simple_sim.manifest import check_checkpoint_profile, hash_file, read_dataset_manifest, resolve_dataset_profile
from simple_sim.metrics import compute_metrics
from simple_sim.model_bundle import bundle_checkpoint_path, upsert_bundle_meta
from simple_sim.models import build_resnet18, freeze_stages, load_checkpoint, trainable_parameters
from simple_sim.rng import resolve_train_seed, seed_everything
from simple_sim.schema import LabelRow, read_jsonl
from simple_sim.telemetry import emit

__all__ = [
    "LR_FLOOR",
    "PROFILE_CLASSIFIER",
    "CheckpointMeta",
    "EpochRun",
    "build_scheduler",
    "build_training",
    "dataset_profile_names",
    "fit_epoch",
    "last_checkpoint_path",
    "load_dataset_config",
    "lr_factor",
    "make_checkpoint",
    "plain_criterion",
    "set_torch_threads",
    "train_epoch",
    "train_model",
    "train_profile_model",
    "validate",
]

log = logging.getLogger(__name__)

#: ``checkpoint_type`` of profile classifiers (defect classifiers carry no type key).
PROFILE_CLASSIFIER = "profile_classifier"

LR_FLOOR = 0.02  # cosine schedule ends at this fraction of train.lr

_RULE = "=" * 60


def _banner(title: str) -> None:
    log.info("\n" + _RULE)
    log.info(title)
    log.info(_RULE)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    frozen: list[nn.Module] | None = None,
) -> tuple[float, int]:
    """Train for one epoch; return ``(average loss, num_samples)``.

    *frozen* modules (``train.freeze``) stay in eval mode so their BatchNorm statistics do not move.
    """
    model.train()
    for m in frozen or ():
        m.eval()
    total_loss = 0.0
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
        total_loss += loss.item() * bs  # sample-weighted: the last, smaller batch counts proportionally
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

    return total_loss / max(1, num_samples), num_samples


def plain_criterion(criterion: nn.Module) -> nn.Module:
    """*criterion* without label smoothing, for validation.

    ``train.label_smoothing`` smooths the training loss only; the validation loss is plain
    cross-entropy so val losses stay comparable between runs with and without smoothing.
    Other criteria are returned unchanged.
    """
    if isinstance(criterion, nn.CrossEntropyLoss) and criterion.label_smoothing > 0:
        return nn.CrossEntropyLoss(weight=criterion.weight, ignore_index=criterion.ignore_index,
                                   reduction=criterion.reduction)
    return criterion


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


@dataclass
class EpochRun:
    """What stays the same for every epoch of one training run: the input of :func:`fit_epoch`.

    Validation uses *val_criterion*, by default :func:`plain_criterion` of *criterion* (no label
    smoothing), so ``val_loss`` is comparable across runs.
    """

    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: nn.Module
    optimizer: optim.Optimizer
    device: torch.device
    class_names: list[str]
    total_epochs: int
    critical_classes: list[str] | None = None
    use_amp: bool = False
    scaler: torch.amp.GradScaler | None = None
    emit_events: bool = True
    frozen: list[nn.Module] | None = None
    val_criterion: nn.Module | None = None

    def __post_init__(self) -> None:
        if self.val_criterion is None:
            self.val_criterion = plain_criterion(self.criterion)


def fit_epoch(run: EpochRun, epoch: int) -> dict[str, Any]:
    """Train + validate one epoch (0-based *epoch*) of *run*, log the summary and return its stats."""
    total_epochs, emit_events = run.total_epochs, run.emit_events
    log.info(f"\nEpoch {epoch + 1}/{total_epochs}")
    if emit_events:
        emit("epoch_start", epoch=int(epoch + 1), epochs=int(total_epochs))

    t_epoch = time.perf_counter()
    train_loss, train_seen = train_epoch(
        run.model, run.train_loader, run.criterion, run.optimizer, run.device,
        use_amp=run.use_amp, scaler=run.scaler, frozen=run.frozen,
    )
    val_loss, val_metrics, val_seen = validate(
        run.model, run.val_loader, run.val_criterion, run.device, run.class_names,
        critical_classes=run.critical_classes, use_amp=run.use_amp,
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

    log.info(f"  Train Loss: {train_loss:.4f}")
    log.info(f"  Val Loss: {val_loss:.4f}")
    log.info(f"  Val Accuracy: {stats['val_accuracy']:.4f}")
    log.info(f"  Val Macro F1: {stats['val_macro_f1']:.4f}")
    log.info(f"  Seen: train={train_seen} val={val_seen}  Epoch time: {stats['epoch_s']:.1f}s")
    if emit_events:
        emit("epoch_end", **stats)
    return stats


# ---------------------------------------------------------------------------
# Setup: config, model, optimizer, schedule, checkpoints
# ---------------------------------------------------------------------------

def load_dataset_config(data_dir: Path) -> dict[str, Any]:
    """Read ``<data_dir>/config.yaml``."""
    return load_yaml(Path(data_dir) / CONFIG_YAML)


def build_training(
    num_classes: int, train_config: dict[str, Any], device: torch.device, use_amp: bool
) -> tuple[nn.Module, nn.Module, optim.Optimizer, torch.amp.GradScaler, list[nn.Module]]:
    """Create ``(model, criterion, optimizer, scaler, frozen_modules)`` from the dataset's ``train`` section.

    ``train.label_smoothing`` (default 0) smooths the training loss only; validation uses plain
    cross-entropy so val losses stay comparable between runs. ``train.freeze`` (default none) lists
    ResNet stages (``stem``, ``layer1`` .. ``layer4``) that keep their (pretrained) weights: the
    optimizer only sees the remaining parameters and backward stops at the first trainable layer.
    """
    log.info("\nInitializing ResNet18...")
    model = build_resnet18(num_classes, pretrained=train_config.get('pretrained', True))
    frozen = freeze_stages(model, train_config.get('freeze'))
    if frozen:
        n_train = sum(p.numel() for p in trainable_parameters(model))
        n_all = sum(p.numel() for p in model.parameters())
        log.info(f"  Frozen stages: {list(train_config['freeze'])} ({n_train / 1e6:.2f} of {n_all / 1e6:.2f} M parameters trainable)")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_config.get('label_smoothing', 0.0) or 0.0))
    optimizer = optim.Adam(trainable_parameters(model), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    return model, criterion, optimizer, scaler, frozen


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


def _seed_and_config(data_dir: Path) -> tuple[dict[str, Any], int]:
    config = load_dataset_config(data_dir)
    # Reproducibility: seed Python/NumPy/PyTorch from the dataset config.
    train_seed = resolve_train_seed(config)
    seed_everything(train_seed)
    return config, train_seed


def _resolve_device(device: str | torch.device, amp: bool) -> tuple[torch.device, bool]:
    dev = resolve_device(device)
    return dev, bool(amp and dev.type == "cuda")


def set_torch_threads(threads: int | None) -> None:
    """``torch.set_num_threads(threads)`` for a positive *threads* (``--threads``); ``None`` keeps torch's choice."""
    if threads is None:
        return
    if int(threads) <= 0:
        raise ValueError("--threads must be a positive integer")
    torch.set_num_threads(int(threads))


# ---------------------------------------------------------------------------
# Defect classifier
# ---------------------------------------------------------------------------

def _checkpoint_targets(output_path_arg: Path, resume_path_arg: Path | None,
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


def _defect_loaders(data_dir: Path, class_names: list[str], train_config: dict[str, Any], train_seed: int,
                    loader_kwargs: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Train (augmented, seeded shuffle) and val loaders of *data_dir*; logs sizes and class distribution."""
    batch_size = train_config['batch_size']
    log.info("\nLoading datasets...")
    # return_id=True so we can show "last seen" sample IDs live in tqdm.
    train_augment = train_config.get('augment') or {}
    if train_augment:
        log.info(f"  Train augmentation: {train_augment}")
    cache_images = bool(train_config.get('cache_images', False))
    if cache_images:
        log.info("  Image cache: decoded images are kept in RAM after the first epoch")
    train_dataset = ROIDataset(data_dir, 'train', class_names, return_id=True,
                               transform=build_train_transform(train_augment), cache_images=cache_images)
    val_dataset = ROIDataset(data_dir, 'val', class_names, return_id=True, cache_images=cache_images)
    log.info(f"Train samples: {len(train_dataset)}")
    log.info(f"Val samples: {len(val_dataset)}")
    _require_samples(data_dir, train=len(train_dataset), val=len(val_dataset))
    log.info(format_distribution("Train class distribution", train_dataset.get_class_distribution()))

    # Shuffle order from its own seeded generator: independent of how much of the global torch RNG
    # the model init / augmentation consumed, so a run is reproducible for a given train seed.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(train_seed), **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _require_samples(data_dir: Path, **split_sizes: int) -> None:
    """Refuse to train with an empty split: without val samples every epoch scores F1 0, so
    ``--out`` silently kept the first epoch."""
    empty = [name for name, n in split_sizes.items() if n == 0]
    if empty:
        raise ValueError(f"Dataset {data_dir} has no usable samples in split(s): {', '.join(empty)}. "
                         "Check splits/*.txt and that the labels match the classes in config.yaml.")


def _restore_resume_state(resume_path: Path, model: nn.Module, optimizer: optim.Optimizer, *, device: torch.device,
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
    if _other_dataset(checkpoint, data_dir, manifest_path):
        log.info(f"  Checkpoint was trained on other data ({checkpoint.get('trained_on_dataset')}); "
                 f"its best val F1 {best_val_f1:.4f} does not count for this dataset.")
        return start_epoch, 0.0, 0
    log.info(f"  Resumed best val F1: {best_val_f1:.4f}")
    return start_epoch, best_val_f1, best_epoch


def _same_file(path: Path, *others: Path) -> bool:
    """True when *path* is one of *others* (resolved)."""
    try:
        target = Path(path).resolve()
        return any(Path(o).resolve() == target for o in others)
    except OSError:
        return False


def _other_dataset(checkpoint: dict[str, Any], data_dir: Path | None, manifest_path: Path | None) -> bool:
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


def _total_epochs(epochs: int, start_epoch: int, extra_epochs: int | None) -> int:
    """Epoch count of the whole run: the config value, or ``start_epoch + extra_epochs`` on resume."""
    if extra_epochs is not None:
        if extra_epochs < 0:
            raise ValueError("--extra-epochs must be >= 0")
        return max(start_epoch + int(extra_epochs), start_epoch)
    return max(int(epochs), start_epoch)


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
    threads: int | None = None,
) -> dict[str, Any]:
    """Train a defect classifier on *data_dir* and write checkpoints to *out*.

    *out* may be a ``.pt`` path or a multi-model bundle directory (existing dir or ``*.bundle``).
    *threads* sets torch's intra-op thread count (CPU training; default: torch's choice).
    A resume checkpoint of another component profile raises
    :class:`simple_sim.manifest.ProfileMismatchError` unless *allow_profile_mismatch*.
    Returns a summary dict (checkpoint paths, best epoch / val metrics, epochs run).
    """
    data_dir = Path(data_dir)
    output_path_arg = Path(out)
    resume_path_arg = Path(resume) if resume else None
    device, use_amp = _resolve_device(device, amp)
    set_torch_threads(threads)

    config, train_seed = _seed_and_config(data_dir)

    # GUARD: Load and validate dataset manifest
    manifest_path = require_manifest(data_dir)
    manifest = read_dataset_manifest(manifest_path)
    # Multi-profile (v2) datasets map to the logical profile "multi" for checkpoint metadata.
    dataset_profile_id, dataset_profile_hash = resolve_dataset_profile(manifest)
    if dataset_profile_id != "multi":
        log.info(f"Dataset Profile: {dataset_profile_id}")
    else:
        pids = [p.get('profile_id') for p in (manifest.get('component_profiles') or []) if isinstance(p, dict)]
        log.info(f"Dataset Profile: multi ({', '.join([x for x in pids if x])})")

    bundle_dir, output_path, resume_path = _checkpoint_targets(output_path_arg, resume_path_arg, dataset_profile_id)

    train_config = config['train']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    loader_kwargs = make_loader_kwargs(train_config.get('num_workers', 0), device)

    # Class names (sorted for consistency)
    class_names = sorted(config['classes'].keys())

    log.info("Training configuration:")
    log.info(f"  Dataset: {data_dir}")
    log.info(f"  Classes: {class_names}")
    log.info(f"  Epochs: {epochs}")
    log.info(f"  Batch size: {batch_size}")
    log.info(f"  Learning rate: {train_config['lr']}")
    log.info(f"  Pretrained: {train_config.get('pretrained', True)}")
    log.info(f"  Seed: {train_seed}")
    log.info(f"  Device: {device}")
    log.info(f"  AMP: {use_amp}")
    if device.type == "cpu":
        log.info(f"  Torch threads: {torch.get_num_threads()}")
    if resume_path:
        log.info(f"  Resume: {resume_path}")
        log.info(f"  Out mode: {out_mode}")

    train_loader, val_loader = _defect_loaders(data_dir, class_names, train_config, train_seed, loader_kwargs)

    model, criterion, optimizer, scaler, frozen = build_training(len(class_names), train_config, device, use_amp)

    start_epoch = 0
    best_val_f1 = 0.0
    best_val_acc: float | None = None
    best_epoch = 0

    if resume_path:
        start_epoch, best_val_f1, best_epoch = _restore_resume_state(
            resume_path, model, optimizer, device=device, class_names=class_names, train_config=train_config,
            profile=(dataset_profile_id, dataset_profile_hash), allow_profile_mismatch=allow_profile_mismatch,
            data_dir=data_dir, manifest_path=manifest_path)

    total_epochs = _total_epochs(int(epochs), start_epoch, extra_epochs)

    # Where to save a "last checkpoint" so resume runs always write something.
    last_path = last_checkpoint_path(output_path)
    # Val F1 of what --out holds: the resumed best when --out is that run's own file, else nothing
    # yet (a new --out, or a best score that was reset for another dataset).
    out_f1: float | None = None
    if resume_path and best_epoch > 0 and output_path.exists() and _same_file(resume_path, output_path, last_path):
        out_f1 = best_val_f1

    # LR schedule over the whole run (resume continues it at start_epoch).
    scheduler = build_scheduler(optimizer, train_config, total_epochs)
    if scheduler is not None:
        log.info(f"  Scheduler: {train_config.get('scheduler')} (warmup {int(train_config.get('warmup_epochs', 0) or 0)} epochs)")
        for _ in range(start_epoch):  # LambdaLR sets lr = base_lr * factor(epoch) on each step
            scheduler.step()
    if float(train_config.get('label_smoothing', 0.0) or 0.0) > 0:
        log.info(f"  Label smoothing: {train_config['label_smoothing']}")

    _banner("TRAINING")
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
        log.info(f"\nNothing to do: start_epoch={start_epoch} total_epochs={total_epochs}")
        return {**summary, "epochs_run": 0, "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                "best_val_accuracy": best_val_acc, "last_epoch": None}

    critical_classes = config.get('eval', {}).get('critical_classes')
    run = EpochRun(
        model, train_loader, val_loader, criterion, optimizer, device, class_names,
        total_epochs=total_epochs, critical_classes=critical_classes, use_amp=use_amp, scaler=scaler, frozen=frozen,
    )
    ckpt_meta = CheckpointMeta(
        class_names=class_names, config=config, profile=(dataset_profile_id, dataset_profile_hash),
        manifest_path=manifest_path, data_dir=data_dir,
    )
    stats: dict[str, Any] = {}
    for epoch in range(start_epoch, total_epochs):
        log.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        stats = fit_epoch(run, epoch)
        val_acc = stats['val_accuracy']
        val_f1 = stats['val_macro_f1']
        if scheduler is not None:
            scheduler.step()

        # Decide "best" before building the checkpoint so it carries the best-so-far score.
        # A fresh run (no resumed best) counts its first epoch as best even at F1 0; a resumed
        # best is only replaced by a better epoch, whatever --out-mode is.
        is_best = val_f1 > best_val_f1 or best_epoch == 0
        if is_best:
            best_val_f1 = float(val_f1)
            best_val_acc = float(val_acc)
            best_epoch = int(epoch + 1)
        # --out: the best epoch ('best'), every epoch ('last'); written at least once either way.
        # 'best' compares with what --out holds, so a resume into a new --out keeps the best epoch
        # of this run there instead of the first one.
        write_out = is_best or out_mode == "last" or out_f1 is None or val_f1 > out_f1 or not output_path.exists()
        if write_out:
            out_f1 = float(val_f1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = make_checkpoint(
            ckpt_meta, epoch=epoch + 1, model=model, optimizer=optimizer, stats=stats, best=(best_val_f1, best_epoch),
        )

        # Always save "last" so resume runs are observable.
        torch.save(ckpt, last_path)
        if is_best:
            log.info(f"  ✓ New best model (F1: {best_val_f1:.4f})")
        if write_out:
            torch.save(ckpt, output_path)

        # Bundle bookkeeping: keep a lightweight index of which profiles exist.
        if bundle_dir is not None:
            try:
                if output_path.exists():
                    upsert_bundle_meta(bundle_dir, dataset_profile_id, output_path)
            except Exception as e:  # noqa: BLE001 - index is best-effort, never fail training
                log.warning(f"Failed to update bundle index in {bundle_dir}: {e}")

    _banner("TRAINING COMPLETE")
    log.info(f"Best model: Epoch {best_epoch}, Val F1: {best_val_f1:.4f}")
    log.info(f"Model saved to: {output_path}")
    log.info(f"Last checkpoint: {last_path}")
    log.info(_RULE)
    return {**summary, "epochs_run": total_epochs - start_epoch, "best_epoch": best_epoch,
            "best_val_f1": best_val_f1, "best_val_accuracy": best_val_acc, "last_epoch": stats}


# ---------------------------------------------------------------------------
# Profile classifier
# ---------------------------------------------------------------------------

def dataset_profile_names(data_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Sorted profile IDs of a dataset: from a v2 manifest, else scanned from labels.jsonl."""
    if manifest.get('manifest_version', 1) == 2:
        return sorted(p['profile_id'] for p in manifest['component_profiles'])
    label_rows = read_jsonl(Path(data_dir) / LABELS_JSONL, LabelRow)
    return sorted({r.profile_id for r in label_rows if r.profile_id})


def train_profile_model(
    data_dir: Path, out: Path, *, device: str | torch.device = "cpu", amp: bool = True,
    threads: int | None = None,
) -> dict[str, Any]:
    """Train a profile classifier on a mixed-profile dataset; return a summary dict.

    Uses the same ``train`` section as :func:`train_model` (including ``train.scheduler`` /
    ``train.warmup_epochs`` / ``train.label_smoothing``); *threads* as there.
    """
    data_dir = Path(data_dir)
    output_path = Path(out)
    device, use_amp = _resolve_device(device, amp)
    set_torch_threads(threads)

    config, train_seed = _seed_and_config(data_dir)

    manifest_path = require_manifest(data_dir)
    manifest = read_dataset_manifest(manifest_path)
    profile_names = dataset_profile_names(data_dir, manifest)
    if len(profile_names) < 2:
        raise ValueError(f"Profile classifier requires 2+ profiles, found: {profile_names}")
    dataset_profile = resolve_dataset_profile(manifest)

    log.info("Profile classifier training")
    log.info(f"  Dataset: {data_dir}")
    log.info(f"  Profiles ({len(profile_names)}): {profile_names}")
    log.info(f"  Device: {device}")
    log.info(f"  AMP: {use_amp}")
    log.info(f"  Seed: {train_seed}")

    train_config = config['train']
    epochs = int(train_config['epochs'])
    batch_size = train_config['batch_size']
    loader_kwargs = make_loader_kwargs(train_config.get('num_workers', 0), device)

    log.info("\nLoading datasets...")
    train_dataset = ProfileDataset(data_dir, 'train', profile_names, return_id=True)
    val_dataset = ProfileDataset(data_dir, 'val', profile_names, return_id=True)
    _require_samples(data_dir, train=len(train_dataset), val=len(val_dataset))
    log.info(f"Train samples: {len(train_dataset)}")
    log.info(f"Val samples: {len(val_dataset)}")

    train_dist = train_dataset.get_class_distribution()
    log.info(format_distribution("Train profile distribution", train_dist))
    for name, count in train_dist.items():
        if count == 0:
            raise ValueError(f"Profile '{name}' has 0 training samples. Check dataset generation.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(train_seed), **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model, criterion, optimizer, scaler, frozen = build_training(len(profile_names), train_config, device, use_amp)

    best_val_f1 = 0.0
    best_val_acc: float | None = None
    best_epoch = 0
    last_path = last_checkpoint_path(output_path)
    scheduler = build_scheduler(optimizer, train_config, epochs)
    if scheduler is not None:
        log.info(f"  Scheduler: {train_config.get('scheduler')} (warmup {int(train_config.get('warmup_epochs', 0) or 0)} epochs)")

    _banner("PROFILE CLASSIFIER TRAINING")
    emit("train_profile_start", dataset_dir=str(data_dir), out_model=str(output_path),
         epochs=epochs, profiles=profile_names)

    run = EpochRun(
        model, train_loader, val_loader, criterion, optimizer, device, profile_names,
        total_epochs=epochs, use_amp=use_amp, scaler=scaler, emit_events=False, frozen=frozen,
    )
    ckpt_meta = CheckpointMeta(
        class_names=profile_names, config=config, profile=dataset_profile, manifest_path=manifest_path,
        data_dir=data_dir, checkpoint_type=PROFILE_CLASSIFIER,
    )
    stats: dict[str, Any] = {}
    for epoch in range(epochs):
        log.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        stats = fit_epoch(run, epoch)
        val_f1 = stats['val_macro_f1']
        if scheduler is not None:
            scheduler.step()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = make_checkpoint(ckpt_meta, epoch=epoch + 1, model=model, optimizer=optimizer, stats=stats)
        torch.save(ckpt, last_path)

        # Always write --out at least once, even if val F1 never rises above 0.
        if val_f1 > best_val_f1 or best_epoch == 0:
            best_val_f1 = float(val_f1)
            best_val_acc = float(stats['val_accuracy'])
            best_epoch = int(epoch + 1)
            log.info(f"  New best model (F1: {best_val_f1:.4f})")
            torch.save(ckpt, output_path)

    _banner("PROFILE CLASSIFIER TRAINING COMPLETE")
    log.info(f"Best model: Epoch {best_epoch}, Val F1: {best_val_f1:.4f}")
    log.info(f"Model saved to: {output_path}")
    log.info(_RULE)
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
