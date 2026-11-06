"""Shared helpers for the train/eval/predict scripts.

Covers batch unpacking, the no-grad inference loop used by validation and evaluation,
softmax prediction, and dataset-manifest / model-bundle resolution.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from simple_sim.manifest import read_dataset_manifest, resolve_dataset_profile
from simple_sim.model_bundle import bundle_checkpoint_path
from simple_sim.telemetry import emit

MANIFEST_NAME = "dataset_manifest.json"


# ---------------------------------------------------------------------------
# Batches / inference loop
# ---------------------------------------------------------------------------

def unpack_batch(batch: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
    """Split a DataLoader batch into ``(images, labels, sample_ids, image_paths)`` (missing parts are None)."""
    images, labels = batch[0], batch[1]
    sample_ids = batch[2] if len(batch) >= 3 else None
    image_paths = batch[3] if len(batch) >= 4 else None
    return images, labels, sample_ids, image_paths


class ThroughputMeter:
    """Exponential moving average of images per second."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.ips: float | None = None

    def update(self, batch_size: int, seconds: float) -> float:
        ips = batch_size / max(1e-9, seconds)
        self.ips = ips if self.ips is None else ((1.0 - self.alpha) * self.ips + self.alpha * ips)
        return self.ips


def amp_autocast(use_amp: bool) -> contextlib.AbstractContextManager[Any]:
    """CUDA autocast context when *use_amp*, else a no-op context."""
    return torch.autocast(device_type="cuda") if use_amp else contextlib.nullcontext()


def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move *t* to *device* (non-blocking on CUDA)."""
    return t.to(device, non_blocking=(getattr(device, "type", "") == "cuda"))


@dataclass
class InferenceResult:
    """Collected labels/predictions of one pass over a loader."""

    y_true: np.ndarray
    y_pred: np.ndarray
    avg_loss: float | None
    num_samples: int


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    criterion: nn.Module | None = None,
    use_amp: bool = False,
    desc: str = "Evaluating",
    leave: bool = True,
    event: str = "eval_batch",
) -> InferenceResult:
    """Run *model* over *loader* without gradients and collect argmax predictions.

    Shows a tqdm progress bar and emits one telemetry *event* per batch. With *criterion*,
    the average loss is computed as well.
    """
    model.eval()
    preds: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    meter = ThroughputMeter()
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    last_img = None

    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=leave)
        for batch in progress:
            images, labels, _ids, image_paths = unpack_batch(batch)
            if image_paths is not None:
                last_img = image_paths[-1]

            t0 = time.perf_counter()
            images = to_device(images, device)
            labels = to_device(labels, device)
            with amp_autocast(use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels) if criterion is not None else None

            bs = int(images.shape[0])
            ips = meter.update(bs, time.perf_counter() - t0)
            num_samples += bs
            postfix: dict[str, str] = {}
            fields: dict[str, Any] = {}
            if loss is not None:
                total_loss += float(loss.item())
                num_batches += 1
                postfix["loss"] = f"{loss.item():.4f}"
                fields["loss"] = float(loss.item())
            postfix["img/s"] = f"{ips:.1f}"
            if last_img is not None:
                postfix["last_img"] = Path(str(last_img)).name
            progress.set_postfix(postfix)
            emit(
                event,
                **fields,
                img_per_s=float(ips),
                last_image_path=str(last_img) if last_img is not None else None,
                batch_size=bs,
            )

            preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    cat = (lambda xs: np.concatenate(xs) if xs else np.array([], dtype=np.int64))
    return InferenceResult(
        y_true=cat(labels_all),
        y_pred=cat(preds),
        avg_loss=(total_loss / max(1, num_batches)) if criterion is not None else None,
        num_samples=num_samples,
    )


def predict_probs(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Softmax probabilities of *model* for batch *x* (shape ``[N, C]``, numpy)."""
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).cpu().numpy()


def top_prediction(probs: np.ndarray, class_names: Sequence[str]) -> tuple[str, float, dict[str, float]]:
    """``(predicted_class, confidence, {class: prob})`` for a single 1-D probability vector."""
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), {n: float(probs[i]) for i, n in enumerate(class_names)}


def checkpoint_eval_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint config's ``eval`` section (``{}`` for ensembles, which carry no top-level config)."""
    return (checkpoint.get("config") or {}).get("eval") or {}


def checkpoint_val_accuracy(checkpoint: dict[str, Any]) -> float:
    """Validation accuracy stored in *checkpoint* (0.0 when absent, e.g. for ensembles)."""
    return float(checkpoint.get("val_accuracy") or 0.0)


def eval_loader(dataset: Dataset, checkpoint: dict[str, Any]) -> DataLoader:
    """Sequential DataLoader using the checkpoint config's ``eval.batch_size`` / ``eval.num_workers``.

    Falls back to batch size 64 / no workers when the checkpoint has no config (ensembles).
    """
    eval_cfg = checkpoint_eval_config(checkpoint)
    return DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=eval_cfg.get("num_workers", 0),
    )


def format_checkpoint_info(checkpoint: dict[str, Any], label: str = "Classes") -> str:
    """The standard checkpoint summary lines (class names, epoch, val metrics).

    Tolerates ensemble checkpoints, which have no epoch / val metrics.
    """
    return "\n".join([
        f"  {label}: {checkpoint.get('class_names')}",
        f"  Trained epoch: {checkpoint.get('epoch', '-')}",
        f"  Val accuracy: {checkpoint_val_accuracy(checkpoint):.4f}",
        f"  Val F1: {float(checkpoint.get('val_f1') or 0.0):.4f}",
    ])


def format_distribution(title: str, dist: dict[str, int]) -> str:
    """``title:`` followed by one indented ``name: count`` line per entry (starts with a blank line)."""
    return "\n".join([f"\n{title}:", *(f"  {name}: {count}" for name, count in dist.items())])


def metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compact metric subset for ``--json`` payloads."""
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "critical_fn_rates": metrics.get("critical_fn_rates", {}),
    }


# ---------------------------------------------------------------------------
# Dataset manifest / bundle resolution
# ---------------------------------------------------------------------------

def require_manifest(data_dir: Path) -> Path:
    """Return ``<data_dir>/dataset_manifest.json`` or raise FileNotFoundError with a backfill hint."""
    manifest_path = Path(data_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path}\n"
            f"This dataset was created before profile support.\n"
            f"Regenerate it, or create the manifest: python -m simple_sim backfill-manifest --data {data_dir}"
        )
    return manifest_path


def dataset_profile(data_dir: Path) -> tuple[str, str]:
    """``(profile_id, profile_hash)`` of a dataset (requires its manifest)."""
    return resolve_dataset_profile(read_dataset_manifest(require_manifest(data_dir)))


def resolve_bundle_checkpoint(model_path: Path, profile_id: str, *, hint: str = "") -> Path:
    """If *model_path* is a multi-model bundle dir, return its best checkpoint for *profile_id*.

    Plain checkpoint paths are returned unchanged.
    """
    model_path = Path(model_path)
    if not model_path.is_dir():
        return model_path
    resolved = bundle_checkpoint_path(model_path, profile_id, kind="best")
    if not resolved.exists():
        raise FileNotFoundError(
            f"Multi-model bundle has no checkpoint for profile '{profile_id}':\n"
            f"  bundle: {model_path}\n"
            f"  expected: {resolved}" + (f"\n{hint}" if hint else "")
        )
    return resolved
