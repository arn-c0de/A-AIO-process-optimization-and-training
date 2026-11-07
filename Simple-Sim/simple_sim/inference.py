"""Single-image classification with a trained checkpoint (plan item L7).

:class:`Classifier` loads a checkpoint, ensemble or bundle checkpoint once and classifies BGR
arrays or image files. Preprocessing is :func:`simple_sim.data_loader.preprocess_bgr`, i.e. the
same transform as training and test-set evaluation (no resize: the GUI used to insert
``Resize((256, 256))``, plan X1, which made its predictions differ from ``predict`` /
``batch-predict`` for ROIs that are not 256 px). :func:`predict_image` is the core of the
``predict`` command (image file or dataset sample id, bundle resolution, profile guard).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simple_sim.cli_utils import UsageError
from simple_sim.data_loader import load_image_bgr, preprocess_bgr
from simple_sim.dataset_files import DATASET_MANIFEST
from simple_sim.dataset_index import DatasetIndex
from simple_sim.devices import resolve_device
from simple_sim.eval_utils import dataset_profile, predict_probs, resolve_bundle_checkpoint
from simple_sim.manifest import check_checkpoint_profile
from simple_sim.models import load_checkpoint_model
from simple_sim.rows import MetaRow

__all__ = [
    "Classifier",
    "Prediction",
    "choose_samples",
    "ground_truth",
    "image_for_sample",
    "predict_image",
    "sample_check",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Result of one classification: top class, its probability and all class probabilities."""

    label: str
    confidence: float
    probs: dict[str, float] = field(default_factory=dict)
    probs_array: np.ndarray | None = field(default=None, compare=False, repr=False)

    def topk(self, k: int) -> list[tuple[str, float]]:
        """The *k* most probable ``(class, prob)`` pairs, best first (ties keep class order)."""
        if self.probs_array is None:
            return sorted(self.probs.items(), key=lambda kv: -kv[1])[:k]
        names = list(self.probs)
        order = np.argsort(-self.probs_array)[:min(k, len(names))]
        return [(names[int(i)], float(self.probs_array[int(i)])) for i in order]


def _prediction(probs: np.ndarray, class_names: Sequence[str]) -> Prediction:
    idx = int(np.argmax(probs))
    return Prediction(
        label=class_names[idx],
        confidence=float(probs[idx]),
        probs={n: float(probs[i]) for i, n in enumerate(class_names)},
        probs_array=probs,
    )


class Classifier:
    """A loaded checkpoint (single model or ensemble) in eval mode on *device*.

    ``class_names`` and the raw ``checkpoint`` dict are kept for callers that need metadata.
    """

    def __init__(self, model_path: Path | str, device: str | torch.device = "cpu") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.device = resolve_device(device)
        self.model, self.class_names, self.checkpoint = load_checkpoint_model(self.model_path, self.device)

    def probs_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        """Softmax probabilities (1-D, ``len(class_names)``) for one BGR image."""
        x = preprocess_bgr(img_bgr).unsqueeze(0).to(self.device)
        return predict_probs(self.model, x)[0]

    def predict_bgr(self, img_bgr: np.ndarray) -> Prediction:
        """Classify one BGR image (as returned by ``cv2.imread``)."""
        return _prediction(self.probs_bgr(img_bgr), self.class_names)

    def predict_path(self, image_path: Path | str) -> Prediction:
        """Classify one image file; raises ``OSError`` if it cannot be read."""
        return self.predict_bgr(load_image_bgr(Path(image_path)))


# ---------------------------------------------------------------------------
# ``predict`` command core
# ---------------------------------------------------------------------------

def ground_truth(data_dir: Path | None, sample_id: str | None) -> str | None:
    """Ground-truth class of *sample_id* from ``labels.jsonl`` (None if unknown)."""
    if not data_dir or not sample_id:
        return None
    return {r.id: r.class_name for r in DatasetIndex(data_dir).label_rows()}.get(sample_id)


def image_for_sample(data_dir: Path, sample_id: str) -> Path:
    """Image path of *sample_id* via ``meta.jsonl``; a bare index like ``42`` maps to ``images/000042.png``."""
    data_dir = Path(data_dir)
    meta = DatasetIndex(data_dir).id_to_meta().get(sample_id)
    if meta is not None:
        return data_dir / meta.image_path
    try:
        image_path = data_dir / "images" / f"{int(str(sample_id).strip()):06d}.png"
    except ValueError:
        image_path = None
    if image_path is not None and image_path.exists():
        return image_path
    raise ValueError(f"Sample ID not found in meta.jsonl: {sample_id}")


def predict_image(
    model_path: Path,
    *,
    image_path: Path | None = None,
    data_dir: Path | None = None,
    sample_id: str | None = None,
    device: str | torch.device = "cpu",
    topk: int = 4,
) -> dict[str, Any]:
    """Classify one image (given directly or as a dataset sample ID).

    Returns ``model_path``, ``image_path``, ``id``, ``pred``, ``confidence``, ``probs``
    (class -> prob), ``topk`` (list of ``[class, prob]``), ``gt`` and ``match`` (None without GT).
    A model dir (bundle) needs *data_dir* to pick the profile's checkpoint.
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir) if data_dir else None
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if image_path:
        image_path = Path(image_path)
    elif data_dir and sample_id:
        image_path = image_for_sample(data_dir, sample_id)
    else:
        raise UsageError("Provide --image OR (--data and --id).")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Multi-model bundle support: if --model is a directory, require --data to resolve profile.
    if model_path.is_dir():
        if not data_dir:
            raise UsageError(
                "When --model points to a multi-model bundle directory, pass --data so the profile can be resolved.\n"
                "Example:\n"
                "  python -m simple_sim predict --model outputs/models/my_multi.bundle --data outputs/sim_data/runs/run_0001 --id <sample_id>"
            )
        model_path = resolve_bundle_checkpoint(model_path, dataset_profile(data_dir)[0])

    clf = Classifier(model_path, device)

    # GUARD: Validate profile compatibility if using dataset (hard fail, no override flag here).
    # Ensemble checkpoints (profile_id "multi") span several profiles and are not checked.
    ckpt_profile_id = (clf.checkpoint.get("component_profile") or {}).get("profile_id")
    if data_dir and ckpt_profile_id != "multi" and (data_dir / DATASET_MANIFEST).exists():
        profile_id, profile_hash = dataset_profile(data_dir)
        check_checkpoint_profile(clf.checkpoint, profile_id, profile_hash, action="predict on")

    pred = clf.predict_path(image_path)
    gt = ground_truth(data_dir, sample_id)
    return {
        "model_path": str(model_path),
        "image_path": str(image_path),
        "id": sample_id,
        "pred": pred.label,
        "confidence": pred.confidence,
        "probs": pred.probs,
        "topk": [[name, prob] for name, prob in pred.topk(topk)],
        "gt": gt,
        "match": (pred.label == gt) if gt is not None else None,
    }


# ---------------------------------------------------------------------------
# ``sample-check`` command core
# ---------------------------------------------------------------------------

def choose_samples(
    data_dir: Path, class_names: list[str], split: str, per_class: int, seed: int
) -> tuple[list[tuple[str, str]], dict[str, MetaRow]]:
    """Pick up to *per_class* random sample IDs per class from *split*.

    Split membership comes from ``splits/<split>.txt`` (the same source as ``ROIDataset``), not from
    the ``/<split>/`` part of the sample id, which is wrong for merged or re-split datasets (plan C1).
    A missing split file selects nothing. Candidates keep the ``labels.jsonl`` order.

    Returns ``([(sample_id, gt_class), ...] sorted by id, id_to_meta)``.
    """
    index = DatasetIndex(data_dir)
    id_to_meta = dict(index.id_to_meta())
    id_to_label = {r.id: r.class_name for r in index.label_rows()}
    try:
        in_split = set(index.split_ids(split))
    except FileNotFoundError:
        in_split = set()

    candidates: dict[str, list[str]] = {c: [] for c in class_names}
    for sid, cls in id_to_label.items():
        if sid in in_split and cls in candidates and sid in id_to_meta:
            candidates[cls].append(sid)

    rng = random.Random(seed)
    chosen: list[tuple[str, str]] = []
    for cls in class_names:
        arr = candidates[cls]
        rng.shuffle(arr)
        chosen.extend((sid, cls) for sid in arr[:per_class])
    # Deterministic order for output readability.
    chosen.sort(key=lambda t: t[0])
    return chosen, id_to_meta


def sample_check(
    model_path: Path,
    data_dir: Path,
    *,
    split: str = "test",
    per_class: int = 5,
    seed: int = 42,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Predict a random per-class sample of *split* and compare with GT.

    Returns ``rows`` (dicts with id/img/gt/pred/ok), ``correct``, ``seen`` and ``accuracy``.
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    clf = Classifier(model_path, device)
    chosen, id_to_meta = choose_samples(data_dir, clf.class_names, split, int(per_class), seed)
    if not chosen:
        raise UsageError("No samples found for the requested split/classes.")

    log.info("Sample Check")
    log.info(f"  Model:   {model_path}")
    log.info(f"  Dataset: {data_dir}")
    log.info(f"  Split:   {split}")
    log.info(f"  Per cls: {per_class}")
    log.info(f"  Seed:    {seed}")
    log.info("")

    rows: list[dict[str, Any]] = []
    bs = max(1, int(batch_size))
    for start in range(0, len(chosen), bs):
        batch = chosen[start:start + bs]
        x = torch.stack(
            [preprocess_bgr(load_image_bgr(data_dir / id_to_meta[sid].image_path)) for sid, _ in batch]
        ).to(clf.device)
        pred_idx = predict_probs(clf.model, x).argmax(axis=1)
        for (sid, gt), pi in zip(batch, pred_idx, strict=True):
            pred = clf.class_names[int(pi)]
            rows.append({"id": sid, "img": Path(id_to_meta[sid].image_path).name, "gt": gt, "pred": pred, "ok": pred == gt})

    correct = sum(r["ok"] for r in rows)
    seen = len(rows)
    return {
        "model_path": str(model_path),
        "dataset_path": str(data_dir),
        "split": split,
        "rows": rows,
        "correct": correct,
        "seen": seen,
        "accuracy": correct / max(1, seen),
    }
