#!/usr/bin/env python3
"""Predict defect class for a single image using a trained checkpoint.

Usage examples:
  .venv/bin/python scripts/predict.py --model outputs/models/run_0001.pt --image /path/to/img.png

  # Predict a dataset sample by ID (loads image_path from meta.jsonl):
  .venv/bin/python scripts/predict.py --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --id run_0001/domain_A/test/000042
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import add_json_flag
from simple_sim.data_loader import load_image_bgr, preprocess_bgr
from simple_sim.eval_utils import (
    MANIFEST_NAME,
    UsageError,
    dataset_profile,
    predict_probs,
    resolve_bundle_checkpoint,
    run_cli,
)
from simple_sim.manifest import check_checkpoint_profile
from simple_sim.models import load_checkpoint_model
from simple_sim.schema import LabelRow, MetaRow, read_jsonl

def maybe_get_ground_truth(data_dir: Path | None, sample_id: str | None) -> str | None:
    """Ground-truth class of *sample_id* from ``labels.jsonl`` (None if unknown)."""
    if not data_dir or not sample_id:
        return None
    label_rows = read_jsonl(Path(data_dir) / "labels.jsonl", LabelRow)
    return {r.id: r.class_name for r in label_rows}.get(sample_id)


def maybe_get_image_from_id(data_dir: Path, sample_id: str) -> Path:
    """Image path of *sample_id* via ``meta.jsonl``; a bare index like ``42`` maps to ``images/000042.png``."""
    data_dir = Path(data_dir)
    id_to_meta = {r.id: r for r in read_jsonl(data_dir / "meta.jsonl", MetaRow)}
    if sample_id in id_to_meta:
        return data_dir / id_to_meta[sample_id].image_path
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
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir) if data_dir else None
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if image_path:
        image_path = Path(image_path)
    elif data_dir and sample_id:
        image_path = maybe_get_image_from_id(data_dir, sample_id)
    else:
        raise UsageError("Provide --image OR (--data and --id).")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(device)
    # Multi-model bundle support: if --model is a directory, require --data to resolve profile.
    if model_path.is_dir():
        if not data_dir:
            raise UsageError(
                "When --model points to a multi-model bundle directory, pass --data so the profile can be resolved.\n"
                "Example:\n"
                "  scripts/predict.py --model outputs/models/my_multi.bundle --data outputs/sim_data/runs/run_0001 --id <sample_id>"
            )
        model_path = resolve_bundle_checkpoint(model_path, dataset_profile(data_dir)[0])

    model, class_names, checkpoint = load_checkpoint_model(model_path, device)

    # GUARD: Validate profile compatibility if using dataset (hard fail, no override flag here).
    if data_dir and (data_dir / MANIFEST_NAME).exists():
        profile_id, profile_hash = dataset_profile(data_dir)
        check_checkpoint_profile(checkpoint, profile_id, profile_hash, action="predict on")

    x = preprocess_bgr(load_image_bgr(image_path)).unsqueeze(0).to(device)
    probs = predict_probs(model, x)[0]
    order = np.argsort(-probs)[:min(topk, len(class_names))]

    pred = class_names[int(order[0])]
    gt = maybe_get_ground_truth(data_dir, sample_id)
    return {
        "model_path": str(model_path),
        "image_path": str(image_path),
        "id": sample_id,
        "pred": pred,
        "confidence": float(probs[int(order[0])]),
        "probs": {n: float(probs[i]) for i, n in enumerate(class_names)},
        "topk": [[class_names[int(i)], float(probs[int(i)])] for i in order],
        "gt": gt,
        "match": (pred == gt) if gt is not None else None,
    }


def print_prediction(result: dict[str, Any]) -> None:
    """Human-readable output of :func:`predict_image`."""
    print("Prediction")
    print(f"  Model:  {result['model_path']}")
    print(f"  Image:  {result['image_path']}")
    if result["id"]:
        print(f"  ID:     {result['id']}")
    if result["gt"] is not None:
        print(f"  GT:     {result['gt']}")
    print(f"  Pred:   {result['pred']}")
    if result["gt"] is not None:
        print(f"  Match:  {result['match']}")
    print("\nTop-k:")
    for name, prob in result["topk"]:
        print(f"  {name:<12} {prob:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict defect class for an image")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--image", help="Path to image file (png/jpg)")
    parser.add_argument("--data", help="Optional: dataset dir to lookup ground truth/ID (has meta.jsonl/labels.jsonl)")
    parser.add_argument("--id", help="Optional: sample ID (to load image from dataset + show ground truth label)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda/cpu")
    parser.add_argument("--topk", type=int, default=4, help="How many classes to show")
    add_json_flag(parser)
    args = parser.parse_args()

    def run() -> dict[str, Any]:
        result = predict_image(
            Path(args.model),
            image_path=Path(args.image) if args.image else None,
            data_dir=Path(args.data) if args.data else None,
            sample_id=args.id,
            device=args.device,
            topk=args.topk,
        )
        print_prediction(result)
        return result

    run_cli("predict", run, args.json)


if __name__ == "__main__":
    main()
