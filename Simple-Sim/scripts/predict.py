#!/usr/bin/env python3
"""Predict defect class for a single image using a trained checkpoint.

Usage examples:
  .venv/bin/python scripts/predict.py --model outputs/models/run_0001.pt --image /path/to/img.png

  # Predict a dataset sample by ID (loads image_path from meta.jsonl):
  .venv/bin/python scripts/predict.py --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --id run_0001/domain_A/test/000042
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.schema import read_jsonl, MetaRow, LabelRow


def load_checkpoint_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, checkpoint


def load_image_bgr(image_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise IOError(f"Failed to load image: {image_path}")
    return img_bgr


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img_rgb)


def maybe_get_ground_truth(data_dir: Optional[Path], sample_id: Optional[str]):
    if not data_dir or not sample_id:
        return None
    label_rows = read_jsonl(data_dir / "labels.jsonl", LabelRow)
    id_to_label = {r.id: r.class_name for r in label_rows}
    return id_to_label.get(sample_id)


def maybe_get_image_from_id(data_dir: Path, sample_id: str) -> Path:
    meta_rows = read_jsonl(data_dir / "meta.jsonl", MetaRow)
    id_to_meta = {r.id: r for r in meta_rows}
    if sample_id not in id_to_meta:
        # Helpful fallback: allow passing a bare numeric index like "42" or "000042".
        try:
            idx = int(str(sample_id).strip())
        except Exception:
            idx = None
        if idx is not None:
            image_path = data_dir / "images" / f"{idx:06d}.png"
            if image_path.exists():
                return image_path
        raise ValueError(f"Sample ID not found in meta.jsonl: {sample_id}")
    return data_dir / id_to_meta[sample_id].image_path


def main():
    parser = argparse.ArgumentParser(description="Predict defect class for an image")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--image", help="Path to image file (png/jpg)")
    parser.add_argument("--data", help="Optional: dataset dir to lookup ground truth/ID (has meta.jsonl/labels.jsonl)")
    parser.add_argument("--id", help="Optional: sample ID (to load image from dataset + show ground truth label)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda/cpu")
    parser.add_argument("--topk", type=int, default=4, help="How many classes to show")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_dir = Path(args.data) if args.data else None

    if args.image:
        image_path = Path(args.image)
    elif data_dir and args.id:
        image_path = maybe_get_image_from_id(data_dir, args.id)
    else:
        raise SystemExit("Provide --image OR (--data and --id).")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(args.device)
    model, class_names, checkpoint = load_checkpoint_model(model_path, device)

    img_bgr = load_image_bgr(image_path)
    x = preprocess(img_bgr).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    topk = min(args.topk, len(class_names))
    order = np.argsort(-probs)[:topk]

    print("Prediction")
    print(f"  Model:  {model_path}")
    print(f"  Image:  {image_path}")
    if args.id:
        print(f"  ID:     {args.id}")

    gt = maybe_get_ground_truth(data_dir, args.id) if data_dir else None
    if gt is not None:
        print(f"  GT:     {gt}")

    pred = class_names[int(order[0])]
    print(f"  Pred:   {pred}")
    if gt is not None:
        print(f"  Match:  {pred == gt}")

    print("\nTop-k:")
    for idx in order:
        print(f"  {class_names[int(idx)]:<12} {probs[int(idx)]:.4f}")


if __name__ == "__main__":
    main()
