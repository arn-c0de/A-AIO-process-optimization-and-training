#!/usr/bin/env python3
"""Sample a few images per class from a dataset split and verify predictions vs GT.

Example:
  .venv/bin/python scripts/sample_check.py \\
    --model outputs/models/run_0001.pt \\
    --data outputs/sim_data/runs/run_0001 \\
    --split test \\
    --per-class 5
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

import sys

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
    return model, class_names


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm(img_rgb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample-check predictions vs GT")
    ap.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--data", required=True, help="Dataset dir")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--per-class", type=int, default=5, help="How many samples per class")
    ap.add_argument("--seed", type=int, default=42, help="Sampling RNG seed")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model_path = Path(args.model)
    data_dir = Path(args.data)
    device = torch.device(args.device)

    model, class_names = load_checkpoint_model(model_path, device)

    # Load meta/labels and build maps.
    meta_rows = read_jsonl(data_dir / "meta.jsonl", MetaRow)
    label_rows = read_jsonl(data_dir / "labels.jsonl", LabelRow)
    id_to_meta = {r.id: r for r in meta_rows}
    id_to_label = {r.id: r.class_name for r in label_rows}

    # Collect candidates per class for the requested split.
    candidates: Dict[str, List[str]] = {c: [] for c in class_names}
    want_token = f"/{args.split}/"
    for sid, cls in id_to_label.items():
        if want_token not in sid:
            continue
        if cls in candidates and sid in id_to_meta:
            candidates[cls].append(sid)

    rng = random.Random(args.seed)
    chosen: List[Tuple[str, str]] = []  # (id, gt)
    for cls in class_names:
        arr = candidates.get(cls, [])
        if not arr:
            continue
        rng.shuffle(arr)
        take = min(int(args.per_class), len(arr))
        for sid in arr[:take]:
            chosen.append((sid, cls))

    if not chosen:
        raise SystemExit("No samples found for the requested split/classes.")

    # Deterministic order for output readability.
    chosen.sort(key=lambda t: t[0])

    print("Sample Check")
    print(f"  Model:   {model_path}")
    print(f"  Dataset: {data_dir}")
    print(f"  Split:   {args.split}")
    print(f"  Per cls: {args.per_class}")
    print(f"  Seed:    {args.seed}")
    print("")

    # Run inference in mini-batches.
    rows = []
    correct = 0
    seen = 0

    bs = max(1, int(args.batch_size))
    for start in range(0, len(chosen), bs):
        batch = chosen[start : start + bs]
        xs = []
        ids = []
        gts = []
        img_names = []
        for sid, gt in batch:
            meta = id_to_meta[sid]
            img_path = data_dir / meta.image_path
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise IOError(f"Failed to load image: {img_path}")
            xs.append(preprocess(img_bgr))
            ids.append(sid)
            gts.append(gt)
            img_names.append(Path(meta.image_path).name)

        x = torch.stack(xs, dim=0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred_idx = torch.argmax(logits, dim=1).detach().cpu().numpy()

        for sid, gt, img_name, pi in zip(ids, gts, img_names, pred_idx):
            pred = class_names[int(pi)]
            ok = pred == gt
            rows.append((sid, img_name, gt, pred, ok))
            seen += 1
            correct += 1 if ok else 0

    # Print table
    print(f"{'id':<34} {'img':<12} {'gt':<10} {'pred':<10} {'ok'}")
    for sid, img_name, gt, pred, ok in rows:
        print(f"{sid:<34} {img_name:<12} {gt:<10} {pred:<10} {str(ok)}")

    acc = correct / max(1, seen)
    print("")
    print(f"Summary: {correct}/{seen} correct  (acc={acc:.4f})")


if __name__ == "__main__":
    main()

