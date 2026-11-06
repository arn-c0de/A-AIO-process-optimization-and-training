#!/usr/bin/env python3
"""Measure how far synthetic ROIs are from real camera ROIs (sim-to-real gap).

Both folders are embedded with a ResNet18 (ImageNet weights, or a trained Simple-Sim checkpoint
via ``--weights``) and compared with two metrics:

* **1-NN accuracy** (two-sample test): pool equally many real and synthetic embeddings and check
  how often an image's nearest neighbour comes from its own domain. 0.5 = indistinguishable,
  1.0 = completely separable.
* **NN distance ratio**: mean distance from each real image to its nearest synthetic image,
  divided by the mean distance to its nearest *other* real image. ~1 = the synthetic set covers the
  real images as closely as they cover each other; larger = gap.

Track both numbers per release to make realism changes measurable instead of eyeballed.

Usage:
    python tools/sim_to_real.py --synthetic data/run_0001/images --real data/real_rois [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.data_loader import preprocess_bgr  # noqa: E402
from simple_sim.generators.opencv2d.photoreal import IMAGE_EXTENSIONS  # noqa: E402
from simple_sim.models import build_resnet18, load_checkpoint_model  # noqa: E402


def load_images(folder: Path, limit: int, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found in {folder}")
    if len(files) > limit:
        files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
    images = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Failed to read {p}")
        images.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
    return images


def build_embedder(weights: Path | None) -> torch.nn.Module:
    model = load_checkpoint_model(weights, "cpu")[0] if weights else build_resnet18(1, pretrained=True)
    for m in model.modules():  # works for single models and ensembles alike
        if hasattr(m, "fc"):
            m.fc = torch.nn.Identity()
    return model.eval()


@torch.no_grad()
def embed(model: torch.nn.Module, images: list[np.ndarray], batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([preprocess_bgr(img) for img in images[i:i + batch]])
        out.append(model(x).numpy())
    feats = np.concatenate(out).astype(np.float64)
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance matrix between unit-normalised rows."""
    return 1.0 - a @ b.T


def one_nn_accuracy(real: np.ndarray, syn: np.ndarray) -> float:
    n = min(len(real), len(syn))
    pooled = np.concatenate([real[:n], syn[:n]])
    labels = np.r_[np.zeros(n), np.ones(n)]
    d = _dist(pooled, pooled)
    np.fill_diagonal(d, np.inf)
    return float((labels[d.argmin(axis=1)] == labels).mean())


def nn_distance_ratio(real: np.ndarray, syn: np.ndarray) -> float:
    rr = _dist(real, real)
    np.fill_diagonal(rr, np.inf)
    return float(_dist(real, syn).min(axis=1).mean() / rr.min(axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--synthetic", type=Path, required=True, help="Folder with synthetic ROI images (searched recursively)")
    ap.add_argument("--real", type=Path, required=True, help="Folder with real camera ROI images (searched recursively)")
    ap.add_argument("--weights", type=Path, help="Simple-Sim checkpoint to embed with (default: ImageNet ResNet18)")
    ap.add_argument("--max-images", type=int, default=500, help="Max images per side (random subset)")
    ap.add_argument("--size", type=int, default=128, help="Images are resized to size x size before embedding")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="Also write the result as JSON")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = build_embedder(args.weights)
    real = embed(model, load_images(args.real, args.max_images, args.size, rng))
    syn = embed(model, load_images(args.synthetic, args.max_images, args.size, rng))
    if len(real) < 2:
        raise SystemExit("Need at least 2 real images")

    result = {
        "real_images": len(real),
        "synthetic_images": len(syn),
        "embedder": str(args.weights) if args.weights else "resnet18_imagenet",
        "one_nn_accuracy": round(one_nn_accuracy(real, syn), 4),
        "nn_distance_ratio": round(nn_distance_ratio(real, syn), 4),
    }
    print(f"Real: {result['real_images']}  Synthetic: {result['synthetic_images']}  Embedder: {result['embedder']}")
    print(f"1-NN accuracy:     {result['one_nn_accuracy']:.3f}  (0.5 = indistinguishable, 1.0 = separable)")
    print(f"NN distance ratio: {result['nn_distance_ratio']:.3f}  (1.0 = synthetic as close as real-to-real)")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
