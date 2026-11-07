#!/usr/bin/env python3
"""Evaluate a Simple-Sim checkpoint on real camera crops (sim-to-real test).

The real crops are sorted into one sub-folder per expected class, named after the class
(``OK``, ``MISSING``, ...). Anything after ``@`` is a sub-group shown separately but scored as the
class (``MISSING@empty_pad``). Every image is resized to the training ROI size (cubic, like the
inspection pipeline upscales its small crops) and classified.

Reported per folder: number of images, accuracy (predicted class == folder class), the share
predicted ``OK`` and the predicted-class histogram; overall: the binary inspection view
(``OK`` vs. any defect): false-call rate on good parts and escape rate on defective ones.

Usage:
    python tools/eval_real.py --checkpoint models/run_0402_c.pt --real data/real_eval [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.data_loader import preprocess_bgr  # noqa: E402
from simple_sim.generators.opencv2d.photoreal import IMAGE_EXTENSIONS  # noqa: E402
from simple_sim.models import load_checkpoint_model  # noqa: E402


def roi_size(checkpoint: dict[str, Any], default: int) -> tuple[int, int]:
    """(width, height) of the training ROIs from the checkpoint's config, else ``default``."""
    roi = ((checkpoint or {}).get("config") or {}).get("roi") or {}
    return int(roi.get("width_px", default)), int(roi.get("height_px", default))


def class_folders(root: Path) -> list[tuple[str, str, list[Path]]]:
    """``(group, class, files)`` for every sub-folder of *root* that holds images (sorted)."""
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if files:
            out.append((folder.name, folder.name.split("@", 1)[0], files))
    if not out:
        raise SystemExit(f"No class sub-folders with images in {root}")
    return out


@torch.no_grad()
def predict(model: torch.nn.Module, files: list[Path], size: tuple[int, int], batch: int = 64) -> list[int]:
    preds: list[int] = []
    for i in range(0, len(files), batch):
        imgs = []
        for p in files[i:i + batch]:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"Failed to read {p}")
            interp = cv2.INTER_CUBIC if img.shape[1] < size[0] else cv2.INTER_AREA
            imgs.append(preprocess_bgr(cv2.resize(img, size, interpolation=interp)))
        preds.extend(model(torch.stack(imgs)).argmax(1).tolist())
    return preds


def evaluate(model: torch.nn.Module, class_names: list[str], root: Path, size: tuple[int, int],
             limit: int, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups: dict[str, Any] = {}
    good = good_flagged = bad = bad_passed = 0
    for group, cls, files in class_folders(root):
        if len(files) > limit:
            files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
        names = [class_names[i] for i in predict(model, files, size)]
        hist = Counter(names)
        n = len(names)
        groups[group] = {"class": cls, "images": n, "known_class": cls in class_names,
                         "accuracy": hist[cls] / n, "predicted_ok": hist["OK"] / n,
                         "predictions": dict(hist.most_common())}
        if cls == "OK":
            good, good_flagged = good + n, good_flagged + n - hist["OK"]
        else:
            bad, bad_passed = bad + n, bad_passed + hist["OK"]
    return {"groups": groups,
            "false_call_rate": good_flagged / good if good else None,
            "escape_rate": bad_passed / bad if bad else None}


def format_report(result: dict[str, Any]) -> str:
    lines = [f"{'folder':24s} {'n':>5s} {'acc':>6s} {'->OK':>6s}  predictions"]
    for group, g in result["groups"].items():
        lines.append(f"{group:24s} {g['images']:5d} {g['accuracy']:6.3f} {g['predicted_ok']:6.3f}  {g['predictions']}")
    fc, esc = result["false_call_rate"], result["escape_rate"]
    lines.append(f"false-call rate (good flagged): {'-' if fc is None else f'{fc:.3f}'}   "
                 f"escape rate (defect passed as OK): {'-' if esc is None else f'{esc:.3f}'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True, help="folder with one sub-folder per class")
    ap.add_argument("--limit", type=int, default=1000, help="max images per sub-folder (random subset)")
    ap.add_argument("--size", type=int, default=256, help="ROI size if the checkpoint has no config")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    model, class_names, checkpoint = load_checkpoint_model(args.checkpoint, "cpu")
    result = evaluate(model, class_names, args.real, roi_size(checkpoint, args.size), args.limit)
    result["checkpoint"] = str(args.checkpoint)
    print(format_report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
