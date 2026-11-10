#!/usr/bin/env python3
"""Evaluate a Simple-Sim checkpoint on real camera crops (sim-to-real test).

The real crops are sorted into one sub-folder per expected class, named after the class
(``OK``, ``MISSING``, ...). Anything after ``@`` is a sub-group shown separately but scored as the
class (``MISSING@empty_pad``). Every image is resized to the training ROI size (cubic, like the
inspection pipeline upscales its small crops) and classified.

Reported per folder: number of images, accuracy (predicted class == folder class), the share
predicted ``OK`` and the predicted-class histogram; overall: the binary inspection view
(``OK`` vs. any defect): false-call rate on good parts and escape rate on defective ones.

``--tta`` averages the softmax over flips and quarter turns; ``--calibration`` adds ECE (15 bins),
NLL and Brier score over the images of known classes (``calibration`` in the result).

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

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.data_loader import preprocess_bgr  # noqa: E402
from simple_sim.eval_utils import model_outputs  # noqa: E402
from simple_sim.images import list_images  # noqa: E402
from simple_sim.metrics import calibration_metrics, format_calibration  # noqa: E402
from simple_sim.models import load_checkpoint_model  # noqa: E402


def roi_size(checkpoint: dict[str, Any], default: int) -> tuple[int, int]:
    """(width, height) of the training ROIs from the checkpoint's config, else ``default``."""
    checkpoint = checkpoint or {}
    config = checkpoint.get("config")
    if not config:  # ensembles keep the config with their sub-models
        first = next(iter(checkpoint.get("models") or []), None) or {}
        config = (first.get("checkpoint") or {}).get("config") if isinstance(first, dict) else None
    roi = (config or {}).get("roi") or {}
    return int(roi.get("width_px", default)), int(roi.get("height_px", default))


def class_folders(root: Path) -> list[tuple[str, str, list[Path]]]:
    """``(group, class, files)`` for every sub-folder of *root* that holds images (sorted)."""
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = list_images(folder)
        if files:
            out.append((folder.name, folder.name.split("@", 1)[0], files))
    if not out:
        raise SystemExit(f"No class sub-folders with images in {root}")
    return out


@torch.no_grad()
def predict_probs(model: torch.nn.Module, files: list[Path], size: tuple[int, int], batch: int = 64,
                  *, tta: bool = False) -> np.ndarray:
    """Softmax probabilities ``[N, C]`` of *files* (with *tta*: averaged over flips and quarter turns)."""
    probs: list[np.ndarray] = []
    for i in range(0, len(files), batch):
        imgs = []
        for p in files[i:i + batch]:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"Failed to read {p}")
            interp = cv2.INTER_CUBIC if img.shape[1] < size[0] else cv2.INTER_AREA
            imgs.append(preprocess_bgr(cv2.resize(img, size, interpolation=interp)))
        probs.append(torch.softmax(model_outputs(model, torch.stack(imgs), tta=tta).float(), dim=1).numpy())
    return np.concatenate(probs) if probs else np.zeros((0, 0), dtype=np.float32)


def predict(model: torch.nn.Module, files: list[Path], size: tuple[int, int], batch: int = 64,
            *, tta: bool = False) -> list[int]:
    return predict_probs(model, files, size, batch, tta=tta).argmax(1).tolist()


def evaluate(model: torch.nn.Module, class_names: list[str], root: Path, size: tuple[int, int],
             limit: int, seed: int = 0, *, tta: bool = False, calibration: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups: dict[str, Any] = {}
    good = good_flagged = bad = bad_passed = 0
    cal_probs: list[np.ndarray] = []
    cal_true: list[int] = []
    for group, cls, files in class_folders(root):
        if 0 < limit < len(files):  # 0 = all, as in mix-real / dv-crops
            files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
        probs = predict_probs(model, files, size, tta=tta)
        names = [class_names[i] for i in probs.argmax(1)]
        hist = Counter(names)
        if cls in class_names:
            cal_probs.append(probs)
            cal_true.extend([class_names.index(cls)] * len(names))
        n = len(names)
        groups[group] = {"class": cls, "images": n, "known_class": cls in class_names,
                         "accuracy": hist[cls] / n, "predicted_ok": hist["OK"] / n,
                         "predictions": dict(hist.most_common())}
        if cls == "OK":
            good, good_flagged = good + n, good_flagged + n - hist["OK"]
        else:
            bad, bad_passed = bad + n, bad_passed + hist["OK"]
    result = {"groups": groups,
              "false_call_rate": good_flagged / good if good else None,
              "escape_rate": bad_passed / bad if bad else None}
    if tta:
        result["tta"] = True
    if calibration:
        probs_all = np.concatenate(cal_probs) if cal_probs else np.zeros((0, len(class_names)))
        result["calibration"] = {**calibration_metrics(probs_all, np.asarray(cal_true, dtype=np.int64)),
                                 "images": int(len(cal_true))}
    return result


def format_report(result: dict[str, Any]) -> str:
    lines = [f"{'folder':24s} {'n':>5s} {'acc':>6s} {'->OK':>6s}  predictions"]
    for group, g in result["groups"].items():
        lines.append(f"{group:24s} {g['images']:5d} {g['accuracy']:6.3f} {g['predicted_ok']:6.3f}  {g['predictions']}")
    fc, esc = result["false_call_rate"], result["escape_rate"]
    lines.append(f"false-call rate (good flagged): {'-' if fc is None else f'{fc:.3f}'}   "
                 f"escape rate (defect passed as OK): {'-' if esc is None else f'{esc:.3f}'}")
    if "calibration" in result:
        lines.append(f"{format_calibration(result['calibration'])}  (known classes, {result['calibration']['images']} images)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True, help="folder with one sub-folder per class")
    ap.add_argument("--limit", type=int, default=1000, help="max images per sub-folder (random subset)")
    ap.add_argument("--size", type=int, default=256, help="ROI size if the checkpoint has no config")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation: average over horizontal/vertical flips and quarter turns")
    ap.add_argument("--calibration", action="store_true", help="report ECE (15 bins), NLL and Brier score")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    model, class_names, checkpoint = load_checkpoint_model(args.checkpoint, "cpu")
    result = evaluate(model, class_names, args.real, roi_size(checkpoint, args.size), args.limit,
                      tta=args.tta, calibration=args.calibration)
    result["checkpoint"] = str(args.checkpoint)
    print(format_report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
