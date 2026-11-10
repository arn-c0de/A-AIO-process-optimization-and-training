#!/usr/bin/env python3
"""Side-by-side montage and gap statistics of real crops vs. a generated dataset, class by class.

The montage that `render.look: quins` was tuned with: one row of real crops per class folder (the
layout of ``tools/real_crops.py`` / ``tools/eval_real.py``) directly above one row of synthetic
images of the matching Simple-Sim class (``OK``, ``MISSING``, ...; a folder's ``@group`` suffix and
``DEFECT@...`` folders have no synthetic row). All tiles are resized to ``--size`` (cubic, as the
inspection pipeline upscales its small crops).

``--stats`` prints, per row, the statistics that separated real from synthetic crops while tuning
(means over the images, nothing else about the images leaves the machine):

* ``L_std`` / ``b_std``: standard deviation of LAB lightness and yellow-blue (contrast, colour
  fringes, tan bodies)
* ``>200`` / ``>150``: share of grey levels above 200 / 150 (metal highlights / bright metal)
* ``p90`` / ``p99``: grey-level percentiles
* ``metal``: median BGR of the pixels above 200 (the colour cast of the solder and terminals)

Usage:
    python tools/look_montage.py --real real_eval/ --data data/run_0402_c_quins --out montage.png [--stats] [--per-row 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.io_utils import iter_json_lines  # noqa: E402
from tools.eval_real import class_folders  # noqa: E402

STAT_KEYS = ("L_std", "b_std", ">200", ">150", "p90", "p99", "metal")


def load_tile(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read {path}")
    interp = cv2.INTER_CUBIC if img.shape[1] < size else cv2.INTER_AREA
    return cv2.resize(img, (size, size), interpolation=interp)


def dataset_images(data: Path) -> dict[str, list[Path]]:
    """Image files of a generated dataset per class (from ``labels.jsonl``, sorted by id)."""
    # meta.jsonl knows the real file of an id (merged datasets: images/train_000003.png) and which rows
    # are real crops mixed in by tools/mix_real.py; those are not synthetic renders and are left out.
    image_paths: dict[str, str] = {}
    real_ids: set[str] = set()
    meta_path = data / "meta.jsonl"
    if meta_path.is_file():
        for meta in iter_json_lines(meta_path):
            if meta.get("render_backend") == "real":
                real_ids.add(str(meta.get("id")))
            elif meta.get("image_path"):
                image_paths[str(meta.get("id"))] = str(meta["image_path"])
    out: dict[str, list[Path]] = {}
    for row in iter_json_lines(data / "labels.jsonl"):
        sample_id = str(row["id"])
        if sample_id in real_ids:
            continue
        rel = image_paths.get(sample_id)
        path = data / rel if rel else data / "images" / (Path(sample_id).name + ".png")
        out.setdefault(str(row["class_name"]), []).append(path)
    return {k: sorted(v) for k, v in out.items()}


def pick(files: list[Path], n: int, rng: np.random.Generator) -> list[Path]:
    if len(files) <= n:
        return files
    return [files[i] for i in sorted(rng.choice(len(files), n, replace=False))]


def image_stats(imgs: list[np.ndarray]) -> dict[str, Any]:
    grey = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]).astype(np.float32)
    lab = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2LAB) for i in imgs]).astype(np.float32)
    bright = np.concatenate([i.reshape(-1, 3)[g.reshape(-1) > 200] for i, g in zip(imgs, grey)])
    return {"L_std": float(np.mean(lab[..., 0].std(axis=(1, 2)))), "b_std": float(np.mean(lab[..., 2].std(axis=(1, 2)))),
            ">200": float(np.mean(grey > 200)), ">150": float(np.mean(grey > 150)),
            "p90": float(np.percentile(grey, 90)), "p99": float(np.percentile(grey, 99)),
            "metal": [int(v) for v in np.median(bright, axis=0)] if len(bright) else None}


def format_stats(rows: dict[str, dict[str, Any]]) -> str:
    lines = [f"{'row':28s}" + "".join(f"{k:>16s}" for k in STAT_KEYS)]
    for name, st in rows.items():
        cells = []
        for k in STAT_KEYS:
            v = st[k]
            cells.append("-" if v is None else (f"{v:.1%}" if k.startswith(">") else (str(tuple(v)) if k == "metal" else f"{v:.1f}")))
        lines.append(f"{name:28s}" + "".join(f"{c:>16s}" for c in cells))
    return "\n".join(lines)


def build(real: Path, data: Path, size: int, per_row: int, stat_limit: int, seed: int = 0,
          with_stats: bool = False) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    synthetic = dataset_images(data)
    rows: list[tuple[str, list[Path]]] = []
    for group, cls, files in class_folders(real):
        rows.append((f"real {group}", files))
        if cls in synthetic and "@" not in group:
            rows.append((f"synthetic {cls}", synthetic[cls]))
    tiles, stats = [], {}
    for name, files in rows:
        row_imgs = [load_tile(p, size) for p in pick(files, per_row, rng)]
        while len(row_imgs) < per_row:
            row_imgs.append(np.zeros((size, size, 3), np.uint8))
        strip = np.concatenate(row_imgs, axis=1)
        cv2.putText(strip, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(strip)
        if with_stats:
            stats[name] = image_stats([load_tile(p, size) for p in pick(files, stat_limit, rng)])
    return np.concatenate(tiles, axis=0), stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real", type=Path, required=True, help="real crops, one sub-folder per class")
    ap.add_argument("--data", type=Path, required=True, help="generated dataset directory")
    ap.add_argument("--out", type=Path, required=True, help="montage PNG")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--per-row", type=int, default=8)
    ap.add_argument("--stats", action="store_true", help="print the gap statistics per row")
    ap.add_argument("--stat-limit", type=int, default=300, help="images per row for the statistics")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not (args.data / "labels.jsonl").is_file():
        raise SystemExit(f"{args.data} is not a generated dataset (no labels.jsonl)")
    montage, stats = build(args.real, args.data, args.size, args.per_row, args.stat_limit, args.seed, args.stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), montage)
    print(f"montage {montage.shape[1]}x{montage.shape[0]} -> {args.out}")
    if stats:
        print(format_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
