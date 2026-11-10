#!/usr/bin/env python3
"""Colour/contrast statistics of image folders, to compare a synthetic look with real crops.

Every image is resized to ``--size`` (cubic: small real crops are upscaled like the inspection
pipeline does) and measured; the table shows the 10th/50th/90th percentile over the images:

* ``mask_h/s/v``  median HSV (OpenCV scale, H 0-180) of the saturated, not too bright pixels
  (the solder mask), per image
* ``p1``, ``p99`` 1st/99th grey-level percentile (shadows / metal highlights)
* ``white``       share of bright neutral pixels (metal, silkscreen, white housings)
* ``spec``        share of near-saturated pixels (V > 235)
* ``sharp``       variance of the Laplacian (resolution and blur)

Only aggregate numbers are printed, so it can be run on production crops that must not leave the
machine. ``render.look: quins`` was tuned with it (``docs/guides/PHOTOREAL_RENDERING.md``).

Usage:
    python tools/look_stats.py data/run_0402_c/images /path/to/real/crops [--limit 600] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.images import list_synthetic_images  # noqa: E402

METRICS = ("mask_h", "mask_s", "mask_v", "p1", "p99", "white", "spec", "sharp")


def image_stats(img: np.ndarray, size: int) -> dict[str, float]:
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv.T
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = {"p1": float(np.percentile(grey, 1)), "p99": float(np.percentile(grey, 99)),
           "white": float(np.mean((v > 180) & (s < 50))), "spec": float(np.mean(v > 235)),
           "sharp": float(cv2.Laplacian(grey, cv2.CV_64F).var())}
    mask = (s > 50) & (v > 30) & (v < 170)
    if mask.sum() > 0.05 * mask.size:
        out.update(mask_h=float(np.median(h[mask])), mask_s=float(np.median(s[mask])), mask_v=float(np.median(v[mask])))
    return out


def folder_stats(folder: Path, limit: int, size: int, seed: int = 0) -> dict[str, list[float] | int]:
    files = list_synthetic_images(folder)  # a dataset's images/ without the mixed-in real crops
    if not files:
        raise SystemExit(f"No images found in {folder}")
    if len(files) > limit:
        rng = np.random.default_rng(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
    rows = [image_stats(img, size) for img in (cv2.imread(str(p), cv2.IMREAD_COLOR) for p in files) if img is not None]
    out: dict[str, list[float] | int] = {"images": len(rows)}
    for key in METRICS:
        values = [r[key] for r in rows if key in r]
        out[key] = [round(float(x), 4) for x in np.percentile(values, (10, 50, 90))] if values else []
    return out


def format_table(results: dict[str, dict]) -> str:
    names = list(results)
    width = max(14, *(len(n) for n in names))
    lines = ["metric".ljust(8) + "".join(n[-width:].rjust(width + 2) for n in names)]
    for key in ("images", *METRICS):
        cells = []
        for n in names:
            v = results[n][key]
            cells.append(str(v) if key == "images" else ("-" if not v else "/".join(f"{x:.3g}" for x in v)))
        lines.append(key.ljust(8) + "".join(c.rjust(width + 2) for c in cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--limit", type=int, default=600, help="random images per folder (default 600)")
    ap.add_argument("--size", type=int, default=256, help="resize to this square size first (default 256)")
    ap.add_argument("--json", type=Path, help="also write the numbers to this file")
    args = ap.parse_args(argv)
    results = {str(f): folder_stats(f, args.limit, args.size) for f in args.folders}
    print("p10/p50/p90 over the images")
    print(format_table(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
