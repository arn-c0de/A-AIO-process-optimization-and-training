#!/usr/bin/env python3
"""Fit the camera model (``render.camera``) to real camera ROIs.

Compares two folders - real ROIs from the production camera and synthetic ROIs rendered with the
current settings - on two low-level statistics and prints the ``render.camera`` values that make
the synthetic images match:

* **Sharpness** (steepness of the strongest edges relative to contrast). The extra Gaussian blur that brings the
  sharper set down to the other one is folded into ``blur_sigma`` (Gaussian blurs add in quadrature).
* **Sensor noise** (Immerkaer estimator on the flattest half of each image). The variance ratio scales ``read_noise``
  and ``shot_noise``.

Both folders must be at the same scale (``roi.mm_per_px``) and show similar content (e.g. the
same package); images are not resized.

Usage:
    python tools/calibrate_camera.py --real data/real_rois --synthetic data/run_0001/images \
        [--config configs/run_0001.yaml] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.generators.opencv2d.photoreal import CAMERA_DEFAULTS, IMAGE_EXTENSIONS  # noqa: E402

_BLUR_GRID = np.arange(0.0, 3.01, 0.05)
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def load_gray(folder: Path, limit: int, rng: np.random.Generator) -> list[np.ndarray]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found in {folder}")
    if len(files) > limit:
        files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) for p in files]


def sharpness(img: np.ndarray) -> float:
    """Steepness of the strongest edges relative to image contrast (~ 1 / total blur sigma).

    Light pre-smoothing and using only the top 5 % gradients keep sensor noise out of the measure.
    """
    sm = cv2.GaussianBlur(img, (0, 0), 0.8)
    grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1))
    top = grad[grad >= np.percentile(grad, 95)].mean()
    contrast = np.percentile(sm, 95) - np.percentile(sm, 5)
    return float(top / (contrast + 1e-6))


def noise_sigma(img: np.ndarray) -> float:
    """Immerkaer noise estimate on the flattest half of the image (edges/texture would read as noise)."""
    resp = np.abs(cv2.filter2D(img, cv2.CV_32F, _IMMERKAER))[1:-1, 1:-1]
    sm = cv2.GaussianBlur(img, (0, 0), 1.5)
    grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1))[1:-1, 1:-1]
    flat = grad <= np.percentile(grad, 50)
    return float(np.sqrt(np.pi / 2.0) * resp[flat].mean() / 6.0)


def _median_sharpness(images: list[np.ndarray], extra_blur: float) -> float:
    if extra_blur <= 0:
        return float(np.median([sharpness(i) for i in images]))
    return float(np.median([sharpness(cv2.GaussianBlur(i, (0, 0), extra_blur)) for i in images]))


def blur_offset(real: list[np.ndarray], syn: list[np.ndarray]) -> float:
    """Signed extra blur for the synthetic images: > 0 blur more, < 0 blur less."""
    s_real, s_syn = _median_sharpness(real, 0.0), _median_sharpness(syn, 0.0)
    sharper, target, sign = (syn, s_real, 1.0) if s_syn > s_real else (real, s_syn, -1.0)
    errors = [abs(_median_sharpness(sharper, s) - target) for s in _BLUR_GRID]
    return sign * float(_BLUR_GRID[int(np.argmin(errors))])


def fit_camera(real: list[np.ndarray], syn: list[np.ndarray], current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measured statistics and the fitted ``render.camera`` (starting from *current*)."""
    cur = {**CAMERA_DEFAULTS, **(current or {})}
    offset = blur_offset(real, syn)
    blur = [float(np.sqrt(max(b * b + np.sign(offset) * offset * offset, 0.05**2))) for b in cur["blur_sigma"]]
    n_real = float(np.median([noise_sigma(i) for i in real]))
    n_syn = float(np.median([noise_sigma(i) for i in syn]))
    k2 = (n_real / max(n_syn, 1e-6)) ** 2
    return {
        "stats": {"sharpness_real": _median_sharpness(real, 0.0), "sharpness_synthetic": _median_sharpness(syn, 0.0),
                  "noise_real": n_real, "noise_synthetic": n_syn, "extra_blur": offset, "noise_variance_ratio": k2},
        "camera": {"blur_sigma": [round(b, 3) for b in blur],
                   "read_noise": [round(v * k2, 3) for v in cur["read_noise"]],
                   "shot_noise": round(float(cur["shot_noise"]) * k2, 4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real", type=Path, required=True, help="Folder with real camera ROIs (searched recursively)")
    ap.add_argument("--synthetic", type=Path, required=True, help="Folder with synthetic ROIs rendered with --config")
    ap.add_argument("--config", type=Path, help="Run config the synthetic images were rendered with (its render.camera)")
    ap.add_argument("--max-images", type=int, default=300, help="Max images per side (random subset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="Also write the result as JSON")
    args = ap.parse_args()
    from simple_sim.cli_utils import configure_cli_logging

    configure_cli_logging()

    current = None
    if args.config:
        from simple_sim.config import load_config

        current = (load_config(args.config).get("render") or {}).get("camera")
    rng = np.random.default_rng(args.seed)
    result = fit_camera(load_gray(args.real, args.max_images, rng), load_gray(args.synthetic, args.max_images, rng), current)

    st = result["stats"]
    print(f"Sharpness  real {st['sharpness_real']:.3f}  synthetic {st['sharpness_synthetic']:.3f}  "
          f"-> extra blur {st['extra_blur']:+.2f} px")
    print(f"Noise      real {st['noise_real']:.2f}  synthetic {st['noise_synthetic']:.2f}  "
          f"-> noise variance x{st['noise_variance_ratio']:.2f}")
    cam = result["camera"]
    print("\nPut this into the run config, re-render and run again until both factors are ~0 / ~1:\n")
    print("render:\n  camera:")
    print(f"    blur_sigma: {cam['blur_sigma']}\n    read_noise: {cam['read_noise']}\n    shot_noise: {cam['shot_noise']}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
