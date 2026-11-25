"""Random subsets, image reading and resizing shared by the sim-to-real tools (``tools/*``).

Kept apart from :mod:`simple_sim.images` (dependency-free folder listing): this module needs
NumPy and OpenCV.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np

_T = TypeVar("_T")


def sample_subset(items: Sequence[_T], limit: int, rng: np.random.Generator) -> list[_T]:
    """Up to *limit* items of *items* (random, original order kept); ``limit <= 0`` or a short
    sequence returns all of them. Draws ``rng.choice(len(items), limit, replace=False)`` only when
    a subset is taken, so seeded runs pick the same items as before."""
    if limit <= 0 or len(items) <= limit:
        return list(items)
    return [items[i] for i in sorted(rng.choice(len(items), limit, replace=False))]


def read_bgr_or_exit(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """``cv2.imread`` of *path*; ``SystemExit`` with a short message when it cannot be read (CLI tools)."""
    img = cv2.imread(str(path), flags)
    if img is None:
        raise SystemExit(f"Failed to read {path}")
    return img


def resize_to(img: np.ndarray, size: tuple[int, int] | int) -> np.ndarray:
    """*img* resized to *size* (``(width, height)`` or a square side): cubic when enlarging (the
    inspection pipeline upscales its small crops), area when shrinking."""
    wh = (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))
    interp = cv2.INTER_CUBIC if img.shape[1] < wh[0] else cv2.INTER_AREA
    return cv2.resize(img, wh, interpolation=interp)


def percentiles_10_50_90(values: Iterable[float], ndigits: int = 3) -> list[float]:
    """10th/50th/90th percentile of *values*, rounded to *ndigits*."""
    return [round(float(v), ndigits) for v in np.percentile(list(values), (10, 50, 90))]
