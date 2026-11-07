"""Bare board: procedural solder mask on laminate (or a real board texture) and the glossy mask reflection.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from simple_sim.images import IMAGE_EXTENSIONS  # also re-exported by the photoreal package

from .context import RenderContext
from .imageops import blend, grid, smooth_noise
from .look import MASK_GLOSS_PROBABILITY  # noqa: F401 (re-exported; the look owns the default)
from .palette import jittered


@lru_cache(maxsize=8)
def texture_files(texture_dir: str) -> tuple[Path, ...]:
    """Sorted image files in *texture_dir* (sorted so a seed always picks the same file)."""
    return tuple(sorted(p for p in Path(texture_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))


TEXTURE_CACHE_SIZE = 8  # decoded board photos kept per process (a 12 Mpx photo is ~36 MB)


@lru_cache(maxsize=TEXTURE_CACHE_SIZE)
def _decoded_texture(path: str, mtime_ns: int, size: int) -> np.ndarray:
    """``cv2.imread`` of *path*, read-only and cached per process. *mtime_ns* / *size* are part of the
    key, so a photo replaced on disk is decoded again."""
    del mtime_ns, size
    tex = cv2.imread(path, cv2.IMREAD_COLOR)
    if tex is None:
        raise OSError(f"Failed to read substrate texture: {path}")
    tex.setflags(write=False)
    return tex


def load_texture(path: Path | str) -> np.ndarray:
    """Decoded BGR photo of *path* (read-only, shared: copy before modifying it)."""
    st = Path(path).stat()
    return _decoded_texture(str(path), st.st_mtime_ns, st.st_size)


def texture_substrate(ctx: RenderContext, h: int, w: int, texture_dir: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random crop of a real photo of empty board area, randomly flipped/rotated, scaled to (h, w)."""
    files = texture_files(texture_dir)
    path = files[int(rng.integers(len(files)))]
    tex = load_texture(path)  # read-only: only views and new arrays below
    tex = np.rot90(tex, int(rng.integers(4)))
    if rng.random() < 0.5:
        tex = tex[:, ::-1]
    th, tw = tex.shape[:2]
    rh, rw = h // ctx.ss, w // ctx.ss  # ROI size at camera resolution
    if th >= rh and tw >= rw:  # same scale as the camera: crop a ROI-sized patch
        y, x = int(rng.integers(th - rh + 1)), int(rng.integers(tw - rw + 1))
        tex = tex[y:y + rh, x:x + rw]
    img = cv2.resize(np.ascontiguousarray(tex), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return img, np.median(img.reshape(-1, 3), axis=0)


def procedural_substrate(ctx: RenderContext, h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    palette = ctx.look.mask_palette
    base = jittered(np.array(palette[int(rng.integers(len(palette)))], dtype=np.float32), rng, 0.92, 1.08)
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = base

    # Low-frequency mottling (resin thickness) + fibre weave + fine grain.
    coarse = smooth_noise((h, w), 32 * ctx.ss, rng)
    period = float(rng.uniform(6.0, 9.0)) * ctx.ss
    # Separable: the same float32 values as on a full 2D grid, computed on two 1D axes.
    xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    weave = np.sin(2 * np.pi * xs / period)[None, :] * np.sin(2 * np.pi * ys / period)[:, None]
    img *= (1.0 + ctx.look.mask_mottle * coarse + ctx.look.mask_weave * weave)[..., None]
    img += rng.normal(0.0, 1.2, (h, w, 1)).astype(np.float32)
    return img, base


def mask_sheen(ctx: RenderContext, img: np.ndarray, base: np.ndarray, rng: np.random.Generator, probability: float) -> None:
    """Glossy solder mask mirroring a light source (phone flash, ring light, window).

    Real photos show broad, soft, whitish-green glare patches on the mask, broken up by its
    orange-peel surface. Applied to the bare board only (parts and joints are drawn on top).
    No random draw when disabled (``probability <= 0``), so turning it off leaves later draws alone.
    """
    if probability <= 0.0 or rng.random() >= probability:
        return
    h, w = img.shape[:2]
    field = 0.6 * smooth_noise((h, w), max(h, w) / float(rng.uniform(0.8, 1.5)), rng)
    a = float(rng.uniform(0.0, 2.0 * np.pi))
    yy, xx = grid(h, w)
    field += 1.2 * ((xx - w / 2.0) * np.cos(a) + (yy - h / 2.0) * np.sin(a)) / max(h, w)  # light comes from one side
    sheen = np.clip((field - float(rng.uniform(0.1, 0.6))) / 0.7, 0.0, 1.0) ** 1.5
    peel = 0.65 + 0.35 * np.clip(smooth_noise((h, w), 3.0 * ctx.ss, rng) * 0.5 + 0.5, 0.0, 1.0)  # orange peel
    glare = np.clip(base * 0.5 + 115.0, 0, 255)  # desaturated, much lighter mask colour
    blend(img, glare, np.clip(sheen * peel * float(rng.uniform(0.3, 0.75)), 0.0, 1.0))
