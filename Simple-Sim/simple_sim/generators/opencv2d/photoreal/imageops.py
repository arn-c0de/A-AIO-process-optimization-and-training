"""Image primitives shared by every photoreal layer: blending, coordinate grids, value noise, text masks,
rotated-rectangle masks, mirror shading of height maps and the contact shadow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .palette import HALF, LIGHT

if TYPE_CHECKING:  # context imports the meniscus model, which imports these primitives
    from .context import RenderContext


def clip8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


_GRID_CACHE_MAX_PX = 1 << 23  # grids up to 8 Mpx are kept (2 x 32 MB); larger ones are built per call
_grid_cache: tuple[np.ndarray, np.ndarray] = (np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32))


def grid(h: int, w: int, y0: int = 0, x0: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(yy, xx)`` float32 pixel coordinates of the window ``[y0:y0+h, x0:x0+w]``: the same values as
    ``np.mgrid[y0:y0+h, x0:x0+w].astype(np.float32)``, as read-only views of one shared grid that
    grows to the largest window asked for (no allocation per body, lead, pad, bulge or shadow)."""
    global _grid_cache
    y1, x1 = y0 + h, x0 + w
    yy, xx = _grid_cache
    if yy.shape[0] < y1 or yy.shape[1] < x1:
        gh, gw = max(y1, yy.shape[0]), max(x1, yy.shape[1])
        if gh * gw > _GRID_CACHE_MAX_PX:
            return tuple(np.mgrid[y0:y1, x0:x1].astype(np.float32))  # type: ignore[return-value]
        yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
        yy.setflags(write=False)
        xx.setflags(write=False)
        _grid_cache = (yy, xx)  # one reference swap: safe for concurrent readers
    return yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]


def blend(img: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> None:
    """In-place ``img = img*(1-a) + color*a`` with a 2D alpha map."""
    # Most layers (vias, traces, text, solder blobs) cover a small part of the board: only touch the
    # bounding box of the non-zero alpha (identical result, outside it alpha is 0).
    x, y, bw, bh = cv2.boundingRect((alpha > 0).astype(np.uint8))
    if bw == 0 or bh == 0:
        return
    sub = img[y:y + bh, x:x + bw]
    a = alpha[y:y + bh, x:x + bw, None]
    if color.ndim == 3 and color.shape[:2] == img.shape[:2]:  # per-pixel colour (shaded layer)
        color = color[y:y + bh, x:x + bw]
    sub *= 1.0 - a
    sub += color * a


def gradient(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``np.gradient(f)`` of a 2D float map (``(gy, gx)``, unit spacing, first-order edges): the same
    float operations in the same order, without its per-call overhead (called twice per joint)."""
    h, w = f.shape
    if h < 2 or w < 2 or not np.issubdtype(f.dtype, np.inexact):
        return np.gradient(f)  # type: ignore[return-value]
    gx, gy = np.empty_like(f), np.empty_like(f)
    gx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0
    gx[:, 0] = f[:, 1] - f[:, 0]
    gx[:, -1] = f[:, -1] - f[:, -2]
    gy[1:-1, :] = (f[2:, :] - f[:-2, :]) / 2.0
    gy[0, :] = f[1, :] - f[0, :]
    gy[-1, :] = f[-1, :] - f[-2, :]
    return gy, gx


def heightmap_normals(hmap: np.ndarray) -> np.ndarray:
    """Unit surface normals (H, W, 3: x, y, up) of a height map in pixels (central differences)."""
    h, w = hmap.shape
    gy, gx = gradient(hmap)
    n = np.empty((h, w, 3), dtype=gx.dtype)
    np.negative(gx, out=n[..., 0])
    np.negative(gy, out=n[..., 1])
    n[..., 2] = 1.0
    # == np.linalg.norm(n, axis=2, keepdims=True): sqrt of the squares summed left to right.
    n /= np.sqrt((gx * gx + gy * gy) + 1.0)[..., None]
    return n


def shade_solder(hmap: np.ndarray, base: np.ndarray, env: np.ndarray, spec_power: float, spec_gain: float) -> np.ndarray:
    """Shade a solder height map as a rough mirror. Returns float BGR (H, W, 3).

    Solder is metallic: mostly a reflection of the surroundings. Flat areas mirror the bright
    overhead illumination, slopes mirror the board (*env*), and slopes facing the light catch a
    highlight. Lead-free solder is frosted rather than a perfect mirror and real cameras (ring
    lights, phone flash, room light from several sides) see joints as bright grey blobs, so the
    diffuse share and the falloff of the sky reflection are generous.
    """
    n = heightmap_normals(hmap)
    up = np.clip(2.0 * n[..., 2] ** 2 - 1.0, 0.0, 1.0)  # z of the reflected view ray
    sky = up**1.5
    spec = np.clip(n @ HALF, 0.0, 1.0) ** spec_power
    out = base * (0.22 + 0.22 * np.clip(n @ LIGHT, 0.0, 1.0))[..., None]
    out += base * (0.7 * sky)[..., None] + env * (0.5 * (1.0 - sky))[..., None]
    out += (255.0 * spec_gain * spec)[..., None]
    return out


def solder_env(base: np.ndarray) -> np.ndarray:
    """What the slopes of a joint mirror: the board, the part's end caps and neighbouring metal, i.e. a light grey-green."""
    return base * 0.6 + 55.0


def smooth_noise(shape: tuple[int, int], cell: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, roughly unit-amplitude value noise with feature size *cell* pixels."""
    h, w = shape
    small = rng.normal(0.0, 1.0, (int(h / cell) + 2, int(w / cell) + 2)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def skip_smooth_noise(shape: tuple[int, int], cell: float, rng: np.random.Generator) -> None:
    """Advance *rng* exactly as :func:`smooth_noise` would, without building the field (for a layer
    that does not use it but must leave the random stream as before)."""
    h, w = shape
    rng.normal(0.0, 1.0, (int(h / cell) + 2, int(w / cell) + 2))


def text_layer(shape: tuple[int, int], text: str, center: tuple[float, float], height_px: float,
                angle_deg: float, thickness: int, font: int = cv2.FONT_HERSHEY_SIMPLEX) -> np.ndarray:
    """Anti-aliased text mask (0..1) rotated by *angle_deg* around *center*."""
    scale = max(0.15, height_px / 22.0)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    canvas = np.zeros((th * 3, tw + th * 2), dtype=np.uint8)
    cv2.putText(canvas, text, (th, th * 2), font, scale, 255, thickness, lineType=cv2.LINE_AA)
    ch, cw = canvas.shape
    m = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), -angle_deg, 1.0)
    m[0, 2] += center[0] - cw / 2.0
    m[1, 2] += center[1] - ch / 2.0
    out = cv2.warpAffine(canvas, m, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    return out.astype(np.float32) / 255.0


def local_coords(h: int, w: int, center: tuple[float, float], angle_deg: float,
                 y0: int = 0, x0: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates ``(u, v)`` of the window ``[y0:y0+h, x0:x0+w]`` in the frame rotated by *angle_deg* about
    *center* (image coordinates). A window gives exactly the values of the same pixels of the full frame."""
    yy, xx = grid(h, w, y0, x0)
    a = np.radians(angle_deg)
    dx, dy = xx - center[0], yy - center[1]
    return dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)


def rect_alpha(h: int, w: int, center, length: float, width: float, angle_deg: float, y0: int = 0, x0: int = 0):
    """Local coords (u along *length*) and an anti-aliased 0..1 mask of a rotated rectangle (window as in
    :func:`local_coords`). The mask is 0 outside the rectangle grown by half a pixel."""
    u, v = local_coords(h, w, center, angle_deg, y0, x0)
    return u, v, np.clip(0.5 - np.maximum(np.abs(u) - length / 2.0, np.abs(v) - width / 2.0), 0.0, 1.0)


WINDOW_MARGIN = 3  # px around a shape's exact extent: covers the AA band, float32 rounding and gradients


def local_window(shape: tuple[int, int], center: tuple[float, float], angle_deg: float, u: tuple[float, float],
                 v: tuple[float, float], margin: float = WINDOW_MARGIN) -> tuple[int, int, int, int] | None:
    """``(y0, y1, x0, x1)``: the axis-aligned window (clipped to *shape*) around the local rectangle
    ``u[0]..u[1]`` x ``v[0]..v[1]`` of the frame of :func:`local_coords`, grown by *margin* pixels;
    ``None`` when it misses the image.

    Elementwise layers computed on this window equal the full-frame computation pixel for pixel, and
    every pixel outside it is untouched by the shape, so drawing the window is the same as drawing
    the frame at a fraction of the cost.
    """
    a = np.radians(angle_deg)
    c, s = float(np.cos(a)), float(np.sin(a))
    us, vs = np.array([u[0], u[1], u[1], u[0]], dtype=np.float64), np.array([v[0], v[0], v[1], v[1]], dtype=np.float64)
    xs, ys = center[0] + us * c - vs * s, center[1] + us * s + vs * c
    h, w = shape
    x0, y0 = max(0, int(np.floor(xs.min() - margin))), max(0, int(np.floor(ys.min() - margin)))
    x1, y1 = min(w, int(np.ceil(xs.max() + margin)) + 1), min(h, int(np.ceil(ys.max() + margin)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, y1, x0, x1


def rect_window(shape: tuple[int, int], center: tuple[float, float], length: float, width: float, angle_deg: float,
                margin: float = WINDOW_MARGIN) -> tuple[int, int, int, int] | None:
    """:func:`local_window` of the rectangle of :func:`rect_alpha` (``length`` x ``width`` about *center*)."""
    return local_window(shape, center, angle_deg, (-length / 2.0, length / 2.0), (-width / 2.0, width / 2.0), margin)


def blur_window(mask: np.ndarray, sigma: float, extra: int = 0) -> tuple[int, int, int, int]:
    """``(y0, y1, x0, x1)`` around the non-zero pixels of *mask* plus the reach of a Gaussian blur of *sigma*
    (and *extra* px); the whole frame when that reaches an image border (where the blur's border
    handling would see the window edge instead of the frame edge)."""
    h, w = mask.shape[:2]
    x, y, bw, bh = cv2.boundingRect(mask)
    reach = gaussian_radius(sigma, mask.dtype) + int(extra) + 1
    x0, y0, x1, y1 = x - reach, y - reach, x + bw + reach, y + bh + reach
    if bw == 0 or bh == 0 or x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return 0, h, 0, w
    return y0, y1, x0, x1


class WindowedNoise:
    """Random fields of a layer drawn for the full *shape* (so the random stream advances exactly as for
    a full-frame layer) and handed out cropped to the computation window ``(y0, y1, x0, x1)``."""

    __slots__ = ("shape", "window")

    def __init__(self, shape: tuple[int, int], window: tuple[int, int, int, int]):
        self.shape, self.window = shape, window

    def _crop(self, a: np.ndarray) -> np.ndarray:
        y0, y1, x0, x1 = self.window
        return a[y0:y1, x0:x1]

    def normal(self, rng: np.random.Generator, sd: float, *, channels: int | None = None) -> np.ndarray:
        """``rng.normal(0, sd, shape[, channels])`` as float32, cropped."""
        size = self.shape if channels is None else (*self.shape, channels)
        return self._crop(rng.normal(0.0, sd, size)).astype(np.float32)

    def smooth(self, rng: np.random.Generator, cell: float) -> np.ndarray:
        """:func:`smooth_noise` of the full shape, cropped."""
        return self._crop(smooth_noise(self.shape, cell, rng))


def gaussian_radius(sigma: float, dtype: np.dtype) -> int:
    """Kernel radius of ``cv2.GaussianBlur(src, (0, 0), sigma)`` (OpenCV: 3 sigma for 8-bit, 4 sigma otherwise)."""
    k = int(round(sigma * (3 if np.dtype(dtype) == np.uint8 else 4) * 2 + 1)) | 1
    return k // 2


def contact_shadow(ctx: RenderContext, img: np.ndarray, mask: np.ndarray, strength: float, spread: float) -> None:
    """Soft shadow under a part, shifted away from the light.

    Computed on the bounding box of *mask* plus a margin that covers the dilation, the blur kernel
    and the shift; outside it the shadow is exactly zero, so the result equals the full-frame one.
    Masks close to the frame edge (where the shift would wrap around) use the full frame.
    """
    h, w = mask.shape[:2]
    strength = strength * ctx.look.shadow  # diffuse scanner light (render.look: quins): fainter shadows
    shift = int(1.5 * ctx.ss)
    sigma = spread * 0.8
    margin = int(spread) + int(round(sigma * 3 * 2 + 1)) // 2 + shift + 2
    x, y, bw, bh = cv2.boundingRect(mask)
    if bw == 0 or bh == 0:
        return
    # Clipped to the frame: inside the window the dilation and blur equal the full-frame ones (the
    # mask is 0 within the margin of the window's inner sides, and a side on the frame edge sees the
    # same border as the full frame).
    x0, y0, x1, y1 = max(0, x - margin), max(0, y - margin), min(w, x + bw + margin), min(h, y + bh + margin)
    sh = cv2.dilate(mask[y0:y1, x0:x1], np.ones((int(spread), int(spread)), np.uint8))
    sh = cv2.GaussianBlur(sh, (0, 0), sigma).astype(np.float32)
    sh /= 255.0
    sh *= -strength  # in place: 1 + (-strength * sh) == 1 - strength * sh bit for bit
    sh += 1.0
    # Shifted away from the light like np.roll(shadow, (shift, shift)) of the full frame: the window
    # lands at rows y0+shift.. and columns x0+shift.. modulo the frame (up to 4 pieces when it wraps);
    # everywhere else the factor is exactly 1.
    for ry0, ry1, sy in _rolled_spans(y0, y1, shift, h):
        for rx0, rx1, sx in _rolled_spans(x0, x1, shift, w):
            img[ry0:ry1, rx0:rx1] *= sh[sy:sy + ry1 - ry0, sx:sx + rx1 - rx0, None]


def _rolled_spans(a0: int, a1: int, shift: int, n: int) -> list[tuple[int, int, int]]:
    """Where the span ``[a0, a1)`` of an axis of length *n* lands after ``np.roll(.., shift)``:
    ``(start, stop, offset)`` pieces, *offset* being the first span index that lands in the piece."""
    start, length = (a0 + shift) % n, a1 - a0
    if start + length <= n:
        return [(start, start + length, 0)]
    return [(start, n, 0), (0, start + length - n, n - start)]
