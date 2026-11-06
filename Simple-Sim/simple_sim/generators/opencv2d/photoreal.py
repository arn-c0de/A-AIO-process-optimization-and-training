"""Photorealistic base rendering for the OpenCV 2D backend (``render.style: photoreal``).

Produces the un-augmented ROI (substrate, pads, solder fillets, component body) with
procedural textures and simple physically motivated shading, rendered at 2x (``render.supersampling``: 1-4) and
downsampled for anti-aliasing, then passed through a simple camera model (lens blur, uneven
illumination, highlight roll-off, sensor noise) so it does not look like a vector drawing. Every random choice is drawn from the passed ``rng``,
so a sample is fully determined by its seed.

Visual targets were calibrated against photos of real assembled boards: dark olive
solder mask with copper traces shimmering through, vias, white silkscreen, bright
concave solder fillets, black resistors with white value codes, tan MLCCs, black
molded SOT/SOIC/QFN bodies with laser markings and metallic gull-wing leads.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from simple_sim.board import DEFAULT_MM_PER_PX, NONE as ROUTING_NONE, STREAM_PLACE, board_options, build_window_board, family_of, footprint_pads, neighbour_layout
from simple_sim.board.placement import pad_ref as _pad_ref
from simple_sim.board.raster import CopperLayers, rasterize

from .draw import _draw_solder_bridge, _param_selector, body_geometry, lifted_corner_geometry

DEFAULT_SUPERSAMPLING = 2
SUPERSAMPLING_CHOICES = (1, 2, 3, 4)  # 4 = "ultra": 16 rendered sub-pixels per camera pixel
_TLS = threading.local()


def _ss() -> int:
    """Supersampling factor of the render running in this thread (see :func:`_supersampling`)."""
    return getattr(_TLS, "factor", DEFAULT_SUPERSAMPLING)


def __getattr__(name: str) -> Any:
    # ``photoreal.SS``: the factor of the current thread (kept for callers/tests of the old module global).
    if name == "SS":
        return _ss()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@contextmanager
def _supersampling(config: dict[str, Any] | None):
    """Render with ``config['supersampling']`` (default 2) as this thread's factor (:func:`_ss`).

    All drawing helpers read the factor at call time. It is thread-local, so renders in parallel
    threads with different factors do not interfere (and do not serialise on a lock).
    """
    factor = int((config or {}).get("supersampling", DEFAULT_SUPERSAMPLING))
    if factor not in SUPERSAMPLING_CHOICES:
        raise ValueError(f"render.supersampling must be one of {SUPERSAMPLING_CHOICES}, got {factor}")
    previous = getattr(_TLS, "factor", None)
    _TLS.factor = factor
    try:
        yield
    finally:
        if previous is None:
            del _TLS.factor
        else:
            _TLS.factor = previous


# Unit light direction (image x, image y, up). Light from the top-left, fairly high.
_LIGHT = np.array([-0.40, -0.40, 0.82], dtype=np.float32)
_LIGHT /= np.linalg.norm(_LIGHT)
_HALF = _LIGHT + np.array([0.0, 0.0, 1.0], dtype=np.float32)
_HALF /= np.linalg.norm(_HALF)

# Solder-mask colours (BGR). Median mask pixels of the real crops of the production boards
# (DeepVision xpa top: (64, 89, 58), bottom: (68, 86, 50); hue ~135-150 deg, HSV saturation ~90-110)
# plus a few darker/lighter and slightly teal variants. The earlier palette (red ~30, saturation ~150)
# rendered a far too saturated, cyan-lime mask.
_MASK_PALETTE = ((64, 89, 58), (68, 86, 50), (58, 80, 46), (72, 96, 56), (54, 78, 50), (76, 98, 60), (70, 92, 44),
                 (62, 84, 40))
# Mask over copper (traces, pours) relative to mask over bare laminate, measured on the same photos.
_COPPER_UNDER_MASK = np.array((1.29, 1.39, 1.40), dtype=np.float32)
_SILKSCREEN = np.array((215, 228, 228), dtype=np.float32)
_SOLDER = np.array((192, 196, 198), dtype=np.float32)  # real joints: bright grey ~(187, 196, 196), p90 V 228
_FINISHES = {"enig": np.array((70, 165, 205), dtype=np.float32), "hasl": np.array((165, 170, 175), dtype=np.float32)}
_LEAD = np.array((150, 153, 156), dtype=np.float32)
_CAP_BODY = np.array((120, 152, 170), dtype=np.float32)  # tan MLCC ceramic (real crops: (100..132, 134..166, 159..177))
_END_CAP = np.array((168, 171, 174), dtype=np.float32)

_RESISTOR_CODES = ("103", "472", "1002", "000", "0", "221", "4R7", "100", "330", "10K", "1R0", "5602")
_REFDES_PREFIX = {"resistor": "R", "capacitor": "C", "sot": "Q", "soic": "U", "qfn": "U"}
_MARK_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
_FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _clip8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _blend(img: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> None:
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


def _shade_solder(hmap: np.ndarray, base: np.ndarray, env: np.ndarray, spec_power: float, spec_gain: float) -> np.ndarray:
    """Shade a solder height map as a rough mirror. Returns float BGR (H, W, 3).

    Solder is metallic: mostly a reflection of the surroundings. Flat areas mirror the bright
    overhead illumination, slopes mirror the board (*env*), and slopes facing the light catch a
    highlight. Lead-free solder is frosted rather than a perfect mirror and real cameras (ring
    lights, phone flash, room light from several sides) see joints as bright grey blobs, so the
    diffuse share and the falloff of the sky reflection are generous.
    """
    gy, gx = np.gradient(hmap)
    n = np.dstack([-gx, -gy, np.ones_like(hmap)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    up = np.clip(2.0 * n[..., 2] ** 2 - 1.0, 0.0, 1.0)  # z of the reflected view ray
    sky = up**1.5
    spec = np.clip(n @ _HALF, 0.0, 1.0) ** spec_power
    out = base * (0.22 + 0.22 * np.clip(n @ _LIGHT, 0.0, 1.0))[..., None]
    out += base * (0.7 * sky)[..., None] + env * (0.5 * (1.0 - sky))[..., None]
    out += (255.0 * spec_gain * spec)[..., None]
    return out


def _solder_env(base: np.ndarray) -> np.ndarray:
    """What the slopes of a joint mirror: the board, the part's end caps and neighbouring metal, i.e. a light grey-green."""
    return base * 0.6 + 55.0


def _smooth_noise(shape: tuple[int, int], cell: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, roughly unit-amplitude value noise with feature size *cell* pixels."""
    h, w = shape
    small = rng.normal(0.0, 1.0, (int(h / cell) + 2, int(w / cell) + 2)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _text_layer(shape: tuple[int, int], text: str, center: tuple[float, float], height_px: float,
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


# --------------------------------------------------------------------------- substrate

@lru_cache(maxsize=8)
def texture_files(texture_dir: str) -> tuple[Path, ...]:
    """Sorted image files in *texture_dir* (sorted so a seed always picks the same file)."""
    return tuple(sorted(p for p in Path(texture_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))


def _texture_substrate(h: int, w: int, texture_dir: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random crop of a real photo of empty board area, randomly flipped/rotated, scaled to (h, w)."""
    files = texture_files(texture_dir)
    path = files[int(rng.integers(len(files)))]
    tex = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if tex is None:
        raise OSError(f"Failed to read substrate texture: {path}")
    tex = np.rot90(tex, int(rng.integers(4)))
    if rng.random() < 0.5:
        tex = tex[:, ::-1]
    th, tw = tex.shape[:2]
    rh, rw = h // _ss(), w // _ss()  # ROI size at camera resolution
    if th >= rh and tw >= rw:  # same scale as the camera: crop a ROI-sized patch
        y, x = int(rng.integers(th - rh + 1)), int(rng.integers(tw - rw + 1))
        tex = tex[y:y + rh, x:x + rw]
    img = cv2.resize(np.ascontiguousarray(tex), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return img, np.median(img.reshape(-1, 3), axis=0)


def _substrate(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base = np.array(_MASK_PALETTE[int(rng.integers(len(_MASK_PALETTE)))], dtype=np.float32)
    base *= float(rng.uniform(0.92, 1.08))
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = base

    # Low-frequency mottling (resin thickness) + fibre weave + fine grain.
    coarse = _smooth_noise((h, w), 32 * _ss(), rng)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    period = float(rng.uniform(6.0, 9.0)) * _ss()
    weave = np.sin(2 * np.pi * xx / period) * np.sin(2 * np.pi * yy / period)
    img *= (1.0 + 0.03 * coarse + 0.03 * weave)[..., None]
    img += rng.normal(0.0, 1.2, (h, w, 1)).astype(np.float32)
    return img, base


MASK_GLOSS_PROBABILITY = 0.5  # default share of images with a glossy reflection on the mask (``render.mask_gloss``)


def _mask_sheen(img: np.ndarray, base: np.ndarray, rng: np.random.Generator, probability: float) -> None:
    """Glossy solder mask mirroring a light source (phone flash, ring light, window).

    Real photos show broad, soft, whitish-green glare patches on the mask, broken up by its
    orange-peel surface. Applied to the bare board only (parts and joints are drawn on top).
    """
    if rng.random() >= probability:
        return
    h, w = img.shape[:2]
    field = 0.6 * _smooth_noise((h, w), max(h, w) / float(rng.uniform(0.8, 1.5)), rng)
    a = float(rng.uniform(0.0, 2.0 * np.pi))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field += 1.2 * ((xx - w / 2.0) * np.cos(a) + (yy - h / 2.0) * np.sin(a)) / max(h, w)  # light comes from one side
    sheen = np.clip((field - float(rng.uniform(0.1, 0.6))) / 0.7, 0.0, 1.0) ** 1.5
    peel = 0.65 + 0.35 * np.clip(_smooth_noise((h, w), 3.0 * _ss(), rng) * 0.5 + 0.5, 0.0, 1.0)  # orange peel
    glare = np.clip(base * 0.5 + 115.0, 0, 255)  # desaturated, much lighter mask colour
    _blend(img, glare, np.clip(sheen * peel * float(rng.uniform(0.3, 0.75)), 0.0, 1.0))


def _copper_colour(base: np.ndarray) -> np.ndarray:
    return np.clip(base * _COPPER_UNDER_MASK, 0, 255)


def _blend_copper(img: np.ndarray, base: np.ndarray, mask: np.ndarray) -> None:
    """Copper *mask* (uint8) shimmering through the solder mask."""
    alpha = cv2.GaussianBlur(mask, (0, 0), 0.6 * _ss()).astype(np.float32) / 255.0 * 0.55
    _blend(img, _copper_colour(base), alpha)


def _draw_via(img: np.ndarray, base: np.ndarray, center: tuple[int, int], r: int, tented: bool) -> None:
    """Via of radius *r* px: copper ring, hole (dark when open, mask-coloured when tented) and mask dome highlight."""
    h, w = img.shape[:2]
    vx, vy = center
    ring = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ring, (vx, vy), r, 255, -1, lineType=cv2.LINE_AA)
    _blend(img, _copper_colour(base), ring.astype(np.float32) / 255.0 * 0.9)
    hole = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hole, (vx, vy), max(1, int(r * 0.45)), 255, -1, lineType=cv2.LINE_AA)
    hole_col = base * 0.5 if tented else np.array((22, 30, 24), dtype=np.float32)
    _blend(img, hole_col, hole.astype(np.float32) / 255.0)
    # Tented via = small dome of mask: highlight on the side facing the light.
    hi = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(hi, (vx, vy), (r, r), 0, 190, 260, 255, max(1, _ss()), lineType=cv2.LINE_AA)
    _blend(img, np.clip(base * 0.6 + 100.0, 0, 255), cv2.GaussianBlur(hi, (0, 0), 0.5 * _ss()).astype(np.float32) / 255.0 * 0.5)


def _draw_copper(img: np.ndarray, base: np.ndarray, copper: CopperLayers) -> None:
    """Rasterised board model (``render.board.routing: escape``): tracks, pours and vias."""
    _blend_copper(img, base, copper.copper)
    for via in copper.vias:
        _draw_via(img, base, (via.x, via.y), via.radius, via.tented)


def _traces_and_vias(img: np.ndarray, base: np.ndarray, pads: list[dict[str, int]], keepout: tuple[int, int, int, int],
                     rng: np.random.Generator) -> None:
    """Legacy copper (``render.board.routing: legacy``): random pours, traces and vias that know nothing of nets."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = keepout
    # Copper feature sizes follow the part scale (a 0603 pad is ~0.8 mm): traces 0.15-0.3 mm, vias
    # ~0.5-0.6 mm across. Fixed pixel sizes made them hairlines and pin pricks next to a chip.
    ref = float(np.clip(_pad_ref(pads), 6.0 * _ss(), 40.0 * _ss())) if pads else 20.0 * _ss()

    def _feature(lo: float, hi: float, lo_px: float, hi_px: float) -> float:
        return float(np.clip(float(rng.uniform(lo, hi)) * ref, lo_px * _ss(), hi_px * _ss()))

    # Copper pours: large planes with straight, 45-degree chamfered edges, kept clear of the part.
    if rng.random() < 0.6:
        pour = np.zeros((h, w), dtype=np.uint8)
        for _ in range(int(rng.integers(1, 3))):
            pw_, ph_ = float(rng.uniform(0.4, 1.2)) * w, float(rng.uniform(0.4, 1.2)) * h
            ax, ay = float(rng.uniform(-0.2 * w, w - 0.5 * pw_)), float(rng.uniform(-0.2 * h, h - 0.5 * ph_))
            c = float(rng.uniform(0.05, 0.25)) * min(pw_, ph_)
            poly = [(ax + c, ay), (ax + pw_ - c, ay), (ax + pw_, ay + c), (ax + pw_, ay + ph_ - c),
                    (ax + pw_ - c, ay + ph_), (ax + c, ay + ph_), (ax, ay + ph_ - c), (ax, ay + c)]
            cv2.fillPoly(pour, [np.round(poly).astype(np.int32)], 255)
        gap = int(round(float(rng.uniform(3.0, 6.0)) * _ss()))
        cv2.rectangle(pour, (x0 - gap, y0 - gap), (x1 + gap, y1 + gap), 0, -1)
        mask = np.maximum(mask, pour)

    def _polyline(p0: tuple[int, int], p1: tuple[int, int], width: int) -> None:
        # Route with one 45-degree jog like real autorouted traces.
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        jog = min(abs(dx), abs(dy))
        mid = (p0[0] + int(np.sign(dx)) * jog, p0[1] + int(np.sign(dy)) * jog)
        cv2.polylines(mask, [np.array([p0, mid, p1], dtype=np.int32)], False, 255, width, lineType=cv2.LINE_AA)

    # Free traces crossing the ROI.
    for _ in range(int(rng.integers(1, 4))):
        width = int(round(_feature(0.1, 0.3, 1.5, 9.0)))
        if rng.random() < 0.5:
            p0 = (0, int(rng.integers(0, h)))
            p1 = (w - 1, int(rng.integers(0, h)))
        else:
            p0 = (int(rng.integers(0, w)), 0)
            p1 = (int(rng.integers(0, w)), h - 1)
        _polyline(p0, p1, width)

    # Traces leaving a few pads towards the ROI border.
    for pad in pads:
        if rng.random() > 0.35:
            continue
        cx, cy = pad["x"] + pad["w"] // 2, pad["y"] + pad["h"] // 2
        outward = np.array([cx - w / 2.0, cy - h / 2.0])
        if abs(outward[0]) >= abs(outward[1]):
            p1 = (0 if outward[0] < 0 else w - 1, cy + int(rng.integers(-h // 6, h // 6)))
        else:
            p1 = (cx + int(rng.integers(-w // 6, w // 6)), 0 if outward[1] < 0 else h - 1)
        _polyline((cx, cy), p1, int(round(_feature(0.15, 0.35, 2.0, 10.0))))

    _blend_copper(img, base, mask)

    # Vias outside the footprint keep-out; most are tented (covered by mask), some open. Production
    # boards are full of them (every passive fans out into one or two).
    n_vias = int(rng.integers(2, 17)) if rng.random() < 0.8 else int(rng.integers(0, 3))
    for _ in range(int(round(n_vias * h * w / (512.0 * 512.0 * _ss() * _ss())))):
        for _try in range(6):
            vx, vy = int(rng.integers(0, w)), int(rng.integers(0, h))
            if not (x0 <= vx <= x1 and y0 <= vy <= y1):
                break
        else:
            continue
        r = int(round(_feature(0.22, 0.36, 3.5, 11.0)))
        _draw_via(img, base, (vx, vy), r, tented=bool(rng.random() < 0.85))


def _silkscreen(img: np.ndarray, keepout: tuple[int, int, int, int], family: str, rng: np.random.Generator) -> None:
    if rng.random() < 0.45:  # dense production boards often have no print next to small parts
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = keepout
    gap = int(3 * _ss())
    layer = np.zeros((h, w), dtype=np.float32)
    thick = max(1, int(round(1.2 * _ss())))
    outline = np.zeros((h, w), dtype=np.uint8)
    if family in ("resistor", "capacitor") and rng.random() < 0.6:
        # Two short bars flanking the body (typical passives outline).
        cv2.line(outline, (x0 - gap, y0 - gap), (x0 - gap, y1 + gap), 255, thick)
        cv2.line(outline, (x1 + gap, y0 - gap), (x1 + gap, y1 + gap), 255, thick)
    else:
        cv2.rectangle(outline, (x0 - gap, y0 - gap), (x1 + gap, y1 + gap), 255, thick)
        if family in ("soic", "qfn", "sot"):
            cv2.circle(outline, (x0 - gap - 3 * _ss(), y0 - gap - 3 * _ss()), int(1.5 * _ss()), 255, -1)
    layer = np.maximum(layer, outline.astype(np.float32) / 255.0)

    refdes = f"{_REFDES_PREFIX[family]}{int(rng.integers(1, 400))}"
    text_h = 9.0 * _ss()
    above = y0 - gap - text_h * 1.2
    cy = above if above > text_h else y1 + gap + text_h * 1.2
    cx = x0 + (x1 - x0) * float(rng.uniform(0.25, 0.75))
    angle = 0.0 if rng.random() < 0.7 else 90.0
    layer = np.maximum(layer, _text_layer((h, w), refdes, (cx, cy), text_h, angle, max(1, _ss())))

    layer = cv2.GaussianBlur(layer, (0, 0), 0.5 * _ss())
    ink = 0.85 + 0.12 * _smooth_noise((h, w), 2 * _ss(), rng)  # screen-printed ink is blotchy, not uniform
    _blend(img, _SILKSCREEN, np.clip(layer * ink, 0, 1))


# --------------------------------------------------------------------------- pads & solder

def thermal_pad_index(pads: list[dict[str, int]]) -> int | None:
    """Index of an exposed thermal pad (QFN): the largest pad, if clearly larger than the others."""
    if len(pads) < 3:
        return None
    areas = [p["w"] * p["h"] for p in pads]
    i = max(range(len(areas)), key=areas.__getitem__)
    return i if areas[i] > 3 * sorted(areas)[len(areas) // 2] else None


def _pads_and_solder(img: np.ndarray, base: np.ndarray, pads: list[dict[str, int]], body_mask: np.ndarray,
                     rng: np.random.Generator, finish: np.ndarray, *, with_solder: bool = True,
                     excess: frozenset[int] = frozenset()) -> None:
    """Pad finish and solder; pads in *excess* get a high, convex (bulging) joint instead of a fillet."""
    h, w = img.shape[:2]
    solder = _SOLDER * float(rng.uniform(0.9, 1.05))
    mask_edge = base * 0.65
    env = _solder_env(base)  # what slopes of the solder mirror: the surrounding board
    # Surface irregularity: real fillets are lumpy, not mathematically smooth.
    lumps = _smooth_noise((h, w), 8 * _ss(), rng) * 0.06
    thermal = thermal_pad_index(pads)

    for i, pad in enumerate(pads):
        x0, y0 = max(0, pad["x"]), max(0, pad["y"])
        x1, y1 = min(w, pad["x"] + pad["w"]), min(h, pad["y"] + pad["h"])
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        # Solder-mask opening: dark rim then copper finish.
        cv2.rectangle(img, (x0 - _ss(), y0 - _ss()), (x1 + _ss() - 1, y1 + _ss() - 1), mask_edge.tolist(), _ss())
        region = img[y0:y1, x0:x1]
        region[:] = finish
        if not with_solder:
            continue

        ph, pw = region.shape[:2]
        covered = body_mask[y0:y1, x0:x1] > 0
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        edge = np.minimum(np.minimum(xx, pw - 1 - xx), np.minimum(yy, ph - 1 - yy))
        rim = np.clip(edge / (1.5 * _ss()), 0.0, 1.0)  # copper visible at the pad rim
        amp = min(0.55 * min(pw, ph), 9.0 * _ss())  # solder height; big (thermal) pads stay flat
        if covered.any():
            dist = cv2.distanceTransform((~covered).astype(np.uint8), cv2.DIST_L2, 3)
            t = 1.0 - dist / max(1.0, float(dist.max()))
            if i in excess:  # too much solder: convex bulge, higher than the terminal
                hmap = np.where(covered, 1.0, t ** 0.5) * rim * amp * 1.5
            else:
                hmap = np.where(covered, 1.0, t ** 1.2) * rim * amp  # concave fillet rising to the terminal
            spec_power, spec_gain = 40.0, 0.9
        else:
            # Reflowed paste, no part. Thermal pads are printed through a window-pane stencil,
            # so the solder pulls into separate flat islands with bare finish in between.
            nx, ny = (max(2, round(3 * pw / min(pw, ph))), max(2, round(3 * ph / min(pw, ph)))) if i == thermal else (1, 1)
            u, v = ((xx / pw * nx) % 1.0) * 2 - 1, ((yy / ph * ny) % 1.0) * 2 - 1
            dome = np.clip(1.0 - (np.abs(u) ** 3 + np.abs(v) ** 3) * (0.85 if nx * ny == 1 else 1.2), 0, 1)
            hmap = dome * rim * min(amp * 0.35, 4.0 * _ss())
            spec_power, spec_gain = 40.0, 0.5
        hmap = hmap * (1.0 + lumps[y0:y1, x0:x1])
        shaded = _shade_solder(hmap, solder, env, spec_power, spec_gain)
        alpha = np.clip(hmap / (0.08 * amp), 0.0, 1.0)
        _blend(region, shaded, alpha)
        img[y0:y1, x0:x1] = region


# --------------------------------------------------------------------------- solder defects

def solder_defect_layout(defect_type: str, pads: list[dict[str, int]], body: np.ndarray | None,
                         rng: np.random.Generator) -> dict[str, Any]:
    """Where the inspected part's solder defects are; shared by the 2D renderer and the 3D bake.

    ``excess``: indices of pads with too much (bulging) solder. ``balls``: ``(x, y, r)`` solder balls on
    the mask next to the joints, never on a pad or under the body (*body*: its 4 corner points).
    """
    out: dict[str, Any] = {"excess": [], "balls": []}
    thermal = thermal_pad_index(pads)
    joints = [i for i in range(len(pads)) if i != thermal]
    if defect_type == "EXCESS_SOLDER":
        k = int(rng.integers(1, min(len(joints), 4) + 1))
        out["excess"] = sorted(int(i) for i in rng.choice(joints, size=k, replace=False))
    elif defect_type == "SOLDER_BALL":
        ref = _pad_ref([pads[i] for i in joints])
        for _ in range(int(rng.integers(1, 5))):
            r = max(float(rng.uniform(0.12, 0.25)) * ref, float(rng.uniform(2.5, 4.5)) * _ss())  # >= ~5-9 camera px across
            for _try in range(30):
                p = pads[joints[int(rng.integers(len(joints)))]]
                a, d = float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0.6, 1.3)) * max(p["w"], p["h"])
                x, y = p["x"] + p["w"] / 2.0 + d * np.cos(a), p["y"] + p["h"] / 2.0 + d * np.sin(a)
                on_pad = any(q["x"] - r < x < q["x"] + q["w"] + r and q["y"] - r < y < q["y"] + q["h"] + r for q in pads)
                under = body is not None and cv2.pointPolygonTest(body.astype(np.float32), (x, y), True) > -r
                if not on_pad and not under:
                    out["balls"].append((float(x), float(y), r))
                    break
    return out


def _draw_solder_balls(img: np.ndarray, base: np.ndarray, balls: list[tuple[float, float, float]]) -> None:
    """Small mirror-like solder spheres with a contact shadow."""
    h, w = img.shape[:2]
    env = _solder_env(base)
    for x, y, r in balls:
        x0, y0, x1, y1 = int(x - 2 * r), int(y - 2 * r), int(x + 2 * r) + 1, int(y + 2 * r) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
        d2 = ((xx - x) ** 2 + (yy - y) ** 2) / (r * r)
        region = img[cy0:cy1, cx0:cx1]
        shadow = np.clip(1.0 - (((xx - x - 0.4 * r) ** 2 + (yy - y - 0.4 * r) ** 2) / (r * r) - 0.6) / 0.8, 0, 1)
        region *= (1.0 - 0.4 * shadow)[..., None]
        hmap = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0)) * r
        _blend(region, _shade_solder(hmap, _SOLDER, env, 60.0, 0.9), np.clip((1.0 - d2) * r, 0.0, 1.0))


def _solder_bulge(img: np.ndarray, base: np.ndarray, center, length: float, width: float, angle_deg: float) -> None:
    """Shiny convex solder blob (too much solder), e.g. over a chip's end cap or a buried lead foot."""
    h, w = img.shape[:2]
    u, v, m = _rect_alpha(h, w, center, length, width, angle_deg)
    bulge = np.clip(1.0 - (u / (length / 2.0)) ** 2 - (v / (width / 2.0)) ** 2, 0.0, 1.0)
    hmap = np.sqrt(bulge) * 0.4 * min(length, width)
    _blend(img, _shade_solder(hmap, _SOLDER, _solder_env(base), 60.0, 0.85), np.clip(bulge * 4.0, 0, 1) * m)


def _draw_excess(img: np.ndarray, base: np.ndarray, family: str, pads: list[dict[str, int]], excess: list[int],
                 center: tuple[float, float], length: float, width: float, angle_deg: float) -> None:
    """Too much solder on the pads in *excess*: chips get it over the end cap, leaded parts over the lead foot."""
    a = np.radians(angle_deg)
    axis = np.array([np.cos(a), np.sin(a)])
    for i in excess:
        p = pads[i]
        pc = np.array([p["x"] + p["w"] / 2.0, p["y"] + p["h"] / 2.0])
        if family in ("resistor", "capacitor"):
            side = 1.0 if (pc - np.array(center)) @ axis >= 0 else -1.0
            _solder_bulge(img, base, np.array(center) + axis * side * length * 0.42, length * 0.3, width * 1.06, angle_deg)
        elif family in ("sot", "soic"):
            _solder_bulge(img, base, pc, p["w"] * 0.95, p["h"] * 0.95, 0.0)
        elif family == "qfn":  # bulges out beyond the package edge
            out = pc - np.array(center)
            out = np.where(np.abs(out) >= np.abs(out).max(), np.sign(out), 0.0)
            _solder_bulge(img, base, pc + out * 0.35 * np.array([p["w"], p["h"]]), p["w"] * 1.1, p["h"] * 1.1, 0.0)


# --------------------------------------------------------------------------- camera

# Camera model parameters (``render.camera``); tools/calibrate_camera.py fits the AOI ones to real ROIs.
# Ranges are sampled per image.
CAMERA_DEFAULTS: dict[str, Any] = {
    "blur_sigma": [0.55, 0.95],    # lens/optics blur (Gaussian sigma in camera pixels)
    "read_noise": [1.0, 4.0],      # read noise variance (8-bit levels^2)
    "shot_noise": 0.04,            # shot noise variance per intensity level
    "tilt_deg": [0.0, 0.0],        # oblique view: angle between optical axis and board normal
    "zoom": [1.0, 1.0],            # >1: camera farther away, the part appears this much smaller
    "defocus": [0.0, 0.0],         # extra blur (sigma) at the far/near edge of a tilted view (depth of field)
    "sharpen": [0.0, 0.0],         # phone ISP edge sharpening (unsharp mask amount)
    "jpeg_quality": [100, 100],    # < 95 compresses
    "capture_scale": [1.0, 1.0],   # sensor pixels per ROI pixel (<1: low-res capture, upscaled to the ROI like a small crop)
    "chroma_shift": [0.0, 0.0],    # lateral chromatic aberration: R/B shift at the ROI border (sensor pixels)
}
# ``render.camera.preset``: a fixed AOI camera (default) or handheld phone photos.
CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    "aoi": {},
    # Phone photos of a whole board, cut into small per-part crops (real 0603 crops are ~50 px wide,
    # i.e. ~10 px/mm) and upscaled to the ROI: soft, blocky, sharpening halos, colour fringes.
    "phone": {"tilt_deg": [5.0, 40.0], "zoom": [1.0, 1.5], "blur_sigma": [0.5, 1.2], "defocus": [0.5, 2.5],
              "read_noise": [4.0, 16.0], "shot_noise": 0.08, "sharpen": [0.3, 0.9], "jpeg_quality": [70, 92],
              "capture_scale": [0.2, 0.7], "chroma_shift": [0.0, 0.8]},
}


def camera_settings(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Effective camera parameters: defaults < preset < explicit ``render.camera`` keys."""
    cfg = dict(cfg or {})
    return {**CAMERA_DEFAULTS, **CAMERA_PRESETS[str(cfg.pop("preset", "aoi"))], **cfg}


def sample_camera_pose(cam: dict[str, Any], rng: np.random.Generator) -> dict[str, float]:
    """Viewing geometry for one image: tilt (deg), its azimuth in the image (deg), zoom."""
    return {"tilt": float(rng.uniform(*cam["tilt_deg"])), "azimuth": float(rng.uniform(0.0, 360.0)),
            "zoom": float(rng.uniform(*cam["zoom"]))}


def view_canvas(size: tuple[int, int], pose: dict[str, float]) -> tuple[int, int]:
    """Board area to render so that *size* is still covered after zooming out and foreshortening."""
    grow = pose["zoom"] / max(np.cos(np.radians(pose["tilt"])), 0.5) * (1.1 if pose["tilt"] > 0 else 1.0)
    return int(np.ceil(size[0] * grow)), int(np.ceil(size[1] * grow))


def apply_view(img: np.ndarray, pose: dict[str, float], size: tuple[int, int]) -> np.ndarray:
    """Look at a top-down render of the board (from :func:`view_canvas`) from farther away and at an angle.

    The board plane is tilted by ``pose['tilt']`` about an in-plane axis perpendicular to
    ``pose['azimuth']`` and projected with a pinhole camera; the output is centred on the part.
    """
    width, height = size
    z = pose["zoom"]
    small = cv2.resize(img, (max(1, int(round(img.shape[1] / z))), max(1, int(round(img.shape[0] / z)))),
                       interpolation=cv2.INTER_AREA) if z > 1.0 else img
    sh, sw = small.shape[:2]
    if pose["tilt"] <= 0.0 and (sw, sh) == (width, height):
        return small
    t, a = np.radians(pose["tilt"]), np.radians(pose["azimuth"])
    d = np.array([np.cos(a), np.sin(a)])                              # tilt direction in the image
    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float64) - [sw / 2.0, sh / 2.0]
    along, across = corners @ d, corners @ np.array([-d[1], d[0]])
    dist = 4.0 * max(width, height)                                   # pinhole distance: mild keystone
    depth = along * np.sin(t)                                         # towards the camera on one side
    scale = dist / (dist - depth)
    proj = (np.outer(along * np.cos(t), d) + np.outer(across, [-d[1], d[0]])) * scale[:, None]
    dst = (proj + [width / 2.0, height / 2.0]).astype(np.float32)
    m = cv2.getPerspectiveTransform((corners + [sw / 2.0, sh / 2.0]).astype(np.float32), dst)
    return cv2.warpPerspective(small, m, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def camera_model(img: np.ndarray, rng: np.random.Generator, cfg: dict[str, Any] | None = None,
                 pose: dict[str, float] | None = None) -> np.ndarray:
    """Turn an ideal render (2D supersampled or 3D) into a camera image. BGR in, uint8 out.

    Lens blur (plus depth-of-field falloff along the tilt of an oblique *pose*), uneven
    illumination, soft highlight roll-off instead of hard clipping, signal-dependent sensor noise
    that is slightly correlated by demosaicing, and for phones lateral chromatic aberration, ISP
    sharpening and JPEG. With ``capture_scale`` < 1 all of this happens at the lower sensor
    resolution and the result is upscaled to the input size (a small crop fed to the model).
    *cfg* is ``render.camera`` (see :func:`camera_settings`).
    """
    c = camera_settings(cfg)
    out_h, out_w = img.shape[:2]
    img = img.astype(np.float32)
    scale = float(rng.uniform(*c["capture_scale"]))
    if scale < 0.999:  # the sensor sampled the part with fewer pixels than the ROI has
        img = cv2.resize(img, (max(8, int(round(out_w * scale))), max(8, int(round(out_h * scale)))),
                         interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    sigma0 = max(1e-3, float(rng.uniform(*c["blur_sigma"])))
    blurred = cv2.GaussianBlur(img, (0, 0), sigma0)
    defocus = float(rng.uniform(*c["defocus"]))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if pose is not None and pose["tilt"] > 0.0 and defocus > 0.0:
        a = np.radians(pose["azimuth"])  # focus on the part; sharpness falls off towards near/far edge
        off = np.abs((xx - w / 2.0) * np.cos(a) + (yy - h / 2.0) * np.sin(a)) / (0.5 * max(h, w))
        weight = np.clip(off * np.sin(np.radians(pose["tilt"])) / np.sin(np.radians(40.0)), 0.0, 1.0)[..., None]
        far = cv2.GaussianBlur(img, (0, 0), float(np.hypot(sigma0, defocus)))
        blurred = blurred * (1.0 - weight) + far * weight
    img = blurred
    shift = float(rng.uniform(*c["chroma_shift"]))
    if shift > 0.0:  # lateral CA: R magnified, B shrunk around the optical centre
        for ch, k in ((2, 1.0), (0, -1.0)):
            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, 1.0 + k * shift / (0.5 * max(h, w)))
            img[..., ch] = cv2.warpAffine(img[..., ch], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = float(rng.uniform(0.0, 2.0 * np.pi))
    ramp = ((xx - w / 2.0) * np.cos(a) + (yy - h / 2.0) * np.sin(a)) / max(h, w)
    img *= (1.0 + float(rng.uniform(0.04, 0.12)) * ramp + 0.03 * _smooth_noise((h, w), max(h, w) / 3.0, rng))[..., None]
    knee = 205.0
    img = np.where(img > knee, knee + (255.0 - knee) * (1.0 - np.exp(-(img - knee) / (255.0 - knee))), img)
    sigma = np.sqrt(float(rng.uniform(*c["read_noise"])) + float(c["shot_noise"]) * np.clip(img, 0.0, None))
    noise = rng.normal(0.0, 1.0, img.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.5) * 1.6  # demosaic correlation, variance roughly kept
    noise += rng.normal(0.0, 0.35, (h, w, 1)).astype(np.float32)  # luminance-dominated
    img = img + sigma * noise
    amount = float(rng.uniform(*c["sharpen"]))
    if amount > 0.0:
        img = img + amount * (img - cv2.GaussianBlur(img, (0, 0), 1.2))
    out = _clip8(img)
    quality = int(round(float(rng.uniform(*c["jpeg_quality"]))))
    if quality < 95:
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)
    if (h, w) != (out_h, out_w):  # small crop resized to the model input, as the inspection pipeline does
        out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    return out


# --------------------------------------------------------------------------- component body

def _local_coords(h: int, w: int, center: tuple[float, float], angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.radians(angle_deg)
    dx, dy = xx - center[0], yy - center[1]
    return dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)


def _rect_alpha(h: int, w: int, center, length: float, width: float, angle_deg: float):
    """Local coords (u along *length*) and an anti-aliased 0..1 mask of a rotated rectangle."""
    u, v = _local_coords(h, w, center, angle_deg)
    return u, v, np.clip(0.5 - np.maximum(np.abs(u) - length / 2.0, np.abs(v) - width / 2.0), 0.0, 1.0)


def _contact_shadow(img: np.ndarray, mask: np.ndarray, strength: float, spread: float) -> None:
    sh = cv2.dilate(mask, np.ones((int(spread), int(spread)), np.uint8))
    sh = cv2.GaussianBlur(sh, (0, 0), spread * 0.8).astype(np.float32) / 255.0
    sh = np.roll(sh, (int(1.5 * _ss()), int(1.5 * _ss())), axis=(0, 1))  # away from the light
    img *= (1.0 - strength * sh)[..., None]


def nominal_body_angle(family: str) -> float:
    """Orientation of the part body (long axis) for zero rotation, in image degrees.

    The SOT-23 footprint puts pins 1 and 2 above each other (left column) and pin 3 on the right,
    so the body's long axis (``component_length``, 2.9 mm) runs vertically. Every other footprint
    has its length along x.
    """
    return 90.0 if family == "sot" else 0.0


def _rot(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float64)


def lead_layout(family: str, pads: list[dict[str, int]], nominal_center: tuple[float, float], length: float,
                width: float) -> list[dict[str, float]]:
    """Gull-wing leads of a SOT/SOIC in the body frame (u along *length*, v across), one per pad.

    Derived from the *nominal* footprint, so the leads move rigidly with the body when it is
    shifted or rotated (a misplaced part's feet miss their pads, they do not stretch towards them).
    Each lead: ``u`` (position along its body edge), ``side`` (+-1 across or along), ``axis``
    ('v' = leaves a long side, 'u' = leaves an end), ``edge`` (body edge), ``tip``, ``w`` (lead width).
    """
    if family not in ("sot", "soic"):
        return []
    inv = _rot(-nominal_body_angle(family))
    hl, hw = length / 2.0, width / 2.0
    rows = []
    for p in pads:
        loc = inv @ (np.array([p["x"] + p["w"] / 2.0, p["y"] + p["h"] / 2.0]) - np.array(nominal_center, dtype=np.float64))
        ext = np.abs(inv @ np.array([p["w"], p["h"]], dtype=np.float64))  # pad size in the body frame
        rows.append((loc, ext))
    lw = 0.6 * min(min(e[0] if abs(l[1]) / hw >= abs(l[0]) / hl else e[1] for l, e in rows), 0.3 * length)
    leads = []
    for loc, ext in rows:
        if abs(loc[1]) / hw >= abs(loc[0]) / hl:  # leaves a long side
            edge, tip = hw, abs(loc[1]) + 0.2 * ext[1]
            leads.append({"u": float(loc[0]), "side": float(np.sign(loc[1]) or 1.0), "axis": "v", "edge": edge,
                          "tip": float(max(tip, edge + 2 * _ss())), "w": float(lw)})
        else:
            edge, tip = hl, abs(loc[0]) + 0.2 * ext[0]
            leads.append({"u": float(loc[1]), "side": float(np.sign(loc[0]) or 1.0), "axis": "u", "edge": edge,
                          "tip": float(max(tip, edge + 2 * _ss())), "w": float(lw)})
    return leads


def _lead_coords(h: int, w: int, center, angle_deg: float, lead: dict[str, float]):
    """(t, s, alpha): 0..1 position from body edge to tip, signed offset across the lead, AA mask."""
    u, v = _local_coords(h, w, center, angle_deg)
    along, across = (v, u) if lead["axis"] == "v" else (u, v)  # 'along' = direction the lead leaves the body
    d = along * lead["side"]
    s = across - lead["u"]
    t = (d - lead["edge"]) / max(1e-3, lead["tip"] - lead["edge"])
    inside = np.maximum(np.maximum(lead["edge"] - d, d - lead["tip"]), np.abs(s) - lead["w"] / 2.0)
    return t, s, np.clip(0.5 - inside, 0.0, 1.0)


def lead_feet_mask(shape: tuple[int, int], leads: list[dict[str, float]], center, angle_deg: float) -> np.ndarray:
    """uint8 mask of the flat lead feet (what the solder wets)."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    for lead in leads:
        t, _, m = _lead_coords(h, w, center, angle_deg, lead)
        out = np.maximum(out, ((m > 0.5) & (t > 0.5)).astype(np.uint8) * 255)
    return out


def _leads(img: np.ndarray, leads: list[dict[str, float]], center: tuple[float, float], angle_deg: float,
           rng: np.random.Generator) -> None:
    """Gull-wing leads (drawn before the body): shoulder bending down out of the mold, then the flat foot.

    Seen from above the shoulder slopes away from the camera (dark, catches a thin highlight on the
    knee), the foot is flat tin-plated metal (bright, a soft ridge along its middle).
    """
    if not leads:
        return
    h, w = img.shape[:2]
    reach = max(max(l["tip"], abs(l["u"]) + l["w"]) for l in leads) + 3 * _ss()
    x0, y0 = max(0, int(center[0] - reach)), max(0, int(center[1] - reach))
    x1, y1 = min(w, int(center[0] + reach) + 1), min(h, int(center[1] + reach) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0:y1, x0:x1]
    c = (center[0] - x0, center[1] - y0)
    metal = _LEAD * float(rng.uniform(0.95, 1.1))
    for lead in leads:
        t, s, m = _lead_coords(sub.shape[0], sub.shape[1], c, angle_deg, lead)
        across = np.clip(np.abs(s) / (lead["w"] / 2.0), 0.0, 1.0)
        shoulder = 0.62 + 0.25 * np.clip(t / 0.4, 0.0, 1.0)
        knee = np.exp(-((t - 0.42) / 0.07) ** 2)
        foot = np.where(t > 0.5, 1.15 + 0.25 * np.exp(-(across / 0.45) ** 2), 0.0)
        shade = np.where(t > 0.5, foot, shoulder + 0.5 * knee) * (1.0 - 0.25 * across**4)
        _blend(sub, metal * shade[..., None], m)
    img[y0:y1, x0:x1] = sub


def _marking(img: np.ndarray, family: str, center: tuple[float, float], length: float, width: float,
             angle_deg: float, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    if family == "capacitor":
        return
    if family == "resistor":
        lines = [str(rng.choice(_RESISTOR_CODES))]
        text_h = width * 0.42
    else:
        n_lines = 1 if family == "sot" else 3
        # SOT-23 carries a 2-3 character marking code, larger packages part number / date / lot lines.
        lo, hi = (2, 4) if family == "sot" else (3, 7)
        lines = ["".join(rng.choice(list(_MARK_CHARS), size=int(rng.integers(lo, hi)))) for _ in range(n_lines)]
        text_h = min(width * 0.13, length * 0.085) if family != "sot" else width * 0.32
    if 90.0 < angle_deg % 360.0 <= 270.0:
        angle_deg += 180.0  # keep laser marking readable from the camera
    layer = np.zeros((h, w), dtype=np.float32)
    a = np.radians(angle_deg)
    perp = np.array([-np.sin(a), np.cos(a)], dtype=np.float32)
    along = np.array([np.cos(a), np.sin(a)], dtype=np.float32)
    font = _FONTS[int(rng.integers(len(_FONTS)))]
    for i, text in enumerate(lines):
        off = (i - (len(lines) - 1) / 2.0) * text_h * 1.6
        # Lines are left-aligned on real parts, so their centres wander a little.
        pos = np.array(center, dtype=np.float32) + perp * off + along * float(rng.uniform(-0.15, 0.15)) * text_h
        layer = np.maximum(layer, _text_layer((h, w), text, (float(pos[0]), float(pos[1])), text_h, angle_deg,
                                              max(1, _ss()), font))
    color = np.array((225, 225, 220), dtype=np.float32) if family == "resistor" else np.array((190, 190, 188), dtype=np.float32)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.4 * _ss())
    # Printed ink (resistors) is blotchy, laser marking (ICs) is a grainy, partly etched matte surface.
    grain = _smooth_noise((h, w), _ss(), rng)
    texture = 0.85 + 0.15 * grain if family == "resistor" else 0.8 + 0.2 * grain
    _blend(img, color, np.clip(layer * texture * (0.9 if family == "resistor" else float(rng.uniform(0.4, 0.65))), 0, 1))


def _draw_body(img: np.ndarray, family: str, center: tuple[float, float], length: float, width: float, angle_deg: float,
               base_color: np.ndarray, rng: np.random.Generator, *, marks: bool = True) -> None:
    """Part body seen from above. ``marks=False``: no marking/pin-1 (tombstone top face, small neighbour parts)."""
    h, w = img.shape[:2]
    u, v, aa_body = _rect_alpha(h, w, center, length, width, angle_deg)
    hl, hw = length / 2.0, width / 2.0
    # Visible slope at the body edge from above: chip parts have rounded edges, moulded SOT/SOIC a
    # small draft angle, sawn QFN packages vertical walls (practically invisible from above).
    inset = max(1.0, {"qfn": 0.012, "soic": 0.035, "sot": 0.05}.get(family, 0.07) * min(length, width))

    inside = (np.abs(u) <= hl) & (np.abs(v) <= hw)
    top = (np.abs(u) <= hl - inset) & (np.abs(v) <= hw - inset)
    d_top = np.maximum(np.abs(u) - (hl - inset), np.abs(v) - (hw - inset))
    aa_top = np.clip(0.5 - d_top, 0.0, 1.0)

    if family == "capacitor":
        body_col = _CAP_BODY * float(rng.uniform(0.9, 1.1))
    else:
        body_col = np.maximum(base_color, 34.0) * float(rng.uniform(0.9, 1.1))

    # Side walls: lit on the light-facing side, dark elsewhere.
    a = np.radians(angle_deg)
    light_u = _LIGHT[0] * np.cos(a) + _LIGHT[1] * np.sin(a)
    light_v = -_LIGHT[0] * np.sin(a) + _LIGHT[1] * np.cos(a)
    facing = np.where(np.abs(u) / hl > np.abs(v) / hw, np.sign(u) * light_u, np.sign(v) * light_v)
    side_shade = 0.45 + 0.35 * np.clip(-facing, 0, 1)
    _blend(img, body_col, aa_body)  # body footprint
    img *= np.where(inside & ~top, side_shade, 1.0)[..., None]

    # Top face: slight gradient towards the light, surface grain, mild gloss.
    grad = 1.0 + 0.10 * np.clip(-(u / hl) * light_u - (v / hw) * light_v, -1, 1)
    grain = 1.0 + rng.normal(0.0, 0.025 if family in ("resistor", "capacitor") else 0.06, (h, w)).astype(np.float32)
    top_img = body_col * (grad * grain)[..., None]
    if family in ("qfn", "soic", "sot"):
        gloss = np.exp(-(((u + hl * 0.35) / (hl * 0.6)) ** 2 + ((v + hw * 0.35) / (hw * 0.6)) ** 2))
        top_img += (28.0 * gloss)[..., None]
    _blend(img, top_img, aa_top)

    if family in ("resistor", "capacitor"):
        cap_len = (0.26 if family == "capacitor" else 0.22) * length
        cap = inside & (np.abs(u) > hl - cap_len)
        cap_col = _END_CAP * float(rng.uniform(0.9, 1.05))
        # Metallic end caps: bright ridge across the middle of each cap, darker towards the body.
        ridge = np.exp(-(((np.abs(u) - (hl - cap_len * 0.45)) / (cap_len * 0.35)) ** 2))
        shade = 0.72 + 0.45 * ridge - 0.25 * np.clip(v / hw, 0, 1) * 0.5
        cap_img = cap_col * shade[..., None] * (1.0 + rng.normal(0.0, 0.02, (h, w, 1)).astype(np.float32))
        _blend(img, cap_img, cap.astype(np.float32) * aa_body)
        if not marks:
            return
    elif not marks:
        return
    else:
        # Pin-1 indicator (molded dimple / laser dot).
        pin1 = np.array([-hl + 0.12 * length, -hw + 0.15 * width], dtype=np.float32)
        r = max(1.5 * _ss(), 0.035 * min(length, width))
        dot = np.clip(1.0 - (np.hypot(u - pin1[0], v - pin1[1]) - r) / _ss(), 0, 1)
        _blend(img, body_col * 0.55, dot * aa_top)
        hi = np.clip(1.0 - (np.hypot(u - pin1[0] + r * 0.4, v - pin1[1] + r * 0.4) - r * 0.5) / _ss(), 0, 1)
        _blend(img, body_col * 1.6 + 25, hi * aa_top)
        if family == "soic":
            # Bevelled pin-1 edge (lighter strip along one long side).
            # Pin 1 sits at (-u, -v); the strip runs along whichever of its two sides is the long one.
            if width > length:
                strip = top & (u < -hl + inset + 0.12 * length)
            else:
                strip = top & (v < -hw + inset + 0.08 * width)
            img *= np.where(strip, 1.25, 1.0)[..., None]

    # Fine highlight on top-face edges facing the light.
    edge_top = (aa_top > 0) & (d_top > -1.2 * _ss())
    img *= np.where(edge_top & (facing < 0), 1.18, 1.0)[..., None]

    _marking(img, family, center, length - 2 * (0.22 * length if family == "resistor" else inset * 2), width - 2 * inset,
             angle_deg, rng)


_THICKNESS_RATIO = 0.55  # body height / width of chip parts (0603: 0.45 / 0.8)


def _draw_body_local(img: np.ndarray, family: str, center: tuple[float, float], length: float, width: float,
                     angle_deg: float, base_color: np.ndarray, rng: np.random.Generator, *, marks: bool = True) -> None:
    """:func:`_draw_body` on just the part's window (cost independent of the canvas size)."""
    h, w = img.shape[:2]
    r = 0.6 * float(np.hypot(length, width)) + 2 * _ss()
    x0, y0 = max(0, int(center[0] - r)), max(0, int(center[1] - r))
    x1, y1 = min(w, int(center[0] + r) + 1), min(h, int(center[1] + r) + 1)
    if x1 > x0 and y1 > y0:
        _draw_body(img[y0:y1, x0:x1], family, (center[0] - x0, center[1] - y0), length, width, angle_deg, base_color, rng,
                   marks=marks)


def _tombstone_pose(geo: dict[str, Any], selector: int, img_shape: tuple[int, int]) -> dict[str, Any]:
    """Part raised on one end, pivoting about the bottom edge of its anchored end.

    Seen from above two faces are visible: the steep top face (length*cos(tilt)), which starts
    thickness*sin(tilt) *behind* the anchor, and the raised end face (thickness*sin(tilt)), which
    points almost straight up. Returns pose + footprint mask.
    """
    length, width = float(geo["length"]), float(geo["width"])
    tilt = np.radians(float(geo["tilt_deg"]))
    angle = float(geo["rotation_deg"])
    a = np.radians(angle)
    raised_sign = 1.0 if selector % 2 == 0 else -1.0
    axis = np.array([np.cos(a), np.sin(a)], dtype=np.float32) * raised_sign
    anchor = np.array(geo["center"], dtype=np.float32) - axis * (length / 2.0)
    top_len = length * np.cos(tilt)
    end_len = width * _THICKNESS_RATIO * np.sin(tilt)
    center = anchor + axis * ((top_len - end_len) / 2.0)
    box = cv2.boxPoints(((float(center[0]), float(center[1])), (float(top_len + end_len), width), angle)).astype(np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)
    return {"center": center, "angle": angle, "box": box, "mask": mask, "raised_sign": raised_sign, "axis": axis,
            "anchor": anchor, "tilt": tilt, "top_len": top_len, "end_len": end_len, "height": length * np.sin(tilt)}


def _draw_tombstone(img: np.ndarray, family: str, pose: dict[str, Any], width: float, base_color: np.ndarray,
                    rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    axis, anchor, tilt, angle = pose["axis"], pose["anchor"], pose["tilt"], pose["angle"]
    perp = np.array([-axis[1], axis[0]], dtype=np.float32) * (width / 2.0)

    # Shadow: projection of the standing part along the light onto the board.
    light_xy = _LIGHT[:2] / np.linalg.norm(_LIGHT[:2])
    shift = -light_xy * pose["height"] * np.linalg.norm(_LIGHT[:2]) / _LIGHT[2]
    raised_end = anchor + axis * pose["top_len"]
    pts = np.vstack([pose["box"], raised_end + perp + shift, raised_end - perp + shift]).astype(np.int32)
    sh = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(sh, cv2.convexHull(pts), 255)
    sh = cv2.GaussianBlur(sh, (0, 0), 2.0 * _ss()).astype(np.float32) / 255.0
    img *= (1.0 - 0.4 * sh)[..., None]
    _contact_shadow(img, pose["mask"], 0.4, 3 * _ss())

    # Steep top face: the foreshortened body, shaded by how much it still faces the light.
    top_c = anchor + axis * (pose["top_len"] / 2.0 - pose["end_len"])
    _draw_body(img, family, (float(top_c[0]), float(top_c[1])), pose["top_len"], width, angle, base_color, rng,
               marks=False)
    n_top = np.array([*(-axis * np.sin(tilt)), np.cos(tilt)], dtype=np.float32)
    _, _, m_top = _rect_alpha(h, w, top_c, pose["top_len"], width, angle)
    img *= (1.0 + (0.35 + 0.65 * max(0.0, float(n_top @ _LIGHT)) - 1.0) * m_top)[..., None]

    # Raised end face, pointing almost straight up: bare metal end cap on chips, mold wall on ICs.
    end_c = anchor + axis * (pose["top_len"] - pose["end_len"] / 2.0)
    u, v, m_end = _rect_alpha(h, w, end_c, pose["end_len"], width, angle)
    n_end = np.array([*(axis * np.cos(tilt)), np.sin(tilt)], dtype=np.float32)
    lit = 0.55 + 0.6 * max(0.0, float(n_end @ _LIGHT))
    if family in ("resistor", "capacitor"):
        # Tin-plated cap, slightly convex: shaded as metal like the solder, tilted with the part.
        su = np.clip(np.abs(u) / (pose["end_len"] / 2.0), 0.0, 1.0)
        sv = np.clip(np.abs(v) / (width / 2.0), 0.0, 1.0)
        bulge = (1.0 - su**4) * (1.0 - sv**4) * 0.12 * min(pose["end_len"], width)
        bulge += 0.04 * _smooth_noise((h, w), 3 * _ss(), rng)  # plating texture
        face = _shade_solder(bulge, _END_CAP * float(rng.uniform(0.95, 1.08)), np.full(3, 70.0, dtype=np.float32),
                             30.0, 0.45) * lit
    else:
        face = (np.maximum(base_color, 34.0) * 1.3 * lit)[None, None, :] * np.ones((h, w, 1), dtype=np.float32)
    _blend(img, face, m_end)


def _draw_neighbours(img: np.ndarray, base: np.ndarray, parts: list[dict[str, Any]], finish: np.ndarray,
                     rng: np.random.Generator) -> None:
    if not parts:
        return
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in parts:
        box = cv2.boxPoints(((p["center"][0], p["center"][1]), (p["length"], p["width"]), p["angle"]))
        cv2.fillPoly(mask, [np.round(box).astype(np.int32)], 255)
    _pads_and_solder(img, base, [pad for p in parts for pad in p["pads"]], mask, rng, finish)
    _contact_shadow(img, mask, 0.35, 3 * _ss())
    dark = np.array((20, 20, 20), dtype=np.float32)
    for p in parts:
        _draw_body_local(img, p["family"], p["center"], p["length"], p["width"], p["angle"], dark, rng, marks=False)


REWORK_PROBABILITY = 0.15  # default share of images with rework residue (``render.rework_artifacts``)
_FLUX = np.array((55, 105, 140), dtype=np.float32)   # amber, translucent
_DIRT = np.array((43, 114, 131), dtype=np.float32)   # olive-yellow smeared contamination, unmixed from a board photo


def _rework(img: np.ndarray, keepout: tuple[int, int, int, int], ref: float, rng: np.random.Generator,
            probability: float) -> None:
    """Residue seen on the production boards: flux residue, smeared yellow contamination, patch wires.

    A nuisance, not a defect class: always kept clear of the inspected part (its keep-out plus a gap)
    so the label is unaffected; the model learns to ignore it.
    """
    if rng.random() >= probability:
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(np.clip(v, 0, lim)) for v, lim in zip(keepout, (w, h, w, h)))
    protected = img[y0:y1, x0:x1].copy()

    def clear(pts: np.ndarray, margin: float) -> bool:
        m = 4 * _ss() + margin  # gap + stroke half-width + blur
        return not np.any((pts[:, 0] > keepout[0] - m) & (pts[:, 0] < keepout[2] + m)
                          & (pts[:, 1] > keepout[1] - m) & (pts[:, 1] < keepout[3] + m))

    def stroke(margin: float) -> np.ndarray | None:
        """Smooth curve (quadratic Bezier) clear of the keep-out, or None."""
        for _try in range(10):
            ctrl = np.stack([rng.uniform(-0.1 * w, 1.1 * w, 3), rng.uniform(-0.1 * h, 1.1 * h, 3)], axis=1)
            t = np.linspace(0.0, 1.0, 60)[:, None]
            curve = (1 - t) ** 2 * ctrl[0] + 2 * (1 - t) * t * ctrl[1] + t**2 * ctrl[2]
            if clear(curve, margin):
                return curve.astype(np.int32)
        return None

    for kind in rng.permutation(["flux", "dirt", "wire"])[: int(rng.integers(1, 3))]:
        layer = np.zeros((h, w), dtype=np.uint8)
        if kind == "flux":
            r = float(rng.uniform(0.08, 0.2)) * min(h, w)
            for _try in range(10):
                c = np.array([rng.uniform(0, w), rng.uniform(0, h)])
                ring = c + r * np.stack([np.cos(np.linspace(0, 2 * np.pi, 24)), np.sin(np.linspace(0, 2 * np.pi, 24))], 1)
                if clear(ring, 6 * _ss()):
                    break
            else:
                continue
            cv2.ellipse(layer, (int(c[0]), int(c[1])), (int(r), int(r * rng.uniform(0.5, 1.0))),
                        float(rng.uniform(0, 180)), 0, 360, 255, -1)
            blob = (layer > 0) & (_smooth_noise((h, w), 0.6 * r, rng) > -0.7)  # irregular outline
            a = cv2.GaussianBlur(blob.astype(np.float32), (0, 0), 2 * _ss())
            rim = np.clip(cv2.GaussianBlur(blob.astype(np.float32), (0, 0), 1.0 * _ss()) - a, 0, 1)  # dried edge
            _blend(img, _FLUX * float(rng.uniform(0.85, 1.1)), a * float(rng.uniform(0.25, 0.45)))
            img *= (1.0 - 0.5 * rim)[..., None]
        elif kind == "dirt":
            width = int(round(float(rng.uniform(0.35, 0.8)) * ref))
            curve = stroke(width / 2 + 4 * _ss())
            if curve is None:
                continue
            cv2.polylines(layer, [curve], False, 255, width, lineType=cv2.LINE_AA)
            a = cv2.GaussianBlur(layer, (0, 0), 1.5 * _ss()).astype(np.float32) / 255.0
            _blend(img, _DIRT * float(rng.uniform(0.85, 1.1)), a * float(rng.uniform(0.5, 0.7)))
        else:
            wd = max(_ss(), int(round(float(rng.uniform(0.08, 0.15)) * ref)))
            curve = stroke(wd + 5 * _ss())  # incl. the offset shadow
            if curve is None:
                continue
            shadow = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(shadow, [curve + int(1.5 * _ss())], False, 255, wd + _ss(), lineType=cv2.LINE_AA)
            img *= (1.0 - 0.35 * cv2.GaussianBlur(shadow, (0, 0), _ss()).astype(np.float32)[..., None] / 255.0)
            cv2.polylines(layer, [curve], False, 255, wd, lineType=cv2.LINE_AA)
            col = np.array((215, 218, 220) if rng.random() < 0.7 else (70, 120, 200), dtype=np.float32)  # white / enamel
            _blend(img, col, layer.astype(np.float32) / 255.0)
            core = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(core, [curve - int(0.3 * wd)], False, 255, max(1, wd // 3), lineType=cv2.LINE_AA)
            _blend(img, np.full(3, 250.0, dtype=np.float32), core.astype(np.float32) / 255.0 * 0.6)  # highlight
    img[y0:y1, x0:x1] = protected  # hard guarantee: the inspected part's area is never touched


def _board(nom: dict[str, float], footprint: str, family: str, shape: tuple[int, int], config: dict[str, Any],
           rng: np.random.Generator, crowd: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]], np.ndarray]:
    """Board around the inspected part at supersampled *shape*: substrate, copper, silkscreen and
    populated neighbour parts. Returns (float image, mask colour, target pads, pad finish colour).

    The keep-out uses the *nominal* body, so nothing on the board follows a defect (a real board's
    silkscreen and neighbours do not move when a part is misplaced).
    """
    h, w = shape
    specs = footprint_pads(footprint, nom, (h, w))
    pads = [p.as_dict() for p in specs]
    hl, hw = nom["component_length"] / 2.0, nom["component_width"] / 2.0
    xs = [p["x"] for p in pads] + [p["x"] + p["w"] for p in pads] + [int(w // 2 - hl), int(w // 2 + hl)]
    ys = [p["y"] for p in pads] + [p["y"] + p["h"] for p in pads] + [int(h // 2 - hw), int(h // 2 + hw)]
    keepout = (min(xs), min(ys), max(xs), max(ys))
    options = board_options(config)

    texture_dir = config.get("texture_dir")
    neighbours: list[dict[str, Any]] | None = None
    if texture_dir:
        img, base = _texture_substrate(h, w, str(texture_dir), rng)  # real board already has its own copper
    else:
        img, base = _substrate(h, w, rng)
        if options.legacy:
            _traces_and_vias(img, base, pads, keepout, rng)
        elif options.routing != ROUTING_NONE:
            # Board model: neighbours are placed first (from their own stream) so the copper can
            # connect them and keep clear of them; one draw from *rng* seeds the model's streams.
            board_seed = int(rng.integers(1 << 62))
            neighbours = neighbour_layout(pads, keepout, (h, w), np.random.default_rng([board_seed, STREAM_PLACE]), crowd)
            px_to_mm = float(config.get("mm_per_px", DEFAULT_MM_PER_PX)) / _ss()
            board, _report = build_window_board(
                footprint=footprint, family=family, target_pads=specs,
                target_body_px=(float(nom["component_length"]), float(nom["component_width"])), shape=(h, w),
                neighbours=neighbours, px_to_mm=px_to_mm, options=options, seed=board_seed)
            _draw_copper(img, base, rasterize(board, (h, w), px_to_mm))
    _silkscreen(img, keepout, family, rng)
    if not texture_dir:  # a real board texture already has its own reflections
        _mask_sheen(img, base, rng, float(config.get("mask_gloss", MASK_GLOSS_PROBABILITY)))
    finish = _FINISHES["enig" if rng.random() < 0.5 else "hasl"] * float(rng.uniform(0.92, 1.06))
    if neighbours is None:
        neighbours = neighbour_layout(pads, keepout, (h, w), rng, crowd)
    _draw_neighbours(img, base, neighbours, finish, rng)
    _rework(img, keepout, _pad_ref(pads), rng, float(config.get("rework_artifacts", REWORK_PROBABILITY)))
    return img, base, pads, finish


def _scaled(nominal: dict[str, float], defect_params: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    nom = {k: float(v) * _ss() for k, v in nominal.items()}
    dp = dict(defect_params)
    dp["shift_x"] = float(dp["shift_x"]) * _ss()
    dp["shift_y"] = float(dp["shift_y"]) * _ss()
    return nom, dp


def _render_board(
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    config: dict[str, Any],
    tolerances: dict[str, Any] | None,
    rng: np.random.Generator,
    footprint: str,
    size: tuple[int, int],
    crowd: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Board texture (substrate, copper, silkscreen, bare pad finish) for the 3D backend, plus the scene layout.

    Returns ``(BGR uint8 image of *size*, scene)``. ``scene`` holds, in camera pixels relative to the image
    centre (x right, y down): ``family``, ``pads`` as ``[cx, cy, w, h]``, ``body`` (``None`` when MISSING)
    with the same pose semantics as the 2D renderer including the tombstone pivot, ``excess`` (pad
    indices with too much solder) and ``balls`` (``[x, y, r]``), from :func:`solder_defect_layout`.
    """
    width, height = size
    h, w = height * _ss(), width * _ss()
    nom, dp = _scaled(nominal, defect_params)
    family = family_of(footprint, str(config.get("component_type", "")))
    img, base, pads, finish = _board(nom, footprint, family, (h, w), config, rng, crowd)
    _pads_and_solder(img, base, pads, np.zeros((h, w), dtype=np.uint8), rng, finish, with_solder=False)
    if dp["type"] == "SOLDER_BRIDGE":  # flat solder between neighbouring pads
        geo = body_geometry(nom, dp, tolerances, (h, w))
        img8 = _clip8(img)
        _draw_solder_bridge(img8, nom, footprint, geo["center"], (int(geo["length"]), int(geo["width"])),
                            _param_selector(defect_params), geo["tilt_deg"])
        img = img8.astype(np.float32)

    scene: dict[str, Any] = {
        "family": family,
        "pads": [[(p["x"] + p["w"] / 2.0 - w / 2.0) / _ss(), (p["y"] + p["h"] / 2.0 - h / 2.0) / _ss(), p["w"] / _ss(), p["h"] / _ss()]
                 for p in pads],
        "thermal_pad": thermal_pad_index(pads),
        "body": None,
    }
    geo_ss = body_geometry(nom, dp, tolerances, (h, w))
    rect = cv2.boxPoints(((float(geo_ss["center"][0]), float(geo_ss["center"][1])),
                          (float(geo_ss["length"]), float(geo_ss["width"])),
                          geo_ss["rotation_deg"] + nominal_body_angle(family)))
    solder = solder_defect_layout(dp["type"], pads, rect, rng)
    scene["excess"] = solder["excess"]
    scene["balls"] = [[(x - w / 2.0) / _ss(), (y - h / 2.0) / _ss(), r / _ss()] for x, y, r in solder["balls"]]
    if dp["type"] != "MISSING":
        geo = body_geometry(nominal, defect_params, tolerances, (height, width))
        # rotation_deg: of the body's length axis (SOT-23: vertical, see nominal_body_angle);
        # lead_frame_deg: that axis for the nominal placement, so leads can stay rigid with the body.
        body = {"center": [geo["center"][0] - width // 2, geo["center"][1] - height // 2], "length": geo["length"],
                "width": geo["width"], "rotation_deg": geo["rotation_deg"] + nominal_body_angle(family),
                "lead_frame_deg": nominal_body_angle(family), "tilt_deg": geo["tilt_deg"],
                "is_tombstone": bool(geo["is_tombstone"])}
        if geo["is_tombstone"]:
            pose = _tombstone_pose(geo, _param_selector(defect_params), (height, width))
            body["raised_sign"] = pose["raised_sign"]
        scene["body"] = body
    return cv2.resize(_clip8(img), (width, height), interpolation=cv2.INTER_AREA), scene


def _render_photoreal_base(
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    config: dict[str, Any],
    tolerances: dict[str, Any] | None,
    rng: np.random.Generator,
    footprint: str,
    roi_size: tuple[int, int],
) -> np.ndarray:
    """Render substrate + pads + solder + component for *defect_params* at *roi_size* (BGR uint8)."""
    view_pose = sample_camera_pose(camera_settings(config.get("camera")), rng)
    canvas = view_canvas(roi_size, view_pose)  # larger board area for a zoomed-out / oblique view
    width, height = canvas
    h, w = height * _ss(), width * _ss()
    nom, dp = _scaled(nominal, defect_params)
    family = family_of(footprint, str(config.get("component_type", "")))
    base_color = np.array(config.get("component_color", (20, 20, 20)), dtype=np.float32)
    defect_type = dp["type"]

    crowd = canvas[0] * canvas[1] / float(roi_size[0] * roi_size[1])
    img, base, pads, finish = _board(nom, footprint, family, (h, w), config, rng, crowd)
    geo = body_geometry(nom, dp, tolerances, (h, w))
    geo["rotation_deg"] += nominal_body_angle(family)  # body frame of the package (labels only use the deviation)
    length, width_px = float(geo["length"]), float(geo["width"])
    cx, cy = geo["center"]
    rect = cv2.boxPoints(((float(cx), float(cy)), (length, width_px), geo["rotation_deg"]))
    selector = _param_selector(defect_params)

    body_mask = np.zeros((h, w), dtype=np.uint8)
    pose = None
    if defect_type != "MISSING" and geo["is_tombstone"]:
        pose = _tombstone_pose(geo, selector, (h, w))
        body_mask = pose["mask"]
    elif defect_type != "MISSING":
        pts = rect
        if defect_type == "CORNER_LIFT":
            pts = lifted_corner_geometry(rect, geo["center"], int(width_px), geo["tilt_deg"], selector)["lifted"]
        cv2.fillPoly(body_mask, [np.round(pts).astype(np.int32)], 255)
    leads = []
    if defect_type != "MISSING" and pose is None:
        leads = lead_layout(family, pads, (w // 2, h // 2), length, width_px)
        if leads:  # solder wets the lead feet, so the fillets rise towards them
            body_mask = np.maximum(body_mask, lead_feet_mask((h, w), leads, (float(cx), float(cy)), geo["rotation_deg"]))

    solder = solder_defect_layout(defect_type, pads, rect, rng)
    _pads_and_solder(img, base, pads, body_mask, rng, finish, excess=frozenset(solder["excess"]))

    if pose is not None:
        _draw_tombstone(img, family, pose, width_px, base_color, rng)
    elif defect_type != "MISSING":
        _contact_shadow(img, body_mask, 0.35, 3 * _ss())
        _leads(img, leads, (float(cx), float(cy)), geo["rotation_deg"], rng)
        _draw_body_local(img, family, (float(cx), float(cy)), length, width_px, geo["rotation_deg"], base_color, rng)
        if defect_type == "CORNER_LIFT":
            lift = lifted_corner_geometry(rect, geo["center"], int(width_px), geo["tilt_deg"], selector)
            extra = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(extra, [np.round(lift["lifted"]).astype(np.int32)], 255)
            cv2.fillPoly(extra, [np.round(rect).astype(np.int32)], 0)
            body_col = np.maximum(base_color, 34.0)
            _blend(img, body_col, extra.astype(np.float32) / 255.0)  # raised corner sticking out
            facet = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(facet, [np.round(lift["facet"]).astype(np.int32)], 255)
            _blend(img, body_col * 1.6 + 30.0, facet.astype(np.float32) / 255.0 * 0.4)  # lit facet
        if defect_type == "SOLDER_BRIDGE":
            img8 = _clip8(img)
            _draw_solder_bridge(img8, nom, footprint, (int(cx), int(cy)), (int(length), int(width_px)), selector, geo["tilt_deg"])
            img = img8.astype(np.float32)
        _draw_excess(img, base, family, pads, solder["excess"], (float(cx), float(cy)), length, width_px, geo["rotation_deg"])
        _draw_solder_balls(img, base, solder["balls"])

    view = apply_view(cv2.resize(img, canvas, interpolation=cv2.INTER_AREA), view_pose, roi_size)
    return camera_model(view, rng, config.get("camera"), view_pose)


def render_board(nominal: dict[str, float], defect_params: dict[str, Any], config: dict[str, Any],
                 tolerances: dict[str, Any] | None, rng: np.random.Generator, footprint: str,
                 size: tuple[int, int], crowd: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    with _supersampling(config):
        return _render_board(nominal, defect_params, config, tolerances, rng, footprint, size, crowd)


render_board.__doc__ = _render_board.__doc__


def render_photoreal_base(nominal: dict[str, float], defect_params: dict[str, Any], config: dict[str, Any],
                          tolerances: dict[str, Any] | None, rng: np.random.Generator, footprint: str,
                          roi_size: tuple[int, int]) -> np.ndarray:
    """Render substrate + pads + solder + component for *defect_params* at *roi_size* (BGR uint8).

    Rendered at ``render.supersampling`` x the camera resolution (default 2, up to 4) and downsampled.
    """
    with _supersampling(config):
        return _render_photoreal_base(nominal, defect_params, config, tolerances, rng, footprint, roi_size)
