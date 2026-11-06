"""Photorealistic base rendering for the OpenCV 2D backend (``render.style: photoreal``).

Produces the un-augmented ROI (substrate, pads, solder fillets, component body) with
procedural textures and simple physically motivated shading, rendered at 2x and
downsampled for anti-aliasing, then passed through a simple camera model (lens blur, uneven
illumination, highlight roll-off, sensor noise) so it does not look like a vector drawing. Every random choice is drawn from the passed ``rng``,
so a sample is fully determined by its seed.

Visual targets were calibrated against photos of real assembled boards: dark olive
solder mask with copper traces shimmering through, vias, white silkscreen, bright
concave solder fillets, black resistors with white value codes, tan MLCCs, black
molded SOT/SOIC/QFN bodies with laser markings and metallic gull-wing leads.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .draw import _compute_pad_positions, _draw_solder_bridge, _param_selector, body_geometry, lifted_corner_geometry

SS = 2  # supersampling factor

# Unit light direction (image x, image y, up). Light from the top-left, fairly high.
_LIGHT = np.array([-0.40, -0.40, 0.82], dtype=np.float32)
_LIGHT /= np.linalg.norm(_LIGHT)
_HALF = _LIGHT + np.array([0.0, 0.0, 1.0], dtype=np.float32)
_HALF /= np.linalg.norm(_HALF)

# Solder-mask greens measured on real boards (BGR).
_MASK_PALETTE = ((44, 86, 34), (50, 96, 40), (38, 76, 30), (56, 104, 46), (48, 90, 42), (42, 82, 38))
_SILKSCREEN = np.array((215, 228, 228), dtype=np.float32)
_SOLDER = np.array((176, 180, 184), dtype=np.float32)
_FINISHES = {"enig": np.array((70, 165, 205), dtype=np.float32), "hasl": np.array((165, 170, 175), dtype=np.float32)}
_LEAD = np.array((150, 153, 156), dtype=np.float32)
_CAP_BODY = np.array((110, 155, 190), dtype=np.float32)  # tan MLCC ceramic
_END_CAP = np.array((168, 171, 174), dtype=np.float32)

_RESISTOR_CODES = ("103", "472", "1002", "000", "0", "221", "4R7", "100", "330", "10K", "1R0", "5602")
_REFDES_PREFIX = {"resistor": "R", "capacitor": "C", "sot": "Q", "soic": "U", "qfn": "U"}
_MARK_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
_FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX)
_TEXTURE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _family(footprint: str, component_type: str) -> str:
    if footprint == "sot23":
        return "sot"
    if footprint.startswith("qfn"):
        return "qfn"
    if footprint == "soic_16":
        return "soic"
    return "capacitor" if "cap" in component_type.lower() else "resistor"


def _clip8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _blend(img: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> None:
    """In-place ``img = img*(1-a) + color*a`` with a 2D alpha map."""
    a = alpha[..., None]
    img *= 1.0 - a
    img += color * a


def _shade_solder(hmap: np.ndarray, base: np.ndarray, env: np.ndarray, spec_power: float, spec_gain: float) -> np.ndarray:
    """Shade a solder height map as a rough mirror. Returns float BGR (H, W, 3).

    Solder is metallic: little diffuse light, mostly a reflection of the surroundings. Flat areas
    mirror the bright overhead illumination, slopes mirror the dark board (*env*), and slopes facing
    the light catch a sharp highlight.
    """
    gy, gx = np.gradient(hmap)
    n = np.dstack([-gx, -gy, np.ones_like(hmap)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    up = np.clip(2.0 * n[..., 2] ** 2 - 1.0, 0.0, 1.0)  # z of the reflected view ray
    sky = up**3
    spec = np.clip(n @ _HALF, 0.0, 1.0) ** spec_power
    out = base * (0.12 + 0.18 * np.clip(n @ _LIGHT, 0.0, 1.0))[..., None]
    out += base * (0.8 * sky)[..., None] + env * (0.45 * (1.0 - sky))[..., None]
    out += (255.0 * spec_gain * spec)[..., None]
    return out


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
    return tuple(sorted(p for p in Path(texture_dir).iterdir() if p.suffix.lower() in _TEXTURE_EXT))


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
    rh, rw = h // SS, w // SS  # ROI size at camera resolution
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
    coarse = _smooth_noise((h, w), 32 * SS, rng)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    period = float(rng.uniform(6.0, 9.0)) * SS
    weave = np.sin(2 * np.pi * xx / period) * np.sin(2 * np.pi * yy / period)
    img *= (1.0 + 0.03 * coarse + 0.03 * weave)[..., None]
    img += rng.normal(0.0, 1.2, (h, w, 1)).astype(np.float32)
    return img, base


def _traces_and_vias(img: np.ndarray, base: np.ndarray, pads: list[dict[str, int]], keepout: tuple[int, int, int, int],
                     rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    trace_col = np.clip(base * np.array((1.30, 1.33, 1.52), dtype=np.float32), 0, 255)
    mask = np.zeros((h, w), dtype=np.uint8)

    def _polyline(p0: tuple[int, int], p1: tuple[int, int], width: int) -> None:
        # Route with one 45-degree jog like real autorouted traces.
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        jog = min(abs(dx), abs(dy))
        mid = (p0[0] + int(np.sign(dx)) * jog, p0[1] + int(np.sign(dy)) * jog)
        cv2.polylines(mask, [np.array([p0, mid, p1], dtype=np.int32)], False, 255, width, lineType=cv2.LINE_AA)

    # Free traces crossing the ROI.
    for _ in range(int(rng.integers(1, 4))):
        width = int(round(float(rng.uniform(1.5, 3.5)) * SS))
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
        _polyline((cx, cy), p1, int(round(float(rng.uniform(2.0, 3.5)) * SS)))

    alpha = cv2.GaussianBlur(mask, (0, 0), 0.6 * SS).astype(np.float32) / 255.0 * 0.7
    _blend(img, trace_col, alpha)

    # Vias outside the footprint keep-out.
    x0, y0, x1, y1 = keepout
    for _ in range(int(rng.integers(0, 4))):
        for _try in range(6):
            vx, vy = int(rng.integers(0, w)), int(rng.integers(0, h))
            if not (x0 <= vx <= x1 and y0 <= vy <= y1):
                break
        else:
            continue
        r = int(round(float(rng.uniform(4.0, 6.5)) * SS))
        ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring, (vx, vy), r, 255, -1, lineType=cv2.LINE_AA)
        _blend(img, trace_col, ring.astype(np.float32) / 255.0 * 0.9)
        hole = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(hole, (vx, vy), max(1, int(r * 0.45)), 255, -1, lineType=cv2.LINE_AA)
        _blend(img, np.array((22, 30, 24), dtype=np.float32), hole.astype(np.float32) / 255.0)


def _silkscreen(img: np.ndarray, keepout: tuple[int, int, int, int], family: str, rng: np.random.Generator) -> None:
    if rng.random() < 0.2:
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = keepout
    gap = int(3 * SS)
    layer = np.zeros((h, w), dtype=np.float32)
    thick = max(1, int(round(1.2 * SS)))
    outline = np.zeros((h, w), dtype=np.uint8)
    if family in ("resistor", "capacitor") and rng.random() < 0.6:
        # Two short bars flanking the body (typical passives outline).
        cv2.line(outline, (x0 - gap, y0 - gap), (x0 - gap, y1 + gap), 255, thick)
        cv2.line(outline, (x1 + gap, y0 - gap), (x1 + gap, y1 + gap), 255, thick)
    else:
        cv2.rectangle(outline, (x0 - gap, y0 - gap), (x1 + gap, y1 + gap), 255, thick)
        if family in ("soic", "qfn", "sot"):
            cv2.circle(outline, (x0 - gap - 3 * SS, y0 - gap - 3 * SS), int(1.5 * SS), 255, -1)
    layer = np.maximum(layer, outline.astype(np.float32) / 255.0)

    refdes = f"{_REFDES_PREFIX[family]}{int(rng.integers(1, 400))}"
    text_h = 9.0 * SS
    above = y0 - gap - text_h * 1.2
    cy = above if above > text_h else y1 + gap + text_h * 1.2
    cx = x0 + (x1 - x0) * float(rng.uniform(0.25, 0.75))
    angle = 0.0 if rng.random() < 0.7 else 90.0
    layer = np.maximum(layer, _text_layer((h, w), refdes, (cx, cy), text_h, angle, max(1, SS)))

    layer = cv2.GaussianBlur(layer, (0, 0), 0.5 * SS)
    ink = 0.85 + 0.12 * _smooth_noise((h, w), 2 * SS, rng)  # screen-printed ink is blotchy, not uniform
    _blend(img, _SILKSCREEN, np.clip(layer * ink, 0, 1))


# --------------------------------------------------------------------------- pads & solder

def _pads_and_solder(img: np.ndarray, base: np.ndarray, pads: list[dict[str, int]], body_mask: np.ndarray,
                     rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    finish = _FINISHES["enig" if rng.random() < 0.5 else "hasl"] * float(rng.uniform(0.92, 1.06))
    solder = _SOLDER * float(rng.uniform(0.9, 1.05))
    mask_edge = base * 0.65
    env = base * 0.8 + 12.0  # what slopes of the solder mirror: the surrounding board
    # Surface irregularity: real fillets are lumpy, not mathematically smooth.
    lumps = _smooth_noise((h, w), 8 * SS, rng) * 0.06

    for pad in pads:
        x0, y0 = max(0, pad["x"]), max(0, pad["y"])
        x1, y1 = min(w, pad["x"] + pad["w"]), min(h, pad["y"] + pad["h"])
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        # Solder-mask opening: dark rim then copper finish.
        cv2.rectangle(img, (x0 - SS, y0 - SS), (x1 + SS - 1, y1 + SS - 1), mask_edge.tolist(), SS)
        region = img[y0:y1, x0:x1]
        region[:] = finish

        ph, pw = region.shape[:2]
        covered = body_mask[y0:y1, x0:x1] > 0
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        edge = np.minimum(np.minimum(xx, pw - 1 - xx), np.minimum(yy, ph - 1 - yy))
        rim = np.clip(edge / (1.5 * SS), 0.0, 1.0)  # copper visible at the pad rim
        amp = min(0.55 * min(pw, ph), 9.0 * SS)  # solder height; big (thermal) pads stay flat
        if covered.any():
            dist = cv2.distanceTransform((~covered).astype(np.uint8), cv2.DIST_L2, 3)
            t = 1.0 - dist / max(1.0, float(dist.max()))
            hmap = np.where(covered, 1.0, t ** 1.8) * rim * amp  # concave fillet rising to the terminal
            spec_power, spec_gain = 60.0, 0.85
        else:
            # Reflowed paste, no part. Big (thermal) pads are printed through a window-pane stencil,
            # so the solder pulls into separate flat islands with bare finish in between.
            big = min(pw, ph) >= 60 * SS
            nx, ny = (max(2, round(pw / (32 * SS))), max(2, round(ph / (32 * SS)))) if big else (1, 1)
            u, v = ((xx / pw * nx) % 1.0) * 2 - 1, ((yy / ph * ny) % 1.0) * 2 - 1
            dome = np.clip(1.0 - (np.abs(u) ** 3 + np.abs(v) ** 3) * (0.85 if nx * ny == 1 else 1.2), 0, 1)
            hmap = dome * rim * min(amp * 0.35, 4.0 * SS)
            spec_power, spec_gain = 40.0, 0.5
        hmap = hmap * (1.0 + lumps[y0:y1, x0:x1])
        shaded = _shade_solder(hmap, solder, env, spec_power, spec_gain)
        alpha = np.clip(hmap / (0.08 * amp), 0.0, 1.0)
        _blend(region, shaded, alpha)
        img[y0:y1, x0:x1] = region


# --------------------------------------------------------------------------- camera

def _camera(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn the ideal (supersampled, perfectly sharp) render into a camera image. Float in, uint8 out.

    Lens blur, uneven illumination, soft highlight roll-off instead of hard clipping, and
    signal-dependent sensor noise that is slightly correlated by demosaicing.
    """
    h, w = img.shape[:2]
    img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.55, 0.95)))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = float(rng.uniform(0.0, 2.0 * np.pi))
    ramp = ((xx - w / 2.0) * np.cos(a) + (yy - h / 2.0) * np.sin(a)) / max(h, w)
    img *= (1.0 + float(rng.uniform(0.04, 0.12)) * ramp + 0.03 * _smooth_noise((h, w), max(h, w) / 3.0, rng))[..., None]
    knee = 205.0
    img = np.where(img > knee, knee + (255.0 - knee) * (1.0 - np.exp(-(img - knee) / (255.0 - knee))), img)
    sigma = np.sqrt(float(rng.uniform(1.0, 4.0)) + 0.04 * np.clip(img, 0.0, None))  # read + shot noise
    noise = rng.normal(0.0, 1.0, img.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.5) * 1.6  # demosaic correlation, variance roughly kept
    noise += rng.normal(0.0, 0.35, (h, w, 1)).astype(np.float32)  # luminance-dominated
    return _clip8(img + sigma * noise)


# --------------------------------------------------------------------------- component body

def _local_coords(h: int, w: int, center: tuple[float, float], angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.radians(angle_deg)
    dx, dy = xx - center[0], yy - center[1]
    return dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)


def _contact_shadow(img: np.ndarray, mask: np.ndarray, strength: float, spread: float) -> None:
    sh = cv2.dilate(mask, np.ones((int(spread), int(spread)), np.uint8))
    sh = cv2.GaussianBlur(sh, (0, 0), spread * 0.8).astype(np.float32) / 255.0
    sh = np.roll(sh, (int(1.5 * SS), int(1.5 * SS)), axis=(0, 1))  # away from the light
    img *= (1.0 - strength * sh)[..., None]


def _leads(img: np.ndarray, pads: list[dict[str, int]], center: tuple[float, float], length: float, width: float,
           angle_deg: float, rng: np.random.Generator) -> None:
    """Gull-wing leads from the body edge to each pad (drawn before the body)."""
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
    c = np.array(center, dtype=np.float32)
    for pad in pads:
        pc = np.array([pad["x"] + pad["w"] / 2.0, pad["y"] + pad["h"] / 2.0], dtype=np.float32)
        loc = rot.T @ (pc - c)
        edge = np.array([np.clip(loc[0], -length / 2, length / 2), np.clip(loc[1], -width / 2, width / 2)])
        start = c + rot @ (edge * 0.9)
        lw = max(SS, int(round(0.55 * min(pad["w"], pad["h"]))))
        p0, p1 = tuple(np.round(start).astype(int)), tuple(np.round(pc).astype(int))
        cv2.line(img, p0, p1, (_LEAD * float(rng.uniform(0.9, 1.05))).tolist(), lw, lineType=cv2.LINE_AA)
        knee = tuple(np.round(start + (pc - start) * 0.35).astype(int))
        cv2.line(img, p0, knee, (_LEAD * 0.55).tolist(), lw, lineType=cv2.LINE_AA)  # bent shoulder in shadow
        cv2.line(img, knee, p1, (_LEAD * 1.35).tolist(), max(1, lw // 3), lineType=cv2.LINE_AA)  # specular ridge


def _marking(img: np.ndarray, family: str, center: tuple[float, float], length: float, width: float,
             angle_deg: float, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    if family == "capacitor":
        return
    if family == "resistor":
        lines = [str(rng.choice(_RESISTOR_CODES))]
        text_h = width * 0.42
    else:
        n_lines = 2 if family == "sot" else 3
        lines = ["".join(rng.choice(list(_MARK_CHARS), size=int(rng.integers(3, 7)))) for _ in range(n_lines)]
        if family == "sot":
            lines = lines[:1]
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
                                              max(1, SS), font))
    color = np.array((225, 225, 220), dtype=np.float32) if family == "resistor" else np.array((190, 190, 188), dtype=np.float32)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.4 * SS)
    # Printed ink (resistors) is blotchy, laser marking (ICs) is a grainy, partly etched matte surface.
    grain = _smooth_noise((h, w), SS, rng)
    texture = 0.85 + 0.15 * grain if family == "resistor" else 0.8 + 0.2 * grain
    _blend(img, color, np.clip(layer * texture * (0.9 if family == "resistor" else float(rng.uniform(0.4, 0.65))), 0, 1))


def _draw_body(img: np.ndarray, family: str, center: tuple[float, float], length: float, width: float, angle_deg: float,
               base_color: np.ndarray, rng: np.random.Generator, *, raised_sign: float = 0.0) -> None:
    h, w = img.shape[:2]
    u, v = _local_coords(h, w, center, angle_deg)
    hl, hw = length / 2.0, width / 2.0
    inset = max(1.0, 0.07 * min(length, width))

    inside = (np.abs(u) <= hl) & (np.abs(v) <= hw)
    top = (np.abs(u) <= hl - inset) & (np.abs(v) <= hw - inset)
    # Soft anti-aliased edges via distance to the rectangle border.
    d_out = np.maximum(np.abs(u) - hl, np.abs(v) - hw)
    aa_body = np.clip(0.5 - d_out, 0.0, 1.0)
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
        if raised_sign:
            return
    elif raised_sign:
        wall = top & (u * raised_sign > hl - inset - 0.18 * length)  # raised side wall facing the camera
        img *= np.where(wall, 1.6, 1.0)[..., None]
        return
    else:
        # Pin-1 indicator (molded dimple / laser dot).
        pin1 = np.array([-hl + 0.12 * length, -hw + 0.15 * width], dtype=np.float32)
        r = max(1.5 * SS, 0.035 * min(length, width))
        dot = np.clip(1.0 - (np.hypot(u - pin1[0], v - pin1[1]) - r) / SS, 0, 1)
        _blend(img, body_col * 0.55, dot * aa_top)
        hi = np.clip(1.0 - (np.hypot(u - pin1[0] + r * 0.4, v - pin1[1] + r * 0.4) - r * 0.5) / SS, 0, 1)
        _blend(img, body_col * 1.6 + 25, hi * aa_top)
        if family == "soic":
            # Bevelled pin-1 edge (lighter strip along one long side).
            strip = top & (v < -hw + inset + 0.08 * width)
            img *= np.where(strip, 1.25, 1.0)[..., None]

    # Fine highlight on top-face edges facing the light.
    edge_top = (aa_top > 0) & (d_top > -1.2 * SS)
    img *= np.where(edge_top & (facing < 0), 1.18, 1.0)[..., None]

    _marking(img, family, center, length - 2 * (0.22 * length if family == "resistor" else inset * 2), width - 2 * inset,
             angle_deg, rng)


def _tombstone_pose(geo: dict[str, Any], selector: int, img_shape: tuple[int, int]) -> dict[str, Any]:
    """Body raised on one end: foreshortened top anchored at the other end. Returns pose + footprint mask."""
    length, width, tilt = float(geo["length"]), float(geo["width"]), float(geo["tilt_deg"])
    angle = float(geo["rotation_deg"])
    a = np.radians(angle)
    raised_sign = 1.0 if selector % 2 == 0 else -1.0
    axis = np.array([np.cos(a), np.sin(a)], dtype=np.float32) * raised_sign
    anchor = np.array(geo["center"], dtype=np.float32) - axis * (length / 2.0)
    thickness = width * 0.55
    proj = length * np.cos(np.radians(tilt)) + thickness * np.sin(np.radians(tilt))
    center = anchor + axis * (proj / 2.0)
    box = cv2.boxPoints(((float(center[0]), float(center[1])), (float(proj), width), angle)).astype(np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)
    return {"center": center, "proj": proj, "angle": angle, "box": box, "mask": mask, "raised_sign": raised_sign,
            "shadow_len": length * np.sin(np.radians(tilt)) * 0.7}


def _draw_tombstone(img: np.ndarray, family: str, pose: dict[str, Any], width: float, base_color: np.ndarray,
                    rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    center, box = pose["center"], pose["box"]
    # Long shadow cast away from the light, proportional to the raised height.
    far = center + np.array([0.6, 0.8], dtype=np.float32) * pose["shadow_len"]
    sh = np.zeros((h, w), dtype=np.uint8)
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        quad = np.array([box[i], box[j], far + (box[j] - center) * 0.6, far + (box[i] - center) * 0.6], dtype=np.int32)
        cv2.fillPoly(sh, [quad], 255)
    sh = cv2.GaussianBlur(sh, (0, 0), 2.5 * SS).astype(np.float32) / 255.0
    img *= (1.0 - 0.45 * sh)[..., None]
    _contact_shadow(img, pose["mask"], 0.4, 3 * SS)
    _draw_body(img, family, (float(center[0]), float(center[1])), pose["proj"], width, pose["angle"], base_color, rng,
               raised_sign=pose["raised_sign"])


def render_photoreal_base(
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    config: dict[str, Any],
    tolerances: dict[str, Any] | None,
    rng: np.random.Generator,
    footprint: str,
    roi_size: tuple[int, int],
) -> np.ndarray:
    """Render substrate + pads + solder + component for *defect_params* at *roi_size* (BGR uint8)."""
    width, height = roi_size
    h, w = height * SS, width * SS
    nom = {k: float(v) * SS for k, v in nominal.items()}
    dp = dict(defect_params)
    dp["shift_x"] = float(dp["shift_x"]) * SS
    dp["shift_y"] = float(dp["shift_y"]) * SS
    family = _family(footprint, str(config.get("component_type", "")))
    base_color = np.array(config.get("component_color", (20, 20, 20)), dtype=np.float32)
    defect_type = dp["type"]

    pads = _compute_pad_positions(footprint, nom, (h, w))
    geo = body_geometry(nom, dp, tolerances, (h, w))
    length, width_px = float(geo["length"]), float(geo["width"])
    cx, cy = geo["center"]
    rect = cv2.boxPoints(((float(cx), float(cy)), (length, width_px), geo["rotation_deg"]))
    selector = _param_selector(defect_params)

    xs = [p["x"] for p in pads] + [p["x"] + p["w"] for p in pads] + [int(rect[:, 0].min()), int(rect[:, 0].max())]
    ys = [p["y"] for p in pads] + [p["y"] + p["h"] for p in pads] + [int(rect[:, 1].min()), int(rect[:, 1].max())]
    keepout = (min(xs), min(ys), max(xs), max(ys))

    texture_dir = config.get("texture_dir")
    if texture_dir:
        img, base = _texture_substrate(h, w, str(texture_dir), rng)  # real board already has its own copper
    else:
        img, base = _substrate(h, w, rng)
        _traces_and_vias(img, base, pads, keepout, rng)
    _silkscreen(img, keepout, family, rng)

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

    _pads_and_solder(img, base, pads, body_mask, rng)

    if pose is not None:
        _draw_tombstone(img, family, pose, width_px, base_color, rng)
    elif defect_type != "MISSING":
        _contact_shadow(img, body_mask, 0.35, 3 * SS)
        if family in ("sot", "soic"):
            _leads(img, pads, (float(cx), float(cy)), length, width_px, geo["rotation_deg"], rng)
        _draw_body(img, family, (float(cx), float(cy)), length, width_px, geo["rotation_deg"], base_color, rng)
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

    return _camera(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA), rng)
