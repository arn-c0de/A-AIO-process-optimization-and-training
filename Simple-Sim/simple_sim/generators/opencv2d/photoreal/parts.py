"""Part bodies seen from above: chips, arrays, LEDs, power inductors, moulded ICs with gull-wing leads,
markings, tombstones and the populated neighbour parts.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .context import RenderContext
from .imageops import blend, blur_window, contact_shadow, local_coords, local_window, rect_alpha, rect_window, shade_solder, smooth_noise, text_layer
from .imageops import WindowedNoise
from .joints import pads_and_solder
from .palette import (ARRAY_CODES, ARRAY_FAMILIES, CAP_ARRAY_BODY, CAP_LEN, CHIPS, FONTS, LEADED,
                      LED_LENS, LIGHT, MARK_CHARS, RESISTOR_CODES, UNMARKED)


def nominal_body_angle(family: str) -> float:
    """Orientation of the part body (long axis) for zero rotation, in image degrees.

    The SOT-23 footprint puts pins 1 and 2 above each other (left column) and pin 3 on the right,
    so the body's long axis (``component_length``, 2.9 mm) runs vertically. Every other footprint
    has its length along x.
    """
    return 90.0 if family == "sot" else 0.0


def rot(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float64)


def lead_layout(family: str, pads: list[dict[str, int]], nominal_center: tuple[float, float], length: float,
                width: float, *, ss: int) -> list[dict[str, float]]:
    """Gull-wing leads of a SOT/SOIC in the body frame (u along *length*, v across), one per pad.

    Derived from the *nominal* footprint, so the leads move rigidly with the body when it is
    shifted or rotated (a misplaced part's feet miss their pads, they do not stretch towards them).
    Each lead: ``u`` (position along its body edge), ``side`` (+-1 across or along), ``axis``
    ('v' = leaves a long side, 'u' = leaves an end), ``edge`` (body edge), ``tip``, ``w`` (lead width). *ss*: supersampling factor of the pixel coordinates (tip clearance).
    """
    if family not in LEADED:
        return []
    inv = rot(-nominal_body_angle(family))
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
                          "tip": float(max(tip, edge + 2 * ss)), "w": float(lw)})
        else:
            edge, tip = hl, abs(loc[0]) + 0.2 * ext[0]
            leads.append({"u": float(loc[1]), "side": float(np.sign(loc[0]) or 1.0), "axis": "u", "edge": edge,
                          "tip": float(max(tip, edge + 2 * ss)), "w": float(lw)})
    return leads


def lead_coords(h: int, w: int, center, angle_deg: float, lead: dict[str, float], y0: int = 0, x0: int = 0):
    """(t, s, alpha): 0..1 position from body edge to tip, signed offset across the lead, AA mask
    (on the window ``[y0:y0+h, x0:x0+w]``, see :func:`local_coords`)."""
    u, v = local_coords(h, w, center, angle_deg, y0, x0)
    along, across = (v, u) if lead["axis"] == "v" else (u, v)  # 'along' = direction the lead leaves the body
    d = along * lead["side"]
    s = across - lead["u"]
    t = (d - lead["edge"]) / max(1e-3, lead["tip"] - lead["edge"])
    inside = np.maximum(np.maximum(lead["edge"] - d, d - lead["tip"]), np.abs(s) - lead["w"] / 2.0)
    return t, s, np.clip(0.5 - inside, 0.0, 1.0)


def lead_window(shape: tuple[int, int], center, angle_deg: float, lead: dict[str, float]) -> tuple[int, int, int, int] | None:
    """Window of *shape* that holds every pixel of *lead* (its AA mask is 0 outside)."""
    along = tuple(sorted((lead["side"] * lead["edge"], lead["side"] * lead["tip"])))
    across = (lead["u"] - lead["w"] / 2.0, lead["u"] + lead["w"] / 2.0)
    u, v = (across, along) if lead["axis"] == "v" else (along, across)
    return local_window(shape, center, angle_deg, u, v)


def lead_feet_mask(shape: tuple[int, int], leads: list[dict[str, float]], center, angle_deg: float) -> np.ndarray:
    """uint8 mask of the flat lead feet (what the solder wets)."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    for lead in leads:
        win = lead_window((h, w), center, angle_deg, lead)
        if win is None:
            continue
        y0, y1, x0, x1 = win
        t, _, m = lead_coords(y1 - y0, x1 - x0, center, angle_deg, lead, y0, x0)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], ((m > 0.5) & (t > 0.5)).astype(np.uint8) * 255)
    return out


def draw_leads(ctx: RenderContext, img: np.ndarray, leads: list[dict[str, float]], center: tuple[float, float],
               angle_deg: float, rng: np.random.Generator) -> None:
    """Gull-wing leads (drawn before the body): shoulder bending down out of the mold, then the flat foot.

    Seen from above the shoulder slopes away from the camera (dark, catches a thin highlight on the
    knee), the foot is flat tin-plated metal (bright, a soft ridge along its middle).
    """
    if not leads:
        return
    h, w = img.shape[:2]
    reach = max(max(l["tip"], abs(l["u"]) + l["w"]) for l in leads) + 3 * ctx.ss
    x0, y0 = max(0, int(center[0] - reach)), max(0, int(center[1] - reach))
    x1, y1 = min(w, int(center[0] + reach) + 1), min(h, int(center[1] + reach) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    # Lead coordinates are relative to this part window (kept as before: its float32 values define the look).
    sub = img[y0:y1, x0:x1]
    c = (center[0] - x0, center[1] - y0)
    metal = ctx.look.lead * float(rng.uniform(0.95, 1.1))
    for lead in leads:
        win = lead_window(sub.shape[:2], c, angle_deg, lead)
        if win is None:
            continue
        ly0, ly1, lx0, lx1 = win
        t, s, m = lead_coords(ly1 - ly0, lx1 - lx0, c, angle_deg, lead, ly0, lx0)
        across = np.clip(np.abs(s) / (lead["w"] / 2.0), 0.0, 1.0)
        shoulder = 0.62 + 0.25 * np.clip(t / 0.4, 0.0, 1.0)
        knee = np.exp(-((t - 0.42) / 0.07) ** 2)
        foot = np.where(t > 0.5, 1.15 + 0.25 * np.exp(-(across / 0.45) ** 2), 0.0)
        shade = np.where(t > 0.5, foot, shoulder + 0.5 * knee) * (1.0 - 0.25 * across**4)
        blend(sub[ly0:ly1, lx0:lx1], metal * shade[..., None], m)


def draw_marking(ctx: RenderContext, img: np.ndarray, family: str, center: tuple[float, float], length: float, width: float,
             angle_deg: float, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    if family in UNMARKED:
        return
    if family == "resistor":
        lines = [str(rng.choice(RESISTOR_CODES))]
        text_h = width * 0.42
    elif family == "array_r":  # 1-3 character code in the black strip between the terminal rows
        lines = [str(rng.choice(ARRAY_CODES))]
        text_h = min(width * 0.85, length / max(1, len(lines[0])) * 0.9)
    else:
        small = family in ("sot", "sot6")
        n_lines = 1 if small else 3
        # SOT-23 carries a 2-3 character marking code, SOT-363/SOT-23-6 3-5 characters, larger
        # packages part number / date / lot lines.
        lo, hi = {"sot": (2, 4), "sot6": (3, 6)}.get(family, (3, 7))
        lines = ["".join(rng.choice(list(MARK_CHARS), size=int(rng.integers(lo, hi)))) for _ in range(n_lines)]
        text_h = width * (0.32 if family == "sot" else 0.22) if small else min(width * 0.13, length * 0.085)
    if 90.0 < angle_deg % 360.0 <= 270.0:
        angle_deg += 180.0  # keep laser marking readable from the camera
    layer = np.zeros((h, w), dtype=np.float32)
    a = np.radians(angle_deg)
    perp = np.array([-np.sin(a), np.cos(a)], dtype=np.float32)
    along = np.array([np.cos(a), np.sin(a)], dtype=np.float32)
    font = FONTS[int(rng.integers(len(FONTS)))]
    for i, text in enumerate(lines):
        off = (i - (len(lines) - 1) / 2.0) * text_h * 1.6
        # Lines are left-aligned on real parts, so their centres wander a little.
        pos = np.array(center, dtype=np.float32) + perp * off + along * float(rng.uniform(-0.15, 0.15)) * text_h
        layer = np.maximum(layer, text_layer((h, w), text, (float(pos[0]), float(pos[1])), text_h, angle_deg,
                                              max(1, ctx.ss), font))
    printed = family in ("resistor", "array_r")
    color = np.array((225, 225, 220), dtype=np.float32) if printed else np.array((190, 190, 188), dtype=np.float32)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.4 * ctx.ss)
    # Printed ink (resistors) is blotchy, laser marking (ICs) is a grainy, partly etched matte surface.
    grain = smooth_noise((h, w), ctx.ss, rng)
    texture = 0.85 + 0.15 * grain if printed else 0.8 + 0.2 * grain
    blend(img, color, np.clip(layer * texture * (0.9 if printed else float(rng.uniform(0.4, 0.65))), 0, 1))


def draw_array_terminals(img: np.ndarray, family: str, u: np.ndarray, v: np.ndarray, inside: np.ndarray,
                         aa_body: np.ndarray, length: float, width: float, rng: np.random.Generator,
                         noise: WindowedNoise, end_cap: np.ndarray) -> None:
    """Terminals of a 4-element R/C array (body *length* across the two pad columns, *width* along them).

    Each element has a tin-plated terminal wrapping over both long edges (u = +-length/2), 0.5 mm
    pitch. Resistor arrays (convex type): metal C-shaped wraps over ~22 % of the top on each side,
    black body between them. Capacitor arrays: small wraps; the ceramic shows faint stripes where
    the inner electrodes of each element run across the body.
    """
    hl, hw = length / 2.0, width / 2.0
    n = 4
    pitch = width / n
    centers = -hw + pitch * (np.arange(n) + 0.5)
    dv = np.min(np.abs(v[..., None] - centers.reshape(1, 1, -1)), axis=-1)  # distance to the nearest element axis
    t_w = (0.5 if family == "array_r" else 0.42) * pitch
    t_len = (0.24 if family == "array_r" else 0.14) * length
    term = inside & (np.abs(u) > hl - t_len) & (dv < t_w / 2.0)
    aa_t = np.clip(0.5 - np.maximum(hl - t_len - np.abs(u), dv - t_w / 2.0), 0.0, 1.0) * aa_body
    col = end_cap * float(rng.uniform(0.95, 1.08))
    ridge = np.exp(-(((np.abs(u) - (hl - t_len * 0.4)) / (t_len * 0.45)) ** 2)) * np.exp(-((dv / (0.5 * t_w)) ** 2))
    shade = 0.7 + 0.5 * ridge
    blend(img, col * shade[..., None] * (1.0 + noise.normal(rng, 0.02, channels=1)),
           np.where(term | (aa_t > 0), aa_t, 0.0).astype(np.float32))
    if family == "array_c":  # inner-electrode stripes across the ceramic
        stripe = np.exp(-((dv / (0.35 * t_w)) ** 2)) * inside * (np.abs(u) <= hl - t_len)
        img *= (1.0 - 0.1 * stripe)[..., None]


def draw_power_inductor_top(ctx: RenderContext, img: np.ndarray, u: np.ndarray, v: np.ndarray, aa_top: np.ndarray,
                            length: float, width: float, body_col: np.ndarray, rng: np.random.Generator,
                            noise: WindowedNoise) -> None:
    """Moulded power inductor seen from above: matte grey iron-powder block with an embossed ring.

    The terminals sit underneath (only the joints at the ends are visible); real 2520 parts show
    a shallow circular imprint of the coil in the middle of the top face.
    """
    r = float(rng.uniform(0.22, 0.3)) * min(length, width)
    off = np.array([rng.uniform(-0.05, 0.05) * length, rng.uniform(-0.05, 0.05) * width])
    d = np.hypot(u - off[0], v - off[1])
    ring = np.exp(-(((d - r) / (0.06 * r + 0.5 * ctx.ss)) ** 2))
    # Embossed: dark on the side away from the light, light towards it.
    lit = np.clip(-((u - off[0]) * LIGHT[0] + (v - off[1]) * LIGHT[1]) / np.maximum(d, 1e-3), -1, 1)
    img *= (1.0 + (0.12 * lit - 0.1) * ring * aa_top)[..., None]
    mottling = 1.0 + 0.05 * noise.smooth(rng, 2.5 * ctx.ss)
    img *= np.where(aa_top > 0, mottling, 1.0)[..., None]


def draw_led_top(ctx: RenderContext, img: np.ndarray, u: np.ndarray, v: np.ndarray, aa_top: np.ndarray, half_len: float, half_wid: float,
             rng: np.random.Generator) -> None:
    """0603 chip LED from above: white housing frame, yellowish epoxy lens with the die and a bond wire."""
    lu, lv = 0.88 * half_len, 0.72 * half_wid
    lens = np.clip(0.5 - np.maximum(np.abs(u) - lu, np.abs(v) - lv), 0.0, 1.0) * aa_top
    depth = 1.0 - 0.25 * np.clip(np.hypot(u / lu, v / lv), 0, 1)  # brighter in the middle of the dome
    blend(img, LED_LENS * float(rng.uniform(0.9, 1.08)) * depth[..., None], lens * 0.85)
    side = 1.0 if rng.random() < 0.5 else -1.0
    die_c = (side * 0.3 * lu, 0.0)
    ds = 0.32 * min(lu, lv)
    die = np.clip(0.5 - np.maximum(np.abs(u - die_c[0]), np.abs(v - die_c[1])) + ds, 0.0, 1.0) * lens
    blend(img, np.array((70, 72, 62), dtype=np.float32), die * 0.8)
    wire = np.clip(0.6 - np.abs(v - 0.15 * (u - die_c[0])) / max(1.0, 0.25 * ctx.ss), 0, 1)
    wire *= ((u * side) < die_c[0] * side) * ((u * side) > -0.8 * lu) * lens
    blend(img, np.array((150, 185, 200), dtype=np.float32), wire * 0.6)
    # Cathode mark: green line across one end of the lens.
    mark = np.clip(0.5 - np.abs(u + side * 0.8 * lu) / (0.08 * lu + 0.5 * ctx.ss), 0.0, 1.0) * lens
    blend(img, np.array((70, 150, 60), dtype=np.float32), mark * 0.7)


def draw_body(ctx: RenderContext, img: np.ndarray, family: str, center: tuple[float, float], length: float, width: float, angle_deg: float,
               base_color: np.ndarray, rng: np.random.Generator, *, marks: bool = True) -> None:
    """Part body seen from above. ``marks=False``: no marking/pin-1 (tombstone top face, small neighbour parts).

    Everything but the marking is computed on the window around the body (the layers are 0 or a
    factor of exactly 1 outside it); random fields are still drawn for the whole *img*, so the random
    stream is consumed exactly as by a full-frame render.
    """
    full = img
    h, w = full.shape[:2]
    y0, y1, x0, x1 = rect_window((h, w), center, length, width, angle_deg) or (0, h, 0, w)
    img = full[y0:y1, x0:x1]
    noise = WindowedNoise((h, w), (y0, y1, x0, x1))
    u, v, aa_body = rect_alpha(y1 - y0, x1 - x0, center, length, width, angle_deg, y0, x0)
    hl, hw = length / 2.0, width / 2.0
    # Visible slope at the body edge from above: chip parts have rounded edges, moulded SOT/SOIC a
    # small draft angle, sawn QFN packages vertical walls (practically invisible from above).
    inset = max(1.0, {"qfn": 0.012, "soic": 0.035, "sot": 0.05, "sot6": 0.05, "array_r": 0.05,
                      "array_c": 0.05}.get(family, 0.07) * min(length, width))

    au, av = np.abs(u), np.abs(v)
    inside = (au <= hl) & (av <= hw)
    top = (au <= hl - inset) & (av <= hw - inset)
    d_top = np.maximum(au - (hl - inset), av - (hw - inset))
    aa_top = np.clip(0.5 - d_top, 0.0, 1.0)

    # Ceramic parts take a light profile colour (component_color_bgr, e.g. brown 1210 MLCCs); the
    # default dark component colour means "standard ceramic".
    ceramic = base_color if float(np.mean(base_color)) > 60.0 else None
    if family == "capacitor":
        body_col = (ctx.look.cap_body if ceramic is None else ceramic) * float(rng.uniform(0.9, 1.1))
    elif family == "array_c":
        body_col = (CAP_ARRAY_BODY if ceramic is None else ceramic) * float(rng.uniform(0.9, 1.08))
    else:
        body_col = ctx.look.body_colour(family, base_color) * float(rng.uniform(0.9, 1.1))

    # Side walls: lit on the light-facing side, dark elsewhere.
    a = np.radians(angle_deg)
    light_u = LIGHT[0] * np.cos(a) + LIGHT[1] * np.sin(a)
    light_v = -LIGHT[0] * np.sin(a) + LIGHT[1] * np.cos(a)
    facing = np.where(au / hl > av / hw, np.sign(u) * light_u, np.sign(v) * light_v)
    side_shade = 0.45 + 0.35 * np.clip(-facing, 0, 1)
    blend(img, body_col, aa_body)  # body footprint
    img *= np.where(inside & ~top, side_shade, 1.0)[..., None]

    # Top face: slight gradient towards the light, surface grain, mild gloss.
    grad = 1.0 + 0.10 * np.clip(-(u / hl) * light_u - (v / hw) * light_v, -1, 1)
    grain_sd = {"resistor": 0.025, "capacitor": 0.025, "led": 0.015, "inductor": 0.05, "power_inductor": 0.07}.get(family, 0.06)
    grain = 1.0 + noise.normal(rng, grain_sd)  # ferrite / iron powder: coarser
    top_img = body_col * (grad * grain)[..., None]
    if family in ("qfn", "soic", "sot", "sot6"):
        gloss = np.exp(-(((u + hl * 0.35) / (hl * 0.6)) ** 2 + ((v + hw * 0.35) / (hw * 0.6)) ** 2))
        top_img += (28.0 * gloss)[..., None]
    blend(img, top_img, aa_top)

    if family == "power_inductor":
        draw_power_inductor_top(ctx, img, u, v, aa_top, length, width, body_col, rng, noise)
        if not marks:
            return
    elif family in CHIPS:
        cap_len = ctx.look.cap_len.get(family, CAP_LEN[family]) * length
        if family == "led":
            draw_led_top(ctx, img, u, v, aa_top, hl - cap_len, hw, rng)
        cap = inside & (au > hl - cap_len)
        cap_col = ctx.look.end_cap * float(rng.uniform(0.9, 1.05))
        v_across = np.clip(v / hw, 0, 1)
        if ctx.look.flat_caps:  # scanner look: the flat tin top mirrors the dark camera (grey); the plating
            # rolling over both cap edges faces the side lights and blazes (bright vertical bands on every terminal)
            d_out, d_in = hl - au, au - (hl - cap_len)
            edges = np.maximum(np.exp(-((d_out / (0.18 * cap_len)) ** 2)), 0.9 * np.exp(-((d_in / (0.16 * cap_len)) ** 2)))
            shade = 0.7 + 0.7 * edges - 0.05 * v_across
        else:
            # Metallic end caps: bright ridge across the middle of each cap, darker towards the body.
            ridge = np.exp(-(((au - (hl - cap_len * 0.45)) / (cap_len * 0.35)) ** 2))
            if ctx.solder_look is not None:  # meniscus look: flat, matte tin top (no tube-like ridge), as in real crops
                shade = 0.88 + 0.12 * ridge - 0.06 * v_across
            else:
                shade = 0.72 + 0.45 * ridge - 0.25 * v_across * 0.5
        cap_img = cap_col * shade[..., None] * (1.0 + noise.normal(rng, 0.02, channels=1))
        blend(img, cap_img, cap.astype(np.float32) * aa_body)
        if not marks:
            return
    elif family in ARRAY_FAMILIES:
        draw_array_terminals(img, family, u, v, inside, aa_body, length, width, rng, noise, ctx.look.end_cap)
        if not marks:
            return
    elif not marks:
        return
    else:
        # Pin-1 indicator (molded dimple / laser dot).
        pin1 = np.array([-hl + 0.12 * length, -hw + 0.15 * width], dtype=np.float32)
        r = max(1.5 * ctx.ss, 0.035 * min(length, width))
        dot = np.clip(1.0 - (np.hypot(u - pin1[0], v - pin1[1]) - r) / ctx.ss, 0, 1)
        blend(img, body_col * 0.55, dot * aa_top)
        hi = np.clip(1.0 - (np.hypot(u - pin1[0] + r * 0.4, v - pin1[1] + r * 0.4) - r * 0.5) / ctx.ss, 0, 1)
        blend(img, body_col * 1.6 + 25, hi * aa_top)
        if family == "soic":
            # Bevelled pin-1 edge (lighter strip along one long side).
            # Pin 1 sits at (-u, -v); the strip runs along whichever of its two sides is the long one.
            if width > length:
                strip = top & (u < -hl + inset + 0.12 * length)
            else:
                strip = top & (v < -hw + inset + 0.08 * width)
            img *= np.where(strip, 1.25, 1.0)[..., None]

    # Fine highlight on top-face edges facing the light.
    edge_top = (aa_top > 0) & (d_top > -1.2 * ctx.ss)
    img *= np.where(edge_top & (facing < 0), 1.18, 1.0)[..., None]

    m_len = length - 2 * (0.22 * length if family == "resistor" else (0.3 * length if family == "array_r" else inset * 2))
    m_wid = width - 2 * inset
    if m_wid > m_len and family not in CHIPS:  # markings run along the long side of the package
        draw_marking(ctx, full, family, center, m_wid, m_len, angle_deg + 90.0, rng)
    else:
        draw_marking(ctx, full, family, center, m_len, m_wid, angle_deg, rng)


THICKNESS_RATIO = 0.55  # body height / width of chip parts (0603: 0.45 / 0.8)


def draw_body_local(ctx: RenderContext, img: np.ndarray, family: str, center: tuple[float, float], length: float, width: float,
                     angle_deg: float, base_color: np.ndarray, rng: np.random.Generator, *, marks: bool = True) -> None:
    """:func:`draw_body` on just the part's window (cost independent of the canvas size)."""
    h, w = img.shape[:2]
    r = 0.6 * float(np.hypot(length, width)) + 2 * ctx.ss
    x0, y0 = max(0, int(center[0] - r)), max(0, int(center[1] - r))
    x1, y1 = min(w, int(center[0] + r) + 1), min(h, int(center[1] + r) + 1)
    if x1 > x0 and y1 > y0:
        draw_body(ctx, img[y0:y1, x0:x1], family, (center[0] - x0, center[1] - y0), length, width, angle_deg, base_color, rng,
                   marks=marks)


def tombstone_pose(geo: dict[str, Any], selector: int, img_shape: tuple[int, int]) -> dict[str, Any]:
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
    end_len = width * THICKNESS_RATIO * np.sin(tilt)
    center = anchor + axis * ((top_len - end_len) / 2.0)
    box = cv2.boxPoints(((float(center[0]), float(center[1])), (float(top_len + end_len), width), angle)).astype(np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)
    return {"center": center, "angle": angle, "box": box, "mask": mask, "raised_sign": raised_sign, "axis": axis,
            "anchor": anchor, "tilt": tilt, "top_len": top_len, "end_len": end_len, "height": length * np.sin(tilt)}


def draw_tombstone(ctx: RenderContext, img: np.ndarray, family: str, pose: dict[str, Any], width: float, base_color: np.ndarray,
                    rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    axis, anchor, tilt, angle = pose["axis"], pose["anchor"], pose["tilt"], pose["angle"]
    perp = np.array([-axis[1], axis[0]], dtype=np.float32) * (width / 2.0)

    # Shadow: projection of the standing part along the light onto the board.
    light_xy = LIGHT[:2] / np.linalg.norm(LIGHT[:2])
    shift = -light_xy * pose["height"] * np.linalg.norm(LIGHT[:2]) / LIGHT[2]
    raised_end = anchor + axis * pose["top_len"]
    pts = np.vstack([pose["box"], raised_end + perp + shift, raised_end - perp + shift]).astype(np.int32)
    sh = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(sh, cv2.convexHull(pts), 255)
    y0, y1, x0, x1 = blur_window(sh, 2.0 * ctx.ss)  # the blurred shadow is exactly 0 outside
    sh = cv2.GaussianBlur(sh[y0:y1, x0:x1], (0, 0), 2.0 * ctx.ss).astype(np.float32) / 255.0
    img[y0:y1, x0:x1] *= (1.0 - 0.4 * ctx.look.shadow * sh)[..., None]
    contact_shadow(ctx, img, pose["mask"], 0.4, 3 * ctx.ss)

    # Steep top face: the foreshortened body, shaded by how much it still faces the light.
    top_c = anchor + axis * (pose["top_len"] / 2.0 - pose["end_len"])
    draw_body(ctx, img, family, (float(top_c[0]), float(top_c[1])), pose["top_len"], width, angle, base_color, rng,
               marks=False)
    n_top = np.array([*(-axis * np.sin(tilt)), np.cos(tilt)], dtype=np.float32)
    win = rect_window((h, w), top_c, pose["top_len"], width, angle)
    if win is not None:
        y0, y1, x0, x1 = win
        _, _, m_top = rect_alpha(y1 - y0, x1 - x0, top_c, pose["top_len"], width, angle, y0, x0)
        img[y0:y1, x0:x1] *= (1.0 + (0.35 + 0.65 * max(0.0, float(n_top @ LIGHT)) - 1.0) * m_top)[..., None]

    # Raised end face, pointing almost straight up: bare metal end cap on chips, mold wall on ICs.
    end_c = anchor + axis * (pose["top_len"] - pose["end_len"] / 2.0)
    y0, y1, x0, x1 = rect_window((h, w), end_c, pose["end_len"], width, angle) or (0, h, 0, w)
    noise = WindowedNoise((h, w), (y0, y1, x0, x1))
    u, v, m_end = rect_alpha(y1 - y0, x1 - x0, end_c, pose["end_len"], width, angle, y0, x0)
    n_end = np.array([*(axis * np.cos(tilt)), np.sin(tilt)], dtype=np.float32)
    lit = 0.55 + 0.6 * max(0.0, float(n_end @ LIGHT))
    if family in CHIPS and family != "power_inductor":
        # Tin-plated cap, slightly convex: shaded as metal like the solder, tilted with the part.
        su = np.clip(np.abs(u) / (pose["end_len"] / 2.0), 0.0, 1.0)
        sv = np.clip(np.abs(v) / (width / 2.0), 0.0, 1.0)
        bulge = (1.0 - su**4) * (1.0 - sv**4) * 0.12 * min(pose["end_len"], width)
        bulge += 0.04 * noise.smooth(rng, 3 * ctx.ss)  # plating texture
        face = shade_solder(bulge, ctx.look.end_cap * float(rng.uniform(0.95, 1.08)), np.full(3, 70.0, dtype=np.float32),
                             30.0, 0.45) * lit
    else:
        face = (np.maximum(base_color, 34.0) * 1.3 * lit)[None, None, :] * np.ones((*u.shape, 1), dtype=np.float32)
    blend(img[y0:y1, x0:x1], face, m_end)


def draw_neighbours(ctx: RenderContext, img: np.ndarray, base: np.ndarray, parts: list[dict[str, Any]], finish: np.ndarray,
                     rng: np.random.Generator) -> None:
    if not parts:
        return
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in parts:
        box = cv2.boxPoints(((p["center"][0], p["center"][1]), (p["length"], p["width"]), p["angle"]))
        cv2.fillPoly(mask, [np.round(box).astype(np.int32)], 255)
    pads_and_solder(ctx, img, base, [pad for p in parts for pad in p["pads"]], mask, rng, finish)
    contact_shadow(ctx, img, mask, 0.35, 3 * ctx.ss)
    dark = np.array((20, 20, 20), dtype=np.float32)
    for p in parts:
        draw_body_local(ctx, img, p["family"], p["center"], p["length"], p["width"], p["angle"], dark, rng, marks=False)


def body_half_extent(half_length: float, half_width: float, angle_deg: float) -> tuple[float, float]:
    """Half extents (x, y) of the axis-aligned box around a body rotated by *angle_deg*."""
    a = np.radians(angle_deg)
    c, s = abs(float(np.cos(a))), abs(float(np.sin(a)))
    return half_length * c + half_width * s, half_length * s + half_width * c
