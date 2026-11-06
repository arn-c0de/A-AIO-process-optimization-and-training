"""Solder joint model for the photoreal 2D renderer (``render.solder: meniscus``).

The legacy joints (``render.solder: legacy``, the default) are a grey slab with a nearly linear ramp,
shaded with a soft grey environment: low contrast, straight pad-shaped outline, no mirror look.
Real reflowed joints (calibrated against the public SolDef_AI crops, CC BY 4.0, see
``docs/guides/PHOTOREAL_RENDERING.md``) differ in four ways, each modelled here:

* **Outline** - the solder wets the pad but its contact line is rounded at the corners and wanders a
  little; a thin rim of bare finish (gold ENIG / grey HASL) is often visible (:func:`wetted_alpha`).
* **Profile** - a concave meniscus: steep at the terminal, running out flat at the toe, with a small
  rounded lip at the contact line; too much solder gives a convex, high bulge (:func:`fillet_height`).
* **Surface** - lead-free solder is wrinkled/frosted: low-frequency lumps plus fine grain that break
  the highlights into streaks and sparkles.
* **Shading** - solder is a mirror. What a pixel shows is what its reflected view ray hits: the light
  (ring light / room lamps -> saturated white streaks), the bright overhead area, the dark solder mask
  and the black part body (-> dark, greenish or blue reflections). That gives the high contrast of
  real joints (:func:`shade`). A frosted share keeps a little diffuse grey.

Every random choice comes from the passed generator; :class:`SolderLook` is drawn once per image so
all joints of one image share the lighting.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

SOLDER_STYLES = ("legacy", "meniscus")
DEFAULT_SOLDER_STYLE = "legacy"

_SOLDER_TINT = np.array((1.0, 0.985, 0.96), dtype=np.float32)  # SAC alloy: neutral, faintly cool (BGR)
_BODY_DARK = np.array((22.0, 22.0, 24.0), dtype=np.float32)    # black resistor body / moulding, mirrored


def solder_style(config: dict | None) -> str:
    """``render.solder`` of a render config (default :data:`DEFAULT_SOLDER_STYLE`)."""
    style = (config or {}).get("solder") or DEFAULT_SOLDER_STYLE
    if style not in SOLDER_STYLES:
        raise ValueError(f"render.solder must be one of {list(SOLDER_STYLES)}, got {style!r}")
    return str(style)


@dataclass(frozen=True)
class SolderLook:
    """Lighting and surface of all joints in one image."""

    ring_elev: float          # elevation (rad) of the light sources as seen in the mirror
    ring_width: float         # angular half width (rad) of the light band
    azimuths: tuple[float, ...]  # light directions (rad, image frame); empty = a full ring light
    azimuth_width: float      # angular half width (rad) of each light around its azimuth
    ring_gain: float          # brightness of the light reflection (8-bit levels, may exceed 255)
    sky: float                # brightness of the area straight above the board (camera, ceiling)
    sky_from: float           # elevation (rad) where the bright overhead area begins
    board_mix: float          # how much the low slopes mirror the mask colour (rest: grey surroundings)
    frost: float              # diffuse share of a frosted, lead-free surface (0 = perfect mirror)
    lumps: float              # low-frequency surface relief, fraction of the joint height
    grain: float              # fine wrinkles, height amplitude in supersampled pixels per ss
    lip: float                # contact-line lip height, fraction of the joint height
    bare_rim: float           # probability that a pad shows a rim of bare finish around the solder
    level: float              # overall brightness factor


def sample_look(rng: np.random.Generator) -> SolderLook:
    n_lights = int(rng.choice((0, 1, 2, 3), p=(0.35, 0.25, 0.25, 0.15)))  # 0: ring light (AOI)
    return SolderLook(
        ring_elev=float(np.radians(rng.uniform(25.0, 60.0))),
        ring_width=float(np.radians(rng.uniform(7.0, 16.0))),
        azimuths=tuple(float(a) for a in rng.uniform(0.0, 2.0 * np.pi, n_lights)),
        azimuth_width=float(np.radians(rng.uniform(25.0, 60.0))),
        ring_gain=float(rng.uniform(260.0, 420.0)),
        sky=float(rng.uniform(160.0, 230.0)),
        sky_from=float(np.radians(rng.uniform(5.0, 25.0))),
        board_mix=float(rng.uniform(0.5, 0.9)),
        frost=float(rng.uniform(0.15, 0.32)),
        lumps=float(rng.uniform(0.04, 0.12)),
        grain=float(rng.uniform(0.02, 0.06)),
        lip=float(rng.uniform(0.04, 0.12)),
        bare_rim=float(rng.uniform(0.2, 0.7)),
        level=float(rng.uniform(0.9, 1.06)),
    )


def side_stream(rng: np.random.Generator, salt: int) -> np.random.Generator:
    """A generator derived from *rng*'s current state without advancing it.

    The meniscus joints draw from it, so the rest of an image (board, part, camera noise) stays
    bit-identical to ``render.solder: legacy`` with the same seed - only the joints differ.
    """
    digest = hashlib.blake2b(repr(rng.bit_generator.state).encode(), digest_size=8).digest()
    return np.random.default_rng([int.from_bytes(digest, "little"), int(salt)])


def _noise(shape: tuple[int, int], cell: float, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    small = rng.normal(0.0, 1.0, (int(h / cell) + 2, int(w / cell) + 2)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def wetted_alpha(ph: int, pw: int, ss: int, look: SolderLook, rng: np.random.Generator) -> np.ndarray:
    """Anti-aliased area the solder covers on a ``ph`` x ``pw`` pad: rounded corners, wavy contact line,
    sometimes inset so a rim of bare finish stays visible."""
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    inset = float(rng.uniform(0.6, 2.2)) * ss if rng.random() < look.bare_rim else 0.0
    r = float(rng.uniform(0.18, 0.45)) * min(pw, ph)
    hx, hy = pw / 2.0 - inset, ph / 2.0 - inset
    qx = np.maximum(np.abs(xx + 0.5 - pw / 2.0) - (hx - r), 0.0)
    qy = np.maximum(np.abs(yy + 0.5 - ph / 2.0) - (hy - r), 0.0)
    inner = np.maximum(np.abs(xx + 0.5 - pw / 2.0) - (hx - r), np.abs(yy + 0.5 - ph / 2.0) - (hy - r))
    sdf = np.where((qx > 0) & (qy > 0), np.hypot(qx, qy), np.maximum(inner, 0.0)) - r  # < 0 inside
    sdf += _noise((ph, pw), 5.0 * ss, rng) * 0.5 * ss  # contact line is not a ruler-straight edge
    return np.clip(0.5 - sdf / ss, 0.0, 1.0)


def fillet_height(covered: np.ndarray, wet: np.ndarray, amp: float, look: SolderLook,
                  rng: np.random.Generator, ss: int, *, excess: bool = False) -> np.ndarray:
    """Height map (pixels) of a joint wetting the part terminal / lead foot in *covered*.

    ``s`` runs from 0 at the terminal to 1 at the contact line on the pad. Normal joints are concave
    (``(1 - s)^p``, p ~ 1.7-2.6); *excess* joints are convex and higher than the terminal.
    """
    inside = (wet > 0.5).astype(np.uint8)
    d_term = cv2.distanceTransform((~covered).astype(np.uint8), cv2.DIST_L2, 3)
    d_edge = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    s = d_term / np.maximum(d_term + d_edge, 1e-3)
    # The meniscus cannot be steeper on average than ~25-30 deg over the visible toe.
    toe = float(d_term[inside > 0].max()) if inside.any() else 0.0
    amp = min(amp, max(0.45 * toe, 1.0))
    if excess:
        h = amp * 1.55 * np.sqrt(np.clip(1.0 - s ** 2.2, 0.0, 1.0))
    else:
        p = float(rng.uniform(1.7, 2.6))
        wetting = float(rng.uniform(0.55, 1.0))  # how far up the terminal the solder climbs
        h = amp * wetting * (1.0 - s) ** p
    # Rounded contact line: the meniscus meets the pad at a small angle, which catches light.
    edge_w = 1.5 * ss
    h += amp * look.lip * (1.0 - np.exp(-d_edge / edge_w))
    return _relief(h, amp, look, rng, ss) * wet  # under the terminal s = 0: the wetting height


def paste_height(ph: int, pw: int, wet: np.ndarray, amp: float, cells: tuple[int, int], look: SolderLook,
                 rng: np.random.Generator, ss: int) -> np.ndarray:
    """Reflowed paste on a pad without a part: a low dome per stencil window (*cells* = (nx, ny))."""
    nx, ny = cells
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    u, v = ((xx / pw * nx) % 1.0) * 2 - 1, ((yy / ph * ny) % 1.0) * 2 - 1
    dome = np.clip(1.0 - (np.abs(u) ** 2.5 + np.abs(v) ** 2.5) * (0.9 if nx * ny == 1 else 1.25), 0, 1) ** 0.6
    return _relief(dome * amp, amp, look, rng, ss) * wet


def _relief(h: np.ndarray, amp: float, look: SolderLook, rng: np.random.Generator, ss: int) -> np.ndarray:
    shape = h.shape
    h = h * (1.0 + look.lumps * _noise(shape, 9.0 * ss, rng))
    h += look.grain * ss * (_noise(shape, 1.6 * ss, rng) + 0.6 * _noise(shape, 3.0 * ss, rng))
    return np.maximum(h, 0.0)


def render_joint(region: np.ndarray, board: np.ndarray, covered: np.ndarray, amp: float, look: SolderLook,
                 rng: np.random.Generator, ss: int, *, excess: bool = False,
                 cells: tuple[int, int] = (1, 1)) -> None:
    """Draw one pad's solder into *region* (float BGR, the pad with its finish) in place.

    *covered*: pixels of the region under the part terminal / lead foot (none: reflowed paste,
    *cells* stencil windows). *amp*: nominal joint height in pixels.
    """
    ph, pw = region.shape[:2]
    wet = wetted_alpha(ph, pw, ss, look, rng)
    toward = None
    if covered.any():
        hmap = fillet_height(covered, wet, amp, look, rng, ss, excess=excess)
        d_term = cv2.distanceTransform((~covered).astype(np.uint8), cv2.DIST_L2, 3)
        gy, gx = np.gradient(d_term)
        norm = np.maximum(np.hypot(gx, gy), 1e-6)
        toward = np.dstack([-gx / norm, -gy / norm])  # from the joint towards the terminal
    else:
        hmap = paste_height(ph, pw, wet, min(amp * 0.35, 4.0 * ss), cells, look, rng, ss)
    shaded = shade(hmap, board, look, toward_body=toward, body_height=np.radians(30.0))
    a = wet[..., None]
    region *= 1.0 - a
    region += shaded * a


def shade(hmap: np.ndarray, board: np.ndarray, look: SolderLook, *, toward_body: np.ndarray | None = None,
          body_height: float = 0.0) -> np.ndarray:
    """Shade a solder height map as a frosted mirror. Returns float BGR (H, W, 3).

    *board*: mask colour (BGR) the low slopes mirror. *toward_body*: unit vectors (H, W, 2) pointing
    from each pixel to the part terminal; slopes facing it mirror the dark body up to *body_height*
    (elevation in radians as seen from the joint).
    """
    gy, gx = np.gradient(hmap)
    n = np.dstack([-gx, -gy, np.ones_like(hmap)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    # Reflected view ray (camera looks straight down, view vector (0, 0, 1)).
    rz = 2.0 * n[..., 2] ** 2 - 1.0
    rx, ry = 2.0 * n[..., 2] * n[..., 0], 2.0 * n[..., 2] * n[..., 1]
    elev = np.arcsin(np.clip(rz, -1.0, 1.0))
    azim = np.arctan2(ry, rx)

    t = np.clip((elev - look.sky_from) / np.radians(35.0), 0.0, 1.0)
    sky = t * t * (3.0 - 2.0 * t)  # smoothstep: bright overhead hemisphere above sky_from
    band = np.exp(-((elev - look.ring_elev) / look.ring_width) ** 2)
    if look.azimuths:
        lobes = np.zeros_like(hmap)
        for a in look.azimuths:
            d = np.angle(np.exp(1j * (azim - a)))
            lobes = np.maximum(lobes, np.exp(-(d / look.azimuth_width) ** 2))
        band = band * lobes
    horizon = np.clip(1.0 - sky - band, 0.0, 1.0)
    board_refl = np.asarray(board, dtype=np.float32) * look.board_mix + (1.0 - look.board_mix) * 95.0

    env = (sky * look.sky)[..., None] * _SOLDER_TINT + (band * look.ring_gain)[..., None] * _SOLDER_TINT
    env += horizon[..., None] * board_refl
    if toward_body is not None and body_height > 0:
        facing = np.clip((rx * toward_body[..., 0] + ry * toward_body[..., 1]) / np.maximum(np.hypot(rx, ry), 1e-6), 0, 1)
        body = (facing ** 2) * (elev < body_height)
        env = env * (1.0 - body)[..., None] + body[..., None] * _BODY_DARK

    diffuse = (0.45 + 0.35 * np.clip(n[..., 2], 0, 1))[..., None] * (_SOLDER_TINT * 175.0)
    out = (1.0 - look.frost) * env + look.frost * diffuse
    return out * look.level
