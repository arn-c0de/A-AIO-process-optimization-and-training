"""Copper under the solder mask: legacy random traces/pours/vias, or the rasterised board model.

``render.board.routing`` selects the source (:data:`COPPER_SOURCES`): ``legacy`` keeps the old
generator bit-for-bit, ``none`` draws no copper, every router of :mod:`simple_sim.board.routing`
builds the board model of the render window.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from simple_sim.board import (DEFAULT_MM_PER_PX, LEGACY, NONE, STREAM_PLACE, BoardOptions, PadSpec, build_window_board,
                              neighbour_layout)
from simple_sim.board.geometry import octagon, polyline45
from simple_sim.board.placement import pad_ref
from simple_sim.board.raster import CopperLayers, rasterize

from .context import RenderContext
from .imageops import blend
from .palette import ARRAY_FAMILIES, CHIPS


def copper_colour(ctx: RenderContext, base: np.ndarray) -> np.ndarray:
    return np.clip(base * ctx.look.copper_under_mask, 0, 255)


def blend_copper(ctx: RenderContext, img: np.ndarray, base: np.ndarray, mask: np.ndarray) -> None:
    """Copper *mask* (uint8) shimmering through the solder mask."""
    alpha = cv2.GaussianBlur(mask, (0, 0), 0.6 * ctx.ss).astype(np.float32) / 255.0 * 0.55
    blend(img, copper_colour(ctx, base), alpha)


def draw_via(ctx: RenderContext, img: np.ndarray, base: np.ndarray, center: tuple[int, int], r: int, tented: bool) -> None:
    """Via of radius *r* px: copper ring, hole (dark when open, mask-coloured when tented) and mask dome highlight.

    Drawn on a window around the via only (identical result to drawing on the full canvas: outside
    the window every layer is 0, and a window clipped by the canvas has the same border handling).
    """
    h, w = img.shape[:2]
    m = int(r) + 4 * max(1, ctx.ss) + 4  # ring/highlight radius + highlight stroke + blur reach
    x0, y0 = max(0, int(center[0]) - m), max(0, int(center[1]) - m)
    x1, y1 = min(w, int(center[0]) + m + 1), min(h, int(center[1]) + m + 1)
    if x1 <= x0 or y1 <= y0:
        return
    draw_via_window(ctx, img[y0:y1, x0:x1], base, (int(center[0]) - x0, int(center[1]) - y0), r, tented)


def draw_via_window(ctx: RenderContext, img: np.ndarray, base: np.ndarray, center: tuple[int, int], r: int, tented: bool) -> None:
    h, w = img.shape[:2]
    vx, vy = center
    ring = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ring, (vx, vy), r, 255, -1, lineType=cv2.LINE_AA)
    blend(img, copper_colour(ctx, base), ring.astype(np.float32) / 255.0 * 0.9)
    hole = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hole, (vx, vy), max(1, int(r * 0.45)), 255, -1, lineType=cv2.LINE_AA)
    hole_col = base * 0.5 if tented else np.array((22, 30, 24), dtype=np.float32)
    blend(img, hole_col, hole.astype(np.float32) / 255.0)
    # Tented via = small dome of mask: highlight on the side facing the light.
    hi = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(hi, (vx, vy), (r, r), 0, 190, 260, 255, max(1, ctx.ss), lineType=cv2.LINE_AA)
    blend(img, np.clip(base * 0.6 + 100.0, 0, 255), cv2.GaussianBlur(hi, (0, 0), 0.5 * ctx.ss).astype(np.float32) / 255.0 * 0.5)


def draw_copper(ctx: RenderContext, img: np.ndarray, base: np.ndarray, copper: CopperLayers) -> None:
    """Rasterised board model (``render.board.routing: escape``): tracks, pours and vias."""
    blend_copper(ctx, img, base, copper.copper)
    for via in copper.vias:
        draw_via(ctx, img, base, (via.x, via.y), via.radius, via.tented)


def draw_legacy_copper(ctx: RenderContext, img: np.ndarray, base: np.ndarray, pads: list[dict[str, int]], keepout: tuple[int, int, int, int],
                     rng: np.random.Generator) -> None:
    """Legacy copper (``render.board.routing: legacy``): random pours, traces and vias that know nothing of nets."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = keepout
    # Copper feature sizes follow the part scale (a 0603 pad is ~0.8 mm): traces 0.15-0.3 mm, vias
    # ~0.5-0.6 mm across. Fixed pixel sizes made them hairlines and pin pricks next to a chip.
    ref = float(np.clip(pad_ref(pads), 6.0 * ctx.ss, 40.0 * ctx.ss)) if pads else 20.0 * ctx.ss

    def _feature(lo: float, hi: float, lo_px: float, hi_px: float) -> float:
        return float(np.clip(float(rng.uniform(lo, hi)) * ref, lo_px * ctx.ss, hi_px * ctx.ss))

    # Copper pours: large planes with straight, 45-degree chamfered edges, kept clear of the part.
    if rng.random() < 0.6:
        pour = np.zeros((h, w), dtype=np.uint8)
        for _ in range(int(rng.integers(1, 3))):
            pw_, ph_ = float(rng.uniform(0.4, 1.2)) * w, float(rng.uniform(0.4, 1.2)) * h
            ax, ay = float(rng.uniform(-0.2 * w, w - 0.5 * pw_)), float(rng.uniform(-0.2 * h, h - 0.5 * ph_))
            c = float(rng.uniform(0.05, 0.25)) * min(pw_, ph_)
            cv2.fillPoly(pour, [np.round(octagon(ax, ay, pw_, ph_, c)).astype(np.int32)], 255)
        gap = int(round(float(rng.uniform(3.0, 6.0)) * ctx.ss))
        cv2.rectangle(pour, (x0 - gap, y0 - gap), (x1 + gap, y1 + gap), 0, -1)
        mask = np.maximum(mask, pour)

    def _polyline(p0: tuple[int, int], p1: tuple[int, int], width: int) -> None:
        # Route with one 45-degree jog like real autorouted traces (the board model's polyline45; the
        # jog of integer pixels is an integral float, cast back to int32 here; no dedupe: see there).
        cv2.polylines(mask, [np.array(polyline45(p0, p1, dedupe=False), dtype=np.int32)], False, 255, width,
                      lineType=cv2.LINE_AA)

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

    blend_copper(ctx, img, base, mask)

    # Vias outside the footprint keep-out; most are tented (covered by mask), some open. Production
    # boards are full of them (every passive fans out into one or two).
    n_vias = int(rng.integers(2, 17)) if rng.random() < 0.8 else int(rng.integers(0, 3))
    for _ in range(int(round(n_vias * h * w / (512.0 * 512.0 * ctx.ss * ctx.ss)))):
        for _try in range(6):
            vx, vy = int(rng.integers(0, w)), int(rng.integers(0, h))
            if not (x0 <= vx <= x1 and y0 <= vy <= y1):
                break
        else:
            continue
        r = int(round(_feature(0.22, 0.36, 3.5, 11.0)))
        draw_via(ctx, img, base, (vx, vy), r, tented=bool(rng.random() < 0.85))


# Design-rule scale of the board model. The presets (simple_sim.board.rules) are tuned next to a
# 0603 pad of RULES_REF_PAD_MM at the default profile scale (0.01 mm/px). roi.mm_per_px is not
# reliable as a physical scale (some profiles are physical, most use the 0.01 profile scale), so the
# rules follow the drawn pad size instead: scale = pad_ref_mm / RULES_REF_PAD_MM. Fine-pitch packages
# (IC pads of ~0.25-0.6 mm real) get FINE_PITCH_RULE_BOOST on top: on real boards their tracks are
# about as wide as the pads, while a 0603's tracks are a fraction of its much larger pads.
RULES_REF_PAD_MM = 0.31


FINE_PITCH_RULE_BOOST = 2.0


RULE_SCALE_LIMITS = (0.3, 6.0)


# Families whose rules follow their (large) pads; the others are fine-pitch ICs and get the boost.
# X8: this is the rule-scale-boost predicate, deliberately NOT board.footprints FINE_PITCH_FAMILIES /
# is_fine_pitch (SOT gets the boost here, chip arrays do not), so do not merge the two.
RULE_CHIP_FAMILIES = (*CHIPS, *ARRAY_FAMILIES)


def board_rule_scale(pads: list[dict[str, int]], px_to_mm: float, family: str) -> float:
    """Factor for ``sample_rules`` so copper sizes follow the drawn part, whatever ``roi.mm_per_px`` says."""
    if not pads:
        return 1.0
    scale = pad_ref(pads) * float(px_to_mm) / RULES_REF_PAD_MM
    if family not in RULE_CHIP_FAMILIES:
        scale *= FINE_PITCH_RULE_BOOST
    return float(np.clip(scale, *RULE_SCALE_LIMITS))


@dataclass(frozen=True, slots=True)
class CopperJob:
    """What a copper source needs to know about the render window (supersampled pixels)."""

    footprint: str
    family: str
    specs: list[PadSpec]                 # target pads
    pads: list[dict[str, int]]           # the same as dicts
    keepout: tuple[int, int, int, int]   # nominal part area
    body_px: tuple[float, float]         # nominal body length, width
    body_angle: float                    # nominal body angle (nominal_body_angle)
    shape: tuple[int, int]               # (h, w)
    options: BoardOptions
    mm_per_px: float                     # roi.mm_per_px (camera pixels)
    crowd: float                         # neighbour density factor of the canvas


#: A copper source draws onto the board and returns the neighbour parts it placed (``None``: the
#: neighbours are placed afterwards from the main stream, as the legacy generator always did).
CopperSource = Callable[[RenderContext, np.ndarray, np.ndarray, CopperJob, np.random.Generator], list[dict[str, Any]] | None]


def legacy_copper(ctx: RenderContext, img: np.ndarray, base: np.ndarray, job: CopperJob,
                  rng: np.random.Generator) -> None:
    draw_legacy_copper(ctx, img, base, job.pads, job.keepout, rng)
    return None


def no_copper(ctx: RenderContext, img: np.ndarray, base: np.ndarray, job: CopperJob, rng: np.random.Generator) -> None:
    return None


def board_model_copper(ctx: RenderContext, img: np.ndarray, base: np.ndarray, job: CopperJob,
                       rng: np.random.Generator) -> list[dict[str, Any]]:
    """Board model: neighbours are placed first (from their own stream) so the copper can connect them
    and keep clear of them; one draw from *rng* seeds the model's streams."""
    board_seed = int(rng.integers(1 << 62))
    neighbours = neighbour_layout(job.pads, job.keepout, job.shape, np.random.default_rng([board_seed, STREAM_PLACE]),
                                  job.crowd, ctx.look.packing)
    px_to_mm = job.mm_per_px / ctx.ss
    built = build_window_board(
        footprint=job.footprint, family=job.family, target_pads=job.specs, target_body_px=job.body_px,
        shape=job.shape, neighbours=neighbours, px_to_mm=px_to_mm, options=job.options, seed=board_seed,
        scale=board_rule_scale(job.pads, px_to_mm, job.family), target_rotation=job.body_angle)
    draw_copper(ctx, img, base, rasterize(built.board, job.shape, px_to_mm))
    return built.neighbours


#: ``render.board.routing`` -> copper source; routings not listed here are routers of the board model.
COPPER_SOURCES: dict[str, CopperSource] = {LEGACY: legacy_copper, NONE: no_copper}


def draw_board_copper(ctx: RenderContext, img: np.ndarray, base: np.ndarray, job: CopperJob,
                      rng: np.random.Generator) -> list[dict[str, Any]] | None:
    """Draw the copper of ``job.options.routing``; returns the neighbours the source placed (or ``None``)."""
    return COPPER_SOURCES.get(job.options.routing, board_model_copper)(ctx, img, base, job, rng)


def mm_per_px(config: dict[str, Any]) -> float:
    """``mm_per_px`` of a render config (merged from ``roi.mm_per_px``), default :data:`DEFAULT_MM_PER_PX`."""
    return float(config.get("mm_per_px", DEFAULT_MM_PER_PX))
