"""Placement: the inspected part and the chip parts around it.

:func:`neighbour_layout` is the legacy neighbour placement (pixel dicts, unchanged random draws,
used by ``routing: legacy`` too); :func:`place_window` turns the target and those neighbours into
:class:`~simple_sim.board.model.Part` objects in mm for the board model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from simple_sim.board.footprints import FAMILIES, PadSpec, chip_pads, part_from_pads
from simple_sim.board.model import Board, Part, Vec

# family -> reference-designator prefix, derived from footprints.FAMILIES (N1; same keys and order as before).
REFDES_PREFIX = {name: fam.refdes for name, fam in FAMILIES.items()}


def refdes_prefix(family: str) -> str:
    return REFDES_PREFIX.get(family, "U")


def pad_ref(pads: list[dict[str, int]]) -> float:
    """Smallest pad dimension: size reference for board details that must fit every package."""
    return float(min(min(p["w"], p["h"]) for p in pads))


@dataclass(frozen=True, slots=True)
class Packing:
    """How densely :func:`neighbour_layout` populates the board. The defaults are the legacy layout
    (same values and random draws as before the class existed)."""

    array_probability: float = 0.7          # share of windows with an orderly row/column of identical parts
    pitch: tuple[float, float] = (1.4, 2.0)  # array cell size relative to a part's pad extent
    clearance: float = 0.35                  # gap around the inspected part (x 2.2 x smallest pad side)
    spacing: float = 0.15                    # gap between neighbours (same unit)
    parts: float = 6.0                       # max neighbours per ROI area (x crowd)
    capacitor_share: float = 0.55            # neighbour family: capacitor, else resistor
    tries: int = 12                          # placement attempts per array part (scatter: 8)


LEGACY_PACKING = Packing()


def neighbour_layout(pads: list[dict[str, int]], keepout: tuple[int, int, int, int], shape: tuple[int, int],
                     rng: np.random.Generator, crowd: float = 1.0,
                     packing: Packing = LEGACY_PACKING) -> list[dict[str, Any]]:
    """Small chip parts around the inspected one: real ROIs are densely populated.

    Sized from the target's pads (a neighbour is about a chip with pads of that size, so it fits
    every package), at 0 or 90 degrees, possibly cut by the window border, never in the keep-out.
    *crowd*: rendered area relative to the ROI (zoomed-out views show proportionally more parts).
    *packing*: density of the layout (:class:`Packing`; default: the legacy layout).
    Returns pixel dicts: ``center, length, width, pads, angle, family``.
    """
    h, w = shape
    ref = 2.2 * pad_ref(pads)
    x0, y0, x1, y1 = keepout
    gap = packing.clearance * ref
    taken = [(x0 - gap, y0 - gap, x1 + gap, y1 + gap)]
    parts: list[dict[str, Any]] = []
    # Real layouts are orderly: passives sit in rows/columns of identical parts (decoupling caps,
    # resistor arrays) on a placement grid aligned with the inspected part. Mostly render such an
    # array; otherwise the older loose scatter of mixed parts.
    array = bool(rng.random() < packing.array_probability)
    if array:
        a_length = float(rng.uniform(0.75, 1.1)) * ref
        a_width = a_length * float(rng.uniform(0.45, 0.6))
        a_vertical = bool(rng.random() < 0.35)
        a_family = "capacitor" if rng.random() < packing.capacitor_share else "resistor"
        a_pitch = float(rng.uniform(*packing.pitch))  # cell size relative to the part's pad extent
        # max(1, ...): a canvas much smaller than the ROI rounds the count to 0 (integers(2, 2) raises).
        n_parts = int(rng.integers(2, max(1, int(round(packing.parts * crowd))) + 2))
    else:
        n_parts = int(rng.integers(0, int(round(packing.parts * crowd)) + 1))
    for _ in range(n_parts):
        length = a_length if array else float(rng.uniform(0.6, 1.1)) * ref
        width = a_width if array else length * float(rng.uniform(0.45, 0.6))
        vertical = a_vertical if array else bool(rng.random() < 0.5)
        pad_l, pad_w, off = 0.45 * length, 1.15 * width, 0.45 * length  # pad size along/across, centre offset
        ex, ey = off + pad_l / 2.0, pad_w / 2.0
        if vertical:
            ex, ey = ey, ex
        for _try in range(packing.tries if array else 8):
            cx, cy = float(rng.uniform(-0.3 * ex, w + 0.3 * ex)), float(rng.uniform(-0.3 * ey, h + 0.3 * ey))
            if array:  # snap to the placement grid through the window centre
                px_, py_ = 2.0 * ex * a_pitch, 2.0 * ey * a_pitch
                cx = w / 2.0 + round((cx - w / 2.0) / px_) * px_
                cy = h / 2.0 + round((cy - h / 2.0) / py_) * py_
            box = (cx - ex, cy - ey, cx + ex, cy + ey)
            sp = packing.spacing * ref
            if all(box[2] + sp < t[0] or box[0] - sp > t[2] or box[3] + sp < t[1] or box[1] - sp > t[3] for t in taken):
                break
        else:
            continue
        taken.append(box)
        parts.append({"center": (cx, cy), "length": length, "width": width,
                      "pads": [p.as_dict() for p in chip_pads((cx, cy), length, width, vertical)],
                      "angle": (90.0 if vertical else 0.0) + float(rng.uniform(-1.5, 1.5)),
                      "family": a_family if array else ("capacitor" if rng.random() < packing.capacitor_share else "resistor")})
    return parts


def place_window(board: Board, *, footprint: str, family: str, target_pads: list[PadSpec], target_body_px: Vec,
                 shape: tuple[int, int], neighbours: list[dict[str, Any]], px_to_mm: float,
                 target_rotation: float = 0.0) -> list[Part]:
    """Add the inspected part (image centre, where the pads and the body are drawn: ``w // 2``) and
    the neighbour chips to *board*; returns the parts. *target_rotation*: body angle (0 / 90)."""
    h, w = shape
    target = part_from_pads(f"{refdes_prefix(family)}1", footprint, family, target_pads, (float(w // 2), float(h // 2)),
                            target_body_px, float(target_rotation), px_to_mm, is_target=True)
    parts = [target]
    counters = dict.fromkeys(REFDES_PREFIX, 1)
    for n in neighbours:
        # angle = 0 / 90 plus up to 1.5 degrees of jitter (round() would miss 91 as vertical)
        vertical = abs(float(n["angle"]) % 180.0 - 90.0) < 45.0
        pads = chip_pads(n["center"], float(n["length"]), float(n["width"]), vertical)
        fam = str(n["family"])
        counters[fam] = counters.get(fam, 1) + 1
        parts.append(part_from_pads(f"{refdes_prefix(fam)}{counters[fam]}", "chip_2pad", fam, pads, n["center"],
                                    (float(n["length"]), float(n["width"])), 90.0 if vertical else 0.0, px_to_mm))
    board.parts.extend(parts)
    return parts
