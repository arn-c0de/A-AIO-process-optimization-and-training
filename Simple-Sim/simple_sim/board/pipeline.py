"""Window-board pipeline: footprints -> placement -> netlist -> zones -> routing -> stitching -> DRC.

:func:`build_window_board` is re-exported by :mod:`simple_sim.board` (lazily: this module loads
cv2 through the DRC, the occupancy grid and the pours).
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np

from simple_sim.board.drc import BoardDrcError, DrcReport, check
from simple_sim.board.footprints import PadSpec, is_fine_pitch
from simple_sim.board.geometry import BBox, Vec
from simple_sim.board.model import Board, window_board
from simple_sim.board.netlist import assign_nets
from simple_sim.board.options import (DRC_STRICT_ATTEMPTS, STREAM_NET, STREAM_ROUTE, STREAM_RULES, STREAM_ZONE, BoardOptions,
                                      record_drc)
from simple_sim.board.placement import place_window
from simple_sim.board.routing import ROUTERS, seed_occupancy
from simple_sim.board.rules import DesignRules, fit_to_pitch, sample_rules
from simple_sim.board.zones import sample_zones, stitch_vias

# The package logger (the pipeline used to live in ``simple_sim/board/__init__.py``).
log = logging.getLogger("simple_sim.board")


def _stream(seed: int, key: int, attempt: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(seed), key, attempt] if attempt else [int(seed), key])


def min_row_pitch(pads: list[PadSpec]) -> float:
    """Smallest centre distance between two pads of one pin row (same escape direction), pixels."""
    best = float("inf")
    for i, a in enumerate(pads):
        if a.escape == (0.0, 0.0):
            continue
        axis = 1 if abs(a.escape[0]) >= abs(a.escape[1]) else 0
        for b in pads[i + 1:]:
            if b.escape == a.escape:
                best = min(best, abs(b.center[axis] - a.center[axis]))
    return best


def fanout_band_px(target_pads: list[PadSpec], rules: DesignRules, px_to_mm: float,
                   to_border: bool = False) -> list[BBox]:
    """Pixel rects in front of every target pad where its fan-out runs (stub, jog, staggered via);
    with *to_border* the whole corridor up to the window border (fine-pitch rows run straight out)."""
    reach = max(rules.fanout_stub * 1.2, 1.5 * rules.clearance + rules.width("power") * 1.15)
    band = (reach + 2.0 * rules.clearance + 2.0 * rules.via_diameter + rules.clearance) / px_to_mm
    if to_border:
        band = float("inf")
    side = 2.0 * rules.clearance / px_to_mm
    rects = []
    for p in target_pads:
        ex, ey = p.escape
        if (ex, ey) == (0.0, 0.0):
            continue
        x0, y0, x1, y1 = float(p.x), float(p.y), float(p.x + p.w), float(p.y + p.h)
        if ex:
            rects.append((x0 - band, y0 - side, x1, y1 + side) if ex < 0 else (x0, y0 - side, x1 + band, y1 + side))
        else:
            rects.append((x0 - side, y0 - band, x1 + side, y1) if ey < 0 else (x0 - side, y0, x1 + side, y1 + band))
    return rects


def _clear_of(n: dict[str, Any], rects: list[BBox]) -> bool:
    xs = [p["x"] for p in n["pads"]] + [p["x"] + p["w"] for p in n["pads"]]
    ys = [p["y"] for p in n["pads"]] + [p["y"] + p["h"] for p in n["pads"]]
    nx0, ny0, nx1, ny1 = min(xs), min(ys), max(xs), max(ys)
    return all(nx1 < r[0] or nx0 > r[2] or ny1 < r[1] or ny0 > r[3] for r in rects)


class WindowBoard(NamedTuple):
    """Result of :func:`build_window_board`."""

    board: Board
    report: DrcReport | None              # ``None`` with ``drc: off``
    neighbours: list[dict[str, Any]]      # the neighbour parts of the board (what the renderer draws)


def build_window_board(*, footprint: str, family: str, target_pads: list[PadSpec], target_body_px: Vec,
                       shape: tuple[int, int], neighbours: list[dict[str, Any]], px_to_mm: float,
                       options: BoardOptions, seed: int, scale: float = 1.0,
                       target_rotation: float = 0.0) -> WindowBoard:
    """Model of the render window around the inspected part: parts, nets, pours, tracks, vias.

    *target_pads* / *neighbours* are in supersampled pixels (what the renderer draws), *px_to_mm*
    converts them to mm. Neighbours that sit in the inspected part's fan-out band (where its
    stubs, jogs and vias must go) are left out: ``WindowBoard.neighbours`` are the parts the renderer
    must draw (*neighbours* itself is not modified). *scale* multiplies every design-rule length (the presets are tuned for
    parts drawn at 0.01 mm/px, see :mod:`simple_sim.board.rules`); *target_rotation* is the
    inspected part's body angle in degrees (0 or 90: a SOT-23 body stands upright).

    With ``options.drc == "strict"`` the routing stages are re-drawn (new sub-streams) until the
    DRC passes; after :data:`DRC_STRICT_ATTEMPTS` failed attempts :class:`BoardDrcError` is
    raised. ``warn`` logs the violations and returns the board with its report. Either report is
    handed to an enclosing :func:`collect_drc`.
    """
    if options.routing not in ROUTERS:
        raise ValueError(f"no router {options.routing!r} (choose from {list(ROUTERS)})")
    h, w = shape
    attempts = DRC_STRICT_ATTEMPTS if options.drc == "strict" else 1
    pitch = min_row_pitch(target_pads) * px_to_mm
    board, report = None, None
    for attempt in range(attempts):
        rules = fit_to_pitch(sample_rules(options.design_rules, _stream(seed, STREAM_RULES, attempt), scale), pitch)
        board = window_board((w * px_to_mm, h * px_to_mm), rules)
        band = fanout_band_px(target_pads, rules, px_to_mm, to_border=is_fine_pitch(family))
        kept = [n for n in neighbours if _clear_of(n, band)]
        place_window(board, footprint=footprint, family=family, target_pads=target_pads, target_body_px=target_body_px,
                     shape=shape, neighbours=kept, px_to_mm=px_to_mm, target_rotation=target_rotation)
        assign_nets(board, _stream(seed, STREAM_NET, attempt))
        sample_zones(board, _stream(seed, STREAM_ZONE, attempt), options.pour_probability)
        occ = seed_occupancy(board)
        rng_route = _stream(seed, STREAM_ROUTE, attempt)
        result = ROUTERS[options.routing]().route(board, occ, rng_route)
        board.tracks, board.vias, board.ports = result.tracks, result.vias, result.ports
        for zone in board.zones:
            board.vias += stitch_vias(board, zone, occ, rng_route)
        if options.drc == "off":
            return WindowBoard(board, None, kept)
        report = check(board)
        if report.ok:
            record_drc(report, options)
            return WindowBoard(board, report, kept)
    assert board is not None and report is not None
    if options.drc == "strict":
        raise BoardDrcError(report, seed)
    record_drc(report, options)
    log.warning("board DRC: %s (seed %s, %s)", report.to_dict(), seed, options.routing)
    return WindowBoard(board, report, kept)
