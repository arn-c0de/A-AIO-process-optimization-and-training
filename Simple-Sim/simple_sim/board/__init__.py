"""Board model for the procedural PCB: parts, nets, tracks, vias and pours as data in mm.

Pipeline (window mode, one inspected part per image):

    footprints -> placement -> netlist -> zones -> routing -> stitching -> DRC -> raster

:func:`build_window_board` runs it for one render window; ``photoreal.scene.draw_board`` rasterises the
result. ``render.board`` in the run config selects the router (``legacy`` keeps the old
``photoreal.copper.draw_legacy_copper`` generator bit-for-bit), the design-rule preset, the pour probability and
whether the DRC runs per sample. See ``docs/refactoring/PCB_TRACE_ROUTING_PLAN.md``.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from simple_sim.board.drc import BoardDrcError, DrcReport, check
from simple_sim.board.footprints import DEFAULT_FOOTPRINT, PadSpec, family_of, footprint_pads, profile_footprint
from simple_sim.board.geometry import BBox, Vec
from simple_sim.board.model import Board, window_board
from simple_sim.board.netlist import assign_nets
from simple_sim.board.placement import DENSE_PACKING, LEGACY_PACKING, Packing, neighbour_layout, place_window
from simple_sim.board.routing import ROUTERS, seed_occupancy
from simple_sim.board.routing.escape import FINE_PITCH_FAMILIES
from simple_sim.board.rules import DEFAULT_PRESET, PRESETS, DesignRules, fit_to_pitch, sample_rules
from simple_sim.board.zones import POUR_PROBABILITY, sample_zones, stitch_vias

log = logging.getLogger(__name__)

LEGACY = "legacy"
NONE = "none"
# ``roi.mm_per_px`` of the default run configs: the renderer's fallback when a caller passes no
# ``mm_per_px`` in the render config (profiles are in pixels at that scale, see rules.py).
DEFAULT_MM_PER_PX = 0.01
BOARD_ROUTERS = (LEGACY, NONE, *ROUTERS)
DRC_MODES = ("off", "warn", "strict")
DRC_STRICT_ATTEMPTS = 4
# Keyed random streams per stage (``default_rng([board_seed, key])``): a change in one stage never
# shifts the draws of another. Keys 1 and 2 are taken by the 3D backend (camera model, camera pose).
STREAM_RULES, STREAM_PLACE, STREAM_NET, STREAM_ZONE, STREAM_ROUTE = 10, 11, 12, 13, 14


@dataclass(frozen=True, slots=True)
class BoardOptions:
    routing: str = LEGACY
    design_rules: str = DEFAULT_PRESET
    pour_probability: float = POUR_PROBABILITY
    drc: str = "off"

    @property
    def legacy(self) -> bool:
        return self.routing == LEGACY


def validate_board_options(raw: Any) -> None:
    """Raise ``ValueError`` unless *raw* is a valid ``render.board`` mapping."""
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ValueError("render.board must be a mapping")
    unknown = set(raw) - {"routing", "design_rules", "pour_probability", "drc"}
    if unknown:
        raise ValueError(f"render.board: unknown keys {sorted(unknown)}")
    if raw.get("routing", LEGACY) not in BOARD_ROUTERS:
        raise ValueError(f"render.board.routing must be one of {list(BOARD_ROUTERS)}")
    if raw.get("design_rules", DEFAULT_PRESET) not in PRESETS:
        raise ValueError(f"render.board.design_rules must be one of {sorted(PRESETS)}")
    p = raw.get("pour_probability", POUR_PROBABILITY)
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError("render.board.pour_probability must be a probability in [0, 1]")
    if raw.get("drc", "off") not in DRC_MODES:
        raise ValueError(f"render.board.drc must be one of {list(DRC_MODES)}")


def board_options(render_config: Mapping[str, Any] | None) -> BoardOptions:
    """``render.board`` of a render config as :class:`BoardOptions` (validated)."""
    raw = (render_config or {}).get("board")
    validate_board_options(raw)
    raw = raw or {}
    return BoardOptions(routing=str(raw.get("routing", LEGACY)), design_rules=str(raw.get("design_rules", DEFAULT_PRESET)),
                        pour_probability=float(raw.get("pour_probability", POUR_PROBABILITY)), drc=str(raw.get("drc", "off")))


#: DRC summaries of the boards built in the current context (see :func:`collect_drc`); ``None`` = nobody listens.
_DRC_SINK: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("board_drc_sink", default=None)


def drc_summary(report: DrcReport, options: BoardOptions) -> dict[str, Any]:
    """Compact, JSON-ready DRC result of one board (``render_meta.board_drc`` in ``meta.jsonl``):
    ``ok``, the number of violations, the counts per violation kind and the ``render.board`` options
    (router, design-rule preset, DRC mode) the board was built with. No messages."""
    return {"ok": report.ok, "violations": len(report.violations), "kinds": report.counts(),
            "routing": options.routing, "design_rules": options.design_rules, "drc": options.drc}


@contextmanager
def collect_drc() -> Iterator[list[dict[str, Any]]]:
    """Collect the :func:`drc_summary` of every board :func:`build_window_board` checks while the
    block runs (``drc: warn | strict``; nothing is collected with ``off``, and a strict board that
    fails every attempt raises instead). The sink is per thread / context, so parallel renders never
    mix, and listening changes neither the boards nor any random stream::

        with collect_drc() as reports:
            img = render_record(spec, record)
        meta["board_drc"] = reports[0]   # one board per 2D render window
    """
    sink: list[dict[str, Any]] = []
    token = _DRC_SINK.set(sink)
    try:
        yield sink
    finally:
        _DRC_SINK.reset(token)


def _record_drc(report: DrcReport, options: BoardOptions) -> None:
    sink = _DRC_SINK.get()
    if sink is not None:
        sink.append(drc_summary(report, options))


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
        band = fanout_band_px(target_pads, rules, px_to_mm, to_border=family in FINE_PITCH_FAMILIES)
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
            _record_drc(report, options)
            return WindowBoard(board, report, kept)
    assert board is not None and report is not None
    if options.drc == "strict":
        raise BoardDrcError(report, seed)
    _record_drc(report, options)
    log.warning("board DRC: %s (seed %s, %s)", report.to_dict(), seed, options.routing)
    return WindowBoard(board, report, kept)


__all__ = [
    "BOARD_ROUTERS", "DEFAULT_FOOTPRINT", "DEFAULT_MM_PER_PX", "DRC_MODES", "LEGACY", "NONE", "BBox", "Board", "BoardDrcError",
    "BoardOptions", "DrcReport", "PadSpec", "Vec", "WindowBoard",
    "board_options", "build_window_board", "collect_drc", "drc_summary", "family_of", "footprint_pads", "neighbour_layout", "profile_footprint", "Packing", "LEGACY_PACKING", "DENSE_PACKING", "validate_board_options",
]
