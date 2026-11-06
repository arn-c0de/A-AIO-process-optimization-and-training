"""Board model for the procedural PCB: parts, nets, tracks, vias and pours as data in mm.

Pipeline (window mode, one inspected part per image):

    footprints -> placement -> netlist -> zones -> routing -> stitching -> DRC -> raster

:func:`build_window_board` runs it for one render window; ``photoreal._board`` rasterises the
result. ``render.board`` in the run config selects the router (``legacy`` keeps the old
``_traces_and_vias`` generator bit-for-bit), the design-rule preset, the pour probability and
whether the DRC runs per sample. See ``docs/refactoring/PCB_TRACE_ROUTING_PLAN.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from simple_sim.board.drc import DrcReport, check
from simple_sim.board.footprints import PadSpec, family_of, footprint_pads
from simple_sim.board.model import Board, Vec, window_board
from simple_sim.board.netlist import assign_nets
from simple_sim.board.placement import neighbour_layout, place_window
from simple_sim.board.routing import ROUTERS, seed_occupancy
from simple_sim.board.rules import DEFAULT_PRESET, PRESETS, sample_rules
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


def _stream(seed: int, key: int, attempt: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(seed), key, attempt] if attempt else [int(seed), key])


def build_window_board(*, footprint: str, family: str, target_pads: list[PadSpec], target_body_px: Vec,
                       shape: tuple[int, int], neighbours: list[dict[str, Any]], px_to_mm: float,
                       options: BoardOptions, seed: int) -> tuple[Board, DrcReport | None]:
    """Model of the render window around the inspected part: parts, nets, pours, tracks, vias.

    *target_pads* / *neighbours* are in supersampled pixels (what the renderer draws), *px_to_mm*
    converts them to mm. With ``options.drc == "strict"`` the routing stages are re-drawn (new
    sub-streams) until the DRC passes or :data:`DRC_STRICT_ATTEMPTS` are used up.
    """
    if options.routing not in ROUTERS:
        raise ValueError(f"no router {options.routing!r} (choose from {list(ROUTERS)})")
    h, w = shape
    attempts = DRC_STRICT_ATTEMPTS if options.drc == "strict" else 1
    board, report = None, None
    for attempt in range(attempts):
        rules = sample_rules(options.design_rules, _stream(seed, STREAM_RULES, attempt))
        board = window_board((w * px_to_mm, h * px_to_mm), rules)
        place_window(board, footprint=footprint, family=family, target_pads=target_pads, target_body_px=target_body_px,
                     shape=shape, neighbours=neighbours, px_to_mm=px_to_mm)
        assign_nets(board, _stream(seed, STREAM_NET, attempt))
        sample_zones(board, _stream(seed, STREAM_ZONE, attempt), options.pour_probability)
        occ = seed_occupancy(board)
        rng_route = _stream(seed, STREAM_ROUTE, attempt)
        result = ROUTERS[options.routing]().route(board, occ, rng_route)
        board.tracks, board.vias, board.ports = result.tracks, result.vias, result.ports
        for zone in board.zones:
            board.vias += stitch_vias(board, zone, occ, rng_route)
        if options.drc == "off":
            return board, None
        report = check(board)
        if report.ok:
            return board, report
    assert board is not None and report is not None
    log.warning("board DRC: %s (seed %s, %s)", report.to_dict(), seed, options.routing)
    return board, report


__all__ = [
    "BOARD_ROUTERS", "DEFAULT_MM_PER_PX", "DRC_MODES", "LEGACY", "NONE", "Board", "BoardOptions", "DrcReport", "PadSpec",
    "board_options", "build_window_board", "family_of", "footprint_pads", "neighbour_layout", "validate_board_options",
]
