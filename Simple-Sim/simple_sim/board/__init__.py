"""Board model for the procedural PCB: parts, nets, tracks, vias and pours as data in mm.

Pipeline (window mode, one inspected part per image):

    footprints -> placement -> netlist -> zones -> routing -> stitching -> DRC -> raster

:func:`build_window_board` runs it for one render window; ``photoreal.scene.draw_board`` rasterises the
result. ``render.board`` in the run config selects the router (``legacy`` keeps the old
``photoreal.copper.draw_legacy_copper`` generator bit-for-bit), the design-rule preset, the pour probability and
whether the DRC runs per sample. See ``docs/refactoring/PCB_TRACE_ROUTING_PLAN.md``.

Importing the package (or the data model: ``board.model``, ``geometry``, ``rules``, ``footprints``)
does not load cv2: the options, footprints and placement names are imported eagerly, the
pipeline (:mod:`simple_sim.board.pipeline`), the DRC, the routers, the pours and the raster
are resolved on first attribute access (PEP 562).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from simple_sim.board.footprints import (DEFAULT_FOOTPRINT, FINE_PITCH_FAMILIES, PadSpec, family_of, footprint_pads,
                                         is_fine_pitch, profile_footprint)
from simple_sim.board.geometry import BBox, Vec
from simple_sim.board.model import Board, window_board
from simple_sim.board.netlist import assign_nets
from simple_sim.board.options import (BOARD_ROUTERS, DEFAULT_MM_PER_PX, DRC_MODES, DRC_STRICT_ATTEMPTS, LEGACY, NONE,
                                      POUR_PROBABILITY, ROUTER_NAMES, STREAM_NET, STREAM_PLACE, STREAM_ROUTE, STREAM_RULES,
                                      STREAM_ZONE, BoardOptions, board_options, collect_drc, drc_summary,
                                      validate_board_options)
from simple_sim.board.placement import LEGACY_PACKING, Packing, neighbour_layout, place_window
from simple_sim.board.rules import DEFAULT_PRESET, PRESETS, DesignRules, fit_to_pitch, sample_rules

if TYPE_CHECKING:
    from simple_sim.board.drc import BoardDrcError, DrcReport, check
    from simple_sim.board.pipeline import WindowBoard, build_window_board, fanout_band_px, min_row_pitch
    from simple_sim.board.routing import ROUTERS, seed_occupancy
    from simple_sim.board.zones import sample_zones, stitch_vias

# cv2-dependent names -> the module that defines them (imported on first access).
_LAZY = {
    "BoardDrcError": "drc", "DrcReport": "drc", "check": "drc",
    "WindowBoard": "pipeline", "build_window_board": "pipeline", "fanout_band_px": "pipeline", "min_row_pitch": "pipeline",
    "ROUTERS": "routing", "seed_occupancy": "routing",
    "sample_zones": "zones", "stitch_vias": "zones",
}
_SUBMODULES = ("drc", "footprints", "geometry", "model", "netlist", "occupancy", "options", "pipeline", "placement", "raster",
               "routing", "rules", "zones")


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY) | set(_SUBMODULES))


__all__ = [
    "BOARD_ROUTERS", "DEFAULT_FOOTPRINT", "DEFAULT_MM_PER_PX", "DEFAULT_PRESET", "DRC_MODES", "DRC_STRICT_ATTEMPTS",
    "FINE_PITCH_FAMILIES", "LEGACY", "LEGACY_PACKING", "NONE", "POUR_PROBABILITY", "PRESETS", "ROUTERS", "ROUTER_NAMES",
    "STREAM_NET", "STREAM_PLACE", "STREAM_ROUTE", "STREAM_RULES", "STREAM_ZONE",
    "BBox", "Board", "BoardDrcError", "BoardOptions", "DesignRules", "DrcReport", "Packing", "PadSpec", "Vec", "WindowBoard",
    "assign_nets", "board_options", "build_window_board", "check", "collect_drc", "drc_summary", "family_of", "fanout_band_px",
    "fit_to_pitch", "footprint_pads", "is_fine_pitch", "min_row_pitch", "neighbour_layout", "place_window", "profile_footprint",
    "sample_rules", "sample_zones", "seed_occupancy", "stitch_vias", "validate_board_options", "window_board",
]
