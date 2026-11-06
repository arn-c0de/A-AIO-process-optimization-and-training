"""Board model -> pixel masks of a render window (supersampled resolution).

The renderer (``photoreal._board``) blends the copper mask under the solder mask exactly as it did
for the legacy generator and draws the vias with its own shading; this module only decides *where*
copper is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simple_sim.board.model import Board
from simple_sim.board.occupancy import Grid
from simple_sim.board.zones import zone_fill


@dataclass(frozen=True, slots=True)
class ViaPx:
    x: int
    y: int
    radius: int
    tented: bool


@dataclass(slots=True)
class CopperLayers:
    copper: np.ndarray          # uint8 (H, W): tracks and pours of the top layer, anti-aliased
    vias: list[ViaPx]


def pixel_grid(shape: tuple[int, int], px_to_mm: float) -> Grid:
    """Grid over an image of *shape* with *px_to_mm* mm per (supersampled) pixel, anti-aliased."""
    return Grid(origin=(0.0, 0.0), res=float(px_to_mm), shape=shape, aa=True)


def rasterize(board: Board, shape: tuple[int, int], px_to_mm: float, layer: str = "top") -> CopperLayers:
    grid = pixel_grid(shape, px_to_mm)
    copper = grid.scratch()
    for zone in board.zones:
        if zone.layer == layer:
            copper = np.maximum(copper, zone_fill(board, zone, grid))
    for track in board.tracks:
        if track.layer == layer:
            grid.polyline(track.points, track.width, mask=copper)
    vias = []
    for via in board.vias:
        x, y = grid.to_cell(via.center)
        vias.append(ViaPx(x, y, max(1, grid.px(via.diameter / 2.0)), via.tented))
    return CopperLayers(copper=copper, vias=vias)
