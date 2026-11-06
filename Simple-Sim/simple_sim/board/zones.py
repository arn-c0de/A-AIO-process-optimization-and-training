"""Copper pours: outlines in the model, filled at raster time with clearance and thermal relief.

A :class:`~simple_sim.board.model.Zone` stores only its outline, net and rule values. The copper
it actually produces depends on everything else on the layer (pads, tracks and vias of other
nets are cut out with ``zone.clearance``; pads of the pour's net get a thermal relief), so
:func:`zone_fill` computes it on any :class:`~simple_sim.board.occupancy.Grid`: the occupancy
grid for stitching vias and DRC, the pixel grid for rendering.
"""

from __future__ import annotations

import numpy as np

from simple_sim.board.model import GND, TOP, Board, ThermalRelief, Vec, Via, Zone, point_in_polygon
from simple_sim.board.occupancy import Grid, Occupancy
from simple_sim.board.routing.base import add, perp, unit

POUR_PROBABILITY = 0.6  # default share of boards with a copper pour (``render.board.pour_probability``)


def octagon(x: float, y: float, w: float, h: float, chamfer: float) -> list[Vec]:
    """Rectangle with 45-degree chamfered corners (the pour shape of autorouted boards)."""
    c = chamfer
    return [(x + c, y), (x + w - c, y), (x + w, y + c), (x + w, y + h - c), (x + w - c, y + h), (x + c, y + h),
            (x, y + h - c), (x, y + c)]


def sample_zones(board: Board, rng: np.random.Generator, probability: float = POUR_PROBABILITY) -> list[Zone]:
    """Add 1-2 ground pours covering 40-120 % of the window to *board* (with *probability*)."""
    if rng.random() >= probability:
        return []
    rules = board.rules
    w, h = board.size
    zones = []
    for _ in range(int(rng.integers(1, 3))):
        pw, ph = float(rng.uniform(0.4, 1.2)) * w, float(rng.uniform(0.4, 1.2)) * h
        ax, ay = float(rng.uniform(-0.2 * w, w - 0.5 * pw)), float(rng.uniform(-0.2 * h, h - 0.5 * ph))
        chamfer = float(rng.uniform(0.05, 0.25)) * min(pw, ph)
        zones.append(Zone(net=GND, layer=TOP, outline=octagon(ax, ay, pw, ph, chamfer),
                          clearance=max(rules.zone_clearance, rules.clearance),
                          thermal=ThermalRelief(gap=rules.thermal_gap, spoke_width=rules.thermal_spoke)))
    board.zones.extend(zones)
    return zones


def zone_fill(board: Board, zone: Zone, grid: Grid) -> np.ndarray:
    """Copper of *zone* on *grid* as a 0/255 mask."""
    fill = grid.polygon(zone.outline)
    foreign = grid.scratch()
    target = board.target
    if target is not None:  # the inspected part's area is kept free (as the real boards: pours end at the part)
        grid.rect(target.keepout, mask=foreign)
    for part in board.parts:
        for pad in part.pads:
            if pad.net != zone.net:
                grid.rect(pad.bbox, mask=foreign)
    for track in board.tracks:
        if track.net != zone.net and track.layer == zone.layer:
            grid.polyline(track.points, track.width, mask=foreign)
    for via in board.vias:
        if via.net != zone.net:
            grid.circle(via.center, via.diameter, mask=foreign)
    keep_clear = grid.inflate(foreign, zone.clearance)
    fill[keep_clear > 0] = 0

    if zone.thermal is not None:
        relief = zone.thermal
        spokes = grid.scratch()
        for part in board.parts:
            for pad in part.pads:
                if pad.net != zone.net or not point_in_polygon(pad.center, zone.outline):
                    continue
                fill[grid.inflate(grid.rect(pad.bbox), relief.gap) > 0] = 0
                d = unit(pad.escape_dir) if pad.escape_dir != (0.0, 0.0) else (1.0, 0.0)
                reach = max(pad.size) / 2.0 + relief.gap + 2.0 * relief.spoke_width
                for direction in (d, perp(d), (-d[0], -d[1]), (-perp(d)[0], -perp(d)[1]))[: relief.spokes]:
                    grid.polyline([pad.center, add(pad.center, direction, reach)], relief.spoke_width, mask=spokes)
        spokes[keep_clear > 0] = 0
        spokes[grid.polygon(zone.outline) == 0] = 0
        fill = np.maximum(fill, spokes)
    return fill


def stitch_vias(board: Board, zone: Zone, occ: Occupancy, rng: np.random.Generator, max_count: int = 24) -> list[Via]:
    """Stitching vias on a jittered grid of ``rules.stitch_pitch`` inside the pour (claimed in *occ*)."""
    rules = board.rules
    if rules.stitch_pitch is None:
        return []
    fill = occ.erode(zone_fill(board, zone, occ), rules.via_diameter / 2.0 + rules.clearance)
    x0, y0, x1, y1 = board.bbox
    pitch = rules.stitch_pitch
    phase = (float(rng.uniform(0, pitch)), float(rng.uniform(0, pitch)))
    vias: list[Via] = []
    for gy in np.arange(y0 + phase[1], y1, pitch):
        for gx in np.arange(x0 + phase[0], x1, pitch):
            c = (float(gx + rng.uniform(-0.15, 0.15) * pitch), float(gy + rng.uniform(-0.15, 0.15) * pitch))
            cx, cy = occ.to_cell(c)
            if not (0 <= cy < fill.shape[0] and 0 <= cx < fill.shape[1]) or fill[cy, cx] == 0:
                continue
            mask = occ.circle(c, rules.via_diameter)
            if occ.conflicts(mask, zone.net, rules.clearance):
                continue
            occ.stamp(mask, zone.net)
            vias.append(Via(net=zone.net, center=c, drill=rules.via_drill, diameter=rules.via_diameter,
                            tented=bool(rng.random() < 0.9)))
            if len(vias) >= max_count:
                return vias
    return vias
