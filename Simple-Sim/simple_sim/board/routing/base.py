"""Router protocol, shared geometry helpers and the occupancy seeded from a placed board."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from simple_sim.board.geometry import BBox, Vec, add, box, clip, perp, polyline45, unit  # noqa: F401 (re-exported)
from simple_sim.board.model import Board, Port, Track, Via
from simple_sim.board.netlist import net_class
from simple_sim.board.occupancy import CopperMap


@dataclass(slots=True)
class RouteResult:
    tracks: list[Track] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)
    unrouted: list[str] = field(default_factory=list)   # "<ref>.<pin>" of pads that could not be connected

    def extend(self, other: RouteResult) -> None:
        self.tracks += other.tracks
        self.vias += other.vias
        self.ports += other.ports
        self.unrouted += other.unrouted


class Router(Protocol):
    def route(self, board: Board, occupancy: CopperMap, rng: np.random.Generator) -> RouteResult: ...


def seed_occupancy(board: Board) -> CopperMap:
    """Copper map with every part's keep-out blocked and every pad as copper of its net (NC pads:
    copper of no net, so every net keeps clear of them).

    Keep-out and pads carry the part's group: a pad's own fan-out stub leaves the keep-out outward
    and only needs the pitch-limited clearance to the rest of its pin row (see the escape router).
    """
    occ = CopperMap(board.bbox)
    for part in board.parts:
        g = occ.group(part.ref)
        occ.add(box(part.keepout), None, group=g, blocked=True)
        for pad in part.pads:
            occ.add(box(pad.bbox), pad.net if pad.role != "nc" else None, group=g)
    return occ


def axial_room(bbox: BBox, direction: Vec, start: Vec) -> float:
    """Distance from *start* to the window border along the dominant axis of *direction*."""
    x0, y0, x1, y1 = bbox
    if abs(direction[0]) >= abs(direction[1]):
        return (start[0] - x0) if direction[0] < 0 else x1 - start[0]
    return (start[1] - y0) if direction[1] < 0 else y1 - start[1]


def runs_along_border(bbox: BBox, points: list[Vec], margin: float) -> bool:
    """True if a segment of *points* runs (within *margin*) along a window edge instead of into it."""
    x0, y0, x1, y1 = bbox
    for a, b in zip(points, points[1:]):
        for lo, hi, i in ((x0, x1, 0), (y0, y1, 1)):
            if a[i] == b[i] and (abs(a[i] - lo) <= margin or abs(a[i] - hi) <= margin):
                return True
    return False


def border_point(bbox: BBox, direction: Vec, through: Vec, lateral: float) -> Vec:
    """Point on the window border reached from *through* along the dominant axis of *direction*,
    shifted by *lateral* along the border."""
    x0, y0, x1, y1 = bbox
    if abs(direction[0]) >= abs(direction[1]):
        x = x0 if direction[0] < 0 else x1
        return (x, clip(through[1] + lateral, y0, y1))
    y = y0 if direction[1] < 0 else y1
    return (clip(through[0] + lateral, x0, x1), y)


def random_border_point(bbox: BBox, edge: int, rng: np.random.Generator) -> Vec:
    """Random point on edge 0..3 (left, right, top, bottom)."""
    x0, y0, x1, y1 = bbox
    if edge == 0:
        return (x0, float(rng.uniform(y0, y1)))
    if edge == 1:
        return (x1, float(rng.uniform(y0, y1)))
    if edge == 2:
        return (float(rng.uniform(x0, x1)), y0)
    return (float(rng.uniform(x0, x1)), y1)


class NetWidths:
    """Track width per net: the rule width of the net class with a little per-net variation."""

    def __init__(self, board: Board, rng: np.random.Generator) -> None:
        self._rules = board.rules
        self._rng = rng
        self._widths: dict[str, float] = {}

    def __call__(self, net: str) -> float:
        if net not in self._widths:
            self._widths[net] = self._rules.width(net_class(net)) * float(self._rng.uniform(0.85, 1.15))
        return self._widths[net]
