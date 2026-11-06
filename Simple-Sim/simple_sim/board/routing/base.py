"""Router protocol, shared geometry helpers and the occupancy seeded from a placed board."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from simple_sim.board.model import Board, Port, Track, Vec, Via
from simple_sim.board.netlist import net_class
from simple_sim.board.occupancy import Occupancy

# Occupancy cells per clearance: 2 keeps clearance checks exact to half a clearance.
CELLS_PER_CLEARANCE = 2


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
    def route(self, board: Board, occupancy: Occupancy, rng: np.random.Generator) -> RouteResult: ...


def seed_occupancy(board: Board) -> Occupancy:
    """Occupancy with every part's keep-out blocked and every pad claimed for its net (NC pads blocked).

    The keep-out (body plus pads) blocks foreign copper between the pads of a part; a pad's own
    fan-out leaves the keep-out outward, so routers check stubs with ``ignore_blocked``.
    """
    occ = Occupancy.for_window(board.bbox, board.rules.clearance / CELLS_PER_CLEARANCE)
    for part in board.parts:
        occ.stamp(occ.rect(part.keepout), None)
        for pad in part.pads:
            occ.stamp(occ.rect(pad.bbox), pad.net if pad.role not in ("nc",) else None)
    return occ


def unit(v: Vec) -> Vec:
    n = float(np.hypot(v[0], v[1]))
    return (v[0] / n, v[1] / n) if n > 0 else (0.0, 0.0)


def perp(v: Vec) -> Vec:
    return (-v[1], v[0])


def add(p: Vec, d: Vec, k: float = 1.0) -> Vec:
    return (p[0] + d[0] * k, p[1] + d[1] * k)


def polyline45(p0: Vec, p1: Vec) -> list[Vec]:
    """*p0* to *p1* with one 45-degree jog first, then straight (like autorouted traces)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    jog = min(abs(dx), abs(dy))
    mid = (p0[0] + float(np.sign(dx)) * jog, p0[1] + float(np.sign(dy)) * jog)
    pts = [p0, mid, p1]
    return [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]


def border_point(bbox: tuple[float, float, float, float], direction: Vec, through: Vec, lateral: float) -> Vec:
    """Point on the window border reached from *through* along the dominant axis of *direction*,
    shifted by *lateral* along the border."""
    x0, y0, x1, y1 = bbox
    if abs(direction[0]) >= abs(direction[1]):
        x = x0 if direction[0] < 0 else x1
        return (x, float(np.clip(through[1] + lateral, y0, y1)))
    y = y0 if direction[1] < 0 else y1
    return (float(np.clip(through[0] + lateral, x0, x1)), y)


def random_border_point(bbox: tuple[float, float, float, float], edge: int, rng: np.random.Generator) -> Vec:
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
