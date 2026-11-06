"""Design-rule checks on the board model (on the occupancy raster, independent of the render).

Used by the tests, optionally per sample (``render.board.drc: warn | strict``) and by ``validate``
through ``render_meta.board_drc`` once datasets carry it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simple_sim.board.model import Board, Track, Vec
from simple_sim.board.occupancy import Occupancy
from simple_sim.board.routing.base import CELLS_PER_CLEARANCE
from simple_sim.board.zones import zone_fill

CLEARANCE = "clearance"          # copper of two nets closer than the rule clearance
DANGLING = "dangling"            # a track end on nothing of its net
FORBIDDEN_PAD = "forbidden_pad"  # copper on a thermal / NC pad that is not its own via
KEEPOUT = "keepout"              # foreign track or via inside the inspected part's keep-out
WIDTH = "width"                  # track width outside its net class


@dataclass(frozen=True, slots=True)
class Violation:
    kind: str
    detail: str


@dataclass(slots=True)
class DrcReport:
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def counts(self) -> dict[str, int]:
        return dict(Counter(v.kind for v in self.violations))

    def to_dict(self) -> dict[str, Any]:
        return {"violations": len(self.violations), "kinds": self.counts()}

    def add(self, kind: str, detail: str) -> None:
        self.violations.append(Violation(kind, detail))


def _dist_point_segment(p: Vec, a: Vec, b: Vec) -> float:
    ab = (b[0] - a[0], b[1] - a[1])
    ap = (p[0] - a[0], p[1] - a[1])
    l2 = ab[0] ** 2 + ab[1] ** 2
    t = 0.0 if l2 == 0 else float(np.clip((ap[0] * ab[0] + ap[1] * ab[1]) / l2, 0.0, 1.0))
    q = (a[0] + t * ab[0], a[1] + t * ab[1])
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _end_connected(board: Board, track: Track, p: Vec) -> bool:
    tol = track.width / 2.0 + board.rules.clearance / 2.0
    x0, y0, x1, y1 = board.bbox
    if min(abs(p[0] - x0), abs(p[0] - x1), abs(p[1] - y0), abs(p[1] - y1)) <= tol:
        return True  # port on the window border
    for part in board.parts:
        for pad in part.pads:
            if pad.net == track.net:
                bx0, by0, bx1, by1 = pad.bbox
                if bx0 - tol <= p[0] <= bx1 + tol and by0 - tol <= p[1] <= by1 + tol:
                    return True
    for via in board.vias:
        if via.net == track.net and np.hypot(p[0] - via.center[0], p[1] - via.center[1]) <= via.diameter / 2.0 + tol:
            return True
    for other in board.tracks:
        if other is track or other.net != track.net or other.layer != track.layer:
            continue
        for a, b in zip(other.points, other.points[1:]):
            if _dist_point_segment(p, a, b) <= tol + other.width / 2.0:
                return True
    return False


def check(board: Board) -> DrcReport:
    """Run every check on *board*."""
    report = DrcReport()
    rules = board.rules
    occ = Occupancy.for_window(board.bbox, rules.clearance / CELLS_PER_CLEARANCE)

    def claim(mask: np.ndarray, net: str, what: str, clearance: float, exempt: np.ndarray | None = None) -> None:
        checked = mask if exempt is None else np.where(exempt > 0, 0, mask).astype(np.uint8)
        if occ.conflicts(checked, net, clearance, ignore_blocked=True):
            report.add(CLEARANCE, what)
        occ.stamp(mask, net)

    # Copper vs. copper with clearance: vias, tracks, then pours (already cut by their clearance).
    # Pads are given (footprint geometry; the pitch may be below the clearance): claimed without a
    # check, and a track's fan-out inside its own pad's surroundings is exempt for the same reason.
    forbidden = occ.scratch()
    own_pads: dict[str, np.ndarray] = {}
    for part in board.parts:
        for pad in part.pads:
            if pad.role in ("thermal", "nc") and pad.net is None:
                occ.rect(pad.bbox, mask=forbidden)
            if pad.net:
                occ.stamp(occ.rect(pad.bbox), pad.net)
                occ.rect(pad.bbox, inflate=rules.clearance, mask=own_pads.setdefault(pad.net, occ.scratch()))
    for via in board.vias:
        claim(occ.circle(via.center, via.diameter), via.net, f"via {via.net} @ {via.center}", rules.clearance,
              own_pads.get(via.net))
    for track in board.tracks:
        claim(occ.polyline(track.points, track.width), track.net, f"track {track.net}", rules.clearance,
              own_pads.get(track.net))
    for zone in board.zones:
        claim(zone_fill(board, zone, occ), zone.net, f"zone {zone.net}", 0.0)

    # Track ends, forbidden pads, keep-out, widths.
    target = board.target
    target_nets = {p.net for p in target.pads if p.net} if target else set()
    keep = occ.rect(target.keepout) if target else occ.scratch()
    for track in board.tracks:
        for end in (track.points[0], track.points[-1]):
            if not _end_connected(board, track, end):
                report.add(DANGLING, f"track {track.net} end {end}")
        mask = occ.polyline(track.points, track.width)
        if np.any((mask > 0) & (forbidden > 0)):
            report.add(FORBIDDEN_PAD, f"track {track.net}")
        if track.net not in target_nets and np.any((mask > 0) & (keep > 0)):
            report.add(KEEPOUT, f"track {track.net}")
        lo = min(rules.neck_down, rules.width("signal")) * 0.5
        hi = rules.width("power") * 1.2
        if not lo <= track.width <= hi:
            report.add(WIDTH, f"track {track.net} width {track.width:.3f}")
    for via in board.vias:
        if via.net not in target_nets and np.any((occ.circle(via.center, via.diameter) > 0) & (keep > 0)):
            report.add(KEEPOUT, f"via {via.net}")
    for part in board.parts:
        for pad in part.pads:
            if pad.role == "thermal" and pad.net:
                if any(_track_touches(track, pad.bbox) for track in board.tracks):
                    report.add(FORBIDDEN_PAD, f"track on thermal pad {part.ref}.{pad.pin}")
    return report


def _track_touches(track: Track, bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    for a, b in zip(track.points, track.points[1:]):
        for t in np.linspace(0.0, 1.0, 16):
            px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            if x0 <= px <= x1 and y0 <= py <= y1:
                return True
    return False
