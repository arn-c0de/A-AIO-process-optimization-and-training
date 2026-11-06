"""Fan-out ("escape") router for window boards.

Every connected pad gets a straight stub out of its pad edge (necked down at fine-pitch parts),
then either a 45-degree route to a port on the window border or a via (the net continues on
another layer). Thermal pads get a via grid instead of a track. Pads on the ground net that lie
inside a ground pour are left to the pour (thermal relief, see :mod:`simple_sim.board.zones`).
A few through-nets cross the window between two ports, optionally as a parallel bus. Every
candidate is checked against the occupancy grid with the rule clearance and dropped on conflict,
so no two nets ever touch on the layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simple_sim.board.footprints import fine_pitch
from simple_sim.board.model import GND, TOP, Board, Pad, Part, Port, Track, Vec, Via, point_in_polygon
from simple_sim.board.netlist import connected_pads
from simple_sim.board.occupancy import Occupancy
from simple_sim.board.routing.base import (NetWidths, RouteResult, add, border_point, perp, polyline45, random_border_point,
                                           unit)


def _inside(bbox: tuple[float, float, float, float], p: Vec) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def _clip(bbox: tuple[float, float, float, float], inside: Vec, outside: Vec) -> Vec:
    """Point where the segment *inside* -> *outside* crosses the window border."""
    x0, y0, x1, y1 = bbox
    t = 1.0
    for a, b, lo, hi in ((inside[0], outside[0], x0, x1), (inside[1], outside[1], y0, y1)):
        if b < lo:
            t = min(t, (a - lo) / (a - b))
        elif b > hi:
            t = min(t, (hi - a) / (b - a))
    return (inside[0] + (outside[0] - inside[0]) * t, inside[1] + (outside[1] - inside[1]) * t)


def _on_same_edge(bbox: tuple[float, float, float, float], on_edge: Vec, p: Vec) -> Vec:
    """*p* moved back onto the window edge that *on_edge* lies on (clipped to the window)."""
    x0, y0, x1, y1 = bbox
    if on_edge[0] in (x0, x1):
        return (on_edge[0], float(np.clip(p[1], y0, y1)))
    return (float(np.clip(p[0], x0, x1)), on_edge[1])


@dataclass(slots=True)
class EscapeRouter:
    via_probability: float = 0.3       # end a fan-out at a via instead of the window border
    through_nets: tuple[int, int] = (1, 3)
    bus_probability: float = 0.3
    attempts: int = 4

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _along_across(pad: Pad) -> tuple[float, float]:
        """Pad extent along its escape direction and across it (mm)."""
        d = pad.escape_dir
        if abs(d[0]) >= abs(d[1]):
            return pad.size[0], pad.size[1]
        return pad.size[1], pad.size[0]

    def _stub(self, board: Board, part: Part, pad: Pad, widths: NetWidths, rng: np.random.Generator) -> tuple[list[Vec], float]:
        rules = board.rules
        along, across = self._along_across(pad)
        width = widths(pad.net or "")
        if fine_pitch(part):
            width = min(width, rules.neck_down)
        width = min(width, 0.8 * across)
        # Long enough that the full-width continuation (plus clearance) starts clear of the pin row.
        stub_len = max(rules.fanout_stub * float(rng.uniform(0.8, 1.2)), 1.5 * rules.clearance + widths(pad.net or ""))
        end = add(pad.center, pad.escape_dir, along / 2.0 + stub_len)
        return [pad.center, end], width

    def _try_track(self, occ: Occupancy, points: list[Vec], width: float, net: str, clearance: float,
                   *, ignore_blocked: bool = False) -> bool:
        mask = occ.polyline(points, width)
        if occ.conflicts(mask, net, clearance, ignore_blocked=ignore_blocked):
            return False
        occ.stamp(mask, net)
        return True

    def _try_via(self, board: Board, occ: Occupancy, center: Vec, net: str, rng: np.random.Generator) -> Via | None:
        rules = board.rules
        x0, y0, x1, y1 = board.bbox
        r = rules.via_diameter / 2.0
        if not (x0 + r <= center[0] <= x1 - r and y0 + r <= center[1] <= y1 - r):
            return None
        mask = occ.circle(center, rules.via_diameter)
        if occ.conflicts(mask, net, rules.clearance):
            return None
        occ.stamp(mask, net)
        return Via(net=net, center=center, drill=rules.via_drill, diameter=rules.via_diameter, tented=bool(rng.random() < 0.85))

    def _in_pour(self, board: Board, pad: Pad) -> bool:
        return any(z.net == pad.net and point_in_polygon(pad.center, z.outline) for z in board.zones)

    # ---- pieces --------------------------------------------------------------------------
    def _thermal_vias(self, board: Board, occ: Occupancy, pad: Pad, rng: np.random.Generator) -> list[Via]:
        rules = board.rules
        pitch = rules.via_diameter + rules.clearance
        n = int(np.clip(int((min(pad.size) - rules.via_diameter) / pitch) + 1, 2, 4))
        vias = []
        for i in range(n):
            for j in range(n):
                c = (pad.center[0] + (i - (n - 1) / 2.0) * pitch, pad.center[1] + (j - (n - 1) / 2.0) * pitch)
                mask = occ.circle(c, rules.via_diameter)
                occ.stamp(mask, pad.net)   # inside the own pad: no clearance check needed
                vias.append(Via(net=pad.net or GND, center=c, drill=rules.via_drill, diameter=rules.via_diameter,
                                tented=bool(rng.random() < 0.6)))
        return vias

    def _fanout(self, board: Board, occ: Occupancy, part: Part, pad: Pad, widths: NetWidths,
                rng: np.random.Generator) -> RouteResult:
        """Stub + (45-degree route to a port | via); empty result with ``unrouted`` when nothing fits."""
        res = RouteResult()
        rules = board.rules
        net = pad.net or ""
        if not _inside(board.bbox, pad.center):
            return res  # pad of a part cut by the window border: nothing to see, nothing to route
        stub, width = self._stub(board, part, pad, widths, rng)
        _, across = self._along_across(pad)
        # Inside the pin row the clearance is whatever the pitch leaves: the stub stays within its
        # pad's column (one cell short, the inflation rounds up) and may touch the part's keep-out.
        stub_clear = max(0.0, min(rules.clearance, (across - width) / 2.0 - occ.res))
        if not _inside(board.bbox, stub[-1]):
            # The stub leaves the window: it ends at a port on the border (the occupancy grid cannot
            # see copper outside the window, so nothing may be routed there).
            stub = [pad.center, _clip(board.bbox, pad.center, stub[-1])]
            if self._try_track(occ, stub, width, net, stub_clear, ignore_blocked=True):
                res.tracks.append(Track(net, TOP, stub, width))
                res.ports.append(Port(net, stub[-1]))
            else:
                res.unrouted.append(f"{part.ref}.{pad.pin}")
            return res
        if not self._try_track(occ, stub, width, net, stub_clear, ignore_blocked=True):
            res.unrouted.append(f"{part.ref}.{pad.pin}")
            return res
        start = stub[-1]
        d = unit(pad.escape_dir)
        # Full-width continuation beyond the pin row (fine-pitch stubs are necked down).
        full = widths(net) if not fine_pitch(part) else min(widths(net), 1.5 * width)

        def to_border() -> bool:
            x0, y0, x1, y1 = board.bbox
            span = (y1 - y0) if abs(d[0]) >= abs(d[1]) else (x1 - x0)
            lateral = [0.0] if fine_pitch(part) else []
            lateral += [float(rng.uniform(-span / 6.0, span / 6.0)) for _ in range(self.attempts)]
            for target in (border_point(board.bbox, d, start, o) for o in lateral):
                pts = polyline45(start, target)
                if self._try_track(occ, pts, full, net, rules.clearance):
                    res.tracks += [Track(net, TOP, stub, width), Track(net, TOP, pts, full)]
                    res.ports.append(Port(net, target))
                    return True
            return False

        def to_via() -> bool:
            lead = add(start, d, rules.via_diameter / 2.0 + rules.clearance)
            via = self._try_via(board, occ, lead, net, rng)
            # The lead is beyond the pin row: full clearance like any other track.
            if via is None or not self._try_track(occ, [start, lead], width, net, rules.clearance):
                return False
            res.tracks.append(Track(net, TOP, [pad.center, start, lead], width))
            res.vias.append(via)
            return True

        first, second = (to_via, to_border) if rng.random() < self.via_probability else (to_border, to_via)
        if first() or second():
            return res
        # Nothing fits: take the stub back so no dangling copper is left.
        occ.clear(occ.polyline(stub, width))
        occ.stamp(occ.rect(pad.bbox), net)
        res.unrouted.append(f"{part.ref}.{pad.pin}")
        return res

    def _through_nets(self, board: Board, occ: Occupancy, widths: NetWidths, rng: np.random.Generator) -> RouteResult:
        res = RouteResult()
        rules = board.rules
        lo, hi = self.through_nets
        for i in range(int(rng.integers(lo, hi + 1))):
            net = f"T{i + 1}" if rng.random() < 0.8 else GND
            width = widths(net) if net != GND else rules.width("power")
            for _ in range(self.attempts):
                e0 = int(rng.integers(4))
                e1 = int(rng.integers(4))
                p0, p1 = random_border_point(board.bbox, e0, rng), random_border_point(board.bbox, e1, rng)
                if e0 == e1 or np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 0.3 * min(board.size):
                    continue
                pts = polyline45(p0, p1)
                if self._try_track(occ, pts, width, net, rules.clearance):
                    res.tracks.append(Track(net, TOP, pts, width))
                    res.ports += [Port(net, p0), Port(net, p1)]
                    if rng.random() < self.bus_probability:
                        res.extend(self._bus(board, occ, pts, width, rng))
                    break
        return res

    def _bus(self, board: Board, occ: Occupancy, pts: list[Vec], width: float, rng: np.random.Generator) -> RouteResult:
        """1-3 parallel signal tracks next to *pts* with equal pitch."""
        res = RouteResult()
        rules = board.rules
        pitch = width + rules.clearance * float(rng.uniform(1.2, 2.0))
        side = 1.0 if rng.random() < 0.5 else -1.0
        normal = perp(unit((pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])))
        for k in range(1, int(rng.integers(2, 4))):
            net = f"B{int(rng.integers(1 << 30))}"
            # Shift the end points along their border edges, so the copy also runs port to port.
            ends = [_on_same_edge(board.bbox, p, add(p, normal, side * k * pitch)) for p in (pts[0], pts[-1])]
            shifted = polyline45(*ends)
            if self._try_track(occ, shifted, width, net, rules.clearance):
                res.tracks.append(Track(net, TOP, shifted, width))
                res.ports += [Port(net, shifted[0]), Port(net, shifted[-1])]
            else:
                break
        return res

    # ---- entry ---------------------------------------------------------------------------
    def route(self, board: Board, occ: Occupancy, rng: np.random.Generator) -> RouteResult:
        widths = NetWidths(board, rng)
        res = RouteResult()
        # Target first (its fan-out matters most), then neighbours, then through-nets in the rest.
        for part in sorted(board.parts, key=lambda p: not p.is_target):
            for pad in part.pads:
                if pad.role == "thermal" and pad.net:
                    res.vias += self._thermal_vias(board, occ, pad, rng)
            for _, pad in connected_pads(part):
                if pad.net == GND and self._in_pour(board, pad):
                    continue  # the pour connects it (thermal relief)
                if pad.escape_dir == (0.0, 0.0):
                    continue
                res.extend(self._fanout(board, occ, part, pad, widths, rng))
        res.extend(self._through_nets(board, occ, widths, rng))
        return res
