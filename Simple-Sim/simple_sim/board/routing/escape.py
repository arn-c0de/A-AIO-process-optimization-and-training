"""Fan-out ("escape") router for window boards.

Every connected pad gets a straight stub out of its pad edge (necked down at fine-pitch parts),
then either a 45-degree route to a port on the window border or a via (the net continues on
another layer). Thermal pads get a via grid instead of a track. Pads of a pour's net that lie
inside the pour get a thermal-relief spoke into it (chip-sized parts; fine-pitch pins fan out
like any other pin, the pour merges with their track). A few through-nets cross the window
between two ports, optionally as a parallel bus.

Every candidate is checked exactly (:mod:`simple_sim.board.geometry`) against the copper map
with the rule clearance and dropped on conflict, so no two nets come closer than the clearance.
The one exception is a part's own pin row: a stub only keeps what the pitch leaves to the pads
and stubs of its own part (the DRC applies the same rule: no overlap inside a part's group).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simple_sim.board.footprints import FINE_PITCH_FAMILIES, is_fine_pitch  # noqa: F401 (FINE_PITCH_FAMILIES re-exported)
from simple_sim.board.geometry import BBox, Vec, add, clip, disc, perp, polyline45, track, unit
from simple_sim.board.model import GND, TOP, Board, Pad, Part, Port, Track, Via, Zone, point_in_polygon
from simple_sim.board.netlist import connected_pads
from simple_sim.board.occupancy import CopperMap
from simple_sim.board.routing.base import (NetWidths, RouteResult, axial_room, border_point, random_border_point,
                                           runs_along_border)


def fine_pitch(part: Part) -> bool:
    """Tracks neck down at the pad, ground pins fan out (:func:`~simple_sim.board.footprints.is_fine_pitch`)."""
    return is_fine_pitch(part.family)


def _inside(bbox: BBox, p: Vec, margin: float = 0.0) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 + margin <= p[0] <= x1 - margin and y0 + margin <= p[1] <= y1 - margin


def _clip(bbox: BBox, inside: Vec, outside: Vec) -> Vec:
    """Point where the segment *inside* -> *outside* crosses the window border."""
    x0, y0, x1, y1 = bbox
    t = 1.0
    for a, b, lo, hi in ((inside[0], outside[0], x0, x1), (inside[1], outside[1], y0, y1)):
        if b < lo:
            t = min(t, (a - lo) / (a - b))
        elif b > hi:
            t = min(t, (hi - a) / (b - a))
    return (inside[0] + (outside[0] - inside[0]) * t, inside[1] + (outside[1] - inside[1]) * t)


def _edge_dir(bbox: BBox, p: Vec) -> Vec:
    """Unit direction along the window edge that *p* lies on."""
    x0, _, x1, _ = bbox
    return (0.0, 1.0) if p[0] in (x0, x1) else (1.0, 0.0)


def _on_same_edge(bbox: BBox, on_edge: Vec, p: Vec) -> Vec:
    """*p* moved back onto the window edge that *on_edge* lies on (clipped to the window)."""
    x0, y0, x1, y1 = bbox
    if on_edge[0] in (x0, x1):
        return (on_edge[0], clip(p[1], y0, y1))
    return (clip(p[0], x0, x1), on_edge[1])


def _row_offset(part: Part, pad: Pad) -> float:
    """Offset of *pad* from the part centre along its pin row, in the axis ``border_point`` shifts
    (+y for a horizontal escape, +x for a vertical one)."""
    axis = 1 if abs(pad.escape_dir[0]) >= abs(pad.escape_dir[1]) else 0
    return pad.center[axis] - part.center[axis]


def _row_pitch(part: Part, pad: Pad) -> float:
    """Centre distance from *pad* to the nearest pad of its row (inf for a lone pad)."""
    axis = 1 if abs(pad.escape_dir[0]) >= abs(pad.escape_dir[1]) else 0
    others = [abs(p.center[axis] - pad.center[axis]) for p in part.pads if p.escape_dir == pad.escape_dir and p != pad]
    return min(others, default=float("inf"))


def _row_position(part: Part, pad: Pad) -> int:
    """Index of *pad* along its pin row (the pads with the same escape direction)."""
    row = [p for p in part.pads if p.escape_dir == pad.escape_dir]
    axis = 1 if abs(pad.escape_dir[0]) >= abs(pad.escape_dir[1]) else 0
    row.sort(key=lambda p: p.center[axis])
    return row.index(pad)


def _keepout_reach(part: Part, pad: Pad, d: Vec) -> float:
    """Distance from the pad centre to the part's keep-out edge along the escape axis *d*."""
    kx0, ky0, kx1, ky1 = part.keepout
    cx, cy = pad.center
    if abs(d[0]) >= abs(d[1]):
        return (kx1 - cx) if d[0] > 0 else (cx - kx0)
    return (ky1 - cy) if d[1] > 0 else (cy - ky0)


@dataclass(slots=True)
class EscapeRouter:
    via_probability: float = 0.3       # end a fan-out at a via instead of the window border
    through_nets: tuple[int, int] = (1, 3)
    bus_probability: float = 0.3
    attempts: int = 4
    # Per-part draws (fine-pitch rows use one stub length and one fan-out spread per part, so the
    # continuations of neighbouring pins start level and never cross).
    _part_draws: dict[str, tuple[float, float]] = field(default_factory=dict)

    def _draws(self, part: Part, rng: np.random.Generator) -> tuple[float, float]:
        if part.ref not in self._part_draws:
            self._part_draws[part.ref] = (float(rng.uniform(0.8, 1.2)), float(rng.uniform(0.2, 0.8)))
        return self._part_draws[part.ref]

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
        k = self._draws(part, rng)[0] if fine_pitch(part) else float(rng.uniform(0.8, 1.2))
        stub_len = max(rules.fanout_stub * k, 1.5 * rules.clearance + widths(pad.net or ""))
        end = add(pad.center, pad.escape_dir, along / 2.0 + stub_len)
        return [pad.center, end], width

    @staticmethod
    def _stub_clear(board: Board, pad: Pad, width: float) -> float:
        """Clearance of a stub to its own pin row: what the pitch leaves (the stub stays inside its
        pad's column), at most the rule clearance."""
        _, across = EscapeRouter._along_across(pad)
        return max(0.0, min(board.rules.clearance, 0.5 * (across - width) / 2.0))

    def _via_fits(self, board: Board, center: Vec) -> bool:
        return _inside(board.bbox, center, board.rules.via_diameter / 2.0)

    # ---- pieces --------------------------------------------------------------------------
    def _thermal_vias(self, board: Board, occ: CopperMap, part: Part, pad: Pad, rng: np.random.Generator) -> list[Via]:
        """Via grid inside the thermal pad (every via fully on the pad, clear of everything else)."""
        rules = board.rules
        d = rules.via_diameter
        pitch = d + rules.clearance
        n = int(np.clip(int((min(pad.size) - d) / pitch) + 1, 1, 4))
        x0, y0, x1, y1 = pad.bbox
        g = occ.group(part.ref)
        vias = []
        for i in range(n):
            for j in range(n):
                c = (pad.center[0] + (i - (n - 1) / 2.0) * pitch, pad.center[1] + (j - (n - 1) / 2.0) * pitch)
                tented = bool(rng.random() < 0.6)
                if not (x0 + d / 2 <= c[0] <= x1 - d / 2 and y0 + d / 2 <= c[1] <= y1 - d / 2):
                    continue
                shape = disc(c, d)
                if occ.conflicts(shape, pad.net, rules.clearance, group=g):
                    continue
                occ.add(shape, pad.net, group=g)
                vias.append(Via(net=pad.net or GND, center=c, drill=rules.via_drill, diameter=d, tented=tented,
                                part=part.ref))
        return vias

    def _relief(self, board: Board, occ: CopperMap, part: Part, pad: Pad, zone: Zone, widths: NetWidths,
                rng: np.random.Generator) -> RouteResult | None:
        """Thermal-relief spoke from the pad out of the part's keep-out into the pour, or ``None``
        when it does not fit (then the pad fans out like any other).

        The spoke reaches past the pour's cut-back around the part and keeps ``zone.clearance``
        plus half a rule clearance from foreign copper, so the pour around its tip survives; the
        tip is reserved in the copper map so later tracks keep that distance too.
        """
        rules = board.rules
        d = unit(pad.escape_dir)
        width = zone.thermal.spoke_width if zone.thermal is not None else rules.width("power")
        stub_len = max(rules.fanout_stub * 1.2, 1.5 * rules.clearance + rules.width("power") * 1.15)
        edge = _keepout_reach(part, pad, d)
        tip_from = edge + max(zone.clearance, stub_len)
        reach = tip_from + rules.clearance + 2.0 * width
        end = add(pad.center, d, reach)
        beyond = add(pad.center, d, reach + width)
        if not _inside(board.bbox, beyond) or not point_in_polygon(beyond, zone.outline):
            return None
        g = occ.group(part.ref)
        spoke = track([pad.center, end], width)
        if occ.conflicts(spoke, pad.net, rules.clearance, group=g, group_clearance=self._stub_clear(board, pad, width)):
            return None
        # Foreign copper keeps one clearance from the reservation: zone.clearance + clearance / 2 from the spoke.
        tip = track([add(pad.center, d, tip_from), end], width + 2.0 * (zone.clearance - 0.5 * rules.clearance))
        if occ.conflicts(tip, pad.net, rules.clearance, group=g, group_clearance=rules.clearance):
            return None
        occ.add(spoke, pad.net, group=g)
        occ.add(tip, pad.net)   # reservation (no copper of its own): keeps the pour at the tip intact
        res = RouteResult()
        res.tracks.append(Track(pad.net or GND, TOP, [pad.center, end], width, part=part.ref))
        return res

    def _fanout(self, board: Board, occ: CopperMap, part: Part, pad: Pad, widths: NetWidths,
                rng: np.random.Generator) -> RouteResult:
        """Stub + (45-degree route to a port | via); empty result with ``unrouted`` when nothing fits."""
        res = RouteResult()
        rules = board.rules
        net = pad.net or ""
        if not _inside(board.bbox, pad.center):
            return res  # pad of a part cut by the window border: nothing to see, nothing to route
        g = occ.group(part.ref)
        stub, width = self._stub(board, part, pad, widths, rng)
        stub_clear = self._stub_clear(board, pad, width)
        if not _inside(board.bbox, stub[-1]):
            # The stub leaves the window: it ends at a port on the border (the copper map cannot
            # see copper outside the window, so nothing may be routed there).
            stub = [pad.center, _clip(board.bbox, pad.center, stub[-1])]
            shape = track(stub, width)
            if occ.conflicts(shape, net, rules.clearance, group=g, group_clearance=stub_clear):
                res.unrouted.append(f"{part.ref}.{pad.pin}")
            else:
                occ.add(shape, net, group=g)
                res.tracks.append(Track(net, TOP, stub, width, part=part.ref))
                res.ports.append(Port(net, stub[-1]))
            return res
        stub_shape = track(stub, width)
        if occ.conflicts(stub_shape, net, rules.clearance, group=g, group_clearance=stub_clear):
            res.unrouted.append(f"{part.ref}.{pad.pin}")
            return res
        stub_token = occ.add(stub_shape, net, group=g)
        start = stub[-1]
        d = unit(pad.escape_dir)
        # Full-width continuation beyond the pin row (fine-pitch stubs are necked down).
        full = widths(net)
        pitch = _row_pitch(part, pad)
        if fine_pitch(part):
            # Necked-down rows: the continuations of neighbouring pins run side by side.
            full = max(width, min(full, 1.5 * width, pitch - 1.02 * rules.clearance))
        stub_track = Track(net, TOP, stub, width, part=part.ref)

        def to_border() -> bool:
            x0, y0, x1, y1 = board.bbox
            span = (y1 - y0) if abs(d[0]) >= abs(d[1]) else (x1 - x0)
            # The 45-degree jog must end before the border, else the track runs along the edge.
            room = max(0.0, axial_room(board.bbox, d, start) - (full + rules.clearance))
            if fine_pitch(part):
                # Fan out: every pin jogs away from the row centre in proportion to its offset, so
                # the 45-degree jogs of a row run parallel instead of crossing.
                offset = _row_offset(part, pad)
                fan = self._draws(part, rng)[1]
                outward = [clip(np.sign(offset) * abs(rng.uniform(0.0, span / 6.0)), -room, room)
                           for _ in range(self.attempts)]
                if 0.70710678 * pitch - full >= rules.clearance:
                    # Parallel 45-degree jogs of neighbouring pins keep the clearance: fan out, and
                    # only ever outward (a jog towards the row centre crosses the next pins).
                    lateral = [clip(offset * f, -room, room) for f in (fan, 0.0, fan / 2.0)] + outward
                else:
                    # Too tight for jogs next to each other: straight out; jog only where the
                    # copper map finds room (an NC neighbour pin, the row end).
                    lateral = [0.0] + outward + [-o for o in outward]
            else:
                lateral = [clip(rng.uniform(-span / 6.0, span / 6.0), -room, room) for _ in range(self.attempts)]
                if np.isfinite(pitch):
                    # Several pads on this side (SOT-23 pins 1/2): jog away from the other pads only.
                    sign = np.sign(_row_offset(part, pad)) or 1.0
                    lateral = [sign * abs(o) for o in lateral]
            for target in (border_point(board.bbox, d, start, o) for o in lateral):
                pts = polyline45(start, target)
                if runs_along_border(board.bbox, pts, full / 2.0 + rules.clearance) and len(pts) > 2:
                    continue
                shape = track(pts, full)
                if not occ.conflicts(shape, net, rules.clearance):
                    occ.add(shape, net)
                    res.tracks += [stub_track, Track(net, TOP, pts, full)]
                    res.ports.append(Port(net, target))
                    return True
            return False

        def to_via() -> bool:
            # Fine-pitch rows stagger their vias in two ranks (the pitch is below a via + clearance).
            rank = _row_position(part, pad) % 2 if fine_pitch(part) else 0
            lead = add(start, d, rules.via_diameter / 2.0 + rules.clearance + rank * (rules.via_diameter + rules.clearance))
            if not self._via_fits(board, lead):
                return False
            # The lead is beyond the pin row: full clearance like any other track.
            lead_shape, via_shape = track([start, lead], width), disc(lead, rules.via_diameter)
            if occ.conflicts(lead_shape, net, rules.clearance) or occ.conflicts(via_shape, net, rules.clearance):
                return False
            occ.add(lead_shape, net)
            occ.add(via_shape, net)
            res.tracks += [stub_track, Track(net, TOP, [start, lead], width)]
            res.vias.append(Via(net=net, center=lead, drill=rules.via_drill, diameter=rules.via_diameter,
                                tented=bool(rng.random() < 0.85)))
            return True

        via_first = rng.random() < self.via_probability
        if fine_pitch(part) and 0.70710678 * pitch < rules.via_diameter / 2.0 + rules.clearance + width:
            via_first = False   # a via is wider than the pitch: only where the straight route is blocked
        first, second = (to_via, to_border) if via_first else (to_border, to_via)
        if first() or second():
            return res
        # Nothing fits: take the stub back (exactly that item) so no dangling copper is left.
        occ.remove(stub_token)
        res.unrouted.append(f"{part.ref}.{pad.pin}")
        return res

    def _route45(self, board: Board, p0: Vec, p1: Vec, width: float) -> list[Vec] | None:
        """45-degree polyline between two border points that does not run along a window edge."""
        margin = width / 2.0 + board.rules.clearance
        for a, b in ((p0, p1), (p1, p0)):
            pts = polyline45(a, b)
            if not runs_along_border(board.bbox, pts, margin):
                return pts if a == p0 else pts[::-1]
        return None

    def _through_nets(self, board: Board, occ: CopperMap, widths: NetWidths, rng: np.random.Generator) -> RouteResult:
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
                pts = self._route45(board, p0, p1, width)
                if pts is None:
                    continue
                shape = track(pts, width)
                if not occ.conflicts(shape, net, rules.clearance):
                    occ.add(shape, net)
                    res.tracks.append(Track(net, TOP, pts, width))
                    res.ports += [Port(net, p0), Port(net, p1)]
                    if rng.random() < self.bus_probability:
                        res.extend(self._bus(board, occ, pts, width, rng))
                    break
        return res

    def _bus(self, board: Board, occ: CopperMap, pts: list[Vec], width: float, rng: np.random.Generator) -> RouteResult:
        """1-3 parallel signal tracks next to *pts*, port to port."""
        res = RouteResult()
        rules = board.rules
        pitch = width + rules.clearance * float(rng.uniform(1.2, 2.0))
        side = 1.0 if rng.random() < 0.5 else -1.0
        normal = perp(unit((pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])))
        ends0 = (pts[0], pts[-1])
        for k in range(1, int(rng.integers(2, 4))):
            net = f"B{int(rng.integers(1 << 30))}"
            # Shift each end point along its own border edge by the full pitch, towards the side of
            # *normal* (an L-shaped route has its ends on edges of different axes).
            ends = []
            for p in ends0:
                e = _edge_dir(board.bbox, p)
                s = 1.0 if (e[0] * normal[0] + e[1] * normal[1]) >= 0 else -1.0
                ends.append(_on_same_edge(board.bbox, p, add(p, e, side * s * k * pitch)))
            shifted = self._route45(board, ends[0], ends[1], width)
            if shifted is None:
                break
            shape = track(shifted, width)
            if occ.conflicts(shape, net, rules.clearance):
                break
            occ.add(shape, net)
            res.tracks.append(Track(net, TOP, shifted, width))
            res.ports += [Port(net, shifted[0]), Port(net, shifted[-1])]
        return res

    # ---- entry ---------------------------------------------------------------------------
    @staticmethod
    def _pour_of(board: Board, pad: Pad) -> Zone | None:
        return next((z for z in board.zones if z.net == pad.net and point_in_polygon(pad.center, z.outline)), None)

    def route(self, board: Board, occ: CopperMap, rng: np.random.Generator) -> RouteResult:
        widths = NetWidths(board, rng)
        res = RouteResult()
        # Target first (its fan-out matters most), then neighbours, then through-nets in the rest.
        for part in sorted(board.parts, key=lambda p: not p.is_target):
            for pad in part.pads:
                if pad.role == "thermal" and pad.net:
                    res.vias += self._thermal_vias(board, occ, part, pad, rng)
            pads = [pad for _, pad in connected_pads(part)]
            if fine_pitch(part):
                # Outermost pins of each row first: they jog furthest out, the inner ones follow.
                pads.sort(key=lambda p: -abs(_row_offset(part, p)))
            for pad in pads:
                if pad.escape_dir == (0.0, 0.0):
                    continue
                zone = self._pour_of(board, pad)
                if zone is not None and not fine_pitch(part) and _inside(board.bbox, pad.center):
                    relief = self._relief(board, occ, part, pad, zone, widths, rng)
                    if relief is not None:
                        res.extend(relief)
                        continue
                res.extend(self._fanout(board, occ, part, pad, widths, rng))
        res.extend(self._through_nets(board, occ, widths, rng))
        return res
