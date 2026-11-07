"""Design-rule checks on the board model (exact geometry, independent of the render).

Clearances are measured with :func:`simple_sim.board.geometry.gap`, the same function the router
uses, so a board the router built cleanly passes. Pours are checked on a raster of
``clearance / 4``. Used by the tests, optionally per sample (``render.board.drc: warn | strict``)
and by ``validate`` through ``render_meta.board_drc`` once datasets carry it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from simple_sim.board.geometry import TOUCH, Shape, bbox_near, box, disc, gap, track
from simple_sim.board.model import Board, Track, Vec
from simple_sim.board.occupancy import Grid
from simple_sim.board.zones import POUR_CELLS_PER_CLEARANCE, zone_fill  # noqa: F401 (re-exported)

CLEARANCE = "clearance"          # copper of two nets closer than the rule clearance
DANGLING = "dangling"            # a track end on nothing of its net
FORBIDDEN_PAD = "forbidden_pad"  # copper on an NC pad, or a track on a thermal pad
KEEPOUT = "keepout"              # foreign track or via inside the inspected part's keep-out
UNCONNECTED = "unconnected"      # a connected pin of the inspected part reached by no track, via or pour
WIDTH = "width"                  # track width outside its net class


class BoardDrcError(RuntimeError):
    """``render.board.drc: strict`` and no attempt produced a clean board."""

    def __init__(self, report: DrcReport, seed: int) -> None:
        super().__init__(f"board DRC failed after every attempt (seed {seed}): {report.to_dict()}")
        self.report = report
        self.seed = seed


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


@dataclass(frozen=True, slots=True)
class _Copper:
    shape: Shape
    net: str | None
    group: str | None              # part ref (pads, fan-out stubs, thermal vias of that part)
    kind: str                      # pad | track | via
    label: str
    role: str = ""
    layer: str = ""                # tracks only
    source: Any = None             # the model object (tracks: to tell a track from itself)


def _copper(board: Board) -> list[_Copper]:
    out = []
    for part in board.parts:
        for pad in part.pads:
            out.append(_Copper(box(pad.bbox), pad.net, part.ref, "pad", f"pad {part.ref}.{pad.pin}", pad.role))
    for t in board.tracks:
        out.append(_Copper(track(t.points, t.width), t.net, t.part, "track", f"track {t.net}", layer=t.layer, source=t))
    for v in board.vias:
        out.append(_Copper(disc(v.center, v.diameter), v.net, v.part, "via", f"via {v.net} @ {v.center}"))
    return out


def _check_clearance(board: Board, items: list[_Copper], report: DrcReport) -> None:
    """Copper of different nets keeps the rule clearance; inside one part's group (pads, stubs and
    thermal vias of the same part) it only must not touch. Pads among themselves are given by the
    footprint and not checked."""
    c = board.rules.clearance
    if not items:
        return
    # Pairs (i < j) whose bounding boxes come within the clearance (``bbox_near`` for every pair at once).
    boxes = np.array([it.shape.bbox for it in items], dtype=np.float64)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    near = ~((x1[:, None] + c < x0[None, :]) | (x1[None, :] + c < x0[:, None])
             | (y1[:, None] + c < y0[None, :]) | (y1[None, :] + c < y0[:, None]))
    for i, j in np.argwhere(np.triu(near, 1)):
        a, b = items[i], items[j]
        if a.kind == "pad" and b.kind == "pad":
            continue
        if a.net is not None and a.net == b.net:
            continue
        same_group = a.group is not None and a.group == b.group
        need = 0.0 if same_group else c
        g = gap(a.shape, b.shape)
        if g < need or g <= TOUCH:
            nc_pad = (a.kind == "pad" and a.role == "nc") or (b.kind == "pad" and b.role == "nc")
            kind = FORBIDDEN_PAD if nc_pad and g <= TOUCH else CLEARANCE
            report.add(kind, f"{a.label} / {b.label}: gap {g:.4f} < {need:.4f}")


def _fills(board: Board, grid: Grid) -> dict[int, np.ndarray]:
    return {i: zone_fill(board, z, grid) for i, z in enumerate(board.zones)}


def _near_fill(fill: np.ndarray, grid: Grid, p: Vec, radius: float) -> bool:
    cx, cy = grid.to_cell(p)
    n = grid.cells(radius) + 1
    y0, y1, x0, x1 = max(0, cy - n), min(fill.shape[0], cy + n + 1), max(0, cx - n), min(fill.shape[1], cx + n + 1)
    return y1 > y0 and x1 > x0 and bool(np.any(fill[y0:y1, x0:x1] > 0))


def _end_connected(board: Board, t: Track, p: Vec, fills: dict[int, np.ndarray], grid: Grid,
                   items: list[_Copper]) -> bool:
    """True if track end *p* touches copper of its net, a pour of its net or the window border.
    *items*: :func:`_copper` of the board (built once per :func:`check`)."""
    tol = t.width / 2.0 + board.rules.clearance / 2.0
    x0, y0, x1, y1 = board.bbox
    if min(abs(p[0] - x0), abs(p[0] - x1), abs(p[1] - y0), abs(p[1] - y1)) <= tol:
        return True  # port on the window border
    end = disc(p, 2.0 * tol)
    eb = end.bbox
    # Copper of the net that touches the end (a gap above TOUCH is ruled out by the bounding boxes already).
    for it in items:
        if it.net != t.net or (it.kind == "track" and (it.source is t or it.layer != t.layer)):
            continue
        if bbox_near(eb, it.shape.bbox, TOUCH) and gap(end, it.shape) <= TOUCH:
            return True
    return any(z.net == t.net and z.layer == t.layer and _near_fill(fills[i], grid, p, tol) for i, z in enumerate(board.zones))


def _check_pours(board: Board, fills: dict[int, np.ndarray], grid: Grid, report: DrcReport) -> None:
    """Every pour cell keeps the clearance (to one cell) from copper of other nets."""
    c = board.rules.clearance
    for i, zone in enumerate(board.zones):
        foreign = grid.scratch()
        for part in board.parts:
            for pad in part.pads:
                if pad.net != zone.net:
                    grid.rect(pad.bbox, mask=foreign)
        for t in board.tracks:
            if t.net != zone.net and t.layer == zone.layer:
                grid.polyline(t.points, t.width, mask=foreign)
        for via in board.vias:
            if via.net != zone.net:
                grid.circle(via.center, via.diameter, mask=foreign)
        if not np.any(foreign):
            continue
        dist = cv2.distanceTransform((foreign == 0).astype(np.uint8), cv2.DIST_L2, 5) * grid.res
        bad = (fills[i] > 0) & (dist + grid.res < c)
        if np.any(bad):
            report.add(CLEARANCE, f"zone {zone.net}: {int(bad.sum())} cells closer than {c:.4f}")


def check(board: Board) -> DrcReport:
    """Run every check on *board*."""
    report = DrcReport()
    rules = board.rules
    items = _copper(board)
    _check_clearance(board, items, report)
    grid = Grid.for_window(board.bbox, rules.clearance / POUR_CELLS_PER_CLEARANCE)
    fills = _fills(board, grid)
    _check_pours(board, fills, grid, report)

    target = board.target
    target_nets = {p.net for p in target.pads if p.net} if target else set()
    keep = box(target.keepout) if target else None
    lo = min(rules.neck_down, rules.width("signal")) * 0.5
    hi = rules.width("power") * 1.2
    for t in board.tracks:
        for end in (t.points[0], t.points[-1]):
            if not _end_connected(board, t, end, fills, grid, items):
                report.add(DANGLING, f"track {t.net} end {end}")
        if keep is not None and t.net not in target_nets and gap(track(t.points, t.width), keep) <= TOUCH:
            report.add(KEEPOUT, f"track {t.net}")
        if not lo <= t.width <= hi:
            report.add(WIDTH, f"track {t.net} width {t.width:.3f}")
    for v in board.vias:
        if keep is not None and v.net not in target_nets and gap(disc(v.center, v.diameter), keep) <= TOUCH:
            report.add(KEEPOUT, f"via {v.net}")
    # Thermal pads carry vias only: any track touching one is wrong (whatever its net).
    for part in board.parts:
        for pad in part.pads:
            if pad.role != "thermal":
                continue
            for t in board.tracks:
                if gap(track(t.points, t.width), box(pad.bbox)) <= TOUCH:
                    report.add(FORBIDDEN_PAD, f"track {t.net} on thermal pad {part.ref}.{pad.pin}")
    if target is not None:
        _check_target_connected(board, fills, grid, report)
    return report


def _check_target_connected(board: Board, fills: dict[int, np.ndarray], grid: Grid, report: DrcReport) -> None:
    """Every connected pin of the inspected part inside the window is reached by a track or via of
    its net, or by a pour of its net."""
    target = board.target
    assert target is not None
    x0, y0, x1, y1 = board.bbox
    for pad in target.pads:
        if not pad.net or pad.role == "nc":
            continue
        if not (x0 <= pad.center[0] <= x1 and y0 <= pad.center[1] <= y1):
            continue
        shape = box(pad.bbox)
        if any(t.net == pad.net and gap(track(t.points, t.width), shape) <= TOUCH for t in board.tracks):
            continue
        if any(v.net == pad.net and gap(disc(v.center, v.diameter), shape) <= TOUCH for v in board.vias):
            continue
        pad_mask = grid.inflated_rect(pad.bbox, grid.res)
        if any(z.net == pad.net and np.any((pad_mask > 0) & (fills[i] > 0)) for i, z in enumerate(board.zones)):
            continue
        report.add(UNCONNECTED, f"pad {target.ref}.{pad.pin} ({pad.net})")
