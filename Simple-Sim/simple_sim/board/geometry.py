"""Exact copper geometry in mm: boxes (pads, keep-outs) and paths (tracks with round ends, vias).

The router and the DRC both measure clearances with :func:`gap` on these shapes, so they agree
to the last digit: a route the router accepts at clearance ``c`` passes the DRC at ``c``, with no
raster rounding in between (the raster :class:`~simple_sim.board.occupancy.Grid` is only used for
pours and for drawing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Vec = tuple[float, float]
BBox = tuple[float, float, float, float]

# Gaps at or below this count as touching (a shared pad edge, a track ending on a pad).
TOUCH = 1e-9


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned rectangle (pads, keep-outs)."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def bbox(self) -> BBox:
        return self.x0, self.y0, self.x1, self.y1


@dataclass(frozen=True, slots=True)
class Path:
    """Polyline swept by a disc of radius *r* (a track; a single point is a via or disc)."""
    points: tuple[Vec, ...]
    r: float

    @property
    def bbox(self) -> BBox:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs) - self.r, min(ys) - self.r, max(xs) + self.r, max(ys) + self.r

    def segments(self) -> list[tuple[Vec, Vec]]:
        if len(self.points) == 1:
            return [(self.points[0], self.points[0])]
        return list(zip(self.points, self.points[1:]))


Shape = Box | Path


def box(bbox: BBox) -> Box:
    return Box(*(float(v) for v in bbox))


def track(points: list[Vec] | tuple[Vec, ...], width: float) -> Path:
    return Path(tuple((float(p[0]), float(p[1])) for p in points), float(width) / 2.0)


def disc(center: Vec, diameter: float) -> Path:
    return Path(((float(center[0]), float(center[1])),), float(diameter) / 2.0)


# ---- primitive distances ---------------------------------------------------------------------

def _pt_seg(p: Vec, a: Vec, b: Vec) -> float:
    abx, aby = b[0] - a[0], b[1] - a[1]
    l2 = abx * abx + aby * aby
    t = 0.0 if l2 == 0.0 else max(0.0, min(1.0, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / l2))
    return math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby))


def _orient(a: Vec, b: Vec, c: Vec) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a: Vec, b: Vec, c: Vec, d: Vec) -> bool:
    d1, d2, d3, d4 = _orient(c, d, a), _orient(c, d, b), _orient(a, b, c), _orient(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0


def _seg_seg(a: Vec, b: Vec, c: Vec, d: Vec) -> float:
    if _segments_cross(a, b, c, d):
        return 0.0
    return min(_pt_seg(a, c, d), _pt_seg(b, c, d), _pt_seg(c, a, b), _pt_seg(d, a, b))


def _pt_box(p: Vec, bx: Box) -> float:
    dx = max(bx.x0 - p[0], 0.0, p[0] - bx.x1)
    dy = max(bx.y0 - p[1], 0.0, p[1] - bx.y1)
    return math.hypot(dx, dy)


def _seg_hits_box(a: Vec, b: Vec, bx: Box) -> bool:
    """Liang-Barsky: does segment *a*-*b* intersect the (closed) box?"""
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for p, q in ((-dx, a[0] - bx.x0), (dx, bx.x1 - a[0]), (-dy, a[1] - bx.y0), (dy, bx.y1 - a[1])):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        t = q / p
        if p < 0.0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _seg_box(a: Vec, b: Vec, bx: Box) -> float:
    if _seg_hits_box(a, b, bx):
        return 0.0
    corners = ((bx.x0, bx.y0), (bx.x1, bx.y0), (bx.x1, bx.y1), (bx.x0, bx.y1))
    return min(_pt_box(a, bx), _pt_box(b, bx), *(_pt_seg(c, a, b) for c in corners))


def _box_box(p: Box, q: Box) -> float:
    dx = max(q.x0 - p.x1, p.x0 - q.x1, 0.0)
    dy = max(q.y0 - p.y1, p.y0 - q.y1, 0.0)
    return math.hypot(dx, dy)


def gap(s: Shape, t: Shape) -> float:
    """Edge-to-edge distance of two shapes in mm (0 when they touch or overlap)."""
    if isinstance(s, Box) and isinstance(t, Box):
        return _box_box(s, t)
    if isinstance(s, Box):
        s, t = t, s
    assert isinstance(s, Path)
    if isinstance(t, Box):
        d = min(_seg_box(a, b, t) for a, b in s.segments())
        return max(0.0, d - s.r)
    d = min(_seg_seg(a, b, c, e) for a, b in s.segments() for c, e in t.segments())
    return max(0.0, d - s.r - t.r)


def bbox_near(p: BBox, q: BBox, margin: float) -> bool:
    return not (p[2] + margin < q[0] or q[2] + margin < p[0] or p[3] + margin < q[1] or q[3] + margin < p[1])


# ---- vectors and shapes shared by the router, the pours and the legacy copper generator ------------

def unit(v: Vec) -> Vec:
    n = float(np.hypot(v[0], v[1]))
    return (v[0] / n, v[1] / n) if n > 0 else (0.0, 0.0)


def perp(v: Vec) -> Vec:
    return (-v[1], v[0])


def add(p: Vec, d: Vec, k: float = 1.0) -> Vec:
    return (p[0] + d[0] * k, p[1] + d[1] * k)


def clip(x: float, lo: float, hi: float) -> float:
    """``float(np.clip(x, lo, hi))`` for finite scalars without the ufunc overhead: numpy clips as
    ``min(max(x, lo), hi)`` keeping *x* on ties (the sign of a zero), and *hi* wins when ``lo > hi``."""
    x = x if x >= lo else lo
    return float(x if x <= hi else hi)


def polyline45(p0: Vec, p1: Vec, *, dedupe: bool = True) -> list[Vec]:
    """*p0* to *p1* with one 45-degree jog first, then straight (like autorouted traces).

    The jog point is ``p0 + sign(d) * min(|dx|, |dy|)`` with a float sign, so integer pixel inputs
    give integral floats. *dedupe* drops a jog point that coincides with its neighbour (the board
    model's tracks); the legacy copper generator keeps all three points (``cv2.polylines`` blends a
    zero-length anti-aliased segment, so the pixels depend on it).
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    jog = min(abs(dx), abs(dy))
    mid = (p0[0] + float(np.sign(dx)) * jog, p0[1] + float(np.sign(dy)) * jog)
    pts = [p0, mid, p1]
    if not dedupe:
        return pts
    return [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]


def octagon(x: float, y: float, w: float, h: float, chamfer: float) -> list[Vec]:
    """Rectangle with 45-degree chamfered corners (the pour shape of autorouted boards), clockwise
    from the top edge."""
    c = chamfer
    return [(x + c, y), (x + w - c, y), (x + w, y + c), (x + w, y + h - c), (x + w - c, y + h), (x + c, y + h),
            (x, y + h - c), (x, y + c)]
