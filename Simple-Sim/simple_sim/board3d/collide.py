"""3D-Board-Renderer: bodies that run into each other, and the outline fix for the ones whose size is not measured.

The DeepVision app gives a connector the outline of its library footprint: often the courtyard (mating space, latch
travel, keep-out), not the housing: two neighbouring wire-to-board headers then get bodies twice their depth that stick
into each other, stacked 1x2 pin headers ones twice their pitch. The photo fit (``parttex.photo_fit``) cannot cut such
an outline back: it trims edges that show board, but here the neighbour's housing fills the excess.

:func:`find_collisions` tests every pair of bodies as boxes (rotated rectangle x z range; separating axes, so it also
returns how deep and along which axis they overlap). :func:`resolve` fixes the *soft* bodies of each collision
(connectors and generic boxes from the app outline, and fitted bodies flagged ``soft`` because they grew past the size
of their package name; other measured chips, ICs etc. are *hard* and only reported):

1. every soft body in a collision takes its datasheet size (part catalog, ``body.l`` / ``body.w``) where its outline is
   larger: slid inside the old outline to where it covers its own pins (plated holes of the scan), overlaps the other
   body least and covers the most part (not board colour) of the photo mosaic;
2. what still overlaps is cut off on the one side that costs the least body area, never closer than
   :data:`PIN_MARGIN_MM` to a pin (a small part inside a large outline is cut out sideways, not along the axis of the
   smallest overlap).

The fixes are kept next to the data (``parttex/collide.json``, ``tools/board3d_collide.py``) and applied after the
photo fit (:func:`simple_sim.board3d.scene.apply_fit`), so every render and the viewer's GLB get them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

#: overlaps shallower than this (mm) are contact, not collision (fitted outlines of neighbouring chips touch)
MIN_DEPTH_MM = 0.1
#: gap left between two bodies after a fix (mm)
CLEARANCE_MM = 0.05
#: a body keeps at least this much housing around its own pins (mm, from the pin centre)
PIN_MARGIN_MM = 0.6
#: the catalog size replaces the outline only when the outline is larger by this factor
CATALOG_SLACK = 1.1
#: part kinds whose outline comes from the app's footprint and may be wrong (``part["board"]`` of a job component)
SOFT_KINDS = ("connector", "box")
#: a pixel counts as board in the photo mosaic within this Delta E of the board colour
BOARD_DE = 14.0


# --------------------------------------------------------------------------- geometry

def _frame(c: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, unit axis along l, unit axis along w) of a job component or board3d component in board mm."""
    p = c["pose"]
    a = math.radians(p["rotDeg"])
    return np.array([p["x"], p["y"]]), np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def _size(c: dict[str, Any]) -> tuple[float, float]:
    b = c.get("body") or c
    return float(b["l"]), float(b["w"])


def corners(c: dict[str, Any]) -> np.ndarray:
    o, u, v = _frame(c)
    l, w = _size(c)
    return np.array([o + su * l / 2 * u + sv * w / 2 * v for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _extent(c: dict[str, Any], axis: np.ndarray) -> tuple[float, float]:
    o, u, v = _frame(c)
    l, w = _size(c)
    r = abs(u @ axis) * l / 2 + abs(v @ axis) * w / 2
    return float(o @ axis - r), float(o @ axis + r)


def overlap(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, np.ndarray] | None:
    """(depth mm, unit axis from *a* towards *b*) of the smallest overlap of two body outlines, None when apart."""
    best = None
    for c in (a, b):
        _o, u, v = _frame(c)
        for axis in (u, v):
            a0, a1 = _extent(a, axis)
            b0, b1 = _extent(b, axis)
            d = min(a1, b1) - max(a0, b0)
            if d <= 0:
                return None
            if best is None or d < best[0] - 1e-9:
                sign = 1.0 if (b0 + b1) >= (a0 + a1) else -1.0
                best = (d, sign * axis)
    return best


def _area(a: dict[str, Any], b: dict[str, Any]) -> float:
    import cv2
    ra, rb = (((c["pose"]["x"], c["pose"]["y"]), _size(c), c["pose"]["rotDeg"]) for c in (a, b))
    ok, pts = cv2.rotatedRectangleIntersection(ra, rb)
    return 0.0 if not ok or pts is None else float(cv2.contourArea(cv2.convexHull(pts)))


def _z(c: dict[str, Any]) -> tuple[float, float]:
    z0 = float(c["pose"].get("z") or 0.0)
    return z0, z0 + float(c.get("height") or 0.0)


def solid(c: dict[str, Any]) -> bool:
    """A job component with a body that can collide (placed, not missing, with height and area)."""
    return bool(c.get("pose")) and not c.get("missing") and float(c.get("height") or 0) > 0 and min(_size(c)) > 0


def is_soft(c: dict[str, Any]) -> bool:
    """Outline that may be wrong: a connector / box from the app, or a body flagged ``soft`` (a fit that grew a part
    past the size its package name gives)."""
    return bool(c.get("soft")) or (c.get("part") or {}).get("board") in SOFT_KINDS


def find_collisions(comps: Iterable[dict[str, Any]], *, min_depth: float = MIN_DEPTH_MM) -> list[dict[str, Any]]:
    """Every pair of job components (:func:`simple_sim.board3d.scene.job_component`) whose bodies overlap deeper than
    *min_depth* in the board plane and whose z ranges overlap: ``[{"a", "b", "depthMm", "axis", "areaMm2", "bbox", "soft"}]``,
    deepest share first. ``axis`` points from a to b (board mm), ``bbox`` is the board-mm box of both bodies,
    ``soft`` the refdes of the pair whose outline :func:`resolve` may change."""
    items = [c for c in comps if solid(c)]
    rad = [0.5 * math.hypot(*_size(c)) for c in items]
    out = []
    for i, a in enumerate(items):
        for j in range(i + 1, len(items)):
            b = items[j]
            if math.dist((a["pose"]["x"], a["pose"]["y"]), (b["pose"]["x"], b["pose"]["y"])) >= rad[i] + rad[j]:
                continue
            za, zb = _z(a), _z(b)
            if min(za[1], zb[1]) - max(za[0], zb[0]) <= 0:
                continue
            hit = overlap(a, b)
            if hit is None or hit[0] < min_depth:
                continue
            pts = np.vstack([corners(a), corners(b)])
            area = _area(a, b)
            out.append({"a": a["refdes"], "b": b["refdes"], "depthMm": round(hit[0], 3), "axis": [round(float(x), 4) for x in hit[1]],
                        "areaMm2": round(area, 3), "share": round(area / max(1e-9, min(np.prod(_size(a)), np.prod(_size(b)))), 3),
                        "bbox": [round(float(v), 2) for v in (*pts.min(0), *pts.max(0))],
                        "soft": [c["refdes"] for c in (a, b) if is_soft(c)]})
    out.sort(key=lambda r: (-r["share"], -r["depthMm"]))
    return out


# --------------------------------------------------------------------------- fix

def _part_axis(c: dict[str, Any], axis: np.ndarray) -> str:
    """``l`` or ``w``: the body axis closest to the board direction *axis*."""
    _o, u, v = _frame(c)
    return "l" if abs(u @ axis) >= abs(v @ axis) else "w"


def _set_span(c: dict[str, Any], dim: str, lo: float, hi: float) -> None:
    """Make the body span ``lo..hi`` (mm along its own *dim* axis, relative to the current centre)."""
    o, u, v = _frame(c)
    d = u if dim == "l" else v
    mid = (lo + hi) / 2
    c["pose"] = {**c["pose"], "x": float(o[0] + mid * d[0]), "y": float(o[1] + mid * d[1])}
    c[dim] = float(hi - lo)
    if c.get("pads"):
        # lands stay on the board: they move back in the part frame (connectors have none, generic boxes may)
        k = 0 if dim == "l" else 1
        c["pads"] = [[*p[:k], p[k] - mid, *p[k + 1:]] for p in c["pads"]]


def _local(c: dict[str, Any], pts: np.ndarray, dim: str) -> np.ndarray:
    o, u, v = _frame(c)
    return (np.asarray(pts, float).reshape(-1, 2) - o) @ (u if dim == "l" else v)


def _pins_of(c: dict[str, Any], holes: np.ndarray, others: list[dict[str, Any]]) -> np.ndarray:
    """Plated holes inside *c*'s outline that are not nearer the pin row of another body (two outlines that overlap
    both contain the pins of both): board mm, N x 2."""
    if not len(holes):
        return holes.reshape(0, 2)
    o, u, v = _frame(c)
    l, w = _size(c)
    rel = holes - o
    inside = (np.abs(rel @ u) <= l / 2) & (np.abs(rel @ v) <= w / 2)
    mine = holes[inside]
    if not len(mine) or not others:
        return mine
    pnp = np.array(c.get("pnp") or [c["pose"]["x"], c["pose"]["y"]], float)
    keep = []
    for h in mine:
        d_me = np.linalg.norm(h - pnp)
        if all(np.linalg.norm(h - np.array(x.get("pnp") or [x["pose"]["x"], x["pose"]["y"]], float)) >= d_me for x in others):
            keep.append(h)
    return np.array(keep).reshape(-1, 2)


def _photo_score(c: dict[str, Any], dim: str, lo: float, hi: float, part_mask: Callable[[np.ndarray], np.ndarray] | None,
                 exclude: Iterable[dict[str, Any]] = ()) -> float:
    """Share of not-board pixels of the photo mosaic in the body spanning ``lo..hi`` on *dim* (0 without a photo);
    points inside the bodies *exclude* do not count (they show the other part, not this one)."""
    if part_mask is None or hi <= lo or c.get("noPhoto"):
        return 0.0                  # noPhoto: the photos do not show this part (another variant): its pins decide
    o, u, v = _frame(c)
    l, w = _size(c)
    d, e = (u, v) if dim == "l" else (v, u)
    other = w if dim == "l" else l
    s = np.linspace(lo, hi, max(4, int((hi - lo) * 10)))
    t = np.linspace(-other / 2, other / 2, max(4, int(other * 10)))
    pts = (o + s[:, None, None] * d + t[None, :, None] * e).reshape(-1, 2)
    keep = np.ones(len(pts), bool)
    for x in exclude:
        xo, xu, xv = _frame(x)
        xl, xw = _size(x)
        rel = pts - xo
        keep &= ~((np.abs(rel @ xu) <= xl / 2) & (np.abs(rel @ xv) <= xw / 2))
    return float(part_mask(pts[keep]).mean()) if keep.any() else 0.0


def _catalog_dims(soft: dict[str, Any], catalog: dict[str, Any] | None) -> dict[str, float]:
    """The datasheet sizes that may replace *soft*'s outline: only axes the catalog names unambiguously (one axis given,
    or both with the same long / short order as the outline: a stacked USB socket's catalog width is the outline's
    depth), and only where the outline is larger by :data:`CATALOG_SLACK`."""
    cat = {k: float(v) for k, v in (catalog or {}).items() if k in ("l", "w") and v}
    l, w = _size(soft)
    if len(cat) == 2 and (cat["l"] - cat["w"]) * (l - w) < 0:
        return {}
    return {k: v for k, v in cat.items() if (l if k == "l" else w) > CATALOG_SLACK * v}


def _span(c: dict[str, Any], dim: str) -> tuple[float, float]:
    size = _size(c)[0 if dim == "l" else 1]
    return -size / 2, size / 2


def _pin_span(c: dict[str, Any], dim: str, holes: np.ndarray, neighbours: list[dict[str, Any]]) -> tuple[float, float] | None:
    pins = _local(c, _pins_of(c, holes, neighbours), dim)
    return (float(pins.min()) - PIN_MARGIN_MM, float(pins.max()) + PIN_MARGIN_MM) if len(pins) else None


#: tie-break of the datasheet placement: score lost per mm the body centre sits off its pin row's centre
CENTRE_WEIGHT = 0.02


def _catalog_fit(soft: dict[str, Any], others: list[dict[str, Any]], *, catalog: dict[str, Any] | None, holes: np.ndarray,
                 part_mask: Callable[[np.ndarray], np.ndarray] | None, neighbours: list[dict[str, Any]]) -> list[str]:
    """Give *soft* its datasheet size on every axis it is too large on: slid inside the old outline to where it covers
    its pins, overlaps *others* least (a hard body with its outline, a soft one, which shrinks too, only with its pin
    row: its old outline is no obstacle) and covers the most part (not board) of the photo; ties go to the placement
    centred on its pins. Returns what was done."""
    done = []
    for dim, cat in _catalog_dims(soft, catalog).items():
        lo, hi = _span(soft, dim)
        pins = _pin_span(soft, dim, holes, neighbours)
        blocks = []
        for o in others:
            if is_soft(o):
                op = _pins_of(o, holes, [x for x in neighbours if x is not o] + [soft])
                cross = "w" if dim == "l" else "l"
                half = _size(soft)[0 if cross == "l" else 1] / 2 + PIN_MARGIN_MM
                op = op[np.abs(_local(soft, op, cross)) <= half] if len(op) else op   # only pins in line with soft
                if len(op):
                    ext = _local(soft, op, dim)
                    blocks.append((float(ext.min()) - PIN_MARGIN_MM, float(ext.max()) + PIN_MARGIN_MM))
                continue
            ext = _local(soft, corners(o), dim)
            blocks.append((float(ext.min()), float(ext.max())))
        centre = (pins[0] + pins[1]) / 2 if pins is not None else (lo + hi) / 2
        best = None
        for start in np.linspace(lo, hi - cat, max(2, int((hi - lo - cat) * 20) + 1)):
            a, b = float(start), float(start) + cat
            if pins is not None and (a > pins[0] or b < pins[1]):
                continue
            hit = sum(max(0.0, min(b, e1) - max(a, e0)) for e0, e1 in blocks)
            score = _photo_score(soft, dim, a, b, part_mask) - hit - CENTRE_WEIGHT * abs((a + b) / 2 - centre)
            if best is None or score > best[0] + 1e-6:
                best = (score, a, b)
        if best is not None:
            _set_span(soft, dim, best[1], best[2])
            done.append(f"{dim}: catalog {cat:g}")
    return done


#: cost of cutting off a strip = its area x (this + share of the strip the photo shows as this part's own)
CUT_AREA_WEIGHT = 0.1


def _cut(soft: dict[str, Any], other: dict[str, Any], *, holes: np.ndarray, neighbours: list[dict[str, Any]],
         part_mask: Callable[[np.ndarray], np.ndarray] | None = None, bodies: list[dict[str, Any]] = ()) -> str | None:
    """Cut *soft* back on one side so it clears *other*: of the four sides the one whose strip costs least (its area,
    weighted by how much of it the photo shows as part, the other *bodies* left out: board or a neighbour is cheap to
    give up, the housing itself is not), never into its pins and never below a third of the dimension."""
    l, w = _size(soft)
    best = None
    for dim in ("l", "w"):
        lo, hi = _span(soft, dim)
        ext = _local(soft, corners(other), dim)
        pins = _pin_span(soft, dim, holes, neighbours)
        cross = w if dim == "l" else l
        for side, new in (("+", float(ext.min()) - CLEARANCE_MM), ("-", float(ext.max()) + CLEARANCE_MM)):
            a, b = (lo, min(hi, new)) if side == "+" else (max(lo, new), hi)
            if b - a < (hi - lo) / 3 or (pins is not None and (a > pins[0] or b < pins[1])):
                continue
            area = ((hi - lo) - (b - a)) * cross
            strip = (b, hi) if side == "+" else (lo, a)
            loss = area * (CUT_AREA_WEIGHT + _photo_score(soft, dim, *strip, part_mask, bodies))
            if area > 0 and (best is None or loss < best[0]):
                best = (loss, dim, a, b, side)
    if best is None:
        return None
    _loss, dim, a, b, side = best
    _set_span(soft, dim, a, b)
    return f"{dim}: cut {side} {_size(soft)[0 if dim == 'l' else 1]:.2f}"


def resolve(comps: list[dict[str, Any]], *, catalog: dict[str, dict[str, Any]] | None = None, holes: Iterable[Iterable[float]] = (),
            part_mask: Callable[[np.ndarray], np.ndarray] | None = None, rounds: int = 4) -> dict[str, Any]:
    """Fix the soft bodies of every collision in place (see the module doc). *catalog*: refdes -> datasheet body
    ``{"l", "w"}`` (either may be missing); *holes*: plated holes ``(x, y, ...)`` in board mm; *part_mask*: board mm N x 2
    -> bool N (True where the photo shows a part, not board). Returns ``{"fixed": {refdes: {...}}, "left": [collision]}``."""
    holes_xy = np.array([list(h)[:2] for h in holes], float).reshape(-1, 2)
    by = {c["refdes"]: c for c in comps}
    fixed: dict[str, dict[str, Any]] = {}
    nb = [x for x in comps if solid(x) and is_soft(x) and x.get("pnp")]

    def note(ref: str, against: list[str], how: str) -> None:
        f = fixed.setdefault(ref, {"against": [], "how": []})
        f["against"] += [r for r in against if r not in f["against"]]
        f["how"].append(how)

    # 1. datasheet sizes for every soft body in a collision (first, so a cut never takes what the datasheet frees)
    partners: dict[str, list[str]] = {}
    for col in find_collisions(comps):
        for s, o in ((col["a"], col["b"]), (col["b"], col["a"])):
            if s in col["soft"]:
                partners.setdefault(s, []).append(o)
    for ref, others in partners.items():
        s = by[ref]
        for how in _catalog_fit(s, [by[o] for o in others], catalog=(catalog or {}).get(ref), holes=holes_xy, part_mask=part_mask,
                                neighbours=[x for x in nb if x is not s]):
            note(ref, others, how)
    # 2. what still overlaps: cut back, the larger soft body of a pair first
    for _ in range(rounds):
        changed = False
        for col in find_collisions(comps):
            a, b = by[col["a"]], by[col["b"]]
            if (hit := overlap(a, b)) is None or hit[0] < MIN_DEPTH_MM:
                continue                              # fixed by an earlier pair of this round
            for s in sorted((c for c in (a, b) if is_soft(c)), key=lambda c: -float(np.prod(_size(c)))):
                o = b if s is a else a
                others = [x for x in comps if x is not s and solid(x)]
                near = [x for x in others if overlap(s, x) is not None]
                if (how := _cut(s, o, holes=holes_xy, neighbours=[x for x in nb if x is not s], part_mask=part_mask, bodies=near)) is not None:
                    changed = True
                    note(s["refdes"], [o["refdes"]], how)
                    if (hit := overlap(a, b)) is None or hit[0] < MIN_DEPTH_MM:
                        break
        if not changed:
            break
    for ref, f in fixed.items():
        c = by[ref]
        f.update({"x": round(c["pose"]["x"], 4), "y": round(c["pose"]["y"], 4), "l": round(c["l"], 4), "w": round(c["w"], 4)})
    return {"fixed": fixed, "left": find_collisions(comps)}


def apply_collide(board3d: dict[str, Any], fixes: dict[str, dict[str, Any]]) -> None:
    """Set the fixed outlines of ``collide.json`` (``{refdes: {"x", "y", "l", "w"}}``) on the components in place."""
    for c in board3d.get("components", []):
        f = fixes.get(c.get("refdes"))
        if f and c.get("pose") and c.get("body"):
            c["pose"].update({"x": f["x"], "y": f["y"]})
            c["body"].update({"l": f["l"], "w": f["w"]})


__all__ = ["MIN_DEPTH_MM", "apply_collide", "corners", "find_collisions", "is_soft", "overlap", "resolve", "solid"]
