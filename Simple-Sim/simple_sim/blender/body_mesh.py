"""Part body meshes in real millimetres (stdlib only: no ``bpy``, unit-testable).

A body is a stack of rings (horizontal outlines) from the board up to its top face:

* the outline is a rounded rectangle (half sizes ``a`` x ``b``, vertical corner radius ``r``); single corners can be
  a straight 45 deg chamfer instead (the polarity corners of an electrolytic's base plate). ``a == b == r`` gives
  a circle (an electrolytic can).
* **flat, sharp bottom** on the board: real parts sit on their terminals / standoff with a square bottom edge.
* optional mould **draft**: moulded IC bodies taper from the parting line (where the leads exit) to the top and to
  the bottom, the top face is visibly smaller than the outline at the leads.
* a small **top edge radius** (``r_top``) as a quarter circle: the offset of the outline shrinks over the radius.

Shading flags per face: only the curved bands (corner arcs, the top edge radius) are smooth, the flat walls and the
top / bottom faces are flat. Smooth shading on a whole bevelled box makes every face look like a cushion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: radius floor: corners never collapse to a point (degenerate quads give broken normals)
MIN_RADIUS = 1e-3


@dataclass(frozen=True)
class BodyShape:
    """Outline and profile of one body, lengths in mm (``l`` along x, ``w`` along y, ``h`` up)."""

    l: float
    w: float
    h: float
    r_corner: float = 0.0          # vertical edge radius
    r_top: float = 0.0             # top edge radius (quarter circle)
    draft_deg: float = 0.0         # mould draft of the walls above and below the parting line
    parting: float = 0.5           # parting line height / h (draft only)
    chamfer: dict[int, float] = field(default_factory=dict)   # corner index -> 45 deg chamfer leg (mm)
    corner_segments: int = 3
    top_segments: int = 3
    grooves: tuple[tuple[float, float, float], ...] = ()      # (z, depth, half width) in mm: rolled beads (can)


@dataclass
class BodyMesh:
    verts: list[tuple[float, float, float]]
    faces: list[tuple[int, ...]]
    smooth: list[bool]


#: corner order of a ring (counter-clockwise seen from above): (+x,+y), (-x,+y), (-x,-y), (+x,-y)
CORNERS = ((1, 1), (-1, 1), (-1, -1), (1, -1))


def _ring(a: float, b: float, r: float, wall: float, d: float, shape: BodyShape,
          seg: int) -> tuple[list[tuple[float, float]], list[bool]]:
    """Outline moved inwards by the draft *wall* inset (corner radius kept) and offset by *d* (the top edge
    radius: the corner radius shrinks with it): points (counter-clockwise) and, per point, whether the edge to the
    next point is curved (an arc segment)."""
    a, b = a - wall - d, b - wall - d
    r = max(min(r - d, a, b), MIN_RADIUS)
    pts, curved = [], []
    for k, (sx, sy) in enumerate(CORNERS):
        c = shape.chamfer.get(k)
        if c:
            # straight 45 deg cut, shrunk by the offset (a parallel line at distance d is tan(22.5) d shorter per leg)
            leg = max(c - d * math.tan(math.pi / 8), MIN_RADIUS)
            p0 = (sx * a, sy * (b - leg)) if sx * sy > 0 else (sx * (a - leg), sy * b)
            p1 = (sx * (a - leg), sy * b) if sx * sy > 0 else (sx * a, sy * (b - leg))
            for m in range(seg + 1):
                t = m / seg
                pts.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))
                curved.append(False)
            continue
        cx, cy = sx * (a - r), sy * (b - r)
        start = k * math.pi / 2
        for m in range(seg + 1):
            ang = start + (math.pi / 2) * m / seg
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
            curved.append(m < seg)
    return pts, curved


def body_mesh(shape: BodyShape, *, center_z: bool = True) -> BodyMesh:
    """Mesh of *shape*, centred in x / y; z from ``-h/2`` to ``h/2`` (*center_z*, like a unit cube scaled to the
    body) or from 0 to ``h``."""
    a, b, h = shape.l / 2, shape.w / 2, shape.h
    lim = 0.45 * min(shape.l, shape.w, h)
    r_top = min(max(shape.r_top, 0.0), lim)
    r = min(max(shape.r_corner, r_top * 1.25, MIN_RADIUS), 0.5 * min(shape.l, shape.w))
    tan_d = math.tan(math.radians(shape.draft_deg))
    zp = shape.parting * h if tan_d > 0 else 0.0
    # the walls must not run into each other: cap the draft inset at a quarter of the smaller half size
    tan_d = min(tan_d, 0.25 * min(a, b) / max(h - zp, zp, 1e-6))

    def wall(z: float) -> float:          # draft inset of the wall at height z
        return abs(z - zp) * tan_d

    levels: list[tuple[float, float, float, bool]] = []   # (z, draft inset, round inset, band below is curved)
    levels.append((0.0, wall(0.0), 0.0, False))
    if tan_d > 0 and 0.0 < zp < h - r_top:
        levels.append((zp, 0.0, 0.0, False))
    for zg, dep, hw in sorted(shape.grooves):
        if levels[-1][0] < zg - hw and zg + hw < h - r_top:
            levels += [(zg - hw, wall(zg - hw), 0.0, False), (zg, wall(zg) + dep, 0.0, True),
                       (zg + hw, wall(zg + hw), 0.0, True)]
    if r_top > 0:
        z0 = h - r_top
        levels.append((z0, wall(z0), 0.0, False))
        for k in range(1, shape.top_segments + 1):
            th = (math.pi / 2) * k / shape.top_segments
            levels.append((z0 + r_top * math.sin(th), wall(z0), r_top * (1 - math.cos(th)), True))
    else:
        levels.append((h, wall(h), 0.0, False))

    dz = -h / 2 if center_z else 0.0
    verts, faces, smooth = [], [], []
    n = 0
    for j, (z, w_in, d, band_curved) in enumerate(levels):
        pts, curved = _ring(a, b, r, w_in, d, shape, shape.corner_segments)
        n = len(pts)
        verts += [(x, y, z + dz) for x, y in pts]
        if j == 0:
            continue
        lo, hi = (j - 1) * n, j * n
        for i in range(n):
            i1 = (i + 1) % n
            faces.append((lo + i, lo + i1, hi + i1, hi + i))
            smooth.append(band_curved or curved[i])
    faces.append(tuple(reversed(range(n))))                         # bottom (normal -z)
    smooth.append(False)
    top = (len(levels) - 1) * n
    faces.append(tuple(range(top, top + n)))                        # top (normal +z)
    smooth.append(False)
    return BodyMesh(verts, faces, smooth)


# --------------------------------------------------------------------------- shapes per part kind

def edge_radius(size: tuple[float, float, float], rel: float, cap: float) -> float:
    """Edge radius *rel* x the smallest dimension, at most *cap* mm."""
    return min(rel * min(size), cap)


def box_shape(size, bevel: float) -> BodyShape:
    """Generic box (``render_batch._box`` with a ``bevel``): small edges, sharp bottom. ``bevel`` (mm) is capped:
    render_batch's relative bevels were meant for a unit cube and overshoot on real bodies."""
    l, w, h = (float(v) for v in size)
    r = min(bevel, 0.05, 0.2 * min(l, w, h)) if bevel > 0 else 0.0
    return BodyShape(l, w, h, r_corner=r, r_top=r, corner_segments=2 if r else 1, top_segments=2 if r else 1)


def chip_body_shape(l: float, w: float, h: float) -> BodyShape:
    """Chip resistor / MLCC ceramic: sawn block, nearly sharp edges (tumbled: a few hundredths of a mm)."""
    r = edge_radius((l, w, h), 0.06, 0.04)
    return BodyShape(l, w, h, r_corner=r, r_top=r, corner_segments=2, top_segments=2)


def chip_cap_shape(l: float, w: float, h: float) -> BodyShape:
    """End termination (dipped silver + nickel / tin): a bit rounder than the ceramic, still flat faces."""
    r = edge_radius((l, w, h), 0.12, 0.08)
    return BodyShape(l, w, h, r_corner=r, r_top=r, corner_segments=3, top_segments=3)


def molded_shape(l: float, w: float, h: float, *, leads: bool) -> BodyShape:
    """Moulded body. With gull-wing leads (SOT / SOIC / QFP): transfer-moulded, 8 deg draft from the parting line
    at the lead exit (45 % of H), small top edge. Without (QFN / DFN): sawn from a strip, vertical walls."""
    if leads:
        return BodyShape(l, w, h, r_corner=min(0.04 * min(l, w), 0.15), r_top=min(0.05 * h, 0.06),
                         draft_deg=8.0, parting=0.45)
    r = edge_radius((l, w, h), 0.03, 0.03)
    return BodyShape(l, w, h, r_corner=r, r_top=r, corner_segments=2, top_segments=2)


def can_base_shape(l: float, w: float, h: float, plus_corners: tuple[int, ...]) -> BodyShape:
    """Electrolytic base plate: square, the two corners on the + side cut at 45 deg (the polarity mark)."""
    r = edge_radius((l, w, h), 0.05, 0.08)
    c = 0.15 * min(l, w)
    return BodyShape(l, w, h, r_corner=r, r_top=r, chamfer={k: c for k in plus_corners})


def can_shape(diameter: float, h: float) -> BodyShape:
    """Aluminium can: a cylinder with the rolled rim at the top (radius ~4 % of the diameter)."""
    return BodyShape(diameter, diameter, h, r_corner=diameter / 2, r_top=min(0.04 * diameter, 0.3),
                     corner_segments=16, top_segments=4)


def plus_corners(pads: list, l: float, w: float) -> tuple[int, ...]:
    """Corner indices (see :data:`CORNERS`) on the side of pad 0 (pin 1 = + of a polarised capacitor)."""
    if not pads:
        return (1, 2)
    px, py = float(pads[0][0]), float(pads[0][1])
    if abs(px) / max(l, 1e-6) >= abs(py) / max(w, 1e-6):
        return (0, 3) if px > 0 else (1, 2)
    return (0, 1) if py > 0 else (2, 3)


# --------------------------------------------------------------------------- assemblies (board3d)
#
# A part is a list of :class:`Piece` meshes in the **body frame**: origin at the body centre on the body's bottom face
# (the part root of render_batch sits at the standoff above the board, so the board is at ``z = -standoff``); x along
# the body length ``l``, y along ``w``. Pads are ``(cx, cy, w, h)`` in the same x / y. Contacts (what the solder
# wets) are ``{"cx", "cy", "hl", "hw", "angle"}`` rectangles in that frame.

#: material roles of the pieces (render_board3d maps them to Blender materials)
ROLES = ("mold", "ceramic", "ferrite", "metal", "alu", "base", "substrate", "lid", "white", "ball", "natural", "insert")


@dataclass
class Piece:
    name: str
    mesh: BodyMesh
    role: str
    # side faces sample the body photo too: ceramic chips / arrays and unknown parts (connectors: colour unknown).
    # Moulded, ferrite and BGA bodies keep their material: the outline of the app's body polygon can be larger than
    # the part, then the photo just inside it is the board (green walls on a power inductor).
    photo_sides: bool = False


@dataclass
class Assembly:
    pieces: list[Piece]
    contacts: list[dict[str, float] | None] | None = None   # per pad, None: keep render_batch's
    solder_height: float | None = None


def moved(m: BodyMesh, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> BodyMesh:
    return BodyMesh([(x + dx, y + dy, z + dz) for x, y, z in m.verts], list(m.faces), list(m.smooth))


def shape_at(shape: BodyShape, x: float = 0.0, y: float = 0.0, z0: float = 0.0) -> BodyMesh:
    """*shape* standing on ``z = z0`` at (x, y)."""
    return moved(body_mesh(shape, center_z=False), x, y, z0)


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> BodyMesh:
    """Axis-aligned flat-shaded box."""
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return BodyMesh(v, f, [False] * 6)


def sphere(cx: float, cy: float, cz: float, r: float, seg: int = 10, rings: int = 6) -> BodyMesh:
    verts = [(cx, cy, cz - r)]
    for j in range(1, rings):
        th = math.pi * j / rings - math.pi / 2
        verts += [(cx + r * math.cos(th) * math.cos(2 * math.pi * i / seg), cy + r * math.cos(th) * math.sin(2 * math.pi * i / seg),
                   cz + r * math.sin(th)) for i in range(seg)]
    verts.append((cx, cy, cz + r))
    faces = [(0, 1 + (i + 1) % seg, 1 + i) for i in range(seg)]
    for j in range(rings - 2):
        a, b = 1 + j * seg, 1 + (j + 1) * seg
        faces += [(a + i, a + (i + 1) % seg, b + (i + 1) % seg, b + i) for i in range(seg)]
    top, last = len(verts) - 1, 1 + (rings - 2) * seg
    faces += [(last + i, last + (i + 1) % seg, top) for i in range(seg)]
    return BodyMesh(verts, faces, [True] * len(faces))


def bend_path(pts: list[tuple[float, float]], radius: float, n: int = 5) -> list[tuple[float, float]]:
    """Polyline with every inner corner replaced by an arc of *radius* tangent to both legs (at most half of either
    leg): a bent sheet keeps its straight runs, only the bends are round (inner bend radius ~ sheet thickness)."""
    out = [pts[0]]
    for k in range(1, len(pts) - 1):
        (x0, z0), (x1, z1), (x2, z2) = pts[k - 1], pts[k], pts[k + 1]
        a = (x0 - x1, z0 - z1)
        b = (x2 - x1, z2 - z1)
        la, lb = math.hypot(*a), math.hypot(*b)
        if la < 1e-9 or lb < 1e-9:
            continue
        ua, ub = (a[0] / la, a[1] / la), (b[0] / lb, b[1] / lb)
        cos_t = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1]))
        theta = math.acos(cos_t)                      # angle between the legs
        if theta > math.pi - 1e-6:
            out.append((x1, z1))
            continue
        tan_len = min(radius / math.tan(theta / 2), 0.5 * la, 0.5 * lb)
        r = tan_len * math.tan(theta / 2)
        p0 = (x1 + ua[0] * tan_len, z1 + ua[1] * tan_len)
        p1 = (x1 + ub[0] * tan_len, z1 + ub[1] * tan_len)
        bis = (ua[0] + ub[0], ua[1] + ub[1])
        lbis = math.hypot(*bis)
        d = r / math.sin(theta / 2)
        c = (x1 + bis[0] / lbis * d, z1 + bis[1] / lbis * d)
        a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
        a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        out += [(c[0] + r * math.cos(a0 + da * i / n), c[1] + r * math.sin(a0 + da * i / n)) for i in range(n + 1)]
    out.append(pts[-1])
    return out


def strip_mesh(path: list[tuple[float, float]], t: float, width: float, origin: tuple[float, float],
               direction: tuple[float, float]) -> BodyMesh:
    """A sheet-metal strip (thickness *t*, *width*) whose centre line runs along *path* = [(s, z)]: s outwards along
    *direction* (unit x / y vector) from *origin*. Bends are smooth-shaded, the flat sides flat."""
    ux, uy = direction
    vx, vy = -uy, ux                      # across the strip
    verts, faces, smooth = [], [], []
    n = len(path)
    for k, (s, z) in enumerate(path):
        a = path[max(k - 1, 0)]
        b = path[min(k + 1, n - 1)]
        ts, tz = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(ts, tz) or 1.0
        ns, nz = -tz / ln, ts / ln            # normal in the (s, z) plane, pointing "up" for an outward run
        for (cs, cn) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):   # (across, normal) corners
            ss, zz = s + cn * ns * t / 2, z + cn * nz * t / 2
            verts.append((origin[0] + ux * ss + vx * cs * width / 2, origin[1] + uy * ss + vy * cs * width / 2, zz))
    for k in range(n - 1):
        a, b = 4 * k, 4 * (k + 1)
        bent = abs((path[k + 1][1] - path[k][1]) * (path[min(k + 2, n - 1)][0] - path[k + 1][0])
                   - (path[k + 1][0] - path[k][0]) * (path[min(k + 2, n - 1)][1] - path[k + 1][1])) > 1e-12
        faces += [(a + 0, a + 1, b + 1, b + 0), (a + 1, a + 2, b + 2, b + 1), (a + 2, a + 3, b + 3, b + 2), (a + 3, a + 0, b + 0, b + 3)]
        smooth += [bent, False, bent, False]
    last = 4 * (n - 1)
    faces += [(3, 2, 1, 0), (last, last + 1, last + 2, last + 3)]
    smooth += [False, False]
    # orientation: the loop order above depends on the run direction; make the faces point outwards
    m = BodyMesh(verts, faces, smooth)
    if _signed_volume(m) < 0:
        m.faces = [tuple(reversed(f)) for f in m.faces]
    return m


def _signed_volume(m: BodyMesh) -> float:
    vol = 0.0
    for f in m.faces:
        p = m.verts[f[0]]
        for i in range(1, len(f) - 1):
            a, b = m.verts[f[i]], m.verts[f[i + 1]]
            vol += (p[0] * (a[1] * b[2] - a[2] * b[1]) - p[1] * (a[0] * b[2] - a[2] * b[0]) + p[2] * (a[0] * b[1] - a[1] * b[0])) / 6
    return vol


def _pad_side(pad, l: float, w: float):
    """(direction, distance of the pad centre beyond the body edge, pad extent along / across that direction, position
    along the edge) of a pad outside the body, or None for a pad under it."""
    px, py, pw, ph = pad
    ox, oy = abs(px) - l / 2, abs(py) - w / 2
    if ox <= 0 and oy <= 0:
        return None
    if ox / max(pw, 1e-6) >= oy / max(ph, 1e-6):
        return (1.0 if px > 0 else -1.0, 0.0), abs(px) - l / 2, pw, ph, py
    return (0.0, 1.0 if py > 0 else -1.0), abs(py) - w / 2, ph, pw, px


def _contact(cx: float, cy: float, hl: float, hw: float, direction) -> dict[str, float]:
    return {"cx": cx, "cy": cy, "hl": hl, "hw": hw, "angle": math.atan2(direction[1], direction[0])}


def lead_path(reach: float, p_len: float, t: float, z_exit: float, z_board: float) -> tuple[list[tuple[float, float]], float, float]:
    """Centre line [(s, z)] of a gull-wing lead (sheet thickness *t*) leaving the body at height *z_exit*, its land
    centre *reach* out and *p_len* long; returns (path, start of the flat foot, its end). Out of the parting line, a
    short shoulder, the leg down at ~73 deg, the foot flat on the land (60 % of the land, up to 20 % past its centre:
    the toe fillet forms beyond), bends of inner radius 1.5 t. All lengths relative (ROI scenes scale ICs down)."""
    z_foot = z_board + 0.55 * t
    s_end = reach + 0.2 * p_len
    s_leg = max(s_end - min(0.6 * p_len, 1.0), 2.5 * t)
    s_sh = min(1.2 * t, 0.45 * s_leg)
    s_sh = max(s_sh, s_leg - 0.3 * (z_exit - z_foot))
    path = bend_path([(-1.5 * t, z_exit), (s_sh, z_exit), (s_leg, z_foot), (s_end, z_foot)], 1.5 * t)
    return path, s_leg, s_end


def gullwing(l: float, w: float, h: float, standoff: float, pads: list) -> Assembly:
    """Moulded body with draft + one bent lead per pad: out of the parting line, a round bend down, the foot flat on
    the land (on the inner ~60 % of it: the toe fillet forms beyond). Lead width / thickness from the land
    (SOIC 0.41 x 0.2, SOT-23 0.4 x 0.13, TSSOP 0.25 x 0.13 mm)."""
    shape = molded_shape(l, w, h, leads=True)
    pieces = [Piece("body", shape_at(shape), "mold")]
    contacts: list[dict[str, float] | None] = []
    t_max = 0.0
    z_exit = shape.parting * h
    for k, pad in enumerate(pads):
        side = _pad_side(pad, l, w)
        if side is None:
            contacts.append(None)
            continue
        d, reach, p_len, p_across, along = side
        lw = min(0.6 * p_across, 0.45)
        t = min(max(0.32 * lw, 0.1), 0.2)
        t_max = max(t_max, t)
        path, s_leg, s_end = lead_path(reach, p_len, t, z_exit, -standoff)
        ox, oy = (d[0] * l / 2, along) if d[0] else (along, d[1] * w / 2)
        pieces.append(Piece(f"lead_{k}", strip_mesh(path, t, lw, (ox, oy), d), "metal"))
        mid = (s_leg + s_end) / 2
        contacts.append(_contact(ox + d[0] * mid, oy + d[1] * mid, (s_end - s_leg) / 2, lw / 2, d))
    return Assembly(pieces, contacts, 0.9 * t_max if t_max else None)


def qfn(l: float, w: float, h: float, standoff: float, pads: list, thermal: int | None) -> Assembly:
    """Sawn QFN / DFN: vertical walls, the lead-frame terminals show as bright rectangles at the foot of the walls
    (0.2 mm lead frame, cut flush with the mould)."""
    pieces = [Piece("body", shape_at(molded_shape(l, w, h, leads=False)), "mold")]
    frame = min(0.2, 0.3 * h)
    for k, (px, py, pw, ph) in enumerate(pads):
        if k == thermal:
            continue
        ox, oy = l / 2 - abs(px), w / 2 - abs(py)          # distance of the pad centre inside the edge
        if min(ox, oy) > 0.5:
            continue
        if ox <= oy:                                        # on an x edge
            sx = 1.0 if px > 0 else -1.0
            x_out, x_in = sx * (l / 2 + 0.004), sx * (l / 2 - min(0.35, pw))
            pieces.append(Piece(f"term_{k}", box(min(x_out, x_in), max(x_out, x_in), py - ph / 2, py + ph / 2, -standoff, frame), "metal"))
        else:
            sy = 1.0 if py > 0 else -1.0
            y_out, y_in = sy * (w / 2 + 0.004), sy * (w / 2 - min(0.35, ph))
            pieces.append(Piece(f"term_{k}", box(px - pw / 2, px + pw / 2, min(y_out, y_in), max(y_out, y_in), -standoff, frame), "metal"))
    return Assembly(pieces)


def array(l: float, w: float, h: float, pads: list, *, role: str) -> Assembly:
    """Chip array: a ceramic chip with one wrap-around termination per pad on its long sides."""
    pieces = [Piece("body", shape_at(chip_body_shape(l, w, h)), role, photo_sides=True)]
    contacts: list[dict[str, float] | None] = []
    for k, (px, py, pw, ph) in enumerate(pads):
        on_y = abs(py) / max(w, 1e-6) >= abs(px) / max(l, 1e-6)
        if on_y:
            sy = 1.0 if py > 0 else -1.0
            depth, width = 0.22 * w, 0.8 * pw
            cy = sy * (w / 2 - depth / 2 + 0.01)
            pieces.append(Piece(f"term_{k}", shape_at(chip_cap_shape(width, depth + 0.02, h * 1.02), px, cy, -0.01 * h), "metal"))
            contacts.append(_contact(px, sy * (w / 2 - depth / 2), width / 2, depth / 2 + 0.01, (0.0, sy)))
        else:
            sx = 1.0 if px > 0 else -1.0
            depth, width = 0.22 * l, 0.8 * ph
            cx = sx * (l / 2 - depth / 2 + 0.01)
            pieces.append(Piece(f"term_{k}", shape_at(chip_cap_shape(depth + 0.02, width, h * 1.02), cx, py, -0.01 * h), "metal"))
            contacts.append(_contact(sx * (l / 2 - depth / 2), py, depth / 2 + 0.01, width / 2, (sx, 0.0)))
    return Assembly(pieces, contacts, 0.5 * h)


def crystal(l: float, w: float, h: float) -> Assembly:
    """SMD crystal / oscillator: white alumina base (60 % of H) with a seam-welded metal lid inset 0.15 mm."""
    hb = 0.6 * h
    inset = min(0.15, 0.08 * min(l, w))
    base = BodyShape(l, w, hb, r_corner=edge_radius((l, w, hb), 0.05, 0.05), r_top=0.02, corner_segments=2, top_segments=1)
    lid = BodyShape(l - 2 * inset, w - 2 * inset, h - hb, r_corner=min(0.3, 0.12 * min(l, w)), r_top=0.03,
                    corner_segments=4, top_segments=2)
    return Assembly([Piece("body_base", shape_at(base), "white"), Piece("lid", shape_at(lid, z0=hb), "lid")])


def bga(l: float, w: float, h: float, standoff: float, pitch: float) -> Assembly:
    """Sawn (MAP) BGA: laminate substrate + full-size mould cap, the outer two rings of balls visible under the edge."""
    hs = min(0.36, 0.3 * h)
    sub = BodyShape(l, w, hs, r_corner=0.02, r_top=0.0, corner_segments=1, top_segments=1)
    cap = BodyShape(l, w, h - hs, r_corner=0.03, r_top=0.03, corner_segments=2, top_segments=2)
    pieces = [Piece("body_sub", shape_at(sub), "substrate"), Piece("body", shape_at(cap, z0=hs), "mold")]
    nx, ny = max(int((l - pitch) / pitch) + 1, 1), max(int((w - pitch) / pitch) + 1, 1)
    r = 0.6 * pitch / 2
    rz = max(standoff, 0.05)
    balls = []
    for i in range(nx):
        for j in range(ny):
            if min(i, j, nx - 1 - i, ny - 1 - j) >= 2:
                continue
            x, y = (i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch
            balls.append(sphere(x, y, -rz / 2, max(r, rz / 2) * 1.0, seg=8, rings=5))
    if balls:
        verts, faces, smooth = [], [], []
        for b in balls:
            o = len(verts)
            verts += [(x, y, max(min(z, 0.0), -standoff)) for x, y, z in b.verts]   # squashed between land and substrate
            faces += [tuple(o + i for i in f) for f in b.faces]
            smooth += b.smooth
        pieces.append(Piece("balls", BodyMesh(verts, faces, smooth), "ball"))
    return Assembly(pieces)


def molded_tabs(l: float, w: float, h: float, standoff: float, pads: list, *, draft: bool, role: str = "mold") -> Assembly:
    """Moulded two-terminal body (SOD / SMA diode, shielded power inductor): flat terminals under the body that
    bend up its end faces (to 45 % of H); the body rests on them (``standoff`` = sheet thickness)."""
    if draft:
        shape = BodyShape(l, w, h, r_corner=min(0.04 * min(l, w), 0.12), r_top=min(0.05 * h, 0.06), draft_deg=6.0, parting=0.3)
    else:
        r = edge_radius((l, w, h), 0.05, 0.15)
        shape = BodyShape(l, w, h, r_corner=r, r_top=r * 0.6)
    pieces = [Piece("body", shape_at(shape), role)]
    t = max(standoff, 0.08)
    contacts: list[dict[str, float] | None] = []
    for k, pad in enumerate(pads):
        px, py, pw, ph = pad
        sx = 1.0 if px > 0 else -1.0
        tw = min(0.75 * w, 0.85 * ph)
        x_in, x_end = sx * (l / 2 - min(0.3 * l, 0.6 * pw)), sx * (l / 2 + t)
        pieces.append(Piece(f"term_{k}_b", box(min(x_in, x_end), max(x_in, x_end), py - tw / 2, py + tw / 2, -standoff, 0.0), "metal"))
        x0, x1 = sx * (l / 2 - 0.002), sx * (l / 2 + t)
        pieces.append(Piece(f"term_{k}_e", box(min(x0, x1), max(x0, x1), py - tw / 2, py + tw / 2, -standoff, 0.45 * h), "metal"))
        contacts.append(_contact(sx * (l / 2 + t / 2), py, t / 2 + 0.05, tw / 2, (sx, 0.0)))
    return Assembly(pieces, contacts, 0.35 * h)


def electrolytic(l: float, w: float, h: float, pads: list) -> Assembly:
    """SMD aluminium electrolytic: black base plate (16 % of H, + corners chamfered) on two flat terminals, the can
    (diameter 95.5 % of the base: 6.3 on 6.6 mm) with a rolled bead just above the base and the curled rim on top."""
    t = 0.1
    base_h = 0.16 * h
    base = can_base_shape(l, w, base_h, plus_corners(pads, l, w))
    dia = 0.955 * min(l, w)
    can_h = h - base_h - t
    can = BodyShape(dia, dia, can_h, r_corner=dia / 2, r_top=min(0.04 * dia, 0.3), corner_segments=16, top_segments=4,
                    grooves=((min(0.12 * can_h, 0.8), 0.015 * dia + 0.03, 0.12),))      # bead ~0.12 mm deep on 6.3 mm
    pieces = [Piece("body_base", shape_at(base, z0=t), "base"), Piece("body_can", shape_at(can, z0=t + base_h), "alu")]
    contacts: list[dict[str, float] | None] = []
    for k, (px, py, pw, ph) in enumerate(pads):
        sx = 1.0 if px > 0 else -1.0
        tw = min(0.1 * min(l, w) + 0.1, 0.7 * ph)
        x0, x1 = sx * 0.15 * l, sx * (l / 2 + min(0.35, 0.3 * pw))
        pieces.append(Piece(f"term_{k}", box(min(x0, x1), max(x0, x1), py - tw / 2, py + tw / 2, 0.0, t), "metal"))
        xo = sx * (l / 2 + min(0.35, 0.3 * pw) / 2)
        contacts.append(_contact(xo, py, min(0.35, 0.3 * pw) / 2 + 0.1, tw / 2, (sx, 0.0)))
    return Assembly(pieces, contacts, 0.25)


# --------------------------------------------------------------------------- connectors (catalog ``kind``)
#
# Built in a canonical frame: x along the mating edge (length ``e``), y the depth (``d``), the mating face at -y; then
# turned so the mating face looks at ``front`` (the nearest board edge for horizontal / right-angle parts).

def _rot(m: BodyMesh, front: tuple[float, float]) -> BodyMesh:
    fx, fy = front
    if (fx, fy) == (0.0, -1.0):
        f = lambda x, y: (x, y)          # noqa: E731
    elif (fx, fy) == (0.0, 1.0):
        f = lambda x, y: (-x, -y)        # noqa: E731
    elif (fx, fy) == (1.0, 0.0):
        f = lambda x, y: (-y, x)         # noqa: E731
    else:
        f = lambda x, y: (y, -x)         # noqa: E731
    return BodyMesh([(*f(x, y), z) for x, y, z in m.verts], list(m.faces), list(m.smooth))


def _posts(e: float, d: float, n: int, rows: int, pitch: float, z0: float, z1: float, side: float) -> list[BodyMesh]:
    """Square pins (``side``) in *rows* rows of *n* along x, centred, from z0 to z1."""
    out = []
    for r in range(rows):
        y = (r - (rows - 1) / 2) * pitch
        for i in range(n):
            x = (i - (n - 1) / 2) * pitch
            out.append(box(x - side / 2, x + side / 2, y - side / 2, y + side / 2, z0, z1))
    return out


def _shroud(e: float, d: float, h: float, floor: float, wall: float) -> list[BodyMesh]:
    """Open-top housing: floor + four walls."""
    return [box(-e / 2, e / 2, -d / 2, -d / 2 + wall, 0, h), box(-e / 2, e / 2, d / 2 - wall, d / 2, 0, h),
            box(-e / 2, -e / 2 + wall, -d / 2 + wall, d / 2 - wall, 0, h), box(e / 2 - wall, e / 2, -d / 2 + wall, d / 2 - wall, 0, h)], \
        box(-e / 2 + wall, e / 2 - wall, -d / 2 + wall, d / 2 - wall, 0, floor)


def _plate_with_holes(x0: float, x1: float, z0: float, z1: float, y0: float, y1: float,
                      holes: list[tuple[float, float, float, float]]) -> list[BodyMesh]:
    """A plate (x0..x1, z0..z1, thickness y0..y1) with rectangular holes [(hx0, hx1, hz0, hz1)]: the grid cells
    between the holes' edges that lie in no hole, as boxes."""
    xs = sorted({x0, x1, *[min(max(v, x0), x1) for h in holes for v in h[:2]]})
    zs = sorted({z0, z1, *[min(max(v, z0), z1) for h in holes for v in h[2:]]})
    out = []
    for xa, xb in zip(xs[:-1], xs[1:]):
        for za, zb in zip(zs[:-1], zs[1:]):
            cx, cz = (xa + xb) / 2, (za + zb) / 2
            if xb - xa < 1e-6 or zb - za < 1e-6 or any(h[0] < cx < h[1] and h[2] < cz < h[3] for h in holes):
                continue
            out.append(box(xa, xb, y0, y1, za, zb))
    return out


def _ported_shell(e: float, d: float, h: float, t: float, ports: list[tuple[float, float, float, float]],
                  depth: float) -> tuple[list[BodyMesh], list[BodyMesh]]:
    """Box shell (sheet *t*) with rectangular *ports* in its -y face, and the insert inside it: solid behind the ports
    (from *depth* on), around them in front. Returns (shell plates, insert blocks)."""
    shell = _plate_with_holes(-e / 2, e / 2, 0.0, h, -d / 2, -d / 2 + t, ports)
    shell += [box(-e / 2, e / 2, d / 2 - t, d / 2, 0.0, h), box(-e / 2, e / 2, -d / 2, d / 2, h - t, h),
              box(-e / 2, -e / 2 + t, -d / 2, d / 2, 0.0, h - t), box(e / 2 - t, e / 2, -d / 2, d / 2, 0.0, h - t)]
    yi0, yc, yi1 = -d / 2 + t, -d / 2 + min(depth, d - 3 * t), d / 2 - t
    ins = [box(-e / 2 + t, e / 2 - t, yc, yi1, 0.0, h - t)]
    ins += _plate_with_holes(-e / 2 + t, e / 2 - t, 0.0, h - t, yi0, yc, ports)
    return shell, ins


#: kinds whose "vertical" catalog orientation means mated from above (others keep their mating face at a board edge)
TOP_MATING = ("wire_to_board", "terminal_strip", "power_header", "pin_header", "usb_b")


#: D-sub (DE-9) flange width, jackscrew spacing, hex standoff across flats and length (mm)
DSUB_FLANGE_MM, DSUB_BOLT_PITCH_MM, DSUB_BOLT_AF_MM, DSUB_BOLT_LEN_MM = 30.8, 25.0, 4.8, 4.8


def hex_prism(cx: float, cz: float, y0: float, y1: float, across_flats: float, sides: int = 6) -> BodyMesh:
    """A prism (hex nut / standoff, or a round hole with many *sides*) along y from y0 to y1, centred at (cx, cz)."""
    r = across_flats / 2 / math.cos(math.pi / sides)
    ring = [(cx + r * math.cos(math.pi / sides + 2 * math.pi * i / sides), cz + r * math.sin(math.pi / sides + 2 * math.pi * i / sides))
            for i in range(sides)]
    verts = [(x, y0, z) for x, z in ring] + [(x, y1, z) for x, z in ring]
    faces = [(i, (i + 1) % sides, sides + (i + 1) % sides, sides + i) for i in range(sides)]
    faces += [tuple(range(sides)), tuple(reversed(range(sides, 2 * sides)))]
    m = BodyMesh(verts, faces, [False] * len(faces))
    if _signed_volume(m) < 0:
        m.faces = [tuple(reversed(f)) for f in m.faces]
    return m


def connector(kind: str, orientation: str, l: float, w: float, h: float, *, pins: int, rows: int, pitch: float,
              total_h: float | None, front: tuple[float, float], cell: bool = False, top_cover: str | None = None) -> Assembly:
    """Connector / socket body from the part catalog's ``kind`` (datasheet height *h* above the board):

    * ``wire_to_board`` / ``terminal_strip`` / ``power_header`` vertical: open-top shrouded housing, pins standing in
      it (JST EH / PH / XH natural nylon, shrouded headers black); right angle: the opening on the mating face.
    * ``pin_header``: black insulator (*h*) with square posts to ``total_h``.
    * ``usb_b`` (vertical) / ``usb_a_dual`` / ``rj45`` / ``card_socket``: nickel shield; the mating opening dark.
    * ``dsub``: black body, nickel flange and D shell on the mating face.
    * ``screw_terminal``: metal block with the screw head; ``battery_holder``: the holder's base, empty as assembled
      (the cell goes in at final assembly); with *cell* (``build_job``: the mount stage is shown, or the catalog says
      ``"cell": true``) also the coin cell and its clip.
    Anything else: a plain housing (``plastic`` black / ``metal`` shield)."""
    vertical = orientation == "vertical" and kind in TOP_MATING
    if vertical:
        front = (0.0, -1.0) if l >= w else (1.0, 0.0)
    e, d = (l, w) if front[1] else (w, l)
    pieces: list[Piece] = []

    def add(name, mesh, role):
        pieces.append(Piece(name, _rot(mesh, front), role))

    side = 0.64 if pitch >= 1.9 else (0.5 if pitch >= 1.4 else 0.32)
    rows = max(1, rows)
    n = max(1, pins // rows)
    if kind in ("wire_to_board", "terminal_strip", "power_header"):
        role = "natural" if kind == "wire_to_board" else "mold"
        wall = min(0.6, 0.14 * min(e, d))
        floor = 0.3 * h
        if vertical:
            walls, fl = _shroud(e, d, h, floor, wall)
            # the front wall with the polarisation notch at its top centre (a wire-to-board header's key slot)
            nw = min(0.5 * pitch, 0.15 * e)
            walls[0:1] = _plate_with_holes(-e / 2, e / 2, 0.0, h, -d / 2, -d / 2 + wall, [(-nw / 2, nw / 2, h - min(0.15 * h, 0.8), h + 1.0)])
            for k, m in enumerate(walls):
                add(f"body_wall{k}", m, role)
            # keys: ribs on the inside of the back wall between the pin pairs
            for i in range(max(0, n - 1)):
                x = (i - (n - 2) / 2) * pitch
                add(f"body_key{i}", box(x - 0.2, x + 0.2, d / 2 - wall - 0.35, d / 2 - wall, floor, h - 0.3), role)
            add("cavity", fl, role)
            for k, m in enumerate(_posts(e, d, n, rows, pitch, floor, h - 0.7, side)):
                add(f"pin_{k}", m, "metal")
        else:
            # right angle: closed top, open towards -y (the mating face), pins horizontal in the opening
            add("body_back", box(-e / 2, e / 2, d / 2 - 0.35 * d, d / 2, 0, h), role)
            add("body_top", box(-e / 2, e / 2, -d / 2, d / 2, h - wall, h), role)
            add("body_floor", box(-e / 2, e / 2, -d / 2, d / 2, 0, floor), role)
            add("body_end0", box(-e / 2, -e / 2 + wall, -d / 2, d / 2, 0, h), role)
            add("body_end1", box(e / 2 - wall, e / 2, -d / 2, d / 2, 0, h), role)
            zc = (floor + h - wall) / 2
            for i in range(n):
                x = (i - (n - 1) / 2) * pitch
                add(f"pin_{i}", box(x - side / 2, x + side / 2, -d / 2 + 0.6, d / 2 - 0.35 * d, zc - side / 2, zc + side / 2), "metal")
    elif kind == "pin_header":
        add("body", box(-e / 2, e / 2, -d / 2, d / 2, 0, h), "mold")
        top = max(total_h or 0.0, h + 3.0)
        for k, m in enumerate(_posts(e, d, n, rows, pitch, h - 0.01, top, side)):
            add(f"pin_{k}", m, "metal")
    elif kind == "usb_b" and vertical:
        # mated from above: shell walls, the opening in the top, the square plastic insert with contacts inside
        t = 0.3
        ow, od = 0.72 * e, 0.62 * d
        for k, m in enumerate(_plate_with_holes(-e / 2, e / 2, -d / 2, d / 2, h - t, h, [(-ow / 2, ow / 2, -od / 2, od / 2)])):
            # the helper builds plates in x-z; here the plate lies in x-y at the top: swap its y and z
            add(f"body_shell_top{k}", BodyMesh([(x, z, y) for x, y, z in m.verts], [tuple(reversed(f)) for f in m.faces], m.smooth), "lid")
        for k, m in enumerate([box(-e / 2, e / 2, -d / 2, -d / 2 + t, 0, h - t), box(-e / 2, e / 2, d / 2 - t, d / 2, 0, h - t),
                               box(-e / 2, -e / 2 + t, -d / 2 + t, d / 2 - t, 0, h - t), box(e / 2 - t, e / 2, -d / 2 + t, d / 2 - t, 0, h - t)]):
            add(f"body_shell{k}", m, "lid")
        add("insert", box(-e / 2 + t, e / 2 - t, -d / 2 + t, d / 2 - t, 0, h - 2.6), "insert")
        tw, td = 0.5 * ow, 0.45 * od
        add("tongue", box(-tw / 2, tw / 2, -td / 2, td / 2, h - 2.6, h - 0.5), "mold")
        for i in range(2):
            for sgn in (-1, 1):
                x = (i - 0.5) * 0.45 * tw
                add(f"pin_c{i}{sgn}", box(x - 0.25, x + 0.25, sgn * td / 2 - (0.05 if sgn > 0 else 0.0), sgn * td / 2 + (0.05 if sgn < 0 else 0.0) + 0.0,
                                         h - 2.4, h - 0.7), "metal")
    elif kind in ("usb_b", "usb_a_dual", "rj45", "card_socket"):
        # mated from the board edge: a sheet shell with the port openings, the insert behind them, tongues / contacts
        t = 0.3 if kind != "rj45" else 0.25
        if kind == "usb_a_dual":
            pw, ph = min(13.1, 0.86 * e), min(5.7, 0.4 * h)
            gap = max(0.12 * h - 0.0, 1.0)
            z0 = (h - 2 * ph - gap) / 2
            ports = [(-pw / 2, pw / 2, z0 + k * (ph + gap), z0 + k * (ph + gap) + ph) for k in range(2)]
            depth = 0.75 * d
        elif kind == "rj45":
            pw, ph = min(11.7, 0.72 * e), min(8.0, 0.58 * h)
            ports = [(-pw / 2, pw / 2, 0.12 * h, 0.12 * h + ph)]
            depth = 0.6 * d
        elif kind == "card_socket":
            ports = [(-0.42 * e, 0.42 * e, 0.25 * h, 0.75 * h)]
            depth = 0.85 * d
        else:
            pw, ph = 0.62 * e, 0.55 * h
            ports = [(-pw / 2, pw / 2, 0.22 * h, 0.22 * h + ph)]
            depth = 0.7 * d
        shell, ins = _ported_shell(e, d, h, t, ports, depth)
        shell_role = "natural" if kind == "rj45" and False else "lid"
        for k, m in enumerate(shell):
            add(f"body_shell{k}", m, shell_role)
        for k, m in enumerate(ins):
            add(f"insert_{k}", m, "insert")
        yb = -d / 2 + depth                     # back of the ports
        for k, (hx0, hx1, hz0, hz1) in enumerate(ports):
            pw_, ph_ = hx1 - hx0, hz1 - hz0
            if kind == "usb_a_dual":
                # the plastic tongue in the upper half of the port, four contacts on its underside
                tz0, tz1 = hz0 + 0.42 * ph_, hz0 + 0.68 * ph_
                add(f"tongue_{k}", box(hx0 + 0.08 * pw_, hx1 - 0.08 * pw_, -d / 2 + t + 0.8, yb, tz0, tz1), "white")
                # two spring latches pressed into the port from its top sheet
                for j, fx in enumerate((0.28, 0.72)):
                    lx = hx0 + fx * pw_
                    m = strip_mesh([(0.0, hz1 - t / 2), (1.6, hz1 - 0.45)], t * 0.8, 1.0, (lx, -d / 2 + t + 0.6), (0.0, 1.0))
                    add(f"latch_{k}_{j}", m, "lid")
                for i in range(4):
                    x = hx0 + pw_ * (0.2 + 0.2 * i)
                    add(f"pin_{k}_{i}", box(x - 0.5, x + 0.5, -d / 2 + t + 1.5, yb, tz0 - 0.06, tz0), "metal")
            elif kind == "rj45":
                # eight spring contacts slanting down from the top of the port, the latch slot below
                for i in range(8):
                    x = hx0 + pw_ * (0.15 + 0.7 * i / 7)
                    zt = hz1 - 0.3
                    m = strip_mesh([(-0.5, zt), (0.35 * depth, zt - 0.35 * ph_)], 0.12, 0.35, (x, yb), (0.0, -1.0))
                    add(f"pin_{i}", m, "metal")
                add("latch", box(-0.2 * pw_, 0.2 * pw_, -d / 2 - 0.01, -d / 2 + t + 0.5, hz0 - 0.9, hz0 + 0.01), "insert")
                for i, sx in enumerate((-1, 1)):
                    add(f"led_{i}", box(sx * 0.4 * e - 0.8, sx * 0.4 * e + 0.8, -d / 2 - 0.03, -d / 2 + t, hz1 + 0.06 * h, hz1 + 0.06 * h + 1.0),
                        "natural")
            elif kind == "card_socket":
                for i in range(8):
                    x = hx0 + pw_ * (0.1 + 0.8 * i / 7)
                    add(f"pin_{i}", box(x - 0.3, x + 0.3, -d / 2 + 0.3 * depth, yb, hz0, hz0 + 0.08), "metal")
            else:
                tw = 0.6 * pw_
                add(f"tongue_{k}", box(-tw / 2, tw / 2, -d / 2 + t + 1.0, yb, hz0 + 0.3 * ph_, hz0 + 0.7 * ph_), "mold")
    elif kind == "dsub":
        # body behind a nickel flange; the D shell sticks out of the mating face, hollow, with the insert and its 9
        # contact holes (female) recessed in it
        add("body", box(-e / 2, e / 2, -d / 2 + 1.0, d / 2, 0, 0.75 * h), "mold")
        if top_cover == "metal":
            # some makers wrap the body's top in a tin-plated sheet (with the print), others leave the black moulding
            add("cover", box(-e / 2, e / 2, -d / 2 + 1.0, d / 2, 0.75 * h, 0.75 * h + 0.3), "lid")
        # the nickel flange (DE-9: 30.8 mm wide, the jackscrew holes 25 mm apart) with the two hex standoffs
        fw = max(e, DSUB_FLANGE_MM)
        add("body_flange", box(-fw / 2, fw / 2, -d / 2 + 0.6, -d / 2 + 1.0, 0, h), "lid")
        for k, sx in enumerate((-1, 1)):
            hx = sx * DSUB_BOLT_PITCH_MM / 2
            add(f"bolt_{k}", hex_prism(hx, 0.5 * h, -d / 2 + 0.6 - DSUB_BOLT_LEN_MM, -d / 2 + 0.6, DSUB_BOLT_AF_MM), "lid")
            add(f"bolt_hole_{k}", hex_prism(hx, 0.5 * h, -d / 2 + 0.6 - DSUB_BOLT_LEN_MM - 0.02, -d / 2 + 0.6 - DSUB_BOLT_LEN_MM + 1.5,
                                            1.6, sides=12), "insert")
        sw, sh, depth, t = 0.72 * e, 0.42 * h, 6.0, 0.3
        zc = 0.55 * h
        yf = -d / 2 + 0.6
        for k, m in enumerate([box(-sw / 2, sw / 2, yf - depth, yf, zc + sh / 2 - t, zc + sh / 2),
                               box(-sw / 2 * 0.88, sw / 2 * 0.88, yf - depth, yf, zc - sh / 2, zc - sh / 2 + t),
                               box(-sw / 2, -sw / 2 + t, yf - depth, yf, zc - sh / 2, zc + sh / 2),
                               box(sw / 2 - t, sw / 2, yf - depth, yf, zc - sh / 2, zc + sh / 2)]):
            add(f"shell_{k}", m, "lid")
        yi = yf - depth + 1.0                   # front face of the insert, recessed 1 mm in the shell
        add("insert", box(-sw / 2 + t, sw / 2 - t, yi, yf, zc - sh / 2 + t, zc + sh / 2 - t), "mold")
        pitch = 2.77

        def disc(x, z, dia, y0, y1):
            m = shape_at(BodyShape(dia, dia, y1 - y0, r_corner=dia / 2, r_top=0.0, corner_segments=4, top_segments=1))
            # a cylinder along z: lay it along y (front at y0)
            return BodyMesh([(x + vx, y1 - vz, z + vy) for vx, vy, vz in m.verts], m.faces, m.smooth)
        for row, (n, dz) in enumerate(((5, 0.25 * sh), (4, -0.25 * sh))):
            for i in range(n):
                x = (i - (n - 1) / 2) * pitch
                # the socket hole: a black well just behind the insert face, a contact spring deep in it
                add(f"socket_{row}_{i}", disc(x, zc + dz, 1.15, yi - 0.02, yi + 0.6), "insert")
                add(f"pin_{row}_{i}", disc(x, zc + dz, 0.6, yi + 0.6, yi + 1.2), "metal")
    elif kind == "screw_terminal":
        add("body", box(-e / 2, e / 2, -d / 2, d / 2, 0, 0.55 * h), "lid")
        r = 0.32 * min(e, d)
        add("screw", shape_at(BodyShape(2 * r, 2 * r, 0.45 * h, r_corner=r, r_top=0.15 * r, corner_segments=8, top_segments=2),
                              z0=0.55 * h), "lid")
    elif kind == "battery_holder":
        add("body", box(-e / 2, e / 2, -d / 2, d / 2, 0, 0.3 * h), "mold")
        if not cell:
            return Assembly(pieces)
        r = 0.42 * min(e, d)
        add("cell", shape_at(BodyShape(2 * r, 2 * r, 0.5 * h, r_corner=r, r_top=0.03, corner_segments=12, top_segments=1),
                             z0=0.3 * h), "alu")
        add("clip", box(-0.15 * e, 0.15 * e, -d / 2, d / 2, 0.8 * h, h), "lid")
    else:
        return Assembly([Piece("body", shape_at(molded_shape(l, w, h, leads=False)), "mold", photo_sides=True)])
    return Assembly(pieces)


# --------------------------------------------------------------------------- through-hole joints (top side)

def revolve(profile: list[tuple[float, float]], cx: float, cy: float, seg: int = 24) -> BodyMesh:
    """Surface of revolution of *profile* [(rho, z)] (outer to inner) about the vertical axis at (cx, cy); the inner
    end is closed by a flat cap. Smooth-shaded."""
    verts, faces = [], []
    n = len(profile)
    for rho, z in profile:
        verts += [(cx + rho * math.cos(2 * math.pi * i / seg), cy + rho * math.sin(2 * math.pi * i / seg), z) for i in range(seg)]
    for k in range(n - 1):
        a, b = k * seg, (k + 1) * seg
        faces += [(a + i, a + (i + 1) % seg, b + (i + 1) % seg, b + i) for i in range(seg)]
    last = (n - 1) * seg
    faces.append(tuple(range(last, last + seg)))
    return BodyMesh(verts, faces, [True] * (len(faces) - 1) + [False])


def tht_joint(x: float, y: float, r_hole: float) -> list[Piece]:
    """Top side of a soldered through-hole pin: the solder wicked up the barrel as a concave meniscus from the copper
    ring (1.55 r) to the pin, the square-ish pin (0.6 r, at most 0.32 mm) standing out of it."""
    r_ring, r_pin = 1.55 * r_hole, min(0.6 * r_hole, 0.32)
    hm = 0.45 * r_hole
    prof = [(r_pin + (r_ring - r_pin) * (1 - k / 8), 0.002 + hm * (k / 8) ** 2) for k in range(9)]
    pin = shape_at(BodyShape(2 * r_pin, 2 * r_pin, hm + 0.6, r_corner=0.25 * r_pin, r_top=0.15 * r_pin, corner_segments=2,
                             top_segments=2), x, y)
    return [Piece("tht_solder", revolve(prof, x, y), "metal"), Piece("tht_pin", pin, "metal")]
