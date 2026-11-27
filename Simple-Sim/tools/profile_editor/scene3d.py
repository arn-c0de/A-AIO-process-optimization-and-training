"""Tk-free scene model of the profile editor's 3D preview: box meshes, the orbit camera basis, pad layout and
hit testing. :class:`tools.profile_editor.preview_3d.Preview3D` draws and edits these on a Tk canvas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from simple_sim.board.footprints import footprint_pads

__all__ = ["LAYOUT_CANVAS", "BoxMesh", "Face", "GeometryUpdates", "Vec3", "box_faces", "box_mesh", "camera_basis",
           "cross", "normalize", "pad_layout", "point_in_poly"]


Vec3 = tuple[float, float, float]
Face = tuple[Vec3, Vec3, Vec3, Vec3]
GeometryUpdates = dict[tuple[str, ...], Any]


@dataclass
class BoxMesh:
    name: str
    center: Vec3
    size: Vec3
    movable: bool
    faces: list[Face]
    color: str
    outline: str


#: Side of the virtual image the shared pad geometry is laid out in (raw profile pixels, centred).
LAYOUT_CANVAS = 4000


def pad_layout(footprint: str, pad_w: float, pad_h: float, pad_sx: float, pad_sy: float,
               body_len_x: float, body_len_y: float, *, nominal_raw: dict[str, float] | None = None,
               scale: float = 1.0) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """``((x, y), (size_x, size_y))`` per pad in pad-number order, same layout as the 2D backends.

    chip_2pad, sot23, soic_16 and qfn keep the editor's own layout (dragging their pads is translated back
    into ``geometry_ranges`` by ``_emit_geometry_updates_from_scene``). Every other footprint (tssop, sot_6,
    chip_array_8, tht_pin_N, ...) takes the shared geometry of :func:`simple_sim.board.footprints.footprint_pads`
    from *nominal_raw* (profile pixels) scaled by *scale*; without *nominal_raw* it falls back to two pads.
    """
    if nominal_raw is not None and footprint != "chip_2pad" and footprint not in ("soic_16", "sot23") \
            and not footprint.startswith("qfn"):
        c = LAYOUT_CANVAS // 2
        return [(((p.center[0] - c) * scale, (p.center[1] - c) * scale), (p.w * scale, p.h * scale))
                for p in footprint_pads(footprint, nominal_raw, (LAYOUT_CANVAS, LAYOUT_CANVAS))]
    if footprint == "soic_16":
        # 8 pads on the left + 8 pads on the right (2D layout, see opencv2d.geometry.pad_positions).
        # Keep pads distributed along package length and outside body width.
        row_x = max(pad_sx * 0.5, (body_len_x * 0.5) + (pad_h * 0.65))
        y_span = max(pad_w * 7.0, body_len_y * 0.82)
        pitch = y_span / 7.0
        y0 = -y_span * 0.5
        return [((x, y0 + i * pitch), (pad_h, pad_w)) for i in range(8) for x in (-row_x, row_x)]
    if footprint.startswith("qfn"):
        return [((sx * pad_sx * 0.45, sy * pad_sy * 0.45), (pad_w, pad_h)) for sy in (-1, 1) for sx in (-1, 1)]
    if footprint == "sot23":
        # Pads 1/2 on the left column (y = -/+ pad_spacing_y/2), pad 3 on the right at y = 0. pad_spacing is
        # the left/right column distance; _emit_geometry_updates_from_scene inverts this.
        # The 2D layout draws the two left pads at 0.8x pad_height.
        return [((-pad_sx * 0.5, -pad_sy * 0.5), (pad_w, pad_h * 0.8)),
                ((-pad_sx * 0.5, pad_sy * 0.5), (pad_w, pad_h * 0.8)),
                ((pad_sx * 0.5, 0.0), (pad_w, pad_h))]
    return [((-pad_sx * 0.5, 0.0), (pad_w, pad_h)), ((pad_sx * 0.5, 0.0), (pad_w, pad_h))]


def box_faces(center: Vec3, size: Vec3) -> list[Face]:
    cx, cy, cz = center
    sx, sy, sz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5

    p000 = (cx - sx, cy - sy, cz - sz)
    p001 = (cx - sx, cy - sy, cz + sz)
    p010 = (cx - sx, cy + sy, cz - sz)
    p011 = (cx - sx, cy + sy, cz + sz)
    p100 = (cx + sx, cy - sy, cz - sz)
    p101 = (cx + sx, cy - sy, cz + sz)
    p110 = (cx + sx, cy + sy, cz - sz)
    p111 = (cx + sx, cy + sy, cz + sz)

    return [
        (p001, p101, p111, p011),
        (p000, p010, p110, p100),
        (p000, p001, p011, p010),
        (p100, p110, p111, p101),
        (p010, p011, p111, p110),
        (p000, p100, p101, p001),
    ]


def box_mesh(name: str, center: Vec3, size: Vec3, color: str, outline: str, movable: bool) -> BoxMesh:
    return BoxMesh(
        name=name,
        center=center,
        size=size,
        movable=movable,
        faces=box_faces(center, size),
        color=color,
        outline=outline,
    )


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-8:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def camera_basis(yaw: float, pitch: float) -> tuple[Vec3, Vec3, Vec3]:
    """``(forward, right, up)`` of an orbit camera at *yaw* / *pitch* (radians) looking at the origin."""
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    forward = normalize((sy * cp, cy * cp, -sp))
    world_up = (0.0, 0.0, 1.0)
    right = normalize(cross(world_up, forward))
    if abs(right[0]) + abs(right[1]) + abs(right[2]) < 1e-6:
        right = normalize(cross((0.0, 1.0, 0.0), forward))
    up = normalize(cross(forward, right))
    return forward, right, up


def point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside
