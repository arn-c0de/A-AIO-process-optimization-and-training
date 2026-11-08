"""Blender batch renderer for Simple-Sim (Cycles or EEVEE).

Run via (normally through ``simple_sim.generators.blender3d.render_blender_batch``):
  blender --background --factory-startup --python render_batch.py -- --jobs jobs.jsonl --out_root <dir> --samples 64

Each JSONL job is a ``PreparedJob`` of ``simple_sim.blender.job_schema`` (the stdlib-only contract this
script shares with the library): the sample fields (``image_path``, ``seed``, ``mm_per_px``,
``roi_width_px``, ``roi_height_px``, ``component_height_mm``, ``augment``, ``render_3d``) plus the
``board`` scene added by the bake (``blender3d/bake.py``): texture, scale, camera pose, family,
pads, thermal pad, excess pads, solder balls, the body and ``part`` (how to build the part: the
family's row of ``board.footprints.FAMILIES``; ``part["body"] == "tht"`` builds a through-hole pin row
from ``board["tht"]``: pins, holes, menisci, a bridge bar).

Layout values are camera pixels relative to the ROI centre, image convention (x right, y down),
identical to the 2D renderer. The scene (1 Blender unit = 1 mm): board plane with the baked
texture, a solder fillet mesh per pad, the part (body, end caps or gull-wing leads, pin-1 mark,
marking text), an AOI ring light and an orthographic camera that frames exactly the ROI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

# The job contract lives next to this script in the simple_sim package, which Blender's Python does
# not have on its path (stdlib-only: nothing else of the package is imported).
_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
from simple_sim.blender.job_schema import (BOARD_SCALE, DEFAULT_CAMERA_POSE, DEFAULT_COMPONENT_HEIGHT_MM,  # noqa: E402
                                           DEFAULT_DEVICE, DEFAULT_ENGINE, BoardScene, PreparedJob)

PadMm = tuple[float, float, float, float]  # (cx, cy, w, h) in mm, Blender axes

# Per-family decisions (height ratio, standoff, end caps, leads, marking) come from the scene's ``part``
# (``board.footprints.PartFamily.blender``, baked by ``photoreal.scene.render_board``; D35).
# Reflowed paste dome |u|^3 + |v|^3; the 2D meniscus uses 2.5 (``photoreal/meniscus.py``): known difference (X12).
_PASTE_DOME_EXPONENT = 3
RING_STRENGTH, TOP_STRENGTH = 6.0, 0.9  # emission strengths, calibrated to the baked board colours


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="Render engine (BLENDER_EEVEE or CYCLES)")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- materials

def _material(name: str, color, *, roughness: float, metallic: float = 0.0, bump: float = 0.0,
              emission: float = 0.0) -> bpy.types.Material:
    """Principled material; *bump* adds fine noise relief (mold speckle, solder grain)."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*[float(c) for c in color[:3]], 1.0)
    bsdf.inputs["Roughness"].default_value = float(roughness)
    bsdf.inputs["Metallic"].default_value = float(metallic)
    if emission:
        bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = float(emission)
    if bump:
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 120.0
        noise.inputs["Detail"].default_value = 4.0
        bmp = nodes.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = float(bump)
        links.new(noise.outputs["Fac"], bmp.inputs["Height"])
        links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _board_material(texture: str) -> bpy.types.Material:
    """Solder mask carrying the baked 2D board (mask, copper, silkscreen, pad finish).

    The baked colours come from real photos and already include the mask's sheen, so the specular
    part is kept low; with the default ring light the board renders at its baked colours.
    """
    mat = _material("mat_board", (1, 1, 1), roughness=0.4)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.1
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(texture)
    tex.interpolation = "Cubic"
    links.new(tex.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def _cfg_material(name: str, cfg: dict[str, Any], default_color, default_roughness: float, default_metallic: float,
                  bump: float = 0.0) -> bpy.types.Material:
    color = cfg.get("base_color", default_color)
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        color = default_color
    return _material(name, color, roughness=float(cfg.get("roughness", default_roughness)),
                     metallic=float(cfg.get("metallic", default_metallic)), bump=bump)


# --------------------------------------------------------------------------- geometry helpers

def _mm(px: float, mmpp: float) -> float:
    """Scene length in camera pixels -> mm (1 Blender unit)."""
    return float(px) * mmpp


def _mm_xy(x: float, y: float, mmpp: float) -> tuple[float, float]:
    """Scene point (camera px relative to the ROI centre, image y down) -> Blender (x, y up) in mm."""
    return float(x) * mmpp, -float(y) * mmpp


def _mesh_object(name: str, verts, faces, mat: bpy.types.Material, *, smooth: bool = True) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def _box(name: str, size, loc, mat: bpy.types.Material, *, parent=None, bevel: float = 0.0) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = tuple(float(s) for s in size)
    obj.data.materials.append(mat)
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        bpy.ops.object.shade_smooth()
    if parent is not None:
        obj.parent = parent
    return obj


def _box_between(name: str, p0: Vector, p1: Vector, width: float, thickness: float, mat) -> bpy.types.Object:
    """Box whose length runs from p0 to p1 (lead segments)."""
    d = p1 - p0
    obj = _box(name, (d.length + thickness * 0.5, width, thickness), (p0 + p1) / 2.0, mat)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = d.normalized().to_track_quat("X", "Z")
    return obj


def _rect_distance(x: float, y: float, rect: dict[str, float]) -> float:
    """Chebyshev-style signed distance of (x, y) to a rotated rectangle (< 0 inside)."""
    dx, dy = x - rect["cx"], y - rect["cy"]
    c, s = math.cos(rect["angle"]), math.sin(rect["angle"])
    return max(abs(dx * c + dy * s) - rect["hl"], abs(-dx * s + dy * c) - rect["hw"])


def _fillet(name: str, pad, contact: dict[str, float] | None, h_term: float, big: bool, mat,
            excess: bool = False) -> bpy.types.Object | None:
    """Solder as a height-field mesh over the pad.

    With a part terminal on the pad: concave fillet rising towards the terminal. Without one
    (MISSING, raised tombstone end, part shifted off the pad): reflowed paste dome; large thermal
    pads get window-pane islands like the stencil prints them.
    """
    if contact is not None and contact.get("tht"):
        return _tht_joint(name, pad, contact, h_term, mat, excess)
    cx, cy, pw, ph = pad
    n = 48 if big else 24
    xs = [cx - pw / 2 + pw * i / n for i in range(n + 1)]
    ys = [cy - ph / 2 + ph * j / n for j in range(n + 1)]
    rim_w = 0.12 * min(pw, ph)
    dist = [[_rect_distance(x, y, contact) if contact else 1.0 for x in xs] for y in ys]
    touched = contact is not None and min(min(row) for row in dist) <= 0.0
    dmax = max(max(row) for row in dist) if touched else 1.0
    nx = ny = 1
    if big and not touched:
        nx, ny = max(2, round(pw / (min(pw, ph) / 3))), max(2, round(ph / (min(pw, ph) / 3)))

    dome_h = 0.1 * min(pw / nx, ph / ny)  # reflowed paste height scales with the paste deposit, as in 2D
    verts = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            rim = min(1.0, min(x - xs[0], xs[-1] - x, y - ys[0], ys[-1] - y) / rim_w)
            if touched:
                t = 1.0 - max(0.0, dist[j][i]) / max(dmax, 1e-6)
                z = h_term * (1.5 * t ** 0.5 if excess else t ** 1.8)  # too much solder: convex bulge
            else:
                u = ((i / n * nx) % 1.0) * 2 - 1
                v = ((j / n * ny) % 1.0) * 2 - 1
                z = dome_h * max(0.0, 1.0 - (abs(u) ** _PASTE_DOME_EXPONENT + abs(v) ** _PASTE_DOME_EXPONENT)
                                   * (0.85 if nx * ny == 1 else 1.2))
            verts.append((x, y, 0.001 + z * rim))
    faces = [(j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i)
             for j in range(n) for i in range(n)]
    return _mesh_object(name, verts, faces, mat)


def _text(name: str, text: str, size: float, loc, mat, parent) -> None:
    bpy.ops.object.text_add(location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.body = text
    obj.data.size = size
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.materials.append(mat)
    obj.parent = parent


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


# --------------------------------------------------------------------------- part

def part_params(pads: list[PadMm], board: BoardScene, mmpp: float, comp_h_mm: float,
                rng: random.Random) -> dict[str, Any] | None:
    """Every per-part number the geometry is built from (pure: no ``bpy``; ``None`` for MISSING).

    The decisions come from ``board["part"]`` (:class:`~simple_sim.blender.job_schema.ScenePart`);
    the family name is only a label. Consumes *rng* exactly as the geometry always did (the marking
    text), so it runs before the meshes are built. Lengths in mm: ``length``, ``width``, ``height``,
    ``standoff``, ``cap`` (end cap length or ``None``), ``marking`` (texts, one per line) and
    ``marking_size``, ``leads`` (``[{"width", "thickness"}]`` per pad, gull-wing parts only),
    ``contacts`` (``"body"``: the solder wets the body / terminal, ``"leads"``: the lead feet,
    ``"none"``: paste only) and ``solder_height``.
    """
    body, part = board["body"], board["part"]
    if body is None:
        return None
    L, W = _mm(body["length"], mmpp), _mm(body["width"], mmpp)
    H = min(comp_h_mm, part["height_ratio"] * min(L, W))
    chip = part["body"] == "chip"
    p: dict[str, Any] = {"length": L, "width": W, "height": H, "standoff": part["standoff"] * H,
                         "body_material": part["body_material"], "end_caps": chip, "cap": None, "marking": [],
                         "marking_size": 0.0, "leads": [], "contacts": "body", "solder_height": 0.0}
    if part["body"] == "tht":
        # Through-hole pin row (solder side): no body, a pin in every plated hole, a conical meniscus around it.
        # Ratios of ``part["pin"]``: pin diameter / pad diameter, protrusion / component_height_mm,
        # meniscus height / pad radius. No rng draw (no marking).
        pin = part["pin"]
        r_pad = min(min(pw, ph) for _, _, pw, ph in pads) / 2.0 if pads else 0.5
        p.update({"contacts": "tht", "pin_radius": 0.5 * float(pin["diameter"]) * 2.0 * r_pad,
                  "pin_height": float(pin["height"]) * comp_h_mm, "solder_height": float(pin["solder_height"]) * r_pad})
        return p
    mark = part["marking"]
    if mark is not None:
        size_w, size_l = mark["size"]
        p["marking_size"] = size_w * W if size_l is None else min(size_w * W, size_l * L)
        if mark["codes"]:
            p["marking"] = [rng.choice(mark["codes"])]
        else:
            lo, hi = mark["length"]
            p["marking"] = ["".join(rng.choice(mark["chars"]) for _ in range(rng.randint(lo, hi)))
                            for _ in range(mark["lines"])]
    if chip:
        p["cap"] = part["cap_len"] * L
        p["solder_height"] = 0.6 * H
    elif part["leads"] and not body.get("is_tombstone"):
        a, b = part["lead_width"]
        t = 0.0
        for _, _, pw, ph in pads:
            lw = min(a * min(pw, ph), b * max(L, W))
            t = 0.25 * lw
            p["leads"].append({"width": lw, "thickness": t})
        p["contacts"], p["solder_height"] = "leads", 0.9 * t
    elif part["leads"]:
        p["contacts"] = "none"  # tombstoned leaded part: leads lifted, paste only
    else:
        p["solder_height"] = 0.3 * H
    return p


def _build_part(pads: list[PadMm], board: BoardScene, mmpp: float, comp_h_mm: float,
                render_3d: dict[str, Any], rng: random.Random) -> tuple[list[bpy.types.Object], list, float]:
    """Body, end caps / leads, pin-1 mark and marking (numbers from :func:`part_params`).

    Returns (objects, per-pad contact rect or None, solder height). The contact is what the solder
    wets: the terminal / body edge for chips and QFN, the lead foot for SOT/SOIC.
    """
    body = board["body"]
    p = part_params(pads, board, mmpp, comp_h_mm, rng)
    if p is None:
        return [], [None] * len(pads), 0.0  # no contacts: every pad gets a paste dome

    mats = render_3d.get("materials") or {}
    L, W, H, standoff = p["length"], p["width"], p["height"], p["standoff"]
    angle = -math.radians(float(body["rotation_deg"]))  # image y-down -> Blender y-up
    cx, cy = _mm_xy(body["center"][0], body["center"][1], mmpp)
    axis = Vector((math.cos(angle), math.sin(angle), 0.0))
    if p["contacts"] == "tht":
        return _build_tht(pads, board, p, mats, (cx, cy), angle)

    root = bpy.data.objects.new("part_root", None)
    bpy.context.collection.objects.link(root)
    contact = {"cx": cx, "cy": cy, "hl": L / 2, "hw": W / 2, "angle": angle}
    offset = 0.0
    if body.get("is_tombstone"):
        sign = float(body.get("raised_sign", 1.0))
        tilt = math.radians(float(body["tilt_deg"]))
        anchor = Vector((cx, cy, 0.0)) - axis * sign * (L / 2)
        root.location = (anchor.x, anchor.y, standoff)
        root.rotation_euler = (0.0, -tilt * sign, angle)
        offset = sign * L / 2  # body extends from the anchored end towards the raised one
        # Footprint seen from above, as in 2D: top face from H*sin behind the anchor to L*cos in front.
        top_len, end_len = L * math.cos(tilt), H * math.sin(tilt)
        mid = anchor + axis * sign * ((top_len - end_len) / 2)
        contact = {"cx": mid.x, "cy": mid.y, "hl": (top_len + end_len) / 2, "hw": W / 2, "angle": angle}
    else:
        root.location = (cx, cy, standoff)
        root.rotation_euler = (0.0, 0.0, angle)

    if p["body_material"] == "ceramic":
        m_body = _material("mat_ceramic", (0.42, 0.26, 0.14), roughness=0.55, bump=0.02)
    else:
        m_body = _cfg_material("mat_body", mats.get("body") or {}, (0.02, 0.02, 0.02), 0.5, 0.0, bump=0.03)
    objs: list[bpy.types.Object] = [root]
    objs.append(_box("body", (L, W, H), (offset, 0.0, H / 2), m_body, parent=root, bevel=0.06 * min(W, H)))

    if p["end_caps"]:
        # Tin plating is satin, not a mirror: it must catch the ring light when it faces the camera.
        m_cap = _material("mat_endcap", (0.75, 0.75, 0.74), roughness=0.45, metallic=1.0, bump=0.05)
        cap = p["cap"]
        for s in (-1, 1):
            # Caps wrap slightly beyond the body, so their end faces never coincide with the body's
            # (an upright tombstone shows the cap's end face to the camera).
            objs.append(_box(f"endcap_{s}", (cap + 0.02 * H, W * 1.02, H * 1.02),
                             (offset + s * ((L - cap) / 2 + 0.01 * H), 0.0, H / 2), m_cap,
                             parent=root, bevel=0.1 * min(W, H)))
        if p["marking"]:
            m_ink = _material("mat_ink", (0.85, 0.85, 0.82), roughness=0.8)
            _text("marking", p["marking"][0], p["marking_size"], (offset, 0.0, H + 0.0005), m_ink, root)
        return objs, [contact] * len(pads), p["solder_height"]

    # ICs: pin-1 dimple and laser marking.
    m_dot = _material("mat_pin1", (0.01, 0.01, 0.01), roughness=0.25)
    bpy.ops.mesh.primitive_cylinder_add(radius=max(0.035 * min(L, W), 0.01), depth=0.0004,
                                        location=(offset - L / 2 + 0.12 * L, W / 2 - 0.15 * W, H + 0.0002))
    dot = bpy.context.active_object
    dot.data.materials.append(m_dot)
    dot.parent = root
    m_laser = _material("mat_laser", (0.22, 0.22, 0.21), roughness=0.9)
    lines, size = len(p["marking"]), p["marking_size"]
    for i, text in enumerate(p["marking"]):
        _text(f"marking_{i}", text, size, (offset, ((lines - 1) / 2 - i) * size * 1.6, H + 0.0005), m_laser, root)

    if p["contacts"] == "leads":
        m_lead = _cfg_material("mat_lead", mats.get("solder") or {}, (0.7, 0.7, 0.72), 0.25, 1.0)
        contacts = []
        # Leads are rigid with the body: their layout comes from the *nominal* placement (part centred,
        # body axis at lead_frame_deg), so a shifted/rotated part's feet miss their pads instead of
        # stretching towards them.
        a0 = -math.radians(float(body.get("lead_frame_deg", 0.0)))
        axis0 = Vector((math.cos(a0), math.sin(a0), 0.0))
        to_world = lambda v: Vector((cx, cy, 0.0)) + axis * v.x + Vector((-axis.y, axis.x, 0.0)) * v.y  # noqa: E731
        for k, ((px, py, pw, ph), lead) in enumerate(zip(pads, p["leads"])):
            rel = Vector((px, py, 0.0))  # scene coordinates are relative to the nominal part centre
            loc = Vector((rel.dot(axis0), rel.x * -axis0.y + rel.y * axis0.x, 0.0))  # pad in body frame
            edge = Vector((max(-L / 2, min(L / 2, loc.x)), max(-W / 2, min(W / 2, loc.y)), 0.0))
            start, foot = to_world(edge * 0.98), to_world(loc)
            lw, t = lead["width"], lead["thickness"]
            shoulder = start + Vector((0, 0, standoff + 0.45 * H))
            knee = start + (foot - start) * 0.3 + Vector((0, 0, standoff + 0.45 * H))
            ankle = start + (foot - start) * 0.55 + Vector((0, 0, t / 2))
            for seg, (a, b) in enumerate(((shoulder, knee), (knee, ankle), (ankle, foot + Vector((0, 0, t / 2))))):
                objs.append(_box_between(f"lead_{k}_{seg}", a, b, lw, t, m_lead))
            d = foot - ankle
            mid = (foot + ankle) / 2
            contacts.append({"cx": mid.x, "cy": mid.y, "hl": d.length / 2, "hw": lw / 2, "angle": math.atan2(d.y, d.x)})
        return objs, contacts, p["solder_height"]
    if p["contacts"] == "none":
        return objs, [None] * len(pads), 0.0  # tombstoned leaded part: leads lifted, paste only
    return objs, [contact] * len(pads), p["solder_height"]


def _build_tht(pads: list[PadMm], board: BoardScene, p: dict[str, Any], mats: dict[str, Any],
               centre: tuple[float, float], angle: float) -> tuple[list[bpy.types.Object], list, float]:
    """Through-hole pin row: a metal pin standing in every plated hole (except ``board["tht"]["missing_pins"]``),
    a dark hole cylinder where the hole shows (missing or unwetted pins) and a solder bar between the two
    pins of ``board["tht"]["bridge"]``. The row follows the body pose (shift / rotation about the nominal
    centre: bent pins); the holes stay where the pads are. Returns the objects, one THT contact per pad for
    :func:`_fillet` (``tht``, pin position, pin radius, ``missing`` / ``unwetted``) and the meniscus height.
    """
    tht = board.get("tht") or {}
    missing = {int(i) for i in tht.get("missing_pins") or []}
    unwetted = bool(tht.get("unwetted"))
    bridge = tht.get("bridge")
    cx, cy = centre
    r_pin, h_pin = p["pin_radius"], p["pin_height"]
    root = bpy.data.objects.new("part_root", None)
    bpy.context.collection.objects.link(root)
    root.location = (cx, cy, 0.0)
    root.rotation_euler = (0.0, 0.0, angle)
    m_pin = _material("mat_pin", (0.78, 0.78, 0.76), roughness=0.35, metallic=1.0, bump=0.02)
    m_hole = _material("mat_hole", (0.01, 0.01, 0.012), roughness=0.9)
    objs: list[bpy.types.Object] = [root]
    contacts: list[dict[str, Any] | None] = []
    pin_xy: list[tuple[float, float]] = []
    for i, (px, py, pw, ph) in enumerate(pads):
        # pin position in the row's frame (pads are relative to the nominal centre), then into the world
        x = cx + px * math.cos(angle) - py * math.sin(angle)
        y = cy + px * math.sin(angle) + py * math.cos(angle)
        pin_xy.append((x, y))
        if i not in missing:
            bpy.ops.mesh.primitive_cylinder_add(radius=r_pin, depth=h_pin, location=(x, y, h_pin / 2.0))
            pin = bpy.context.active_object
            pin.name = f"pin_{i}"
            pin.data.materials.append(m_pin)
            pin.parent = root
            objs.append(pin)
        if i in missing or unwetted:  # the hole shows: dark bore slightly wider than the pin
            bpy.ops.mesh.primitive_cylinder_add(radius=1.35 * r_pin, depth=0.4, location=(px, py, -0.19))
            hole = bpy.context.active_object
            hole.name = f"hole_{i}"
            hole.data.materials.append(m_hole)
            objs.append(hole)
        contacts.append({"tht": True, "cx": x, "cy": y, "pin_r": r_pin, "missing": i in missing, "unwetted": unwetted})
    if bridge and len(bridge) == 2 and all(0 <= int(k) < len(pads) for k in bridge):
        a_i, b_i = (int(k) for k in bridge)
        if a_i not in missing and b_i not in missing:
            m_solder = _cfg_material("mat_tht_bridge", mats.get("solder") or {}, (0.7, 0.7, 0.72), 0.18, 1.0)
            z = 0.55 * p["solder_height"]
            objs.append(_box_between("tht_bridge", Vector((*pin_xy[a_i], z)), Vector((*pin_xy[b_i], z)),
                                     1.6 * r_pin, 1.4 * r_pin, m_solder))
    return objs, contacts, p["solder_height"]


def _tht_joint(name: str, pad: PadMm, contact: dict[str, Any], h: float, mat, excess: bool) -> bpy.types.Object | None:
    """Solder meniscus of a through-hole joint: a polar ring mesh from the pin (height *h*) out to the annular
    ring (height 0), concave ``(1 - t)^1.7``, convex and 1.5x higher for *excess*. ``None`` for a missing or
    unwetted pin: the bare ring of the baked texture stays visible."""
    if contact.get("missing") or contact.get("unwetted"):
        return None
    _, _, pw, ph = pad
    r_pad = min(pw, ph) / 2.0
    r_pin = min(contact["pin_r"], 0.9 * r_pad)
    px, py = contact["cx"], contact["cy"]
    n_ang, n_rad = 48, 12
    verts = []
    for i in range(n_rad + 1):
        t = i / n_rad
        r = r_pin + (r_pad - r_pin) * t
        z = h * (1.5 * math.sqrt(max(0.0, 1.0 - t * t)) if excess else (1.0 - t) ** 1.7)
        for j in range(n_ang):
            a = 2.0 * math.pi * j / n_ang
            verts.append((px + r * math.cos(a), py + r * math.sin(a), 0.001 + z))
    faces = [(i * n_ang + j, i * n_ang + (j + 1) % n_ang, (i + 1) * n_ang + (j + 1) % n_ang, (i + 1) * n_ang + j)
             for i in range(n_rad) for j in range(n_ang)]
    return _mesh_object(name, verts, faces, mat)


# --------------------------------------------------------------------------- scene

_GPU_BACKEND: str | None = None


def _enable_cycles_gpu() -> bool:
    """Enable a Cycles GPU backend (OptiX, CUDA, HIP, oneAPI) once per Blender process.

    Setting ``scene.cycles.device = "GPU"`` alone is not enough: without compute devices
    enabled in the Cycles add-on preferences Blender silently renders on the CPU.
    Returns False (render on CPU) if no GPU backend is available.
    """
    global _GPU_BACKEND
    if _GPU_BACKEND is None:
        _GPU_BACKEND = ""
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dev_type in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
                try:
                    prefs.compute_device_type = dev_type
                except TypeError:
                    continue
                prefs.refresh_devices()
                gpus = [d for d in prefs.devices if d.type == dev_type]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == dev_type
                    _GPU_BACKEND = f"{dev_type}: {', '.join(d.name for d in gpus)}"
                    break
        except Exception as exc:  # noqa: BLE001 - fall back to CPU and say why
            print(f"[render_batch] GPU setup failed, rendering on CPU: {exc}", flush=True)
        print(f"[render_batch] Cycles device: {_GPU_BACKEND or 'CPU'}", flush=True)
    return bool(_GPU_BACKEND)


def _configure_render(*, width_px: int, height_px: int, samples: int, seed: int, device: str, engine: str) -> None:
    scene = bpy.context.scene
    engine = str(engine or "").strip().upper()
    if engine in ("", "EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except TypeError:
            scene.render.engine = "BLENDER_EEVEE_NEXT"  # Blender 4.2-4.x enum name
        scene.eevee.taa_render_samples = max(64, int(samples))
    else:
        scene.render.engine = engine
        if engine == "CYCLES":
            scene.cycles.samples = int(samples)
            scene.cycles.seed = int(seed) % (2**31)  # deterministic per job; IntProperty is 32-bit
            scene.cycles.use_denoising = True
            want_gpu = str(device).upper() != "CPU"
            scene.cycles.device = "GPU" if want_gpu and _enable_cycles_gpu() else "CPU"

    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.resolution_x = int(width_px)
    scene.render.resolution_y = int(height_px)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    # Plain sRGB output: the baked board texture is sRGB and must come out unchanged where lit evenly.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.25, 0.25, 0.25, 1.0)  # dim AOI booth
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.1


def _aoi_lights(render_3d: dict[str, Any], roi_mm: float) -> None:
    """AOI lighting: low-angle ring light plus a diffuse top light around the optical axis.

    ``render_3d.lighting.ring_light`` (strength, height_mm, radius_mm) and ``.top_light`` (strength).
    The ring gives slopes and edges their highlights; the top light makes flat metal (solder tops,
    upright end caps) bright as on real AOI images. Default strengths are calibrated so the board
    renders at its baked (photo-calibrated) colours.
    """
    cfg = render_3d.get("lighting") or {}
    ring_cfg, top_cfg = cfg.get("ring_light") or {}, cfg.get("top_light") or {}
    height = float(ring_cfg.get("height_mm", 2.0 * roi_mm))
    radius = float(ring_cfg.get("radius_mm", height * math.tan(math.radians(35.0))))
    bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=0.06 * radius, location=(0.0, 0.0, height))
    ring = bpy.context.active_object
    ring.name = "ring_light"
    ring.data.materials.append(_material("mat_ring", (0, 0, 0), roughness=1.0,
                                         emission=float(ring_cfg.get("strength", RING_STRENGTH))))
    ring.visible_camera = False

    bpy.ops.mesh.primitive_circle_add(vertices=48, radius=2.0 * radius, fill_type="NGON",
                                      location=(0.0, 0.0, 1.2 * height), rotation=(math.pi, 0.0, 0.0))
    top = bpy.context.active_object
    top.name = "top_light"
    top.data.materials.append(_material("mat_top", (0, 0, 0), roughness=1.0,
                                        emission=float(top_cfg.get("strength", TOP_STRENGTH))))
    top.visible_camera = False


def _build_scene_for_job(job: PreparedJob, *, samples: int, device: str, engine: str) -> None:
    seed = int(job.get("seed", 0))
    rng = random.Random(seed)
    mmpp = float(job["mm_per_px"])
    w_px, h_px = int(job["roi_width_px"]), int(job["roi_height_px"])
    roi_w, roi_h = w_px * mmpp, h_px * mmpp
    board = job["board"]
    render_3d = job.get("render_3d") or {}
    _configure_render(width_px=w_px, height_px=h_px, samples=samples, seed=seed, device=device, engine=engine)

    scale = float(board.get("scale", BOARD_SCALE))
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "board"
    plane.scale = (roi_w * scale, roi_h * scale, 1.0)
    plane.data.materials.append(_board_material(board["texture"]))
    scene_objs = [plane]

    pads: list[PadMm] = [(*_mm_xy(p[0], p[1], mmpp), _mm(p[2], mmpp), _mm(p[3], mmpp)) for p in board["pads"]]
    part, contacts, h_solder = _build_part(pads, board, mmpp,
                                           float(job.get("component_height_mm") or DEFAULT_COMPONENT_HEIGHT_MM), render_3d, rng)
    scene_objs += [o for o in part if o.parent is None]
    m_solder = _cfg_material("mat_solder", (render_3d.get("materials") or {}).get("solder") or {},
                             (0.7, 0.7, 0.72), 0.18, 1.0, bump=0.02)
    for i, (pad, contact) in enumerate(zip(pads, contacts)):
        big = i == board.get("thermal_pad")  # window-pane stencil, as in 2D
        joint = _fillet(f"solder_{i}", pad, contact, h_solder, big, m_solder, excess=i in board.get("excess", []))
        if joint is not None:  # None: a bare through-hole ring (missing / unwetted pin)
            scene_objs.append(joint)
    for k, (bx, by, br) in enumerate(board.get("balls", [])):  # solder balls on the mask
        r = _mm(br, mmpp)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(*_mm_xy(bx, by, mmpp), 0.7 * r), segments=16, ring_count=8)
        ball = bpy.context.active_object
        ball.name = f"solder_ball_{k}"
        ball.data.materials.append(m_solder)
        bpy.ops.object.shade_smooth()
        scene_objs.append(ball)

    # Global in-plane orientation (cardinal 0/90/180/270 + jitter), like the 2D rotation.
    rot = float((job.get("augment") or {}).get("rotation_deg", 0.0))
    if abs(rot) > 1e-6:
        root = bpy.data.objects.new("global_rotation_root", None)
        bpy.context.collection.objects.link(root)
        for obj in scene_objs:
            obj.parent = root
        root.rotation_euler[2] = math.radians(rot)

    roi_mm = max(roi_w, roi_h)
    _aoi_lights(render_3d, roi_mm)

    # Orthographic camera framing exactly the ROI (top-down unless the profile tilts it).
    cam_cfg = render_3d.get("camera") or {}
    loc = cam_cfg.get("location_mm")
    loc = tuple(float(v) for v in loc) if isinstance(loc, (list, tuple)) and len(loc) == 3 else (0.0, 0.0, 9.0)
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    pose = board.get("pose") or DEFAULT_CAMERA_POSE
    cam.data.ortho_scale = roi_mm * float(pose["zoom"])  # zoomed out: farther away (render.camera.zoom)
    cam.data.clip_end = 1000.0
    look = cam_cfg.get("look_at_mm")
    look = Vector(tuple(float(v) for v in look)) if isinstance(look, (list, tuple)) and len(look) == 3 else Vector()
    if float(pose["tilt"]) > 0.0:  # oblique view (render.camera.tilt_deg); azimuth is in image coords (y down)
        t, a = math.radians(float(pose["tilt"])), math.radians(float(pose["azimuth"]))
        dist = max((Vector(loc) - look).length, 1.0)
        cam.location = look + Vector((math.sin(t) * math.cos(a), -math.sin(t) * math.sin(a), math.cos(t))) * dist
    _look_at(cam, look)
    bpy.context.scene.camera = cam


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in Path(args.jobs).read_text(encoding="utf-8").splitlines() if line.strip()]

    for i, job in enumerate(jobs, 1):
        out_path = out_root / str(job["image_path"])
        try:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            _build_scene_for_job(job, samples=int(args.samples), device=str(args.device), engine=str(args.engine))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            bpy.context.scene.render.filepath = str(out_path)
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            # Fail fast with a non-zero exit; a partially rendered dataset is worse than stopping.
            print(f"[render][error] job {i}/{len(jobs)} failed: {exc}")
            raise SystemExit(2) from exc
        if i == 1 or i % 50 == 0 or i == len(jobs):
            print(f"[render] {i}/{len(jobs)} -> {out_path}")


if __name__ == "__main__":
    main()
