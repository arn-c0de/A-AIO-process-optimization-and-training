"""Blender script: render a 3D-Board-Renderer window (many parts in one scene).

Run via ``simple_sim.board3d.render.render_window`` (``python -m simple_sim board3d-render``):
  blender --background --factory-startup --python render_board3d.py -- --job job.json

The job (written by :func:`simple_sim.board3d.scene.build_job`, stdlib JSON) holds the substrate texture (bare board
in board mm), the photo texture (the rectified scan with the parts, for the body tops), the window and every
component in it with its pose, nominal pose, body size, pads (part frame, mm) and ``part`` spec.

Geometry per component comes from the single-part builders of ``render_batch.py`` (body, end caps, gull-wing
leads, solder fillets), built around the origin and then parented to an empty at the component pose, so
1 Blender unit = 1 mm and Blender x/y = board x/y (y up). Body tops and end caps carry the real scan pixels:
their UVs are the board coordinates at the **nominal** pose, so a moved part takes its own photo with it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
from simple_sim.blender import render_batch as rb  # noqa: E402


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    return ap.parse_args(argv)


#: image texture filter of the photo / substrate textures (job ``interpolation``). Blender's "Cubic" is a cubic B-spline
#: (smooths: the render was softer than the scan); "Linear" keeps the scan's sharpness where texels and pixels align.
INTERPOLATION = "Linear"


def _texture_material(name: str, image_path: str, *, roughness: float, specular: float, metallic: float = 0.0):
    mat = rb._material(name, (1, 1, 1), roughness=roughness, metallic=metallic)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Specular IOR Level"].default_value = specular
    tex = nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.get(Path(image_path).name) or bpy.data.images.load(image_path)
    tex.image = img
    tex.interpolation = INTERPOLATION
    tex.extension = "EXTEND"
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _unlit_texture_material(name: str, image_path: str, gain=(1.0, 1.0, 1.0)):
    """Emission = the image colour x *gain*, independent of the lights (the scan already holds the scanner's light;
    *gain* is the ``room`` preset's colour calibration, applied like the world light)."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.get(Path(image_path).name) or bpy.data.images.load(image_path)
    tex.interpolation = INTERPOLATION
    tex.extension = "EXTEND"
    emit = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    if tuple(gain) != (1.0, 1.0, 1.0):
        mul = nodes.new("ShaderNodeVectorMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = tuple(float(g) for g in gain)
        links.new(tex.outputs["Color"], mul.inputs[0])
        links.new(mul.outputs["Vector"], emit.inputs["Color"])
    else:
        links.new(tex.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _room_gain(job: dict) -> tuple[float, float, float]:
    g = (job.get("light") or {}).get("gainRgb")
    if g:
        return tuple(float(v) for v in g)
    v = float((job.get("light") or {}).get("gain", 1.0))
    return (v, v, v)


def _board_plane(sub: dict, board: dict | None, mat):
    """The substrate plane with UVs in the substrate raster. With ``board.clip`` the plane is the board outline
    (0..W x 0..H) instead of the whole raster: the mounted source's frame also shows the desk around the board,
    which a free camera must not see as a flat print."""
    x0, y0, w, h = sub["x0"], sub["y0"], sub["widthMm"], sub["heightMm"]
    if board and board.get("clip"):
        bx0, by0, bx1, by1 = 0.0, 0.0, float(board["widthMm"]), float(board["heightMm"])
    else:
        bx0, by0, bx1, by1 = x0, y0, x0 + w, y0 + h
    verts = [(bx0, by0, 0.0), (bx1, by0, 0.0), (bx1, by1, 0.0), (bx0, by1, 0.0)]
    me = bpy.data.meshes.new("board")
    me.from_pydata(verts, [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        vx, vy, _ = verts[loop.vertex_index]
        uv.data[loop.index].uv = ((vx - x0) / w, (vy - y0) / h)
    me.update()
    obj = bpy.data.objects.new("board", me)
    bpy.context.collection.objects.link(obj)
    me.materials.append(mat)
    return obj


def _board_slab(board: dict) -> None:
    """The laminate under the top surface (``thicknessMm``, FR4 edge colour): its edge is visible from any oblique
    camera, a zero-thickness plane looks like paper."""
    t = float(board.get("thicknessMm", 1.6))
    if t <= 0:
        return
    w, h = float(board["widthMm"]), float(board["heightMm"])
    edge = rb._material("mat_fr4_edge", tuple(board.get("edgeRgb", (0.10, 0.17, 0.09))), roughness=0.75)
    _fast_box("board_edge", (w, h, t - 0.002), (w / 2, h / 2, -t / 2 - 0.002), edge)


def _shadow_catcher(job: dict) -> None:
    """Transparent film + a shadow-catcher floor under the board: the PNG keeps the alpha (board and shadows), the
    caller lays it over the real photo (``backplate``)."""
    board = job.get("board") or {}
    t = float(board.get("thicknessMm", 1.6))
    w, h = float(board.get("widthMm", 100.0)), float(board.get("heightMm", 100.0))
    floor = _fast_box("shadow_floor", (6 * w, 6 * h, 0.01), (w / 2, h / 2, -t - float(board.get("standoffMm", 0.0)) - 0.01),
                      rb._material("mat_floor", (0.5, 0.5, 0.5), roughness=1.0))
    floor.is_shadow_catcher = True
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"


# Unit cube of ``bpy.ops.mesh.primitive_cube_add(size=1)``: same vertices, outward quads, a UV layer.
_CUBE_V = [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5), (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]
_CUBE_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def _fast_box(name: str, size, loc, mat, *, parent=None, bevel: float = 0.0):
    """``render_batch._box`` through the data API: ``bpy.ops`` re-evaluates the scene on every call (77 ms per box
    with ~1000 objects, 20 of 22 s of a 97-part build). Same geometry, scale, bevel and smooth shading."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(_CUBE_V, [], _CUBE_F)
    me.uv_layers.new(name="UVMap")
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    obj.location = tuple(float(v) for v in loc)
    obj.scale = tuple(float(v) for v in size)
    me.materials.append(mat)
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        me.shade_smooth()
    if parent is not None:
        obj.parent = parent
    return obj


def _cylinder_mesh(name: str, radius: float, depth: float, segments: int = 32):
    verts, faces = [], []
    for z in (-depth / 2, depth / 2):
        verts += [(radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments), z) for i in range(segments)]
    faces += [(i, (i + 1) % segments, segments + (i + 1) % segments, segments + i) for i in range(segments)]
    faces += [tuple(reversed(range(segments))), tuple(range(segments, 2 * segments))]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    return me


class _MeshOps:
    """``bpy.ops.mesh.primitive_cylinder_add`` for ``render_batch``'s builders through the data API (the operator
    costs ~150 ms in a full board scene). The new object becomes the active one, as with the operator."""

    def __getattr__(self, name):
        return getattr(bpy.ops.mesh, name)

    @staticmethod
    def primitive_cylinder_add(radius: float = 1.0, depth: float = 2.0, location=(0.0, 0.0, 0.0), **_kw):
        obj = bpy.data.objects.new("Cylinder", _cylinder_mesh("Cylinder", float(radius), float(depth)))
        bpy.context.collection.objects.link(obj)
        obj.location = tuple(float(v) for v in location)
        bpy.context.view_layer.objects.active = obj
        return {"FINISHED"}


class _Ops:
    mesh = _MeshOps()

    def __getattr__(self, name):
        return getattr(bpy.ops, name)


class _BpyBoardMode:
    """What ``render_batch`` sees as ``bpy`` in board mode: the real module except the slow operators above."""
    ops = _Ops()

    def __getattr__(self, name):
        return getattr(bpy, name)


#: too much solder: land grown by this factor, blob this many times as high as the nominal fillet
EXCESS_GROW, EXCESS_HEIGHT = 1.3, 2.2

rb._box = _fast_box   # board mode only; render_batch's own ROI renders keep the operator versions
rb.bpy = _BpyBoardMode()


def _descendants(obj):
    for child in obj.children:
        yield child
        yield from _descendants(child)


def _uv_from_board(obj, raster: dict, to_nominal: Matrix | None = None) -> None:
    """UVs = board position of every vertex in *raster*; *to_nominal* maps the vertex (world) back to where it
    was in the real photo first (a moved body keeps its own pixels). Vectorised (numpy ships with Blender)."""
    import numpy as np
    me = obj.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    mw = obj.matrix_world if to_nominal is None else to_nominal @ obj.matrix_world
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    m = np.array(mw, np.float64)
    world = co @ m[:3, :3].T + m[:3, 3]
    idx = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", idx)
    uv = np.empty((len(me.loops), 2), np.float32)
    uv[:, 0] = (world[idx, 0] - raster["x0"]) / raster["widthMm"]
    uv[:, 1] = (world[idx, 1] - raster["y0"]) / raster["heightMm"]
    me.uv_layers.active.data.foreach_set("uv", uv.ravel())


def _local_offset(nominal: dict, pose: dict) -> tuple[float, float, float]:
    """Body displacement in the nominal part frame: (dx, dy) in mm, dRot in degrees (CCW)."""
    a = math.radians(nominal["rotDeg"])
    dx, dy = pose["x"] - nominal["x"], pose["y"] - nominal["y"]
    lx, ly = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
    drot = (pose["rotDeg"] - nominal["rotDeg"] + 180.0) % 360.0 - 180.0
    return lx, ly, drot


def _build_component(c: dict, job: dict, mats: dict, rng: random.Random, pending: list) -> None:
    """Built in the nominal part frame (root empty at the nominal pose): pads and solder stay on the board's lands,
    the body is displaced by the pose difference (shift / rotation / lift / tombstone), so the fillets follow the
    terminals the way ``render_batch`` makes them for a misplaced part."""
    fp_pads = [tuple(float(v) for v in p) for p in c["pads"]]
    height = float(c["height"])
    nominal, pose = c["nominal"], c["pose"]
    lx, ly, drot = _local_offset(nominal, pose)
    # Every part is built into its own collection: its new objects are then that collection's objects (comparing
    # against all of bpy.data.objects per part made the build quadratic).
    coll = bpy.data.collections.new(f"part_{c['refdes']}")
    bpy.context.scene.collection.children.link(coll)
    layer = bpy.context.view_layer
    previous = layer.active_layer_collection
    layer.active_layer_collection = layer.layer_collection.children[coll.name]
    contacts, h_solder = [None] * len(fp_pads), 0.0
    if height > 0.0 and not c.get("missing") and c["part"].get("body") == "can":
        contacts, h_solder = _build_can(c, lx, ly, drot, height, mats, len(fp_pads))
    elif height > 0.0 and not c.get("missing"):
        tomb = c.get("tombstone")
        # render_batch scene convention: image axes (y down), rotation in image degrees (clockwise positive)
        board = {"body": {"center": [lx, -ly], "length": float(c["l"]), "width": float(c["w"]), "rotation_deg": -drot,
                          "lead_frame_deg": 0.0, "tilt_deg": float(tomb["tiltDeg"]) if tomb else 0.0,
                          "is_tombstone": bool(tomb), "raised_sign": float(tomb["raisedSign"]) if tomb else 1.0},
                 "part": c["part"], "pads": [], "thermal_pad": c.get("thermal")}
        _objs, contacts, h_solder = rb._build_part(fp_pads, board, 1.0, height, {}, rng)
    excess = set(c.get("excess") or [])
    joints = []
    for i, (pad, contact) in enumerate(zip(fp_pads, contacts)):
        if i in excess:
            # too much solder: a convex blob on a land grown by EXCESS_GROW, EXCESS_HEIGHT x the nominal fillet,
            # in shiny tin (the photo has no blob to show)
            px, py, pw, ph = pad
            _blob(f"solder_{c['refdes']}_{i}", px, py, pw * EXCESS_GROW, ph * EXCESS_GROW,
                  max(h_solder, 0.12) * EXCESS_HEIGHT, mats["solder_excess"])
            continue
        j = rb._fillet(f"solder_{c['refdes']}_{i}", pad, contact, h_solder, i == c.get("thermal"), mats["solder"])
        if j is not None:
            joints.append(j)
    layer.active_layer_collection = previous
    new = list(coll.objects)
    for o in [o for o in new if o.name.startswith("Cylinder")]:   # synthetic pin-1 dimple: the photo has the real mark
        bpy.data.objects.remove(o, do_unlink=True)
        new.remove(o)
    root = bpy.data.objects.new(f"cmp_{c['refdes']}", None)
    coll.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    root.location = (nominal["x"], nominal["y"], 0.0)
    root.rotation_euler = (0.0, 0.0, math.radians(nominal["rotDeg"]))
    part_root = next((o for o in new if o.name.startswith("part_root")), None)
    if part_root is not None and float(pose.get("z", 0.0)):
        part_root.location.z += float(pose["z"])
    if mats.get("photo") and c.get("photoTexture", True):
        # UVs need world matrices: collected here, assigned after ONE view-layer update for the whole scene
        # (an update per part made the build quadratic: 24 s of 28 s for 97 parts).
        pending.append((c, root, part_root, joints, float(pose.get("z", 0.0))))


def _blob(name: str, cx: float, cy: float, w: float, h: float, height: float, mat):
    """Too much solder: a rounded cap (superellipse footprint, exponent 2.5) over the land, ``height`` at the
    centre: it swallows the terminal instead of wetting it as a concave fillet."""
    n = 32
    verts, radius = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            u, v = 2 * i / n - 1, 2 * j / n - 1
            r = (abs(u) ** 2.5 + abs(v) ** 2.5) ** (1 / 2.5)
            z = height * max(0.0, 1 - r * r) ** 0.6
            verts.append((cx + u * w / 2, cy + v * h / 2, 0.001 + z))
            radius.append(r)
    quads = [(j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i) for j in range(n) for i in range(n)]
    faces = [q for q in quads if min(radius[k] for k in q) < 1.0]      # only the cap, not the flat corners
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.shade_smooth()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    me.materials.append(mat)
    return obj


def _build_can(c: dict, lx: float, ly: float, drot: float, height: float, mats: dict, n_pads: int):
    """SMD aluminium electrolytic: square plastic base plate + round can (``part.body == "can"``). Names start
    with ``body`` so the photo texture lands on them (the can's top shows the real print and vent)."""
    l, w = float(c["l"]), float(c["w"])
    root = bpy.data.objects.new("part_root", None)
    bpy.context.collection.objects.link(root)
    root.location = (lx, ly, 0.0)
    root.rotation_euler = (0.0, 0.0, math.radians(drot))
    base_h = 0.16 * height
    m_base = rb._material("mat_can_base", (0.03, 0.03, 0.03), roughness=0.6)
    m_can = rb._material("mat_can", (0.78, 0.78, 0.8), roughness=0.3, metallic=1.0)
    _fast_box("body_base", (l, w, base_h), (0.0, 0.0, base_h / 2), m_base, parent=root, bevel=0.04 * min(l, w))
    r = 0.49 * min(l, w)
    can = bpy.data.objects.new("body_can", _cylinder_mesh("body_can", r, height - base_h, segments=64))
    bpy.context.collection.objects.link(can)
    can.location = (0.0, 0.0, base_h + (height - base_h) / 2)
    can.data.materials.append(m_can)
    can.data.shade_smooth()
    can.parent = root
    contact = {"cx": lx, "cy": ly, "hl": l / 2, "hw": w / 2, "angle": math.radians(drot)}
    return [contact] * n_pads, 0.25


def _bridge_mesh(name: str, p0: Vector, p1: Vector, width: float, height: float, n: int = 16):
    """A solder bar: half-elliptic cross-section (``width`` x ``height``) swept straight from p0 to p1, with
    rounded ends (the profile shrinks over the last half width)."""
    d = p1 - p0
    length = d.length
    ax = d.normalized()
    side = Vector((-ax.y, ax.x, 0.0))
    steps = 24
    verts, faces = [], []
    for k in range(steps + 1):
        t = k / steps
        along = t * length
        end = min(along, length - along)
        scale = math.sqrt(max(0.0, min(1.0, end / (0.5 * width)))) if width > 0 else 1.0
        for i in range(n + 1):
            a = math.pi * i / n
            q = p0 + ax * along + side * (math.cos(a) * width / 2 * max(scale, 0.05))
            verts.append((q.x, q.y, 0.002 + math.sin(a) * height * scale))
    for k in range(steps):
        for i in range(n):
            a, b = k * (n + 1) + i, (k + 1) * (n + 1) + i
            faces.append((a, a + 1, b + 1, b))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.shade_smooth()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    return obj


def _build_bridges(job: dict, mats: dict) -> None:
    """Solder bars between two lands (op ``bridge``): land centre to land centre, as wide as 85 % of the narrower
    land, 0.2 mm high, in shiny tin."""
    for k, b in enumerate(job.get("bridges") or []):
        (ax, ay, aw, ah), (bx, by, bw, bh) = b["a"], b["b"]
        bar = _bridge_mesh(f"bridge_{k}", Vector((ax, ay, 0.0)), Vector((bx, by, 0.0)), 0.85 * min(aw, ah, bw, bh), 0.2)
        bar.data.materials.append(mats["solder_excess"])


def _apply_photo_uvs(pending: list, job: dict, mats: dict) -> None:
    bpy.context.view_layer.update()
    raster = job["photo"]["raster"]
    for c, root, part_root, joints, z in pending:
        # Solder stays on the lands: the joints show the scan pixels where they are. Defect solder keeps the metal.
        for o in joints:
            o.data.materials.clear()
            o.data.materials.append(mats.get("photo_joint") or mats["photo"])
            _uv_from_board(o, raster)
        if part_root is None:
            continue
        # Body and end caps: the photo of the part where it really sat (nominal pose, flat on its standoff).
        # A tombstone's body sits at +-L/2 in its pivot frame (render_batch: ``offset``); shift it back to the centre.
        tomb = c.get("tombstone")
        offset = float(tomb["raisedSign"]) * float(c["l"]) / 2 if tomb else 0.0
        nominal_frame = root.matrix_world @ Matrix.Translation((-offset, 0.0, part_root.location.z - z))
        to_nominal = nominal_frame @ part_root.matrix_world.inverted()
        # gull-wing leads are rigid with the body (render_batch builds them at the displaced pose): same mapping
        leads = [o for o in root.children if o.name.startswith("lead_")]
        for o in [*_descendants(part_root), *leads]:
            if o.type == "MESH" and (o.name.startswith(("body", "endcap", "lead_"))):
                o.data.materials.clear()
                o.data.materials.append(mats["photo"])
                _uv_from_board(o, raster, to_nominal)


def _lights(job: dict) -> None:
    """``flatbed``: a disc light straight above, invisible to the camera, seen from the board under a cone of
    ``coneDeg`` half-angle (the scanner lights steeply from above: little occlusion next to tall parts, flat metal
    facing up turns bright), plus a dim white surround (``world``). The disc's strength makes a flat diffuse surface
    render at its texture colour: irradiance = pi * L * (sin^2(cone) + world). ``aoi``: the ring + top rig of
    ``render_batch``."""
    light = job.get("light") or {}
    world = bpy.context.scene.world
    bg = world.node_tree.nodes["Background"]
    win = job["window"]
    cx, cy = win["x0"] + win["widthMm"] / 2, win["y0"] + win["heightMm"] / 2
    if light.get("preset") == "room":
        # phone photos under room light: the textures (median of phone photos) already hold that light; a uniform
        # white surround renders flat diffuse surfaces at their texture colour from any viewpoint
        bg.inputs["Color"].default_value = (*_room_gain(job), 1.0)
        bg.inputs["Strength"].default_value = 1.0
        return
    if light.get("preset", "flatbed") != "flatbed":
        rb._aoi_lights({}, max(win["widthMm"], win["heightMm"]))
        return
    ambient = float(light.get("world", 0.15))
    gain = float(light.get("gain", 1.0))
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = ambient * gain
    cone = math.radians(float(light.get("coneDeg", 20.0)))
    height = 4.0 * max(win["widthMm"], win["heightMm"]) + 100.0
    radius = height * math.tan(cone)
    bpy.ops.mesh.primitive_circle_add(vertices=64, radius=radius, fill_type="NGON", location=(cx, cy, height),
                                      rotation=(math.pi, 0.0, 0.0))
    disc = bpy.context.active_object
    disc.name = "flatbed_light"
    strength = gain * (1.0 - ambient) / max(math.sin(cone) ** 2, 1e-4)
    disc.data.materials.append(rb._material("mat_flatbed_light", (0, 0, 0), roughness=1.0, emission=strength))
    disc.visible_camera = False
    disc.visible_shadow = False


def _export_glb(path: str) -> None:
    """Board + components as one GLB for the browser viewer (glTF is y-up: board (x, y) -> (x, -z)). Every component is
    an empty ``cmp_<refdes>`` at its nominal pose with body / leads / joints below it; modifiers (bevels) applied.
    Lights and camera are left out: the viewer lights the scene itself."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", export_apply=True, export_cameras=False,
                              export_lights=False, export_extras=True, export_yup=True)


def main() -> None:
    t_start = time.monotonic()
    global INTERPOLATION
    job = json.loads(Path(_parse_args().job).read_text(encoding="utf-8"))
    INTERPOLATION = str(job.get("interpolation", INTERPOLATION))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    win = job["window"]
    rb._configure_render(width_px=int(job["widthPx"]), height_px=int(job["heightPx"]), samples=int(job.get("samples", 64)),
                         seed=int(job.get("seed", 0)), device=str(job.get("device", "GPU")), engine=str(job.get("engine", "CYCLES")))
    scene = bpy.context.scene
    if scene.render.engine == "CYCLES":
        # The photo textures are noise-free; the denoiser would only smear them. Pixel filter as narrow as the scan.
        scene.cycles.use_denoising = bool(job.get("denoise", False))
        scene.cycles.filter_width = float(job.get("filterWidthPx", 1.0))
    sub = job["substrate"]["raster"]
    # The scan already holds the scanner's sheen: photo-textured surfaces get no extra specular (else the steep
    # light's reflection is counted twice).
    _board_plane(sub, job.get("board"), _texture_material("mat_substrate", job["substrate"]["path"], roughness=0.9, specular=0.0))
    if job.get("board"):
        _board_slab(job["board"])

    mats = {"solder": rb._cfg_material("mat_solder", {}, (0.72, 0.72, 0.74), 0.22, 1.0, bump=0.02)}
    if job.get("photo"):
        mats["photo"] = _texture_material("mat_photo", job["photo"]["path"], roughness=0.9, specular=0.0)
        if (job.get("light") or {}).get("preset", "flatbed") in ("flatbed", "room"):
            # Flatbed: nominal joints show the scanned solder as it was lit (slopes would darken under the steep
            # light, while the scanner made them bright). Other presets light them (T4).
            gain = _room_gain(job) if (job.get("light") or {}).get("preset") == "room" else (1.0, 1.0, 1.0)
            mats["photo_joint"] = _unlit_texture_material("mat_photo_joint", job["photo"]["path"], gain)
    rng = random.Random(int(job.get("seed", 0)))
    mats["solder_excess"] = rb._material("mat_solder_excess", (0.72, 0.72, 0.73), roughness=0.5, metallic=0.55, bump=0.02)
    pending: list = []
    for c in job["components"]:
        _build_component(c, job, mats, rng, pending)
    if pending:
        _apply_photo_uvs(pending, job, mats)
    _build_bridges(job, mats)

    if job.get("exportGlb"):
        _export_glb(job["exportGlb"])
        if not job.get("render", True):
            print(f"[render_board3d] {len(job['components'])} components -> {job['exportGlb']}", flush=True)
            return

    _lights(job)
    if job.get("shadowCatcher"):
        _shadow_catcher(job)
    cam_cfg = job.get("camera")
    if cam_cfg and cam_cfg.get("type") == "PERSP":
        # a real photo's camera (simple_sim.board3d.camera.blender_camera): pose, lens, full photo frame
        cam_data = bpy.data.cameras.new("photo_camera")
        cam = bpy.data.objects.new("photo_camera", cam_data)
        bpy.context.scene.collection.objects.link(cam)
        rot = Matrix([list(row) for row in cam_cfg["rotation"]])
        cam.matrix_world = Matrix.Translation(Vector(cam_cfg["locationMm"])) @ rot.to_4x4()
        cam_data.sensor_fit = "AUTO"
        cam_data.sensor_width = float(cam_cfg["sensorWidthMm"])
        cam_data.lens = float(cam_cfg["lensMm"])
        cam_data.clip_start, cam_data.clip_end = 1.0, 5000.0
    else:
        cx, cy = win["x0"] + win["widthMm"] / 2, win["y0"] + win["heightMm"] / 2
        bpy.ops.object.camera_add(location=(cx, cy, 30.0))
        cam = bpy.context.active_object
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = max(win["widthMm"], win["heightMm"])
        cam.data.clip_end = 1000.0
        rb._look_at(cam, Vector((cx, cy, 0.0)))
    bpy.context.scene.camera = cam
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(out)
    t_build = time.monotonic()
    bpy.ops.render.render(write_still=True)
    t_end = time.monotonic()
    print(f"[render_board3d] {len(job['components'])} components -> {out} | build {t_build - t_start:.1f} s, "
          f"render {t_end - t_build:.1f} s ({bpy.context.scene.cycles.device if scene.render.engine == 'CYCLES' else scene.render.engine})",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # Blender exits 0 after a script error: make the failure visible to the caller
        import traceback
        traceback.print_exc()
        sys.exit(2)
