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
from simple_sim.blender import body_mesh as bm  # noqa: E402
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


def _unlit_colour_material(name: str, bgr, gain=(1.0, 1.0, 1.0)):
    """Emission of one 8-bit sRGB colour (BGR, as OpenCV measured it) x *gain*: like :func:`_unlit_texture_material`."""
    lin = lambda v: ((v / 255.0 + 0.055) / 1.055) ** 2.4 if v / 255.0 > 0.04045 else v / 255.0 / 12.92  # noqa: E731
    rgb = [lin(float(v)) * float(g) for v, g in zip(reversed(list(bgr)), gain)]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emit, out = nodes.new("ShaderNodeEmission"), nodes.new("ShaderNodeOutputMaterial")
    emit.inputs["Color"].default_value = (*rgb, 1.0)
    mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _solder_photo_material(name: str, bgr, gain=(1.0, 1.0, 1.0), glossy: float = 0.3):
    """Defect solder (bridges, blobs, tinned lands) under the flatbed / room presets: the bright solder colour of the
    real image (*bgr*, :func:`simple_sim.board3d.render.scan_solder_colour`) as emission, brightest where the surface
    faces the camera and ~55 % at its grazing rim (a dome's highlight and dark edge), plus *glossy* x a lit tin BSDF for
    the reflections. The nominal joints show the photo's own solder unlit; lit metal under the room preset's dim
    mirror world rendered the defect solder grey-pink, the reviewed real bridges are near white."""
    lin = lambda v: ((v / 255.0 + 0.055) / 1.055) ** 2.4 if v / 255.0 > 0.04045 else v / 255.0 / 12.92  # noqa: E731
    rgb = [lin(float(v)) * float(g) for v, g in zip(reversed(list(bgr)), gain)]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    lw = nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    shade = nodes.new("ShaderNodeMath")
    shade.operation = "MULTIPLY_ADD"          # 1 - 0.45 * facing
    shade.inputs[1].default_value = -0.45
    shade.inputs[2].default_value = 1.0
    links.new(lw.outputs["Facing"], shade.inputs[0])
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*rgb, 1.0)
    links.new(shade.outputs["Value"], emit.inputs["Strength"])
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.25
    mixw = nodes.new("ShaderNodeMixShader")
    mixw.inputs["Fac"].default_value = float(glossy)
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(emit.outputs["Emission"], mixw.inputs[1])
    links.new(bsdf.outputs["BSDF"], mixw.inputs[2])
    links.new(mixw.outputs["Shader"], out.inputs["Surface"])
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


def _shape_mesh(name: str, shape: bm.BodyShape):
    """Mesh of a :class:`body_mesh.BodyShape` in mm (centred, scale 1): flat walls, smooth curved bands only."""
    body = bm.body_mesh(shape)
    me = bpy.data.meshes.new(name)
    me.from_pydata(body.verts, [], body.faces)
    me.uv_layers.new(name="UVMap")
    me.polygons.foreach_set("use_smooth", body.smooth)
    me.update()
    return me


#: render_batch's box builds through the data API in real mm now (it used ``bpy.ops`` and a bevel modifier on a scaled
#: unit cube: 77 ms per box in a full board scene, and the bevel stretched with the body)
_fast_box = rb._box


def _set_mesh(obj, shape: bm.BodyShape) -> None:
    """Swap *obj*'s mesh for *shape* (materials kept)."""
    old = obj.data
    me = _shape_mesh(old.name, shape)
    for m in old.materials:
        me.materials.append(m)
    obj.data = me
    bpy.data.meshes.remove(old)


#: material per piece role (``body_mesh.ROLES``): (rgb, roughness, metallic)
ROLE_MATERIALS = {
    "mold": ((0.03, 0.03, 0.03), 0.55, 0.0), "ceramic": ((0.42, 0.26, 0.14), 0.55, 0.0),
    "ferrite": ((0.07, 0.07, 0.075), 0.75, 0.0), "metal": ((0.75, 0.75, 0.74), 0.35, 1.0),
    "alu": ((0.78, 0.78, 0.8), 0.3, 1.0), "base": ((0.03, 0.03, 0.03), 0.6, 0.0),
    "substrate": ((0.10, 0.14, 0.07), 0.6, 0.0), "lid": ((0.70, 0.70, 0.72), 0.35, 1.0),
    "white": ((0.85, 0.84, 0.80), 0.5, 0.0), "ball": ((0.72, 0.72, 0.74), 0.25, 1.0),
    # natural nylon (JST rows): off-white, as the phone photos show it under a neutral sky (the old cream 0.78 / 0.72 /
    # 0.58 came out yellow once the sky was no longer tinted by the per-channel gain, b* +8 against -3..-8)
    "natural": ((0.80, 0.79, 0.74), 0.6, 0.0), "insert": ((0.015, 0.015, 0.015), 0.8, 0.0),
    # an electrolytic's printed PVC sleeve: black, a little glossy
    "sleeve": ((0.02, 0.02, 0.024), 0.4, 0.0),
}
#: names of the meshes that carry the photo on their top faces (``_apply_photo_uvs``). ``cavity`` / ``pin_``: the floor
#: and pins of an open-top connector, which the top-down scan sees (dark floor, thin pin tips); untextured they rendered
#: as a grey slab with saturated pin squares.
PHOTO_PREFIXES = ("body", "endcap", "lead_", "term_", "lid", "cavity", "pin_")
#: connector kinds whose sheet shell shows its photographed top (clean look; other shells stay plain metal). Measured
#: on the phone views 811d1c / f48a23 / da70eb, body dE plain -> photo top: microSD J500 28 / 31 / 13 -> 11 / 16 / 12;
#: the dual USB-A J1001 got worse (5 / 8 / 4 -> 20 / 13 / 17), the RJ45 mixed (its MAC sticker is per unit)
PHOTO_TOP_SHELLS = ("card_socket",)


def _role_material(role: str):
    mat = bpy.data.materials.get(f"mat_role_{role}")
    if mat is None:
        rgb, rough, metal = ROLE_MATERIALS[role]
        mat = rb._material(f"mat_role_{role}", rgb, roughness=rough, metallic=metal, bump=0.02 if metal else 0.0)
    return mat


def _piece_object(piece: bm.Piece, parent, loc) -> object:
    me = bpy.data.meshes.new(piece.name)
    me.from_pydata(piece.mesh.verts, [], piece.mesh.faces)
    me.uv_layers.new(name="UVMap")
    me.polygons.foreach_set("use_smooth", piece.mesh.smooth)
    me.update()
    me.materials.append(_role_material(piece.role))
    obj = bpy.data.objects.new(piece.name, me)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    obj.parent = parent
    obj["photo_sides"] = piece.photo_sides
    return obj


def _board_kind(part: dict) -> str:
    if part.get("board"):
        return str(part["board"])
    if part.get("body") in ("chip", "can"):
        return str(part["body"])
    return "gullwing" if part.get("leads") else "box"


def _front(c: dict, board: dict | None) -> tuple[float, float]:
    """Part-frame axis (unit +-x / +-y) towards the nearest board edge: where a horizontal connector's mating face
    looks (the job has no mating side; connectors sit at the edge they are plugged from)."""
    pose = c["nominal"]
    bw, bh = float((board or {}).get("widthMm", 0.0)), float((board or {}).get("heightMm", 0.0))
    if bw <= 0 or bh <= 0:
        return (0.0, -1.0)
    x, y = float(pose["x"]), float(pose["y"])
    dist = {(-1.0, 0.0): x, (1.0, 0.0): bw - x, (0.0, -1.0): y, (0.0, 1.0): bh - y}
    wx, wy = min(dist, key=dist.get)
    a = -math.radians(float(pose["rotDeg"]))
    px, py = wx * math.cos(a) - wy * math.sin(a), wx * math.sin(a) + wy * math.cos(a)
    return (1.0 if px > 0 else -1.0, 0.0) if abs(px) >= abs(py) else (0.0, 1.0 if py > 0 else -1.0)


def _assembly(kind: str, c: dict, h: float, standoff: float, pads: list, board: dict | None = None) -> bm.Assembly | None:
    l, w, part = float(c["l"]), float(c["w"]), c["part"]
    if kind == "gullwing":
        return bm.gullwing(l, w, h, standoff, pads)
    if kind == "qfn":
        return bm.qfn(l, w, h, standoff, pads, c.get("thermal"))
    if kind == "array":
        return bm.array(l, w, h, pads, role="ceramic" if part.get("body_material") == "ceramic" else "mold")
    if kind == "crystal":
        return bm.crystal(l, w, h)
    if kind == "bga":
        return bm.bga(l, w, h, standoff, float(part.get("pitch", 0.8)))
    if kind == "tabs":
        # flat-lead diodes (SOD-123F / SMAF) are near vertical: a drafted wall showed as a dark rim in the scan match
        return bm.molded_tabs(l, w, h, standoff, pads, draft=False,
                              role="ferrite" if part.get("body_material") == "ferrite" else "mold")
    if kind == "connector":
        spec = part["connector"]
        total = spec.get("totalHeightMm")
        return bm.connector(spec["kind"], spec.get("orientation", ""), l, w, h, pins=int(spec["pins"]), rows=int(spec["rows"]),
                            pitch=float(spec["pitch"]), total_h=None if total is None else float(total), front=_front(c, board),
                            cell=bool(spec.get("cell", False)), top_cover=spec.get("topCover"))
    if kind == "box":
        return bm.Assembly([bm.Piece("body", bm.shape_at(bm.molded_shape(l, w, h, leads=False)), "mold", photo_sides=True)])
    return None


def _to_part_frame(contact: dict, lx: float, ly: float, drot: float) -> dict:
    """A body-frame contact (``body_mesh``) in the nominal part frame of the fillets."""
    a = math.radians(drot)
    x, y = contact["cx"], contact["cy"]
    return {"cx": lx + x * math.cos(a) - y * math.sin(a), "cy": ly + x * math.sin(a) + y * math.cos(a),
            "hl": contact["hl"], "hw": contact["hw"], "angle": contact["angle"] + a}


def _reshape_part(objs: list, c: dict, kind: str, h: float, standoff: float, pads: list, lx: float, ly: float,
                  drot: float, tomb: bool, board: dict | None = None):
    """Datasheet-like geometry on render_batch's part root: chips keep render_batch's end caps (reshaped), every
    other kind is rebuilt from a ``body_mesh`` assembly (render_batch's plain body box becomes its first piece).
    Returns (contacts, solder height) overriding render_batch's, or None."""
    body = next((o for o in objs if o.type == "MESH" and o.name.startswith("body")), None)
    if body is None:
        return None
    if kind == "chip":
        role = "ceramic" if c["part"].get("body_material") == "ceramic" else "mold"
        for o in objs:
            if o.type != "MESH" or not o.name.startswith(("body", "endcap")):
                continue
            co = [v.co for v in o.data.vertices]
            l, w, hh = (max(q[i] for q in co) - min(q[i] for q in co) for i in range(3))
            if o.name.startswith("endcap"):
                _set_mesh(o, bm.chip_cap_shape(l, w, hh))
            else:
                _set_mesh(o, bm.chip_body_shape(l, w, hh))
                o.data.materials.clear()
                o.data.materials.append(_role_material(role))
                o["photo_sides"] = True
        return None
    asm = _assembly(kind, c, h, standoff, pads, board)
    if asm is None:
        return None
    loc = (body.location.x, body.location.y, 0.0)        # body frame: bottom face of the (tombstone-offset) body
    parent = body.parent
    first, *rest = asm.pieces
    old = body.data
    me = bpy.data.meshes.new(first.name)
    me.from_pydata(first.mesh.verts, [], first.mesh.faces)
    me.uv_layers.new(name="UVMap")
    me.polygons.foreach_set("use_smooth", first.mesh.smooth)
    me.update()
    me.materials.append(_role_material(first.role))
    body.data = me
    bpy.data.meshes.remove(old)
    body.location = loc
    body["photo_sides"] = first.photo_sides
    for piece in rest:
        _piece_object(piece, parent, loc)
    if tomb or asm.contacts is None:
        return None
    rb_like = [None if k is None else _to_part_frame(k, lx, ly, drot) for k in asm.contacts]
    return rb_like, asm.solder_height


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

rb.bpy = _BpyBoardMode()


def _descendants(obj):
    for child in obj.children:
        yield child
        yield from _descendants(child)


def _ensure_uv(me) -> None:
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")


def _with_materials(me, *extra) -> None:
    """Slot 0 the mesh's own material (a moulded one if it has none), then *extra* as slots 1, 2, ..."""
    if not me.materials:
        me.materials.append(_role_material("mold"))
    while len(me.materials) > 1:
        me.materials.pop(index=len(me.materials) - 1)
    for mat in extra:
        me.materials.append(mat)


def _vertex_co(me):
    """Vertex positions (N x 3, float32, object space)."""
    import numpy as np
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _loop_vertices(me):
    """Vertex index of every loop (face corner)."""
    import numpy as np
    idx = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", idx)
    return idx


def _poly_normals(me):
    """Face normals (N x 3, float32, object space)."""
    import numpy as np
    n = np.empty(len(me.polygons) * 3, np.float32)
    me.polygons.foreach_get("normal", n)
    return n.reshape(-1, 3)


def _poly_loops(me):
    """First loop and loop count of every face."""
    import numpy as np
    starts = np.empty(len(me.polygons), np.int32)
    totals = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_start", starts)
    me.polygons.foreach_get("loop_total", totals)
    return starts, totals


def _set_board_uv(me, world, raster: dict) -> None:
    """UVs of the loops at board positions *world* (loops x 3, mm) in *raster*."""
    import numpy as np
    uv = np.empty((len(me.loops), 2), np.float32)
    uv[:, 0] = (world[:, 0] - raster["x0"]) / raster["widthMm"]
    uv[:, 1] = (world[:, 1] - raster["y0"]) / raster["heightMm"]
    me.uv_layers.active.data.foreach_set("uv", uv.ravel())


def _uv_from_board(obj, raster: dict, to_nominal: Matrix | None = None) -> None:
    """UVs = board position of every vertex in *raster*; *to_nominal* maps the vertex (world) back to where it
    was in the real photo first (a moved body keeps its own pixels). Vectorised (numpy ships with Blender)."""
    import numpy as np
    me = obj.data
    _ensure_uv(me)
    mw = obj.matrix_world if to_nominal is None else to_nominal @ obj.matrix_world
    m = np.array(mw, np.float64)
    world = _vertex_co(me) @ m[:3, :3].T + m[:3, 3]
    _set_board_uv(me, world[_loop_vertices(me)], raster)


#: faces whose (nominal-frame) normal points up more than this take the photo; the rest keep their own material
PHOTO_UP = 0.5
#: walls tilted up more than this (normal z) are mould draft seen from above (``_photo_on_top``: slope_photo)
SLOPE_MIN = 0.05
#: photo faces flatter than this (normal z) are lit; the curved / sloped ones below it (a can's curled rim, rounded end
#: caps) take the unlit scan too: the scan holds their slope's shading, lit again they rendered as a dark ring
#: (electrolytic -1.8 L*)
FLAT_UP = 0.97
#: side faces of moulded / ceramic bodies take the photo inside the outline, at SIDE_INSET x the smaller half size, at
#: most SIDE_INSET_MAX_MM, one colour per wall face: a column of edge pixels per wall showed the marking (inset 30 %)
#: or the leads' highlights bleeding into the body edge (0.2 mm) as vertical streaks
SIDE_INSET, SIDE_INSET_MAX_MM = 0.3, 0.4


def _photo_on_top(obj, photo, raster: dict, to_nominal: Matrix, slope_photo=None, flat_up: float | None = None) -> None:
    """The scan on the faces that look up (slot 1), the piece's own material on the walls (slot 0): the photo of the
    top edge stretched down the walls made vertical streaks (the whole can of an electrolytic). Walls of moulded /
    ceramic bodies (``photo_sides``) take the photo too, sampled inside the outline: their own colour, no streaks.
    With *slope_photo* (the unlit scan, flatbed / room presets) the sloped walls seen from above (mould draft) show
    the scan as scanned: the scanner lights them from the side at ~45 deg, the steep render disc left them as a dark
    rim. Vertical walls stay lit (the unlit scan on every wall measured worse). *flat_up*: faces flatter than this stay
    lit (default :data:`FLAT_UP`; above 1: every face that looks up takes the unlit scan)."""
    import numpy as np
    me = obj.data
    _with_materials(me, photo, *([slope_photo] if slope_photo is not None else []))
    m = np.array(to_nominal @ obj.matrix_world, np.float64)
    nz = _poly_normals(me) @ m[:3, :3].T
    nz = nz[:, 2] / np.maximum(np.linalg.norm(nz, axis=1), 1e-9)
    up = nz > PHOTO_UP
    sides = bool(obj.get("photo_sides", False))
    index = np.where(up | sides, 1, 0)
    slope = (nz > SLOPE_MIN) & (nz < (FLAT_UP if flat_up is None else flat_up))
    if slope_photo is not None:
        index[slope] = 2          # a draft slope shows the scan at its own place: what the scanner saw there
    me.polygons.foreach_set("material_index", index.astype(np.int32))
    _ensure_uv(me)
    co = _vertex_co(me).astype(np.float64)
    lco = co[_loop_vertices(me)]
    if sides:
        starts, totals = _poly_loops(me)
        loop_up = np.zeros(len(me.loops), bool)
        for k in np.nonzero(up | slope)[0]:
            loop_up[starts[k]:starts[k] + totals[k]] = True
        side = ~loop_up
        d = min(SIDE_INSET * float(np.abs(co[:, :2]).max(axis=0).min()), SIDE_INSET_MAX_MM)
        lco[side, :2] = np.sign(lco[side, :2]) * np.maximum(np.abs(lco[side, :2]) - d, 0.0)
        face_of = np.repeat(np.arange(len(me.polygons)), totals)
        for k in range(2):                                   # one point per wall face: its loops' mean
            mean = np.bincount(face_of, lco[:, k]) / np.maximum(totals, 1)
            lco[side, k] = mean[face_of[side]]
    _set_board_uv(me, lco @ m[:3, :3].T + m[:3, 3], raster)


def _atlas_uv(obj, entry: dict, atlas_size, mat, to_nominal: Matrix, *, top_only: bool = False) -> None:
    """Per-part face atlas (``board3d.parttex``): every face of *obj* on its part's tile by its normal (in the part's
    nominal frame): up -> ``top`` (x, y), else the dominant wall ``+x`` / ``-x`` (y, z) or ``+y`` / ``-y`` (x, z),
    clamped to the tile (walls of a drafted body sit a little inside the tile's plane). *top_only*: only the faces
    looking up take the atlas, the rest keep the mesh's own material."""
    import numpy as np
    me = obj.data
    _with_materials(me, mat)
    _ensure_uv(me)
    m = np.array(to_nominal @ obj.matrix_world, np.float64)
    px, py, rot = (float(v) for v in entry["pose"])
    a = -math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)
    n = _poly_normals(me) @ m[:3, :3].T
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    nx, ny = n[:, 0] * ca - n[:, 1] * sa, n[:, 0] * sa + n[:, 1] * ca
    faces = entry["faces"]
    names = np.where(n[:, 2] > 0.5, "top", np.where(np.abs(nx) >= np.abs(ny), np.where(nx > 0, "+x", "-x"),
                                                     np.where(ny > 0, "+y", "-y")))
    # a wall no photo saw (the stills are near-frontal: each sees the walls facing away from its centre) takes the
    # opposite wall, else any seen wall (parts are mostly symmetric), else its own material; never the top stretched
    walls = [f for f in ("+x", "-x", "+y", "-y") if f in faces]
    opposite = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"}

    def _wall(f):
        if f in faces:
            return f
        if f == "top":
            return "top"
        return opposite[f] if opposite[f] in faces else (walls[0] if walls else "")
    own_x = np.isin(names, ("+x", "-x"))       # the face's own wall axis (before the fallback below)
    names = np.array([_wall(f) for f in names])
    l_, w_ = (float(v) for v in entry["size"][:2])
    half_l, half_w = max(l_ / 2, 1e-6), max(w_ / 2, 1e-6)
    world = _vertex_co(me).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
    lx = (world[:, 0] - px) * ca - (world[:, 1] - py) * sa
    ly = (world[:, 0] - px) * sa + (world[:, 1] - py) * ca
    lz = world[:, 2]
    idx = _loop_vertices(me)
    _starts, totals = _poly_loops(me)
    face_of = np.repeat(np.arange(len(me.polygons)), totals)
    fname = names[face_of]
    uv = np.zeros((len(me.loops), 2), np.float32)
    aw, ah = (float(v) for v in atlas_size)
    for f, info in faces.items():
        sel = fname == f
        if not sel.any() or "rect" not in info:
            continue
        u0, u1, v0, v1 = info["bounds"]
        rx, ry, rw, rh = info["rect"]
        vi = idx[sel]
        if f == "top":
            u, v = lx[vi], ly[vi]
        else:
            # along the loop's own wall (a borrowed tile of the other axis is stretched over this wall's length)
            on_x = own_x[face_of[sel]]
            half = np.where(on_x, half_w, half_l)
            along = np.where(on_x, ly[vi], lx[vi])
            u = u0 + (along + half) / (2 * half) * (u1 - u0)
            v = lz[vi]
        s = np.clip((u - u0) / max(u1 - u0, 1e-6), 0.0, 1.0)
        t = np.clip((v - v0) / max(v1 - v0, 1e-6), 0.0, 1.0)
        col = rx + 0.5 + s * (rw - 1.0)
        row = ry + 0.5 + (1.0 - t) * (rh - 1.0)
        uv[sel, 0] = col / aw
        uv[sel, 1] = 1.0 - row / ah
    me.uv_layers.active.data.foreach_set("uv", uv.ravel())
    on_atlas = (names == "top") if top_only else (names != "")
    me.polygons.foreach_set("material_index", on_atlas.astype(np.int32))


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
    kind = _board_kind(c["part"])
    if height > 0.0 and not c.get("missing") and kind == "can":
        contacts, h_solder = _build_can(c, lx, ly, drot, height)
    elif height > 0.0 and not c.get("missing"):
        tomb = c.get("tombstone")
        # The catalog height is the top above the board: the body is that minus its standoff (render_batch stacks
        # the body height on top of the standoff, every part stood a standoff too tall).
        standoff = float(c["part"].get("standoff", 0.0)) * height
        h_body = max(height - standoff, 0.05)
        part = {**c["part"], "standoff": standoff / h_body}
        if kind == "gullwing" and not tomb:
            part["leads"] = False            # the bent leads come from body_mesh.gullwing
        # render_batch scene convention: image axes (y down), rotation in image degrees (clockwise positive)
        board = {"body": {"center": [lx, -ly], "length": float(c["l"]), "width": float(c["w"]), "rotation_deg": -drot,
                          "lead_frame_deg": 0.0, "tilt_deg": float(tomb["tiltDeg"]) if tomb else 0.0,
                          "is_tombstone": bool(tomb), "raised_sign": float(tomb["raisedSign"]) if tomb else 1.0},
                 "part": part, "pads": [], "thermal_pad": c.get("thermal")}
        _objs, contacts, h_solder = rb._build_part(fp_pads, board, 1.0, h_body, {}, rng)
        override = _reshape_part(_objs, c, kind, h_body, standoff, fp_pads, lx, ly, drot, bool(tomb), job.get("board"))
        if override is not None:
            new_contacts, new_h = override
            contacts = [n if n is not None else o for n, o in zip(new_contacts, contacts)]
            h_solder = new_h if new_h is not None else h_solder
    excess = set(c.get("excess") or [])
    grow, tall = float(c.get("excessGrow", EXCESS_GROW)), float(c.get("excessHeight", EXCESS_HEIGHT))
    residue = c.get("residue") or {}
    tomb_dome = float((c.get("tombstone") or {}).get("domeMm") or 0.0)
    joints = []
    for i, (pad, contact) in enumerate(zip(fp_pads, contacts)):
        px, py, pw, ph = pad
        if i in excess:
            # too much solder: a convex blob on a land grown by ``grow``, ``tall`` x the nominal fillet, in shiny tin
            # (the photo has no blob to show)
            _blob(f"solder_{c['refdes']}_{i}", px, py, pw * grow, ph * grow, max(h_solder, 0.12) * tall, mats["solder_excess"])
            continue
        if c.get("missing") and residue.get("kind") == "tinned" and not c.get("dnp"):
            # part pulled off by hand: the lands keep a shiny solder dome (not the flat, scanned paste)
            _blob(f"solder_{c['refdes']}_{i}", px, py, pw * 0.95, ph * 0.95, float(residue["domeMm"]), mats["solder_excess"])
            continue
        if tomb_dome and contact is None:
            # the land the raised end left: its solder balled up into a shiny dome
            _blob(f"solder_{c['refdes']}_{i}", px, py, pw * 0.9, ph * 0.9, tomb_dome, mats["solder_excess"])
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


def _build_can(c: dict, lx: float, ly: float, drot: float, height: float):
    """SMD aluminium electrolytic (``part.body == "can"``, :func:`body_mesh.electrolytic`): base plate with the
    polarity chamfers on two flat terminals, the can with its bead and curled rim. The photo lands on the top faces
    (the can's print and vent), the walls keep aluminium / black plastic."""
    root = bpy.data.objects.new("part_root", None)
    bpy.context.collection.objects.link(root)
    root.location = (lx, ly, 0.0)
    root.rotation_euler = (0.0, 0.0, math.radians(drot))
    asm = bm.electrolytic(float(c["l"]), float(c["w"]), height, c["pads"])
    for piece in asm.pieces:
        _piece_object(piece, root, (0.0, 0.0, 0.0))
    return [_to_part_frame(k, lx, ly, drot) for k in asm.contacts], asm.solder_height


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


#: zinc-plated steel of an edge bracket (linear RGB): neutral. Fitted to the front photos it came out warm (0.43, 0.17,
#: 0.11), but that tint is the wooden table mirrored in the plate, which the scene does not have: the plate's back then
#: rendered copper. Neutral 0.30 costs the front views 0.26 Delta E and gives the back views as much.
BRACKET_RGB = (0.30, 0.30, 0.30)


def _build_brackets(job: dict) -> None:
    """Edge brackets (``job["brackets"]``, :func:`simple_sim.board3d.scene.bracket_job`): a plate with the connectors'
    cut-outs, zinc-plated steel. Built along x with its thickness along y, turned for an x edge."""
    mat = None
    for k, b in enumerate(job.get("brackets") or []):
        mat = mat or rb._material("mat_bracket", BRACKET_RGB, roughness=0.45, metallic=1.0)
        edge, t, inner = b["edge"], float(b["thicknessMm"]), float(b["innerMm"])
        n0, n1 = (inner, inner + t) if edge.endswith("+") else (inner - t, inner)
        holes = [tuple(float(v) for v in h) for h in b.get("holes", [])]
        for i, m in enumerate(bm._plate_with_holes(float(b["fromMm"]), float(b["toMm"]), float(b["bottomMm"]), float(b["topMm"]),
                                                   n0, n1, holes)):
            verts, faces = m.verts, m.faces
            if edge[0] == "x":               # swap x / y: a mirror, so the faces turn round to stay outward
                verts, faces = [(y, x, z) for x, y, z in verts], [tuple(reversed(f)) for f in faces]
            me = bpy.data.meshes.new(f"bracket_{k}_{i}")
            me.from_pydata(verts, [], faces)
            me.update()
            me.materials.append(mat)
            bpy.context.collection.objects.link(bpy.data.objects.new(f"bracket_{k}_{i}", me))


def _build_sheets(job: dict) -> None:
    """Flat plates parallel to the board (``job["sheets"]``, :func:`simple_sim.board3d.scene.sheet_job`): an insulating
    foil or base plate under the board, a rounded rectangle of its thickness."""
    for k, sh in enumerate(job.get("sheets") or []):
        l, w, t = sh["x1"] - sh["x0"], sh["y1"] - sh["y0"], sh["thicknessMm"]
        r = min(sh["cornerMm"], 0.49 * min(l, w))
        m = bm.shape_at(bm.BodyShape(l, w, t, r_corner=r, corner_segments=6 if r > 0 else 1, top_segments=1),
                        (sh["x0"] + sh["x1"]) / 2, (sh["y0"] + sh["y1"]) / 2, sh["zTopMm"] - t)
        me = bpy.data.meshes.new(f"sheet_{k}")
        me.from_pydata(m.verts, [], m.faces)
        me.update()
        mat = rb._material(f"mat_sheet_{k}", tuple(sh["rgb"]), roughness=sh["roughness"], metallic=sh["metallic"])
        mat.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = sh["specular"]
        me.materials.append(mat)
        bpy.context.collection.objects.link(bpy.data.objects.new(f"sheet_{k}", me))


def _build_tht_joints(job: dict, mats: dict) -> None:
    """The top side of every soldered through-hole pin of the job (``thtJoints``, the scan's plated holes under the
    populated THT parts): solder meniscus and pin end, in tin."""
    for k, h in enumerate(job.get("thtJoints") or []):
        for piece in bm.tht_joint(float(h["x"]), float(h["y"]), float(h["r"])):
            me = bpy.data.meshes.new(f"{piece.name}_{k}")
            me.from_pydata(piece.mesh.verts, [], piece.mesh.faces)
            me.polygons.foreach_set("use_smooth", piece.mesh.smooth)
            me.update()
            me.materials.append(mats["solder"] if piece.name == "tht_solder" else _role_material("metal"))
            obj = bpy.data.objects.new(f"{piece.name}_{h.get('refdes', '')}_{k}", me)
            bpy.context.collection.objects.link(obj)


def _build_bridges(job: dict, mats: dict) -> None:
    """Solder bars between two lands (op ``bridge``): land centre to land centre, ``widthScale`` (default 85 %) of the
    narrower land wide, ``heightMm`` (default 0.2) high, in shiny tin."""
    for k, b in enumerate(job.get("bridges") or []):
        (ax, ay, aw, ah), (bx, by, bw, bh) = b["a"], b["b"]
        width = float(b.get("widthScale", 0.85)) * min(aw, ah, bw, bh)
        height = float(b.get("heightMm", 0.2))
        bar = _bridge_mesh(f"bridge_{k}", Vector((ax, ay, 0.0)), Vector((bx, by, 0.0)), width, height)
        bar.data.materials.append(mats["solder_excess"])


def _apply_photo_uvs(pending: list, job: dict, mats: dict) -> None:
    bpy.context.view_layer.update()
    raster = job["photo"]["raster"]
    atlas = job.get("partAtlas") if "atlas" in mats else None
    for c, root, part_root, joints, z in pending:
        # Solder stays on the lands: the joints show the scan pixels where they are. Defect solder keeps the metal.
        for o in joints:
            if (job.get("partAtlas") or {}).get("look") == "clean":
                break                        # clean look: the joints keep the solder metal
            if c.get("missing"):
                # removed part: the scan shows its body over the lands (an electrolytic's can came back as strips on
                # the pads); the lands' scanned finish, unlit (the solder metal under the steep light saturated)
                if "pad_unlit" in mats:
                    o.data.materials.clear()
                    o.data.materials.append(mats["pad_unlit"])
                continue
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
        entry = (atlas or {}).get("parts", {}).get(c["refdes"])
        if entry is not None and "top" not in entry["faces"]:
            entry = None                     # gated off (tall part whose top predicts badly): the board photo as before
        clean = (atlas or {}).get("look") == "clean"
        # Flatbed: a part at its nominal place shows the scan unlit on every face that looks up: lit, its top took the
        # shadows of the tall parts next to it (an electrolytic's can on its own base: -10 L*), which the scan holds
        # already. A part a defect moves, lifts or tips stays lit: its shading has to change. ``light.scanNominal: false``
        # lights every part (checks the body / light model, ``board3d_render.py --model-check``).
        light = job.get("light") or {}
        nominal = (c["pose"] == c["nominal"] and not c.get("tombstone")
                   and light.get("preset", "flatbed") == "flatbed" and light.get("scanNominal", True))
        flat_up = 2.0 if nominal else None
        for o in [*_descendants(part_root), *leads]:
            if o.type != "MESH" or not o.name.startswith(PHOTO_PREFIXES):
                continue                     # openings, inserts, shells keep their material
            if (clean and o.name.startswith("body_shell") and entry is not None and "top" in entry["faces"]
                    and (c["part"].get("connector") or {}).get("kind") in PHOTO_TOP_SHELLS):
                # a card socket's sheet shell: the photographed top on the faces looking up (embossing, print), the
                # walls stay metal
                _atlas_uv(o, entry, atlas["size"], mats["atlas"], to_nominal, top_only=True)
                continue
            if clean and (not o.name.startswith(("body", "lid")) or o.name.startswith(("body_shell", "body_flange"))
                          or c["part"].get("body") == "can" and o.name.startswith("body_can")):
                continue                     # clean look: caps, leads, terminals, shields, cans stay metal
            if clean and o.name.startswith(("body_wall", "body_key")):
                # an open-top connector's shroud keeps its nylon: seen from above its "top" is the dark opening, so the
                # atlas's body colour came out grey (white JST rows rendered 12 L* too dark on the phone views)
                continue
            if o.name.startswith(("cavity", "pin_")) and mats.get("photo_joint") is not None:
                # down an open connector's shaft the scan already holds the shaft's shading: lit again under the
                # steep light the floor came out ~4 L* too dark. The unlit scan, as for the joints.
                _photo_on_top(o, mats["photo_joint"], raster, to_nominal)
                continue
            if entry is not None:
                # per-part face textures from all photos (board3d.parttex)
                _atlas_uv(o, entry, atlas["size"], mats["atlas"], to_nominal)
            else:
                _photo_on_top(o, mats["photo"], raster, to_nominal, mats.get("photo_joint"), flat_up=flat_up)


def _lights(job: dict) -> None:
    """``flatbed``: a disc light straight above, invisible to the camera, seen from the board under a cone of
    ``coneDeg`` half-angle (the scanner lights steeply from above: little occlusion next to tall parts, flat metal
    facing up turns bright), plus a dim white surround (``world``). The disc's strength makes a flat diffuse surface
    render at its texture colour: irradiance = pi * (L * sin^2(cone) + world * cos^2(cone)) (the disc hides the
    part of the sky it covers; the old ``L = (1 - world) / sin^2`` rendered 2 % too dark at 20 deg / 0.15, more at
    wider cones or more ambient). ``aoi``: the ring + top rig of
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
        if light.get("groundRgb"):
            # below the horizon the desk (its colour measured on the photo): flat surfaces facing up never see it, so
            # the texture calibration holds; walls and metal now mirror the desk instead of a white sky (RJ45 / USB
            # shields, the coin cell rendered 10-30 L* too bright)
            _desk_horizon(world, bg, [float(v) for v in light["groundRgb"]], _room_gain(job))
        glossy = float(light.get("glossyWorld", 1.0))
        if glossy != 1.0:
            _glossy_world(world, bg, glossy)
        return
    if light.get("preset", "flatbed") != "flatbed":
        rb._aoi_lights({}, max(win["widthMm"], win["heightMm"]))
        return
    ambient = float(light.get("world", 0.15))
    gain = float(light.get("gain", 1.0))
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = ambient * gain
    cone = math.radians(float(light.get("coneDeg", 20.0)))
    # far above: a whole-board window lost ~1.5 % irradiance towards its corners (off-axis view of the disc) at 4x
    height = 12.0 * max(win["widthMm"], win["heightMm"]) + 100.0
    radius = height * math.tan(cone)
    bpy.ops.mesh.primitive_circle_add(vertices=64, radius=radius, fill_type="NGON", location=(cx, cy, height),
                                      rotation=(math.pi, 0.0, 0.0))
    disc = bpy.context.active_object
    disc.name = "flatbed_light"
    strength = gain * (1.0 - ambient * math.cos(cone) ** 2) / max(math.sin(cone) ** 2, 1e-4)
    disc.data.materials.append(rb._material("mat_flatbed_light", (0, 0, 0), roughness=1.0, emission=strength))
    disc.visible_camera = False
    disc.visible_shadow = False


def _desk_horizon(world, bg, ground_rgb: list[float], sky_rgb) -> None:
    """World colour *ground_rgb* below the horizon, *sky_rgb* above (a narrow blend at z = 0)."""
    nodes, links = world.node_tree.nodes, world.node_tree.links
    coord, sep = nodes.new("ShaderNodeTexCoord"), nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = -0.05, 0.05
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*ground_rgb, 1.0)
    mix.inputs["B"].default_value = (*sky_rgb, 1.0)
    links.new(coord.outputs["Generated"], sep.inputs["Vector"])
    links.new(sep.outputs["Z"], ramp.inputs["Value"])
    links.new(ramp.outputs["Result"], mix.inputs["Factor"])
    links.new(mix.outputs["Result"], bg.inputs["Color"])


def _glossy_world(world, bg, factor: float) -> None:
    """The world as a mirror sees it: *factor* x its strength for glossy rays, unchanged for diffuse ones. A uniform
    sky lights a flat diffuse surface at its texture colour (the calibration), but a room is not uniform: a few
    bright lamps, the rest dimmer, and a mirror mostly sees the dim part. With the full sky metal and glossy moulds
    turned milky white in phone views (coin cell, microSD shell, black BGA tops)."""
    nodes, links = world.node_tree.nodes, world.node_tree.links
    path = nodes.new("ShaderNodeLightPath")
    mul = nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY_ADD"            # strength = is_glossy * (factor - 1) + 1
    mul.inputs[1].default_value = factor - 1.0
    mul.inputs[2].default_value = 1.0
    links.new(path.outputs["Is Glossy Ray"], mul.inputs[0])
    links.new(mul.outputs["Value"], bg.inputs["Strength"])


def _export_glb(path: str) -> None:
    """Board + components as one GLB for the browser viewer (glTF is y-up: board (x, y) -> (x, -z)). Every component is
    an empty ``cmp_<refdes>`` at its nominal pose with body / leads / joints below it; modifiers (bevels) applied.
    Lights and camera are left out: the viewer lights the scene itself."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", export_apply=True, export_cameras=False,
                              export_lights=False, export_extras=True, export_yup=True)


def _configure(job: dict) -> None:
    """Empty scene, render size / samples / device; Cycles without denoiser and with a scan-narrow pixel filter."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rb._configure_render(width_px=int(job["widthPx"]), height_px=int(job["heightPx"]), samples=int(job.get("samples", 64)),
                         seed=int(job.get("seed", 0)), device=str(job.get("device", "GPU")), engine=str(job.get("engine", "CYCLES")))
    scene = bpy.context.scene
    if scene.render.engine == "CYCLES":
        # The photo textures are noise-free; the denoiser would only smear them. Pixel filter as narrow as the scan.
        scene.cycles.use_denoising = bool(job.get("denoise", False))
        scene.cycles.filter_width = float(job.get("filterWidthPx", 1.0))


def _light_preset(job: dict) -> str:
    return (job.get("light") or {}).get("preset", "flatbed")


def _build_board(job: dict) -> None:
    """The substrate plane (the scan) and, with ``board``, the board slab under it."""
    # The scan already holds the scanner's sheen: photo-textured surfaces get no extra specular (else the steep
    # light's reflection is counted twice).
    if _light_preset(job) == "flatbed" and not (job.get("light") or {}).get("boardShadows", False):
        # The substrate is the scan outside the parts: it already holds the scanner's shading next to every part.
        # Lit again, the parts' shadows were counted twice (-5..-7 L* at a tall part's edge). The unlit scan is a lit
        # board without shadows (same radiance onto the parts' walls); a moved / removed part keeps the shading of its
        # nominal place (``boardShadows: true``: the lit board, as before).
        board_mat = _unlit_texture_material("mat_substrate", job["substrate"]["path"], _room_gain(job))
    else:
        board_mat = _texture_material("mat_substrate", job["substrate"]["path"], roughness=0.9, specular=0.0)
    _board_plane(job["substrate"]["raster"], job.get("board"), board_mat)
    if job.get("board"):
        _board_slab(job["board"])


def _part_materials(job: dict) -> dict:
    """The shared materials of the components: solder, the photo (joints, pads) and the part atlas."""
    mats = {"solder": rb._cfg_material("mat_solder", {}, (0.72, 0.72, 0.74), 0.22, 1.0, bump=0.02)}
    preset = _light_preset(job)
    if job.get("photo"):
        mats["photo"] = _texture_material("mat_photo", job["photo"]["path"], roughness=0.9, specular=0.0)
        if preset in ("flatbed", "room"):
            # Flatbed: nominal joints show the scanned solder as it was lit (slopes would darken under the steep
            # light, while the scanner made them bright). Other presets light them (T4).
            gain = _room_gain(job) if preset == "room" else (1.0, 1.0, 1.0)
            mats["photo_joint"] = _unlit_texture_material("mat_photo_joint", job["photo"]["path"], gain)
            if job.get("padBgr"):
                mats["pad_unlit"] = _unlit_colour_material("mat_pad_unlit", job["padBgr"], gain)
    if job.get("photo") and job.get("partAtlas"):
        if job["partAtlas"].get("look") == "clean":
            # clean look: a real moulded / ceramic surface (glossy, specular) coloured by the atlas
            mats["atlas"] = _texture_material("mat_photo_atlas_clean", job["partAtlas"]["path"], roughness=0.42, specular=0.5)
        else:
            mats["atlas"] = _texture_material("mat_photo_atlas", job["partAtlas"]["path"], roughness=0.9, specular=0.0)
    return mats


def _build_parts(job: dict, mats: dict) -> None:
    """Components (photo UVs in one pass at the end), bridges, through-hole joints, brackets and sheets."""
    rng = random.Random(int(job.get("seed", 0)))
    mats["solder_excess"] = rb._material("mat_solder_excess", (0.72, 0.72, 0.73), roughness=0.5, metallic=0.55, bump=0.02)
    preset = _light_preset(job)
    if job.get("solderBgr") and preset in ("flatbed", "room"):
        gain = _room_gain(job) if preset == "room" else (1.0, 1.0, 1.0)
        mats["solder_excess"] = _solder_photo_material("mat_solder_photo", job["solderBgr"], gain)
    pending: list = []
    for c in job["components"]:
        _build_component(c, job, mats, rng, pending)
    if pending:
        _apply_photo_uvs(pending, job, mats)
    _build_bridges(job, mats)
    _build_tht_joints(job, mats)
    _build_brackets(job)
    _build_sheets(job)


def _camera(job: dict):
    """A real photo's camera (``camera.type == "PERSP"``, :func:`simple_sim.board3d.camera.blender_camera`: pose,
    lens, full photo frame), else an orthographic camera straight above the window."""
    cam_cfg = job.get("camera")
    if cam_cfg and cam_cfg.get("type") == "PERSP":
        cam_data = bpy.data.cameras.new("photo_camera")
        cam = bpy.data.objects.new("photo_camera", cam_data)
        bpy.context.scene.collection.objects.link(cam)
        rot = Matrix([list(row) for row in cam_cfg["rotation"]])
        cam.matrix_world = Matrix.Translation(Vector(cam_cfg["locationMm"])) @ rot.to_4x4()
        cam_data.sensor_fit = "AUTO"
        cam_data.sensor_width = float(cam_cfg["sensorWidthMm"])
        cam_data.lens = float(cam_cfg["lensMm"])
        cam_data.clip_start, cam_data.clip_end = 1.0, 5000.0
        return cam
    win = job["window"]
    cx, cy = win["x0"] + win["widthMm"] / 2, win["y0"] + win["heightMm"] / 2
    bpy.ops.object.camera_add(location=(cx, cy, 30.0))
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(win["widthMm"], win["heightMm"])
    cam.data.clip_end = 1000.0
    rb._look_at(cam, Vector((cx, cy, 0.0)))
    return cam


def main() -> None:
    t_start = time.monotonic()
    global INTERPOLATION
    job = json.loads(Path(_parse_args().job).read_text(encoding="utf-8"))
    INTERPOLATION = str(job.get("interpolation", INTERPOLATION))
    _configure(job)
    scene = bpy.context.scene
    _build_board(job)
    _build_parts(job, _part_materials(job))

    if job.get("exportGlb"):
        _export_glb(job["exportGlb"])
        if not job.get("render", True):
            print(f"[render_board3d] {len(job['components'])} components -> {job['exportGlb']}", flush=True)
            return

    _lights(job)
    if job.get("shadowCatcher"):
        _shadow_catcher(job)
    scene.camera = _camera(job)
    if job.get("cropPx"):
        # only a region of the frame ([u0, v0, u1, v1] in px, y down), written at its own size: a part's crop of a
        # photo view (defect fit) costs its few pixels instead of the whole frame
        u0, v0, u1, v1 = (float(v) for v in job["cropPx"])
        w, h = float(job["widthPx"]), float(job["heightPx"])
        scene.render.use_border, scene.render.use_crop_to_border = True, True
        scene.render.border_min_x, scene.render.border_max_x = max(0.0, u0 / w), min(1.0, u1 / w)
        scene.render.border_min_y, scene.render.border_max_y = max(0.0, 1.0 - v1 / h), min(1.0, 1.0 - v0 / h)
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out)
    t_build = time.monotonic()
    bpy.ops.render.render(write_still=True)
    t_end = time.monotonic()
    print(f"[render_board3d] {len(job['components'])} components -> {out} | build {t_build - t_start:.1f} s, "
          f"render {t_end - t_build:.1f} s ({scene.cycles.device if scene.render.engine == 'CYCLES' else scene.render.engine})",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # Blender exits 0 after a script error: make the failure visible to the caller
        import traceback
        traceback.print_exc()
        sys.exit(2)
