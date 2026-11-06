"""Blender batch renderer for Simple-Sim (Cycles or EEVEE).

Run via (normally through ``simple_sim.generators.blender3d.render_blender_batch``):
  blender --background --factory-startup --python render_batch.py -- --jobs jobs.jsonl --out_root <dir> --samples 64

Each JSONL job carries the usual sample fields (``image_path``, ``seed``, ``mm_per_px``,
``roi_width_px``, ``roi_height_px``, ``component_height_mm``, ``augment``, ``render_3d``) plus a
``board`` entry added by the runner (see ``blender3d/bake.py``)::

  "board": {"texture": "<png>", "scale": 2, "family": "resistor|capacitor|sot|soic|qfn",
            "pads": [[cx, cy, w, h], ...], "thermal_pad": index | null,
            "body": null | {"center": [x, y], "length", "width", "rotation_deg", "tilt_deg",
                            "is_tombstone", "raised_sign"}}

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

# Height / body-width ratios of real packages (0603: 0.45/0.8, SOT-23: 1.0/1.3, SOIC: 1.5/3.9, QFN: 0.9/5).
_HEIGHT_RATIO = {"resistor": 0.56, "capacitor": 0.56, "sot": 0.75, "soic": 0.38, "qfn": 0.18}
_RESISTOR_CODES = ("103", "472", "1002", "000", "0", "221", "4R7", "100", "330", "10K", "1R0", "5602")
_MARK_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--engine", default="BLENDER_EEVEE", help="Render engine (BLENDER_EEVEE or CYCLES)")
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


def _fillet(name: str, pad, contact: dict[str, float] | None, h_term: float, big: bool, mat) -> bpy.types.Object:
    """Solder as a height-field mesh over the pad.

    With a part terminal on the pad: concave fillet rising towards the terminal. Without one
    (MISSING, raised tombstone end, part shifted off the pad): reflowed paste dome; large thermal
    pads get window-pane islands like the stencil prints them.
    """
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

    verts = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            rim = min(1.0, min(x - xs[0], xs[-1] - x, y - ys[0], ys[-1] - y) / rim_w)
            if touched:
                t = 1.0 - max(0.0, dist[j][i]) / max(dmax, 1e-6)
                z = h_term * t ** 1.8
            else:
                u = ((i / n * nx) % 1.0) * 2 - 1
                v = ((j / n * ny) % 1.0) * 2 - 1
                z = 0.3 * h_term * max(0.0, 1.0 - (abs(u) ** 3 + abs(v) ** 3) * (0.85 if nx * ny == 1 else 1.2))
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

def _build_part(pads: list[tuple[float, float, float, float]], board: dict[str, Any], mmpp: float, comp_h_mm: float,
                render_3d: dict[str, Any], rng: random.Random) -> tuple[list[bpy.types.Object], list, float]:
    """Body, end caps / leads, pin-1 mark and marking.

    Returns (objects, per-pad contact rect or None, solder height). The contact is what the solder
    wets: the terminal / body edge for chips and QFN, the lead foot for SOT/SOIC.
    """
    family, body = board["family"], board["body"]
    if body is None:
        return [], [None] * len(pads), 0.25 * min(min(p[2], p[3]) for p in pads)

    mats = render_3d.get("materials") or {}
    L, W = float(body["length"]) * mmpp, float(body["width"]) * mmpp
    H = min(comp_h_mm, _HEIGHT_RATIO[family] * min(L, W))
    angle = -math.radians(float(body["rotation_deg"]))  # image y-down -> Blender y-up
    cx, cy = float(body["center"][0]) * mmpp, -float(body["center"][1]) * mmpp
    axis = Vector((math.cos(angle), math.sin(angle), 0.0))
    standoff = 0.08 * H if family in ("sot", "soic") else 0.01 * H

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
        proj = L * math.cos(tilt) + H * math.sin(tilt)
        mid = anchor + axis * sign * (proj / 2)
        contact = {"cx": mid.x, "cy": mid.y, "hl": proj / 2, "hw": W / 2, "angle": angle}
    else:
        root.location = (cx, cy, standoff)
        root.rotation_euler = (0.0, 0.0, angle)

    if family == "capacitor":
        m_body = _material("mat_ceramic", (0.42, 0.26, 0.14), roughness=0.55, bump=0.02)
    else:
        m_body = _cfg_material("mat_body", mats.get("body") or {}, (0.02, 0.02, 0.02), 0.5, 0.0, bump=0.03)
    objs: list[bpy.types.Object] = [root]
    objs.append(_box("body", (L, W, H), (offset, 0.0, H / 2), m_body, parent=root, bevel=0.06 * min(W, H)))

    if family in ("resistor", "capacitor"):
        m_cap = _material("mat_endcap", (0.75, 0.75, 0.74), roughness=0.3, metallic=1.0, bump=0.05)
        cap = (0.26 if family == "capacitor" else 0.22) * L
        for s in (-1, 1):
            objs.append(_box(f"endcap_{s}", (cap, W * 1.02, H * 1.02), (offset + s * (L - cap) / 2, 0.0, H / 2), m_cap,
                             parent=root, bevel=0.1 * min(W, H)))
        if family == "resistor":
            m_ink = _material("mat_ink", (0.85, 0.85, 0.82), roughness=0.8)
            _text("marking", rng.choice(_RESISTOR_CODES), 0.42 * W, (offset, 0.0, H + 0.0005), m_ink, root)
        return objs, [contact] * len(pads), 0.6 * H

    # ICs: pin-1 dimple and laser marking.
    m_dot = _material("mat_pin1", (0.01, 0.01, 0.01), roughness=0.25)
    bpy.ops.mesh.primitive_cylinder_add(radius=max(0.035 * min(L, W), 0.01), depth=0.0004,
                                        location=(offset - L / 2 + 0.12 * L, W / 2 - 0.15 * W, H + 0.0002))
    dot = bpy.context.active_object
    dot.data.materials.append(m_dot)
    dot.parent = root
    m_laser = _material("mat_laser", (0.22, 0.22, 0.21), roughness=0.9)
    lines = 1 if family == "sot" else 3
    size = 0.32 * W if family == "sot" else min(0.13 * W, 0.085 * L)
    for i in range(lines):
        text = "".join(rng.choice(_MARK_CHARS) for _ in range(rng.randint(3, 6)))
        _text(f"marking_{i}", text, size, (offset, ((lines - 1) / 2 - i) * size * 1.6, H + 0.0005), m_laser, root)

    if family in ("sot", "soic") and not body.get("is_tombstone"):
        m_lead = _cfg_material("mat_lead", mats.get("solder") or {}, (0.7, 0.7, 0.72), 0.25, 1.0)
        contacts, t = [], 0.0
        for k, (px, py, pw, ph) in enumerate(pads):
            rel = Vector((px - cx, py - cy, 0.0))
            loc = Vector((rel.dot(axis), rel.x * -axis.y + rel.y * axis.x, 0.0))  # pad in body frame
            edge = Vector((max(-L / 2, min(L / 2, loc.x)), max(-W / 2, min(W / 2, loc.y)), 0.0))
            to_world = lambda v: Vector((cx, cy, 0.0)) + axis * v.x + Vector((-axis.y, axis.x, 0.0)) * v.y
            start, foot = to_world(edge * 0.98), Vector((px, py, 0.0))
            lw = min(0.55 * min(pw, ph), 0.15 * max(L, W))
            t = 0.25 * lw
            shoulder = start + Vector((0, 0, standoff + 0.45 * H))
            knee = start + (foot - start) * 0.3 + Vector((0, 0, standoff + 0.45 * H))
            ankle = start + (foot - start) * 0.55 + Vector((0, 0, t / 2))
            for seg, (a, b) in enumerate(((shoulder, knee), (knee, ankle), (ankle, foot + Vector((0, 0, t / 2))))):
                objs.append(_box_between(f"lead_{k}_{seg}", a, b, lw, t, m_lead))
            d = foot - ankle
            mid = (foot + ankle) / 2
            contacts.append({"cx": mid.x, "cy": mid.y, "hl": d.length / 2, "hw": lw / 2, "angle": math.atan2(d.y, d.x)})
        return objs, contacts, 0.9 * t
    if family in ("sot", "soic"):
        return objs, [None] * len(pads), 0.5 * H  # tombstoned leaded part: leads lifted, paste only
    return objs, [contact] * len(pads), 0.3 * H


# --------------------------------------------------------------------------- scene

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
            scene.cycles.device = "GPU" if str(device).upper() != "CPU" else "CPU"

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


def _ring_light(render_3d: dict[str, Any], roi_mm: float) -> None:
    """AOI ring light around the optical axis (``render_3d.lighting.ring_light``: strength, height_mm, radius_mm).

    The default strength is calibrated so the board renders at its baked (photo-calibrated) colours.
    """
    ring_cfg = (render_3d.get("lighting") or {}).get("ring_light") or {}
    height = float(ring_cfg.get("height_mm", 2.0 * roi_mm))
    radius = float(ring_cfg.get("radius_mm", height * math.tan(math.radians(35.0))))
    bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=0.06 * radius, location=(0.0, 0.0, height))
    ring = bpy.context.active_object
    ring.name = "ring_light"
    ring.data.materials.append(_material("mat_ring", (0, 0, 0), roughness=1.0,
                                         emission=float(ring_cfg.get("strength", 14.0))))
    ring.visible_camera = False


def _build_scene_for_job(job: dict[str, Any], *, samples: int, device: str, engine: str) -> None:
    seed = int(job.get("seed", 0))
    rng = random.Random(seed)
    mmpp = float(job["mm_per_px"])
    w_px, h_px = int(job["roi_width_px"]), int(job["roi_height_px"])
    roi_w, roi_h = w_px * mmpp, h_px * mmpp
    board = job["board"]
    render_3d = job.get("render_3d") or {}
    _configure_render(width_px=w_px, height_px=h_px, samples=samples, seed=seed, device=device, engine=engine)

    scale = float(board.get("scale", 2))
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "board"
    plane.scale = (roi_w * scale, roi_h * scale, 1.0)
    plane.data.materials.append(_board_material(board["texture"]))
    scene_objs = [plane]

    pads = [(p[0] * mmpp, -p[1] * mmpp, p[2] * mmpp, p[3] * mmpp) for p in board["pads"]]
    part, contacts, h_solder = _build_part(pads, board, mmpp, float(job.get("component_height_mm") or 0.45),
                                           render_3d, rng)
    scene_objs += [o for o in part if o.parent is None]
    m_solder = _cfg_material("mat_solder", (render_3d.get("materials") or {}).get("solder") or {},
                             (0.7, 0.7, 0.72), 0.18, 1.0, bump=0.02)
    for i, (pad, contact) in enumerate(zip(pads, contacts)):
        big = i == board.get("thermal_pad")  # window-pane stencil, as in 2D
        scene_objs.append(_fillet(f"solder_{i}", pad, contact, h_solder, big, m_solder))

    # Global in-plane orientation (cardinal 0/90/180/270 + jitter), like the 2D rotation.
    rot = float((job.get("augment") or {}).get("rotation_deg", 0.0))
    if abs(rot) > 1e-6:
        root = bpy.data.objects.new("global_rotation_root", None)
        bpy.context.collection.objects.link(root)
        for obj in scene_objs:
            obj.parent = root
        root.rotation_euler[2] = math.radians(rot)

    roi_mm = max(roi_w, roi_h)
    _ring_light(render_3d, roi_mm)

    # Orthographic camera framing exactly the ROI (top-down unless the profile tilts it).
    cam_cfg = render_3d.get("camera") or {}
    loc = cam_cfg.get("location_mm")
    loc = tuple(float(v) for v in loc) if isinstance(loc, (list, tuple)) and len(loc) == 3 else (0.0, 0.0, 9.0)
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = roi_mm
    cam.data.clip_end = 1000.0
    look = cam_cfg.get("look_at_mm")
    look = Vector(tuple(float(v) for v in look)) if isinstance(look, (list, tuple)) and len(look) == 3 else Vector()
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
