"""Blender batch renderer for Simple-Sim (Cycles).

Run via:
  blender --background --factory-startup --python render_batch.py -- --jobs jobs.jsonl --out_root <dir> --samples 64

Jobs file format (JSONL), one per line:
{
  "image_path": "images/000123.png",
  "seed": 123,
  "mm_per_px": 0.01,
  "roi_width_px": 256,
  "roi_height_px": 256,
  "footprint": "chip_2pad|sot23|qfn_32",
  "component_height_mm": 0.45,
  "nominal": {...},
  "defect": {...},
  "render_3d": {...}   # optional profile defaults (camera/lighting/materials)
}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bpy
from mathutils import Vector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--device", default="CPU")
    # Blender passes custom args after a `--` separator.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    return parser.parse_args(argv)


def _clean_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _new_material(name: str, *, base_color=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0)
        bsdf.inputs["Roughness"].default_value = float(roughness)
        bsdf.inputs["Metallic"].default_value = float(metallic)
    return mat


def _apply_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj.data is None:
        return
    if not getattr(obj.data, "materials", None):
        return
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def _mk_box(name: str, *, size_xyz: Tuple[float, float, float], loc_xyz: Tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc_xyz)
    obj = bpy.context.active_object
    obj.name = name
    sx, sy, sz = size_xyz
    obj.scale = (sx / 2.0, sy / 2.0, sz / 2.0)
    return obj


def _mk_plane(name: str, *, size_xy: Tuple[float, float], loc_xyz: Tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc_xyz)
    obj = bpy.context.active_object
    obj.name = name
    sx, sy = size_xy
    obj.scale = (sx / 2.0, sy / 2.0, 1.0)
    return obj


def _look_at(obj: bpy.types.Object, target: Vector, *, track_axis: str = "-Z", up_axis: str = "Y") -> None:
    """Point an object at a target location."""
    direction = target - obj.location
    rot_quat = direction.to_track_quat(track_axis, up_axis)
    obj.rotation_euler = rot_quat.to_euler()


def _pad_positions_mm(footprint: str, nominal: Dict[str, float], *, mm_per_px: float) -> List[Tuple[float, float, float, float]]:
    # Returns list of (cx_mm, cy_mm, w_mm, h_mm) in a local XY plane.
    pad_w = float(nominal["pad_width"]) * mm_per_px
    pad_h = float(nominal["pad_height"]) * mm_per_px
    pad_spacing = float(nominal["pad_spacing"]) * mm_per_px

    if footprint == "sot23":
        pad_spacing_y = float(nominal.get("pad_spacing_y", 0.0)) * mm_per_px
        small_h = pad_h * 0.8
        # left-top, left-bottom, right-center
        return [
            (-pad_spacing / 2.0, -pad_spacing_y / 2.0, pad_w, small_h),
            (-pad_spacing / 2.0, +pad_spacing_y / 2.0, pad_w, small_h),
            (+pad_spacing / 2.0, 0.0, pad_w, pad_h),
        ]
    if footprint.startswith("qfn"):
        pad_spacing_y = float(nominal.get("pad_spacing_y", 0.0)) * mm_per_px
        # left/right vertical pads + top/bottom horizontal (swap dims)
        return [
            (-pad_spacing / 2.0, 0.0, pad_w, pad_h),
            (+pad_spacing / 2.0, 0.0, pad_w, pad_h),
            (0.0, -pad_spacing_y / 2.0, pad_h, pad_w),
            (0.0, +pad_spacing_y / 2.0, pad_h, pad_w),
        ]
    # chip_2pad default
    return [
        (-pad_spacing / 2.0, 0.0, pad_w, pad_h),
        (+pad_spacing / 2.0, 0.0, pad_w, pad_h),
    ]


def _configure_cycles(*, width_px: int, height_px: int, samples: int, seed: int, device: str) -> bpy.types.Scene:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    # Cycles seed must fit a signed 32-bit int.
    seed32 = int(seed) % 2147483647
    scene.cycles.seed = int(seed32)
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.resolution_x = int(width_px)
    scene.render.resolution_y = int(height_px)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    # Determinism: keep device explicit. If GPU is requested but unavailable, Blender may error.
    prefs = bpy.context.preferences
    try:
        cprefs = prefs.addons["cycles"].preferences
        cprefs.compute_device_type = "NONE" if str(device).upper() == "CPU" else cprefs.compute_device_type
        scene.cycles.device = "CPU" if str(device).upper() == "CPU" else "GPU"
    except Exception:
        scene.cycles.device = "CPU"

    # Setup world environment for realistic lighting
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_nodes.clear()

    # Environment texture (subtle gradient)
    node_bg = world_nodes.new(type="ShaderNodeBackground")
    node_bg.inputs["Color"].default_value = (0.15, 0.15, 0.18, 1.0)  # Subtle gray-blue
    node_bg.inputs["Strength"].default_value = 0.4  # Subtle ambient light

    node_output = world_nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(node_bg.outputs["Background"], node_output.inputs["Surface"])

    return scene


def _as_rgb(v) -> Tuple[float, float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return (0.5, 0.5, 0.5)
    return (float(v[0]), float(v[1]), float(v[2]))


def _build_scene_for_job(job: Dict[str, Any], *, samples: int, device: str) -> None:
    seed = int(job.get("seed", 0))
    random.seed(seed)

    mm_per_px = float(job["mm_per_px"])
    width_px = int(job["roi_width_px"])
    height_px = int(job["roi_height_px"])
    footprint = str(job.get("footprint", "chip_2pad"))
    nominal = job["nominal"]
    defect = job["defect"]
    render_3d = job.get("render_3d") or {}

    _configure_cycles(width_px=width_px, height_px=height_px, samples=samples, seed=seed, device=device)

    # Materials (defaults if missing)
    mats_cfg = (render_3d.get("materials") or {})
    m_sub = _new_material("mat_substrate", base_color=_as_rgb(mats_cfg.get("substrate", {}).get("base_color", [0.05, 0.2, 0.05])),
                          roughness=float(mats_cfg.get("substrate", {}).get("roughness", 0.85)),
                          metallic=float(mats_cfg.get("substrate", {}).get("metallic", 0.0)))
    m_cu = _new_material("mat_copper", base_color=_as_rgb(mats_cfg.get("copper", {}).get("base_color", [0.8, 0.45, 0.1])),
                         roughness=float(mats_cfg.get("copper", {}).get("roughness", 0.25)),
                         metallic=float(mats_cfg.get("copper", {}).get("metallic", 1.0)))
    m_body = _new_material("mat_body", base_color=_as_rgb(mats_cfg.get("body", {}).get("base_color", [0.02, 0.02, 0.02])),
                           roughness=float(mats_cfg.get("body", {}).get("roughness", 0.6)),
                           metallic=float(mats_cfg.get("body", {}).get("metallic", 0.0)))

    # Substrate plane
    board_w_mm = max(1e-6, float(width_px) * mm_per_px)
    board_h_mm = max(1e-6, float(height_px) * mm_per_px)
    sub = _mk_plane("substrate", size_xy=(board_w_mm, board_h_mm), loc_xyz=(0.0, 0.0, 0.0))
    _apply_material(sub, m_sub)

    # Pads (thin boxes)
    pad_th = 0.05
    for i, (cx, cy, pw, ph) in enumerate(_pad_positions_mm(footprint, nominal, mm_per_px=mm_per_px), 1):
        pad = _mk_box(f"pad_{i}", size_xyz=(pw, ph, pad_th), loc_xyz=(cx, cy, pad_th / 2.0))
        _apply_material(pad, m_cu)

    # Component body
    comp_type = str(defect.get("type", "OK"))
    comp_h = float(job.get("component_height_mm") or 0.45)
    comp_l = float(nominal["component_length"]) * mm_per_px
    comp_w = float(nominal["component_width"]) * mm_per_px

    comp_obj = None
    if comp_type != "MISSING":
        comp_obj = _mk_box("component", size_xyz=(comp_l, comp_w, comp_h), loc_xyz=(0.0, 0.0, pad_th + comp_h / 2.0))
        _apply_material(comp_obj, m_body)

        # Apply misalignment / rotation
        sx = float(defect.get("shift_x", 0.0)) * mm_per_px
        sy = float(defect.get("shift_y", 0.0)) * mm_per_px
        rz = math.radians(float(defect.get("rotation_deg", 0.0)))
        comp_obj.location.x += sx
        comp_obj.location.y += sy
        comp_obj.rotation_euler[2] = rz

        # Tombstone: rotate around X axis (tilt), crude but gives shadow cue.
        tilt = float(defect.get("tilt_deg", 0.0))
        if tilt != 0.0:
            comp_obj.rotation_euler[0] = math.radians(tilt)

    # Lights
    light_cfg = (render_3d.get("lighting") or {})
    key = light_cfg.get("key_light") or {}
    fill = light_cfg.get("fill_light") or {}

    def add_area(name: str, loc_mm, power_w: float, size_mm: float, target_mm=(0.0, 0.0, 0.0)) -> None:
        bpy.ops.object.light_add(type="AREA", location=(float(loc_mm[0]), float(loc_mm[1]), float(loc_mm[2])))
        l = bpy.context.active_object
        l.name = name
        l.data.energy = float(power_w)
        l.data.size = float(size_mm)
        # Point the light at the target
        _look_at(l, Vector(target_mm), track_axis="-Z", up_axis="Y")

    target = Vector((0.0, 0.0, pad_th + comp_h / 2.0 if comp_obj else 0.0))
    add_area(
        "key_light",
        key.get("location_mm", [4.0, -4.0, 10.0]),
        float(key.get("power_w", 300.0)),
        float(key.get("size_mm", 10.0)),
        target,
    )
    add_area(
        "fill_light",
        fill.get("location_mm", [-4.0, 4.0, 8.0]),
        float(fill.get("power_w", 120.0)),
        float(fill.get("size_mm", 12.0)),
        target,
    )

    # Camera
    cam_cfg = (render_3d.get("camera") or {})
    bpy.ops.object.camera_add(location=(0.0, -7.0, 9.0))
    cam = bpy.context.active_object
    cam.data.lens = float(cam_cfg.get("focal_length_mm", 35.0))
    loc = cam_cfg.get("location_mm")
    if isinstance(loc, (list, tuple)) and len(loc) == 3:
        cam.location = (float(loc[0]), float(loc[1]), float(loc[2]))
    look = cam_cfg.get("look_at_mm")
    if isinstance(look, (list, tuple)) and len(look) == 3:
        _look_at(cam, Vector((float(look[0]), float(look[1]), float(look[2]))))
    bpy.context.scene.camera = cam


def main() -> None:
    args = _parse_args()
    jobs_path = Path(args.jobs)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load jobs
    jobs: List[Dict[str, Any]] = []
    with open(jobs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))

    # One blender process; rebuild a minimal scene per job.
    errors = 0
    for i, job in enumerate(jobs, 1):
        try:
            _clean_scene()
            _build_scene_for_job(job, samples=int(args.samples), device=str(args.device))
            out_path = out_root / str(job["image_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            bpy.context.scene.render.filepath = str(out_path)
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            errors += 1
            print(f"[render][error] job {i}/{len(jobs)} failed: {exc}")
            # Fail fast; leaving a partially-rendered dataset is worse than stopping.
            break

        # Flush stdout-ish progress for long runs.
        if i == 1 or i % 50 == 0 or i == len(jobs):
            print(f"[render] {i}/{len(jobs)} -> {out_path}")

    if errors:
        # Ensure Blender returns non-zero so upstream can abort.
        raise SystemExit(2)


if __name__ == "__main__":
    main()
