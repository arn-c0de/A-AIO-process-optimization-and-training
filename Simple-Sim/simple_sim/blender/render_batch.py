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
    nodes = mat.node_tree.nodes
    nodes.clear()  # Clear any default nodes

    # Create Principled BSDF explicitly
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs["Base Color"].default_value = (float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0)
    bsdf.inputs["Roughness"].default_value = float(roughness)
    bsdf.inputs["Metallic"].default_value = float(metallic)

    # Create output node
    output = nodes.new(type='ShaderNodeOutputMaterial')

    # Link them
    links = mat.node_tree.links
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


def _apply_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj is None or obj.data is None:
        print(f"[ERROR] Object or data is None!")
        return

    # FIX: Use obj.data directly, materials is a collection
    if obj.type == 'MESH':
        # Clear existing materials and add new one
        obj.data.materials.clear()
        obj.data.materials.append(mat)


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


def _mk_chip_resistor(name: str, *, length: float, width: float, height: float, loc_xyz: Tuple[float, float, float]) -> bpy.types.Object:
    """Create realistic chip resistor with rounded ends and end caps."""
    # Main body (slightly smaller than total length for end caps)
    body_length = length * 0.7
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc_xyz)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (body_length / 2.0, width / 2.0, height / 2.0)

    # Add bevel modifier for rounded edges (realistic!)
    bevel_mod = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel_mod.width = min(width, height) * 0.15
    bevel_mod.segments = 2

    return obj


def _mk_sot23_transistor(name: str, *, length: float, width: float, height: float, loc_xyz: Tuple[float, float, float]) -> bpy.types.Object:
    """Create realistic SOT-23 transistor package with tapered shape."""
    # Keep sharp edges (no bevel). AOI-style renders look wrong with rounded corners.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc_xyz)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (length / 2.0, width / 2.0, height / 2.0)

    return obj


def _mk_qfn_package(name: str, *, length: float, width: float, height: float, loc_xyz: Tuple[float, float, float]) -> bpy.types.Object:
    """Create realistic QFN IC package."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc_xyz)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (length / 2.0, width / 2.0, height / 2.0)

    return obj


def _add_solder_joint(name: str, *, pos_xyz: Tuple[float, float, float], size: float, material: bpy.types.Material) -> bpy.types.Object:
    """Create realistic solder fillet/joint."""
    # Use UV sphere for smooth solder fillet appearance
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=pos_xyz, segments=8, ring_count=6)
    solder = bpy.context.active_object
    solder.name = name

    # Scale to create fillet shape (flattened sphere)
    solder.scale = (1.0, 1.0, 0.4)

    # Apply material
    _apply_material(solder, material)

    return solder


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
        # QFN: treat pad_height as the *radial* pad length (towards/away from package),
        # and pad_width as the *tangential* dimension along the package edge.
        return [
            # Left/right pads: radial along X, tangential along Y.
            (-pad_spacing / 2.0, 0.0, pad_h, pad_w),
            (+pad_spacing / 2.0, 0.0, pad_h, pad_w),
            # Top/bottom pads: tangential along X, radial along Y.
            (0.0, -pad_spacing_y / 2.0, pad_w, pad_h),
            (0.0, +pad_spacing_y / 2.0, pad_w, pad_h),
        ]
    # chip_2pad default
    return [
        (-pad_spacing / 2.0, 0.0, pad_w, pad_h),
        (+pad_spacing / 2.0, 0.0, pad_w, pad_h),
    ]


def _configure_cycles(*, width_px: int, height_px: int, samples: int, seed: int, device: str) -> bpy.types.Scene:
    scene = bpy.context.scene

    # Try EEVEE instead of Cycles for better color handling
    scene.render.engine = "BLENDER_EEVEE"

    # EEVEE settings for quality
    scene.eevee.taa_render_samples = max(64, int(samples))
    scene.eevee.use_gtao = True  # Ambient occlusion
    scene.eevee.use_bloom = False  # No bloom (more realistic)
    scene.eevee.use_ssr = True  # Screen space reflections

    # Disable compositor and sequencer
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.resolution_x = int(width_px)
    scene.render.resolution_y = int(height_px)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = 'RGB'  # Force RGB, not grayscale!
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = False

    # Color management: CRITICAL - override AgX (Blender 4.0+ default) with Raw
    # AgX can desaturate colors significantly
    scene.view_settings.view_transform = 'Raw'  # No tone mapping
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.view_settings.use_curve_mapping = False
    scene.display_settings.display_device = 'sRGB'

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

    # Environment texture - realistic AOI booth lighting
    node_bg = world_nodes.new(type="ShaderNodeBackground")
    node_bg.inputs["Color"].default_value = (0.25, 0.25, 0.27, 1.0)  # Neutral inspection booth
    node_bg.inputs["Strength"].default_value = 1.2  # Soft ambient like real AOI

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

    # Solder material: shiny metallic silver (for solder joints)
    solder_col_raw = _as_rgb(mats_cfg.get("solder", {}).get("base_color", [0.7, 0.7, 0.72]))
    solder_col = (
        min(1.0, solder_col_raw[0] * 1.0),
        min(1.0, solder_col_raw[1] * 1.0),
        min(1.0, solder_col_raw[2] * 1.0)
    )
    m_solder = _new_material("mat_solder", base_color=solder_col,
                             roughness=float(mats_cfg.get("solder", {}).get("roughness", 0.15)),
                             metallic=float(mats_cfg.get("solder", {}).get("metallic", 1.0)))

    # Substrate: Authentic PCB green solder mask (darker, realistic)
    sub_col_raw = _as_rgb(mats_cfg.get("substrate", {}).get("base_color", [0.05, 0.2, 0.05]))
    # Realistic PCB green: darker but visible under AOI lighting
    sub_col = (
        min(0.6, sub_col_raw[0] * 1.8),   # R: limited to avoid oversaturation
        min(0.6, sub_col_raw[1] * 2.8),   # G: dominant but not neon
        min(0.6, sub_col_raw[2] * 1.6)    # B: minimal for true green
    )
    m_sub = _new_material("mat_substrate", base_color=sub_col,
                          roughness=float(mats_cfg.get("substrate", {}).get("roughness", 0.8)),
                          metallic=float(mats_cfg.get("substrate", {}).get("metallic", 0.0)))

    # Copper pads: Authentic copper/gold finish - warm metallic
    cu_col_raw = _as_rgb(mats_cfg.get("copper", {}).get("base_color", [0.8, 0.45, 0.1]))
    # Real copper/ENIG finish color
    cu_col = (
        min(0.9, cu_col_raw[0] * 1.0),    # R: copper orange
        min(0.7, cu_col_raw[1] * 1.0),    # G: reduce for warmer tone
        min(0.3, cu_col_raw[2] * 0.5)     # B: minimal for copper look
    )
    m_cu = _new_material("mat_copper", base_color=cu_col,
                         roughness=float(mats_cfg.get("copper", {}).get("roughness", 0.15)),
                         metallic=float(mats_cfg.get("copper", {}).get("metallic", 0.9)))

    # Component body: Realistic dark component (SMD resistor/IC black)
    body_col_raw = _as_rgb(mats_cfg.get("body", {}).get("base_color", [0.02, 0.02, 0.02]))
    # Real component body: very dark but not pure black
    body_col = (
        min(0.15, body_col_raw[0] * 4.0),
        min(0.15, body_col_raw[1] * 4.0),
        min(0.15, body_col_raw[2] * 4.0)
    )
    m_body = _new_material("mat_body", base_color=body_col,
                           roughness=float(mats_cfg.get("body", {}).get("roughness", 0.75)),
                           metallic=float(mats_cfg.get("body", {}).get("metallic", 0.0)))

    # Pad thickness and component dims (used for sizing + placement)
    pad_th = 0.05
    comp_h = float(job.get("component_height_mm") or 0.45)

    # Compute base geometry in mm.
    pad_w = float(nominal["pad_width"]) * mm_per_px
    pad_h = float(nominal["pad_height"]) * mm_per_px
    comp_l_raw = float(nominal["component_length"]) * mm_per_px
    comp_w_raw = float(nominal["component_width"]) * mm_per_px

    # For QFN, keep pads close to the body. Some configs sample pad_spacing larger than
    # the body size, which looks like pads "floating away".
    nominal_pads = dict(nominal)
    if footprint.startswith("qfn"):
        pad_spacing_px = float(nominal_pads["pad_spacing"])
        pad_spacing_y_px = float(nominal_pads.get("pad_spacing_y", pad_spacing_px))
        pad_radial_px = float(nominal_pads["pad_height"])
        comp_l_px = float(nominal_pads["component_length"])
        comp_w_px = float(nominal_pads["component_width"])

        # Keep pads directly at the package edges for QFN.
        # pad_spacing is center-to-center distance between opposite pads.
        # For pads to align with package edge: spacing = component_size - pad_radial_length
        # This positions pad outer edges exactly at the component edges.
        desired_spacing_px = comp_l_px - pad_radial_px
        desired_spacing_y_px = comp_w_px - pad_radial_px
        nominal_pads["pad_spacing"] = min(pad_spacing_px, desired_spacing_px)
        nominal_pads["pad_spacing_y"] = min(pad_spacing_y_px, desired_spacing_y_px)
        print(f"[QFN] YAML pad_spacing={pad_spacing_px:.1f}px, desired={desired_spacing_px:.1f}px, FINAL={nominal_pads['pad_spacing']:.1f}px")
        print(f"[QFN] YAML pad_spacing_y={pad_spacing_y_px:.1f}px, desired={desired_spacing_y_px:.1f}px, FINAL={nominal_pads['pad_spacing_y']:.1f}px")

    # Pre-compute pad positions and ensure the board (substrate) is large enough.
    # Some profiles use large geometry ranges relative to ROI; if we keep the board
    # fixed to ROI size, pads/components can overhang and look wrong.
    pad_positions = _pad_positions_mm(footprint, nominal_pads, mm_per_px=mm_per_px)

    # Use the actual spacing used for pad placement.
    pad_spacing = float(nominal_pads["pad_spacing"]) * mm_per_px

    # Component sizing rules:
    # - 2-pad parts need generous overlap so they read as soldered.
    # - QFN should NOT be stretched to cover the pads; keep nominal body size.
    if footprint.startswith("qfn"):
        comp_l = comp_l_raw
        comp_w = comp_w_raw
    else:
        # Component must span far enough to visually reach the pad copper.
        # Pad outer-edge distance (left-to-right) is: pad_spacing + pad_w.
        # Use a generous overlap so the part looks soldered even with small jitter.
        # (pad_spacing + 2.5*pad_w) means ~0.75*pad_w overhang beyond each outer pad edge.
        required_length = pad_spacing + (2.5 * pad_w)
        comp_l = max(comp_l_raw, required_length)
        comp_w = comp_w_raw

    # SOT-23 has 2 pads on one side separated in Y; ensure the body isn't narrower
    # than the pad cluster, otherwise it can look "not connected" even in OK.
    if footprint == "sot23":
        pad_spacing_y = float(nominal.get("pad_spacing_y", 0.0)) * mm_per_px
        required_width = (pad_spacing_y + pad_h) * 1.05
        comp_w = max(comp_w, required_width)

    print(
        f"[COMP] Length={comp_l:.3f}mm (pad_spacing={pad_spacing:.3f}mm, pad_w={pad_w:.3f}mm) "
        f"Width={comp_w:.3f}mm (raw={comp_w_raw:.3f}mm)"
    )

    # Substrate plane
    board_w_mm = max(1e-6, float(width_px) * mm_per_px)
    board_h_mm = max(1e-6, float(height_px) * mm_per_px)
    if pad_positions:
        min_x = min((cx - pw / 2.0) for (cx, cy, pw, ph) in pad_positions)
        max_x = max((cx + pw / 2.0) for (cx, cy, pw, ph) in pad_positions)
        min_y = min((cy - ph / 2.0) for (cx, cy, pw, ph) in pad_positions)
        max_y = max((cy + ph / 2.0) for (cx, cy, pw, ph) in pad_positions)
        pad_span_x = max(1e-6, max_x - min_x)
        pad_span_y = max(1e-6, max_y - min_y)
    else:
        pad_span_x = pad_span_y = 1e-6

    # Size around the larger of pad cluster and component, with margin.
    # Use a generous margin (or full ROI size) so pads are well within the PCB, not at edges.
    needed_span_x = max(pad_span_x, comp_l)
    needed_span_y = max(pad_span_y, comp_w)
    margin = max(3.0, max(pad_w, pad_h, 0.8) * 3.0)  # Much larger margin for realistic PCB
    board_w_mm = max(board_w_mm, needed_span_x + 2.0 * margin)
    board_h_mm = max(board_h_mm, needed_span_y + 2.0 * margin)

    sub = _mk_plane("substrate", size_xy=(board_w_mm, board_h_mm), loc_xyz=(0.0, 0.0, 0.0))
    _apply_material(sub, m_sub)

    # Pads (thin boxes)
    for i, (cx, cy, pw, ph) in enumerate(pad_positions, 1):
        pad = _mk_box(f"pad_{i}", size_xyz=(pw, ph, pad_th), loc_xyz=(cx, cy, pad_th / 2.0))
        _apply_material(pad, m_cu)

    # Component body
    comp_type = str(defect.get("type", "OK"))

    comp_obj = None
    if comp_type != "MISSING":
        # Create realistic 3D component based on footprint type
        comp_z = pad_th + comp_h / 2.0

        # Place the body centered over the pad cluster. For SOT-23 the pad cluster
        # centroid is not at (0,0) because there are 2 pads on one side and 1 on the other.
        base_x, base_y = 0.0, 0.0
        if footprint == "sot23":
            if pad_positions:
                base_x = sum(p[0] for p in pad_positions) / float(len(pad_positions))
                base_y = sum(p[1] for p in pad_positions) / float(len(pad_positions))

        if footprint == "chip_2pad":
            # Chip resistor/capacitor with realistic rounded shape
            comp_obj = _mk_chip_resistor("component", length=comp_l, width=comp_w, height=comp_h, loc_xyz=(base_x, base_y, comp_z))

        elif footprint == "sot23":
            # SOT-23 transistor package
            comp_obj = _mk_sot23_transistor("component", length=comp_l, width=comp_w, height=comp_h, loc_xyz=(base_x, base_y, comp_z))

            # Add solder joints at the 3 SOT-23 pins
            # SOT-23: 2 pins on one side (left), 1 pin on other side (right)
            pad_w = float(nominal["pad_width"]) * mm_per_px
            pad_spacing_val = float(nominal["pad_spacing"]) * mm_per_px
            pad_spacing_y = float(nominal.get("pad_spacing_y", 0.0)) * mm_per_px

            solder_size = min(pad_w, comp_h) * 0.3
            solder_z = pad_th + solder_size * 0.3

            # Left side: 2 pins
            for y_pos in [-pad_spacing_y / 2.0, pad_spacing_y / 2.0]:
                _add_solder_joint(f"solder_L_{y_pos}", pos_xyz=(-pad_spacing_val / 2.0, y_pos, solder_z), size=solder_size, material=m_solder)

            # Right side: 1 pin (center)
            _add_solder_joint("solder_R", pos_xyz=(pad_spacing_val / 2.0, 0.0, solder_z), size=solder_size, material=m_solder)
        elif footprint.startswith("qfn"):
            # QFN IC package
            comp_obj = _mk_qfn_package("component", length=comp_l, width=comp_w, height=comp_h, loc_xyz=(base_x, base_y, comp_z))

            # Add solder joints at the 4 QFN pads (left, right, top, bottom)
            pad_w = float(nominal["pad_width"]) * mm_per_px
            pad_spacing_val = float(nominal_pads["pad_spacing"]) * mm_per_px
            pad_spacing_y = float(nominal_pads.get("pad_spacing_y", pad_spacing_val)) * mm_per_px

            solder_size = min(pad_w, comp_h) * 0.25
            solder_z = pad_th + solder_size * 0.3

            # Left and right pads (along X axis)
            _add_solder_joint("solder_L", pos_xyz=(-pad_spacing_val / 2.0, 0.0, solder_z), size=solder_size, material=m_solder)
            _add_solder_joint("solder_R", pos_xyz=(pad_spacing_val / 2.0, 0.0, solder_z), size=solder_size, material=m_solder)

            # Top and bottom pads (along Y axis)
            _add_solder_joint("solder_T", pos_xyz=(0.0, -pad_spacing_y / 2.0, solder_z), size=solder_size, material=m_solder)
            _add_solder_joint("solder_B", pos_xyz=(0.0, pad_spacing_y / 2.0, solder_z), size=solder_size, material=m_solder)
        else:
            # Fallback: simple box
            comp_obj = _mk_box("component", size_xyz=(comp_l, comp_w, comp_h), loc_xyz=(base_x, base_y, comp_z))

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

    # Lights - boost power for small-scale scenes
    light_cfg = (render_3d.get("lighting") or {})
    key = light_cfg.get("key_light") or {}
    fill = light_cfg.get("fill_light") or {}

    # EEVEE lighting: AOI inspection lighting (bright but natural)
    power_boost = 8.0

    def add_area(name: str, loc_mm, power_w: float, size_mm: float, target_mm=(0.0, 0.0, 0.0)) -> None:
        bpy.ops.object.light_add(type="AREA", location=(float(loc_mm[0]), float(loc_mm[1]), float(loc_mm[2])))
        l = bpy.context.active_object
        l.name = name
        l.data.energy = float(power_w) * power_boost
        l.data.size = float(size_mm) * 0.5  # Smaller area lights for focused illumination
        l.data.shape = 'SQUARE'
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

    # Camera - use orthographic for AOI-style top-down inspection
    cam_cfg = (render_3d.get("camera") or {})
    bpy.ops.object.camera_add(location=(0.0, -7.0, 9.0))
    cam = bpy.context.active_object

    # Set orthographic projection with scale matching ROI
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = max(board_w_mm, board_h_mm) * 1.2  # 20% margin

    # Position from config
    loc = cam_cfg.get("location_mm")
    if isinstance(loc, (list, tuple)) and len(loc) == 3:
        cam.location = (float(loc[0]), float(loc[1]), float(loc[2]))

    # Point camera at scene center (slightly above substrate)
    look = cam_cfg.get("look_at_mm")
    if isinstance(look, (list, tuple)) and len(look) == 3:
        _look_at(cam, Vector((float(look[0]), float(look[1]), float(look[2]))))
    else:
        # Default: look at component center
        target_z = pad_th + (comp_h / 2.0 if comp_obj else 0.0)
        _look_at(cam, Vector((0.0, 0.0, target_z)))

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
