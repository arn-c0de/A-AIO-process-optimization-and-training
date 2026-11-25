"""Blender building blocks shared by ``render_batch.py`` and ``render_board3d.py`` (runs inside Blender's Python).

Materials, render settings (engine, Cycles GPU, colour management, world) and the AOI lights. Imports only
the stdlib, ``bpy`` and ``mathutils``. ``render_batch`` re-exports these under its former private names
(``_material``, ``_configure_render``, ...). The part builders and ``_box`` stay in ``render_batch``:
``render_board3d`` swaps ``render_batch._box`` / ``render_batch.bpy`` for faster data-API versions in board mode.
"""

from __future__ import annotations

import math
from typing import Any

import bpy
from mathutils import Vector

RING_STRENGTH, TOP_STRENGTH = 6.0, 0.9  # emission strengths, calibrated to the baked board colours


def material(name: str, color, *, roughness: float, metallic: float = 0.0, bump: float = 0.0,
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


def board_material(texture: str) -> bpy.types.Material:
    """Solder mask carrying the baked 2D board (mask, copper, silkscreen, pad finish).

    The baked colours come from real photos and already include the mask's sheen, so the specular
    part is kept low; with the default ring light the board renders at its baked colours.
    """
    mat = material("mat_board", (1, 1, 1), roughness=0.4)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.1
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(texture)
    tex.interpolation = "Cubic"
    links.new(tex.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def cfg_material(name: str, cfg: dict[str, Any], default_color, default_roughness: float, default_metallic: float,
                  bump: float = 0.0) -> bpy.types.Material:
    color = cfg.get("base_color", default_color)
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        color = default_color
    return material(name, color, roughness=float(cfg.get("roughness", default_roughness)),
                     metallic=float(cfg.get("metallic", default_metallic)), bump=bump)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


_GPU_BACKEND: str | None = None


def enable_cycles_gpu() -> bool:
    """Enable a Cycles GPU backend (OptiX, CUDA, HIP, oneAPI) once per Blender process.

    Setting ``scene.cycles.device = "GPU"`` alone is not enough: without compute devices
    enabled in the Cycles add-on preferences Blender silently renders on the CPU.
    Returns False (render on CPU) if no GPU backend is available.
    """
    global _GPU_BACKEND
    # main() resets to factory settings before every job, which also resets these preferences: apply
    # them on every call and only report the device once.
    first = _GPU_BACKEND is None
    if first or _GPU_BACKEND:
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
        if first:
            print(f"[render_batch] Cycles device: {_GPU_BACKEND or 'CPU'}", flush=True)
    return bool(_GPU_BACKEND)


def configure_render(*, width_px: int, height_px: int, samples: int, seed: int, device: str, engine: str) -> None:
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
            scene.cycles.device = "GPU" if want_gpu and enable_cycles_gpu() else "CPU"

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


def aoi_lights(render_3d: dict[str, Any], roi_mm: float) -> None:
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
    ring.data.materials.append(material("mat_ring", (0, 0, 0), roughness=1.0,
                                         emission=float(ring_cfg.get("strength", RING_STRENGTH))))
    ring.visible_camera = False

    bpy.ops.mesh.primitive_circle_add(vertices=48, radius=2.0 * radius, fill_type="NGON",
                                      location=(0.0, 0.0, 1.2 * height), rotation=(math.pi, 0.0, 0.0))
    top = bpy.context.active_object
    top.name = "top_light"
    top.data.materials.append(material("mat_top", (0, 0, 0), roughness=1.0,
                                        emission=float(top_cfg.get("strength", TOP_STRENGTH))))
    top.visible_camera = False
