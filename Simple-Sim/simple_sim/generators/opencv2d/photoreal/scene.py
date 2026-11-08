"""The photoreal pipeline: board around the part, joints, part, defects, view and camera.

:func:`render_photoreal_base` renders one camera image of the inspected part (2D backend),
:func:`render_board` the board texture plus scene layout for the Blender backend. Both render at
``render.supersampling`` x the camera resolution; every random choice comes from the passed ``rng``,
so a sample is fully determined by its seed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from simple_sim.board import family_of, footprint_pads, neighbour_layout
from simple_sim.board.footprints import scene_part
from simple_sim.board.placement import pad_ref
from simple_sim.classes import DefectClass

from ..geometry import body_geometry, draw_solder_bridge, lifted_corner_geometry, param_selector
from .camera import (apply_view, camera_model, camera_settings, effective_supersampling, fit_crop, sample_camera_pose,
                     view_canvas)
from .context import RenderContext
from .copper import CopperJob, draw_board_copper
from .imageops import blend, clip8, contact_shadow
from .joints import pads_and_solder, thermal_pad_index
from .palette import FINISHES
from .parts import (body_half_extent, draw_body_local, draw_leads, draw_neighbours, draw_tombstone, lead_feet_mask,
                    lead_layout, nominal_body_angle, tombstone_pose)
from .rework import draw_rework
from .silkscreen import draw_silkscreen
from .solder_defects import draw_excess, draw_solder_balls, solder_defect_layout
from .substrate import mask_sheen, procedural_substrate, texture_substrate


def draw_board(ctx: RenderContext, nom: dict[str, float], footprint: str, family: str, shape: tuple[int, int],
           rng: np.random.Generator, crowd: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]], np.ndarray]:
    """Board around the inspected part at supersampled *shape*: substrate, copper, silkscreen and
    populated neighbour parts. Returns (float image, mask colour, target pads, pad finish colour).

    The keep-out uses the *nominal* body, so nothing on the board follows a defect (a real board's
    silkscreen and neighbours do not move when a part is misplaced). The render options come from
    *ctx* (:meth:`RenderContext.for_render`).
    """
    h, w = shape
    specs = footprint_pads(footprint, nom, (h, w))
    pads = [p.as_dict() for p in specs]
    body_angle = nominal_body_angle(family)
    hl, hw = body_half_extent(nom["component_length"] / 2.0, nom["component_width"] / 2.0, body_angle)
    xs = [p["x"] for p in pads] + [p["x"] + p["w"] for p in pads] + [int(w // 2 - hl), int(w // 2 + hl)]
    ys = [p["y"] for p in pads] + [p["y"] + p["h"] for p in pads] + [int(h // 2 - hw), int(h // 2 + hw)]
    keepout = (min(xs), min(ys), max(xs), max(ys))
    texture_dir = ctx.texture_dir
    neighbours: list[dict[str, Any]] | None = None
    if texture_dir:
        img, base = texture_substrate(ctx, h, w, texture_dir, rng)  # real board already has its own copper
    else:
        img, base = procedural_substrate(ctx, h, w, rng)
        job = CopperJob(footprint=footprint, family=family, specs=specs, pads=pads, keepout=keepout,
                        body_px=(float(nom["component_length"]), float(nom["component_width"])), body_angle=body_angle,
                        shape=(h, w), options=ctx.board, mm_per_px=ctx.mm_per_px, crowd=crowd)
        neighbours = draw_board_copper(ctx, img, base, job, rng)
    draw_silkscreen(ctx, img, keepout, family, rng)
    if not texture_dir:  # a real board texture already has its own reflections
        mask_sheen(ctx, img, base, rng, ctx.mask_gloss)
    finish = FINISHES["enig" if rng.random() < 0.5 else "hasl"] * float(rng.uniform(0.92, 1.06))
    if neighbours is None:
        neighbours = neighbour_layout(pads, keepout, (h, w), rng, crowd, ctx.look.packing)
    draw_neighbours(ctx, img, base, neighbours, finish, rng)
    draw_rework(ctx, img, keepout, pad_ref(pads), rng, ctx.rework)
    return img, base, pads, finish


def scaled_geometry(ctx: RenderContext, nominal: dict[str, float], defect_params: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    nom = {k: float(v) * ctx.ss for k, v in nominal.items()}
    dp = dict(defect_params)
    dp["shift_x"] = float(dp["shift_x"]) * ctx.ss
    dp["shift_y"] = float(dp["shift_y"]) * ctx.ss
    return nom, dp


@dataclass(frozen=True, slots=True)
class SceneGeometry:
    """The common scene prologue of both pipelines (:func:`_scene_setup`), supersampled pixels.

    ``geo`` is :func:`body_geometry` of the scaled defect with ``rotation_deg`` already in the body
    frame of the package (``+ nominal_body_angle(family)``), ``rect`` its ``cv2.boxPoints``; ``img`` is
    the board image as :func:`draw_board` returned it (the pipelines keep drawing into it).
    """

    h: int
    w: int
    nom: dict[str, float]
    dp: dict[str, Any]
    family: str
    img: np.ndarray
    base: np.ndarray
    pads: list[dict[str, int]]
    finish: np.ndarray
    geo: dict[str, Any]
    rect: np.ndarray


def _scene_setup(ctx: RenderContext, nominal: dict[str, float], defect_params: dict[str, Any],
                 tolerances: dict[str, Any] | None, rng: np.random.Generator, footprint: str, canvas: tuple[int, int],
                 crowd: float) -> SceneGeometry:
    """Scaled geometry, family, board (:func:`draw_board`, the only rng consumer here), body pose and box.

    *canvas*: (width, height) in camera pixels. ``body_geometry`` and ``boxPoints`` draw nothing, so
    computing them right after the board keeps the random stream of both callers unchanged.
    """
    width, height = canvas
    h, w = height * ctx.ss, width * ctx.ss
    nom, dp = scaled_geometry(ctx, nominal, defect_params)
    family = family_of(footprint, ctx.component_type)
    img, base, pads, finish = draw_board(ctx, nom, footprint, family, (h, w), rng, crowd)
    geo = body_geometry(nom, dp, tolerances, (h, w))
    geo["rotation_deg"] += nominal_body_angle(family)  # body frame of the package (labels only use the deviation)
    rect = cv2.boxPoints(((float(geo["center"][0]), float(geo["center"][1])), (float(geo["length"]), float(geo["width"])),
                          geo["rotation_deg"]))
    return SceneGeometry(h=h, w=w, nom=nom, dp=dp, family=family, img=img, base=base, pads=pads, finish=finish, geo=geo,
                         rect=rect)


def _render_board(ctx: RenderContext, 
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    tolerances: dict[str, Any] | None,
    rng: np.random.Generator,
    footprint: str,
    size: tuple[int, int],
    crowd: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """:func:`render_board` with an explicit context."""
    width, height = size
    sg = _scene_setup(ctx, nominal, defect_params, tolerances, rng, footprint, size, crowd)
    h, w, nom, dp, family, img, base, pads = sg.h, sg.w, sg.nom, sg.dp, sg.family, sg.img, sg.base, sg.pads
    pads_and_solder(ctx, img, base, pads, np.zeros((h, w), dtype=np.uint8), rng, sg.finish, with_solder=False)
    if dp["type"] == DefectClass.SOLDER_BRIDGE:  # flat solder between neighbouring pads
        geo = sg.geo
        img8 = clip8(img)
        draw_solder_bridge(img8, nom, footprint, geo["center"], (int(geo["length"]), int(geo["width"])),
                            param_selector(defect_params), geo["tilt_deg"])
        img = img8.astype(np.float32)

    scene: dict[str, Any] = {
        "family": family,
        "pads": [[(p["x"] + p["w"] / 2.0 - w / 2.0) / ctx.ss, (p["y"] + p["h"] / 2.0 - h / 2.0) / ctx.ss, p["w"] / ctx.ss, p["h"] / ctx.ss]
                 for p in pads],
        "thermal_pad": thermal_pad_index(pads),
        "body": None,
        "part": scene_part(family),  # how render_batch.py builds the part (FAMILIES[family].blender, D35)
    }
    solder = solder_defect_layout(dp["type"], pads, sg.rect, rng, ss=ctx.ss)
    scene["excess"] = solder["excess"]
    scene["balls"] = [[(x - w / 2.0) / ctx.ss, (y - h / 2.0) / ctx.ss, r / ctx.ss] for x, y, r in solder["balls"]]
    if dp["type"] != DefectClass.MISSING:
        geo = body_geometry(nominal, defect_params, tolerances, (height, width))
        # rotation_deg: of the body's length axis (SOT-23: vertical, see nominal_body_angle);
        # lead_frame_deg: that axis for the nominal placement, so leads can stay rigid with the body.
        body = {"center": [geo["center"][0] - width // 2, geo["center"][1] - height // 2], "length": geo["length"],
                "width": geo["width"], "rotation_deg": geo["rotation_deg"] + nominal_body_angle(family),
                "lead_frame_deg": nominal_body_angle(family), "tilt_deg": geo["tilt_deg"],
                "is_tombstone": bool(geo["is_tombstone"])}
        if geo["is_tombstone"]:
            pose = tombstone_pose(geo, param_selector(defect_params), (height, width))
            body["raised_sign"] = pose["raised_sign"]
        scene["body"] = body
    return cv2.resize(clip8(img), (width, height), interpolation=cv2.INTER_AREA), scene


def _draw_part(ctx: RenderContext, img: np.ndarray, sg: SceneGeometry, footprint: str, pose: dict[str, Any] | None,
               body_mask: np.ndarray, leads: list[dict[str, float]], base_color: np.ndarray, selector: int,
               solder: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """The inspected part and its part-level defects on *img* (after the joints); returns the image
    (a solder bridge replaces it by its uint8 round trip). *pose*: tombstone pose or ``None``;
    *body_mask*: footprint of body and lead feet (contact shadow)."""
    h, w, nom, family, base, pads, geo, rect = sg.h, sg.w, sg.nom, sg.family, sg.base, sg.pads, sg.geo, sg.rect
    defect_type = sg.dp["type"]
    length, width_px = float(geo["length"]), float(geo["width"])
    cx, cy = geo["center"]
    if pose is not None:
        draw_tombstone(ctx, img, family, pose, width_px, base_color, rng)
    elif defect_type != DefectClass.MISSING:
        contact_shadow(ctx, img, body_mask, 0.35, 3 * ctx.ss)
        draw_leads(ctx, img, leads, (float(cx), float(cy)), geo["rotation_deg"], rng)
        # render.marking_probability (from the profile): small parts such as 0402 resistors carry no code.
        p_mark = ctx.marking_probability
        marks = True if p_mark is None else bool(rng.random() < p_mark)
        draw_body_local(ctx, img, family, (float(cx), float(cy)), length, width_px, geo["rotation_deg"], base_color, rng,
                         marks=marks)
        if defect_type == DefectClass.CORNER_LIFT:
            lift = lifted_corner_geometry(rect, geo["center"], int(width_px), geo["tilt_deg"], selector)
            extra = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(extra, [np.round(lift["lifted"]).astype(np.int32)], 255)
            cv2.fillPoly(extra, [np.round(rect).astype(np.int32)], 0)
            body_col = np.maximum(base_color, 34.0)
            blend(img, body_col, extra.astype(np.float32) / 255.0)  # raised corner sticking out
            facet = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(facet, [np.round(lift["facet"]).astype(np.int32)], 255)
            blend(img, body_col * 1.6 + 30.0, facet.astype(np.float32) / 255.0 * 0.4)  # lit facet
        if defect_type == DefectClass.SOLDER_BRIDGE:
            img8 = clip8(img)
            draw_solder_bridge(img8, nom, footprint, (int(cx), int(cy)), (int(length), int(width_px)), selector, geo["tilt_deg"])
            img = img8.astype(np.float32)
        draw_excess(ctx, img, base, family, pads, solder["excess"], (float(cx), float(cy)), length, width_px, geo["rotation_deg"])
        draw_solder_balls(ctx, img, base, solder["balls"])
    return img


def _render_photoreal_base(ctx: RenderContext, 
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    tolerances: dict[str, Any] | None,
    rng: np.random.Generator,
    footprint: str,
    roi_size: tuple[int, int],
    post: dict[str, float] | None = None,
    camera_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """:func:`render_photoreal_base` with an explicit context (``ctx.post_roll_deg``: the caller's rotation).
    *camera_rng*: generator of the sensor model's draws (default: *rng*, see :func:`render_photoreal_base`)."""
    post = post or {}
    camera = ctx.camera  # render.camera, or the preset of render.look
    cam = camera_settings(camera)
    view_pose = sample_camera_pose(cam, rng, float(post.get("rotation_deg", 0.0)))
    view_pose = fit_crop(view_pose, cam, rng, float(nominal["component_length"]), roi_size)
    ss = effective_supersampling(ctx.ss, view_pose, cam)  # camera.max_oversampling: cheaper zoomed-out views
    if ss != ctx.ss:
        ctx = replace(ctx, ss=ss)
    canvas = view_canvas(roi_size, view_pose)  # larger board area for a zoomed-out / oblique / rolled view
    base_color = np.array(ctx.component_color, dtype=np.float32)

    crowd = canvas[0] * canvas[1] / float(roi_size[0] * roi_size[1])  # rendered area / ROI area
    sg = _scene_setup(ctx, nominal, defect_params, tolerances, rng, footprint, canvas, crowd)
    h, w, family, img, base, pads, geo, rect = sg.h, sg.w, sg.family, sg.img, sg.base, sg.pads, sg.geo, sg.rect
    defect_type = sg.dp["type"]
    length, width_px = float(geo["length"]), float(geo["width"])
    cx, cy = geo["center"]
    selector = param_selector(defect_params)

    body_mask = np.zeros((h, w), dtype=np.uint8)
    pose = None
    if defect_type != DefectClass.MISSING and geo["is_tombstone"]:
        pose = tombstone_pose(geo, selector, (h, w))
        body_mask = pose["mask"]
    elif defect_type != DefectClass.MISSING:
        pts = rect
        if defect_type == DefectClass.CORNER_LIFT:
            pts = lifted_corner_geometry(rect, geo["center"], int(width_px), geo["tilt_deg"], selector)["lifted"]
        cv2.fillPoly(body_mask, [np.round(pts).astype(np.int32)], 255)
    leads = []
    if defect_type != DefectClass.MISSING and pose is None:
        leads = lead_layout(family, pads, (w // 2, h // 2), length, width_px, ss=ctx.ss)
        if leads:  # solder wets the lead feet, so the fillets rise towards them
            body_mask = np.maximum(body_mask, lead_feet_mask((h, w), leads, (float(cx), float(cy)), geo["rotation_deg"]))

    solder = solder_defect_layout(defect_type, pads, rect, rng, ss=ctx.ss)
    pads_and_solder(ctx, img, base, pads, body_mask, rng, sg.finish, excess=frozenset(solder["excess"]))

    img = _draw_part(ctx, img, sg, footprint, pose, body_mask, leads, base_color, selector, solder, rng)

    view = apply_view(cv2.resize(img, canvas, interpolation=cv2.INTER_AREA), view_pose, roi_size)
    # sensor_px_per_mm: one output pixel covers mm_per_px x zoom of the board
    return camera_model(view, rng if camera_rng is None else camera_rng, camera, view_pose,
                        mm_per_px=ctx.mm_per_px * view_pose["zoom"], extra_blur=float(post.get("blur_sigma", 0.0)),
                        extra_noise_var=float(post.get("noise_stddev", 0.0)) ** 2, axis_roll_deg=ctx.post_roll_deg)


def render_board(nominal: dict[str, float], defect_params: dict[str, Any], config: dict[str, Any],
                 tolerances: dict[str, Any] | None, rng: np.random.Generator, footprint: str,
                 size: tuple[int, int], crowd: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Board texture (substrate, copper, silkscreen, bare pad finish) for the 3D backend, plus the scene layout.

    Returns ``(BGR uint8 image of *size*, scene)``. ``scene`` holds, in camera pixels relative to the image
    centre (x right, y down): ``family``, ``pads`` as ``[cx, cy, w, h]``, ``body`` (``None`` when MISSING)
    with the same pose semantics as the 2D renderer including the tombstone pivot, ``excess`` (pad
    indices with too much solder) and ``balls`` (``[x, y, r]``), from :func:`solder_defect_layout`, and
    ``part`` (:func:`~simple_sim.board.footprints.scene_part`: the Blender build of the family).
    """
    ctx = RenderContext.for_render(config, rng)
    return _render_board(ctx, nominal, defect_params, tolerances, rng, footprint, size, crowd)


def render_photoreal_base(nominal: dict[str, float], defect_params: dict[str, Any], config: dict[str, Any],
                          tolerances: dict[str, Any] | None, rng: np.random.Generator, footprint: str,
                          roi_size: tuple[int, int], post: dict[str, float] | None = None,
                          light_roll_deg: float | None = None, *, camera_rng: np.random.Generator | None = None) -> np.ndarray:
    """Render substrate + pads + solder + component for *defect_params* at *roi_size* (BGR uint8).

    Rendered at ``render.supersampling`` x the camera resolution (default 2, up to 4) and downsampled.
    *post* (``render.post_filters: camera``): ``rotation_deg`` is applied as camera roll before the
    camera model, ``blur_sigma`` / ``noise_stddev`` go into the camera model (see :func:`camera_model`).
    *light_roll_deg*: rotation the caller applies to the finished image (``render.post_filters:
    legacy``), so a look with a fixed light direction (``quins``) and a fixed ``camera.channel_axis``
    can compensate; default: the roll in *post* (which is applied before shading, so nothing to compensate).
    *camera_rng*: a separate generator for the sensor model (``None``: *rng*, bit-identical to before). The view
    pose, the board, the neighbours and the joint light are drawn from *rng* before any defect-dependent
    draw, so two renders with the same *rng* seed and *camera_rng* but different defects show one photo of
    one board in which only the inspected part differs.
    """
    if light_roll_deg is None:
        roll, after = float((post or {}).get("rotation_deg", 0.0)), 0.0
    else:
        roll = after = float(light_roll_deg)
    ctx = RenderContext.for_render(config, rng, roll, post_roll_deg=after)
    return _render_photoreal_base(ctx, nominal, defect_params, tolerances, rng, footprint, roi_size, post, camera_rng)
