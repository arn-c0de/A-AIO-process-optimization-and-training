"""Photorealistic base rendering for the OpenCV 2D backend (``render.style: photoreal``).

Produces the un-augmented ROI (substrate, pads, solder fillets, component body) with procedural
textures and simple physically motivated shading, rendered at 2x (``render.supersampling``: 1-4) and
downsampled for anti-aliasing, then passed through a camera model (lens blur, uneven illumination,
highlight roll-off, sensor noise) so it does not look like a vector drawing. Every random choice is
drawn from the passed ``rng``, so a sample is fully determined by its seed.

Layers, in drawing order (one module each):

========================  =====================================================================
:mod:`.substrate`         solder mask on laminate (or a real board texture), mask gloss
:mod:`.copper`            legacy traces/pours/vias or the rasterised board model (``render.board``)
:mod:`.silkscreen`        outline, pin-1 dot and reference designator
:mod:`.parts`             populated neighbours, the inspected part, leads, markings, tombstones
:mod:`.joints`            pad finish and solder joints (``render.solder``)
:mod:`.solder_defects`    solder balls and excess solder
:mod:`.rework`            flux residue, contamination, patch wires
:mod:`.camera`            view pose and sensor model (shared with the Blender backend)
:mod:`.look`              target look (``render.look``): materials, camera preset, joint model
========================  =====================================================================

:mod:`.scene` composes them; :mod:`.context` holds the per-render :class:`RenderContext`,
:mod:`.imageops` and :mod:`.palette` the shared primitives and colours.
"""

from __future__ import annotations

from .camera import CAMERA_DEFAULTS, CAMERA_PRESETS, apply_view, camera_model, camera_settings, sample_camera_pose, view_canvas
from .context import DEFAULT_SUPERSAMPLING, SUPERSAMPLING_CHOICES, RenderContext, supersampling
from .copper import FINE_PITCH_RULE_BOOST, RULE_SCALE_LIMITS, RULES_REF_PAD_MM, board_rule_scale
from .joints import thermal_pad_index
from .look import DEFAULT_LOOK, LOOKS, Look, camera_config, get_look, look_name, mask_gloss, rework_probability
from .parts import lead_feet_mask, lead_layout, nominal_body_angle
from .rework import REWORK_PROBABILITY
from .scene import render_board, render_photoreal_base
from .solder_defects import solder_defect_layout
from .substrate import IMAGE_EXTENSIONS, MASK_GLOSS_PROBABILITY, texture_files

__all__ = [
    "CAMERA_DEFAULTS",
    "CAMERA_PRESETS",
    "DEFAULT_LOOK",
    "DEFAULT_SUPERSAMPLING",
    "FINE_PITCH_RULE_BOOST",
    "IMAGE_EXTENSIONS",
    "LOOKS",
    "Look",
    "MASK_GLOSS_PROBABILITY",
    "REWORK_PROBABILITY",
    "RULES_REF_PAD_MM",
    "RULE_SCALE_LIMITS",
    "SUPERSAMPLING_CHOICES",
    "RenderContext",
    "apply_view",
    "board_rule_scale",
    "camera_config",
    "camera_model",
    "camera_settings",
    "get_look",
    "lead_feet_mask",
    "lead_layout",
    "look_name",
    "mask_gloss",
    "nominal_body_angle",
    "render_board",
    "render_photoreal_base",
    "rework_probability",
    "sample_camera_pose",
    "solder_defect_layout",
    "supersampling",
    "texture_files",
    "thermal_pad_index",
    "view_canvas",
]
