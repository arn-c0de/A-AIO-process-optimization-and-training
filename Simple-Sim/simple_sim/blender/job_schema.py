"""The job contract between the library and the Blender batch script (stdlib only).

One JSONL line per image: :class:`BlenderJob` is what ``generation.blender.blender_job`` writes,
:class:`PreparedJob` what ``generators.blender3d.bake.prepare_job`` hands to ``render_batch.py``
after baking the board (``board`` then holds the :class:`BoardScene` instead of the ``render.board``
options). Both sides import this module, so the keys and their defaults exist once.

Layout values of the scene are camera pixels relative to the ROI centre, image convention (x right,
y down), identical to the 2D renderer; ``render_batch.py`` scales them by ``mm_per_px`` (1 Blender
unit = 1 mm) and flips y.
"""

from __future__ import annotations

from typing import Any, TypedDict

# Defaults shared by the library (``generation.blender.blender_settings``, the runner) and the script.
DEFAULT_BLENDER_EXECUTABLE = "blender"
DEFAULT_CYCLES_SAMPLES = 64
DEFAULT_DEVICE = "CPU"
DEFAULT_ENGINE = "BLENDER_EEVEE"
# ``component.nominal_dims_mm.height`` of a profile when unset: a 0603 chip.
DEFAULT_COMPONENT_HEIGHT_MM = 0.45
# The baked board covers at least this many ROIs per side, so global rotations never reveal its edge.
BOARD_SCALE = 2


class CameraPose(TypedDict):
    """Camera view sampled at bake time (``photoreal.camera.sample_camera_pose``: ``render.camera``)."""

    tilt: float      # degrees off the optical axis (0: top-down)
    azimuth: float   # degrees, image coordinates (y down)
    zoom: float      # > 1: zoomed out
    roll: float      # in-plane roll of the finished image in degrees (the 2D camera model's channel/light axis)


DEFAULT_CAMERA_POSE: CameraPose = {"tilt": 0.0, "azimuth": 0.0, "zoom": 1.0, "roll": 0.0}


class SceneBody(TypedDict, total=False):
    """The placed part body (``None`` in the scene for MISSING)."""

    center: list[float]      # [x, y] camera px relative to the ROI centre
    length: float            # px along the body axis
    width: float             # px across
    rotation_deg: float      # of the body's length axis (SOT-23: vertical)
    lead_frame_deg: float    # that axis for the nominal placement (leads stay rigid with the body)
    tilt_deg: float
    is_tombstone: bool
    raised_sign: float       # tombstones only: which end of the axis is lifted (+1 / -1)


class BoardScene(TypedDict):
    """``board`` of a :class:`PreparedJob`: the baked texture and the layout Blender builds geometry for."""

    texture: str                 # absolute path of the baked board PNG
    scale: float                 # board size in ROIs per side (>= BOARD_SCALE)
    pose: CameraPose
    family: str                  # resistor | capacitor | sot | soic | qfn | ... (board.footprints.FAMILIES)
    pads: list[list[float]]      # [cx, cy, w, h] per pad, camera px relative to the ROI centre
    thermal_pad: int | None      # index of the exposed pad (window-pane stencil)
    excess: list[int]            # pad indices with too much solder
    balls: list[list[float]]     # [x, y, r] solder balls on the mask
    body: SceneBody | None


class JobFields(TypedDict):
    """Fields every job carries (``generation.blender.blender_job``)."""

    image_path: str              # relative to the output root
    seed: int
    mm_per_px: float             # ``roi.mm_per_px``: the one scale of part, scene and board
    roi_width_px: int
    roi_height_px: int
    footprint: str
    component_height_mm: float
    nominal: dict[str, float]
    defect: dict[str, Any]
    augment: dict[str, float]    # ``rotation_deg`` is applied in Blender, the rest by the 2D post filters
    render_3d: dict[str, Any]    # the profile's ``render_3d`` (materials, lighting, camera)


class BakeOptions(TypedDict, total=False):
    """Optional fields of the board bake and camera model (``bake.bake_fields``, ``bake.profile_job_fields``)."""

    component_type: str
    tolerances: dict[str, Any] | None
    texture_dir: str | None
    camera: dict[str, Any] | None        # ``render.camera`` (preset name or settings)
    rework_artifacts: float
    mask_gloss: float
    supersampling: int | None
    solder: dict[str, Any] | None
    look: str | None


class BlenderJob(JobFields, BakeOptions, total=False):
    """A job before the bake: ``board`` are the ``render.board`` options (routing, rules, ...)."""

    board: dict[str, Any] | None


class PreparedJob(JobFields, BakeOptions):
    """A job after :func:`~simple_sim.generators.blender3d.bake.prepare_job`: ``board`` is the scene."""

    board: BoardScene


__all__ = [
    "BOARD_SCALE", "DEFAULT_BLENDER_EXECUTABLE", "DEFAULT_CAMERA_POSE", "DEFAULT_COMPONENT_HEIGHT_MM",
    "DEFAULT_CYCLES_SAMPLES", "DEFAULT_DEVICE", "DEFAULT_ENGINE",
    "BakeOptions", "BlenderJob", "BoardScene", "CameraPose", "JobFields", "PreparedJob", "SceneBody",
]
