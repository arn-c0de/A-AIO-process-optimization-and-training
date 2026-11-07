"""Per-render state of the photoreal renderer, passed explicitly to every drawing step.

Earlier versions kept the supersampling factor and the solder look in thread-local globals that
every helper read at call time. :class:`RenderContext` holds the same two values for one render; it
is immutable, so renders in parallel threads or processes cannot interfere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simple_sim.board import DEFAULT_MM_PER_PX, BoardOptions, board_options

from . import meniscus as solder_model
from .look import LEGACY_LOOK, Look, camera_config, get_look, mask_gloss, rework_probability

DEFAULT_COMPONENT_COLOR = (20, 20, 20)  # render.component_color when the config has none

# 4 = "ultra": 16 rendered sub-pixels per camera pixel (defined in the cv2-free generators.options, I7)
from simple_sim.generators.options import DEFAULT_SUPERSAMPLING, SUPERSAMPLING_CHOICES  # noqa: E402


def supersampling(config: dict[str, Any] | None) -> int:
    """``render.supersampling`` of a render config (default :data:`DEFAULT_SUPERSAMPLING`), validated."""
    factor = int((config or {}).get("supersampling", DEFAULT_SUPERSAMPLING))
    if factor not in SUPERSAMPLING_CHOICES:
        raise ValueError(f"render.supersampling must be one of {SUPERSAMPLING_CHOICES}, got {factor}")
    return factor


@dataclass(frozen=True, slots=True)
class RenderContext:
    """``ss``: supersampling factor (rendered pixels per camera pixel and axis).
    ``solder_look``: joint lighting of ``render.solder: meniscus``; ``None`` = legacy joints.
    ``look``: materials of ``render.look`` (:mod:`.look`).
    ``post_roll_deg``: rotation the caller applies to the finished camera image (``render.post_filters:
    legacy``; counter-clockwise like ``cv2.getRotationMatrix2D``), so a fixed ``camera.channel_axis``
    can be turned against it; 0 when the roll happens before shading (camera roll of the view pose).

    The remaining fields are the render config's per-image reads, resolved once by :meth:`for_render`
    (the defaults are those of an empty config): ``texture_dir`` (real board photos, ``None`` =
    procedural), ``component_type``, ``component_color`` (profile body colour, BGR),
    ``marking_probability`` (``None`` = always marked), ``mm_per_px`` (ROI scale), ``camera``
    (``render.camera`` with the look's preset, :func:`.look.camera_config`), ``mask_gloss`` and
    ``rework`` (probabilities, config else look) and ``board`` (validated ``render.board``)."""

    ss: int = DEFAULT_SUPERSAMPLING
    solder_look: solder_model.SolderLook | None = None
    look: Look = LEGACY_LOOK
    post_roll_deg: float = 0.0
    texture_dir: str | None = None
    component_type: str = ""
    component_color: tuple[float, ...] = DEFAULT_COMPONENT_COLOR
    marking_probability: float | None = None
    mm_per_px: float = DEFAULT_MM_PER_PX
    camera: dict[str, Any] | None = None
    mask_gloss: float = LEGACY_LOOK.mask_gloss
    rework: float = LEGACY_LOOK.rework
    board: BoardOptions = field(default_factory=BoardOptions)

    @classmethod
    def for_render(cls, config: dict[str, Any] | None, rng: np.random.Generator, roll_deg: float = 0.0, *,
                   post_roll_deg: float = 0.0) -> RenderContext:
        """Context of one render of *config*. The meniscus look is drawn from a side stream of *rng*
        (not advancing it), so everything but the joints matches the legacy render of the seed.
        ``render.solder`` unset: the look's joint model (``quins``: meniscus under the scanner's line
        lights, whose direction is fixed in the final image). *roll_deg*: total roll of the board in
        the final image (camera roll before shading or rotation after it); *post_roll_deg*: the part
        of it applied to the finished image (see the class doc)."""
        look = get_look(config)
        style = solder_model.solder_style(config) if (config or {}).get("solder") or look.solder is None else look.solder
        joints = None
        if style == "meniscus":
            joints = look.joint_sampler(solder_model.side_stream(rng, 0), roll_deg)
        cfg = config or {}
        p_mark = cfg.get("marking_probability")
        return cls(ss=supersampling(config), solder_look=joints, look=look, post_roll_deg=float(post_roll_deg),
                   texture_dir=str(cfg["texture_dir"]) if cfg.get("texture_dir") else None,
                   component_type=str(cfg.get("component_type", "")),
                   component_color=tuple(cfg.get("component_color", DEFAULT_COMPONENT_COLOR)),
                   marking_probability=None if p_mark is None else float(p_mark),
                   mm_per_px=float(cfg.get("mm_per_px", DEFAULT_MM_PER_PX)),
                   camera=camera_config(config, look), mask_gloss=mask_gloss(config, look=look),
                   rework=rework_probability(config, look=look), board=board_options(config))
