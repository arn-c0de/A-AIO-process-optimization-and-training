"""Per-render state of the photoreal renderer, passed explicitly to every drawing step.

Earlier versions kept the supersampling factor and the solder look in thread-local globals that
every helper read at call time. :class:`RenderContext` holds the same two values for one render; it
is immutable, so renders in parallel threads or processes cannot interfere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import meniscus as solder_model
from .look import LEGACY_LOOK, Look, get_look

DEFAULT_SUPERSAMPLING = 2
SUPERSAMPLING_CHOICES = (1, 2, 3, 4)  # 4 = "ultra": 16 rendered sub-pixels per camera pixel


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
    ``look``: materials of ``render.look`` (:mod:`.look`)."""

    ss: int = DEFAULT_SUPERSAMPLING
    solder_look: solder_model.SolderLook | None = None
    look: Look = LEGACY_LOOK

    @classmethod
    def for_render(cls, config: dict[str, Any] | None, rng: np.random.Generator) -> RenderContext:
        """Context of one render of *config*. The meniscus look is drawn from a side stream of *rng*
        (not advancing it), so everything but the joints matches the legacy render of the seed.
        ``render.solder`` unset: the look's joint model (``quins``: meniscus under a scanner ring light)."""
        look = get_look(config)
        style = solder_model.solder_style(config) if (config or {}).get("solder") or look.solder is None else look.solder
        joints = None
        if style == "meniscus":
            sample = solder_model.sample_scanner_look if look.name == "quins" else solder_model.sample_look
            joints = sample(solder_model.side_stream(rng, 0))
        return cls(ss=supersampling(config), solder_look=joints, look=look)
