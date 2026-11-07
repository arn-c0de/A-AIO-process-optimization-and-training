"""The options of a run config that decide how its images look, with every default resolved.

``generate --extend`` appends images to an existing dataset; images rendered with another
``render.style``, supersampling, solder model, look, camera, board router, ... would silently mix
two looks (``RENDERER_VERSION`` only covers code changes). :func:`look_options` resolves those
options the way the renderer does (a missing key equals its default), :func:`look_differences`
lists what differs between the dataset's ``config.yaml`` and the new config.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from simple_sim.board import board_options
from simple_sim.config import defect_units, domain_camera, placement_sampling
from simple_sim.generators.backends import backend_of
from simple_sim.generators.opencv2d.photoreal import (MASK_GLOSS_PROBABILITY, REWORK_PROBABILITY, camera_settings,
                                                      supersampling)
from simple_sim.generators.opencv2d.photoreal.look import camera_config, look_name, mask_gloss, rework_probability
from simple_sim.generators.opencv2d.photoreal.meniscus import solder_style
from simple_sim.generators.opencv2d.render import post_filter_mode


def _camera(render: Mapping[str, Any]) -> dict[str, Any]:
    return camera_settings(camera_config(dict(render)))


def look_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolved look options of a run config (``render`` merged with the profile, as in ``config.yaml``)."""
    render = dict(config.get("render") or {})
    info = backend_of(render)
    backend = info.backend.value
    out: dict[str, Any] = {
        "render.backend": backend,
        "render.style": str(render.get("style", "photoreal")),
        "render.supersampling": supersampling(render),
        "render.solder": solder_style(render),
        "render.look": look_name(render),
        "render.board": board_options(render),
        "render.texture_dir": render.get("texture_dir") or None,
        "render.mask_gloss": mask_gloss(render, MASK_GLOSS_PROBABILITY),
        "render.rework_artifacts": rework_probability(render, REWORK_PROBABILITY),
        "render.component_color": list(render.get("component_color") or []) or None,
        "render.camera": _camera(render),
        "defects.placement_sampling": placement_sampling(dict(config)),
        "defects.units": defect_units(dict(config)),
    }
    if info.post_filters_camera:
        out["render.post_filters"] = post_filter_mode(render)
    if info.domain_camera:
        for name, dcfg in (config.get("domains") or {}).items():
            camera = domain_camera(render, dcfg)
            if camera is not None:
                out[f"domains.{name}.camera"] = _camera({**render, "camera": camera})
    if info.needs_blender:
        out["render.blender"] = dict(render.get("blender") or {})
    return out


def look_differences(existing: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """``"key: existing -> new"`` for every look option that differs (empty: same look)."""
    a, b = look_options(existing), look_options(new)
    return [f"{k}: {a.get(k)!r} -> {b.get(k)!r}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
