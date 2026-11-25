"""Configuration loading and helpers; validation lives in :mod:`simple_sim.config_validation` (re-exported)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


from simple_sim.classes import VALID_CLASSES  # noqa: F401  (re-exported for existing importers)
from simple_sim.config_validation import (  # noqa: F401  (re-exported: validate_config and the section validators)
    RENDER_STYLES, core_section_values, _SUPPORTED_SCHEMA_VERSIONS, _SUPPORTED_MODELS, _SUPPORTED_OPTIMIZERS,
    validate_config, validate_splits, validate_tolerances, _validate_run, _required_sections, _validate_roi,
    _validate_classes, _validate_geometry_ranges, _validate_domains, _validate_defects, _validate_render,
    _is_probability, _validate_missing_residue, _validate_board_look, _validate_board,
    _validate_supersampling, _validate_solder, _validate_look, _validate_post_filters, _validate_camera,
    _validate_render_2d, _validate_render_3d, _validate_augment, _validate_train, _TRAIN_SCHEDULERS,
    _TRAIN_FREEZE_STAGES, _TRAIN_AUGMENT_FLAGS, _TRAIN_AUGMENT_JITTER, _validate_train_extras,
    _validate_train_augment, _validate_eval,
)
from simple_sim.io_utils import load_yaml

log = logging.getLogger(__name__)

DEFAULT_COMPONENT_PROFILE = "chip_0603_resistor@2"


def merge_profile_render(render_cfg: dict[str, Any], profile: dict[str, Any],
                         roi: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render config with the profile's defaults filled in (schema v2: colour/type come from the profile).

    With *roi* (the run config's ``roi`` section) the ROI scale ``mm_per_px`` is passed along too:
    the board model (``render.board``) works in mm, profiles are in pixels at that scale.
    """
    out = dict(render_cfg)
    prof_render = profile.get("render") or {}
    if "component_color" not in out and "component_color_bgr" in prof_render:
        out["component_color"] = list(prof_render["component_color_bgr"])
    out.setdefault("component_type", str((profile.get("component") or {}).get("type", "")))
    if "marking_probability" in prof_render:
        out.setdefault("marking_probability", float(prof_render["marking_probability"]))
    if roi and roi.get("mm_per_px"):
        out["mm_per_px"] = float(roi["mm_per_px"])
    return out

def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = load_yaml(path)

    if config is None:
        raise ValueError(f"Empty config file: {path}")

    return config


def domain_camera(render: dict[str, Any], domain_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    """Effective ``render.camera`` for a sample of a domain: the domain's ``camera`` keys override the
    render section's (a ``preset`` in the domain replaces the render preset)."""
    base = render.get("camera")
    override = (domain_cfg or {}).get("camera")
    if not override:
        return base
    return {**(base or {}), **override}


def placement_sampling(cfg: dict[str, Any]) -> str:
    """``defects.placement_sampling`` of a run config (default ``legacy``)."""
    from simple_sim.defects import DEFAULT_PLACEMENT_SAMPLING

    return str((cfg.get('defects') or {}).get('placement_sampling') or DEFAULT_PLACEMENT_SAMPLING)


def defect_units(cfg: dict[str, Any]) -> str:
    """``defects.units`` of a run config (default ``px``)."""
    from simple_sim.defects import DEFAULT_DEFECT_UNITS

    return str((cfg.get('defects') or {}).get('units') or DEFAULT_DEFECT_UNITS)
