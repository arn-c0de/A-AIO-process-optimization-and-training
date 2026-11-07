"""Filter-popup preview rendering (tk-free, shared by both preview panels of the popup).

Renders one OK sample of a component profile with the popup's *current* filter values, using the
same run config (ROI, render style, camera, textures, supersampling) as dataset generation, so the
preview shows what the generated images will look like.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import numpy as np


PREVIEW_SEED = 42

# Visible baseline so toggled-on filters show an effect even when the sampled value is neutral.
_BASE_AUGMENT: dict[str, float] = {
    "blur_sigma": 1.0,
    "noise_stddev": 8.0,
    "brightness_factor": 1.1,
    "contrast_factor": 1.1,
    "rotation_deg": 15.0,
}


def _profile_and_config(sim_root: Path, profile_id: str, backend: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    from simple_sim.preview.render import load_preview_config
    from simple_sim.profile_hash import load_profile

    profile = load_profile(profile_id, sim_root / "configs" / "profiles")
    cfg, src = load_preview_config(forced_cfg=None, sim_root=sim_root, configs_dir=sim_root / "configs",
                                   profile_id=profile_id, backend=backend)
    return profile, cfg, src


def _augment(filter_vals: dict[str, Any], rng: np.random.Generator) -> dict[str, float]:
    from simple_sim.generators.opencv2d import apply_image_filter_overrides

    return apply_image_filter_overrides(dict(_BASE_AUGMENT), filter_vals, rng)


def render_filter_preview_2d(sim_root: Path, profile_id: str, filter_vals: dict[str, Any]) -> tuple[np.ndarray, str]:
    """BGR image of an OK sample with *filter_vals*, plus the config source used."""
    from simple_sim.config import merge_profile_render
    from simple_sim.defects import sample_defect_params
    from simple_sim.generators.opencv2d import RenderSpec, render_sample, sample_nominal_geometry

    sim_root = Path(sim_root)
    profile, cfg, src = _profile_and_config(sim_root, profile_id, "opencv_2d")
    roi = cfg["roi"]
    tolerances = profile.get("tolerances")
    footprint = str((profile.get("component") or {}).get("footprint", "chip_2pad"))
    spec = RenderSpec(roi_size=(int(roi["width_px"]), int(roi["height_px"])), footprint=footprint, tolerances=tolerances,
                      render=merge_profile_render(copy.deepcopy(cfg.get("render") or {}), profile, cfg.get("roi")))

    rng = np.random.default_rng(PREVIEW_SEED)
    nominal = sample_nominal_geometry(roi, rng, geometry_ranges=profile.get("geometry_ranges") or {})
    defect = sample_defect_params("OK", rng, tolerances=tolerances)
    augment = _augment(filter_vals, rng)
    img = render_sample(spec, nominal, defect, augment, rng)
    return img, src


def render_filter_preview_3d(sim_root: Path, profile_id: str, filter_vals: dict[str, Any],
                             out_root: Path, *, max_samples: int = 32) -> tuple[np.ndarray, str]:
    """Blender render of an OK sample with *filter_vals* (samples capped for speed), plus the config source.

    Uses the dataset pipeline (board bake, camera model, post filters; Blender itself applies the
    rotation), so the preview matches generated 3D images.
    """
    import cv2

    from simple_sim.defects import sample_defect_params
    from simple_sim.generation.blender import blender_job, blender_settings, run_blender_jobs
    from simple_sim.generators.blender3d.bake import bake_fields, profile_job_fields
    from simple_sim.generators.opencv2d import sample_nominal_geometry

    sim_root = Path(sim_root)
    profile, cfg, src = _profile_and_config(sim_root, profile_id, "blender_3d")
    roi = cfg["roi"]
    render_cfg = cfg.get("render") or {}
    settings = blender_settings(cfg)
    settings["samples"] = min(int(max_samples), int(settings["samples"]))
    if shutil.which(settings["executable"]) is None and not Path(settings["executable"]).exists():
        raise FileNotFoundError(f"Blender not found: {settings['executable']}")
    component = profile.get("component") or {}
    tolerances = profile.get("tolerances") or {}

    rng = np.random.default_rng(PREVIEW_SEED)
    nominal = sample_nominal_geometry(roi, rng, geometry_ranges=profile.get("geometry_ranges") or {})
    defect = sample_defect_params("OK", rng, tolerances=tolerances)
    augment = _augment(filter_vals, rng)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rel = f"preview_{profile_id}.png"
    bake = {**profile_job_fields(profile), **bake_fields(render_cfg, roi)}
    record = {"image_path": rel, "seed": PREVIEW_SEED, "nominal": nominal, "defect": defect, "augment": augment}
    job = blender_job(record, mm_per_px=float(roi["mm_per_px"]), roi_width=int(roi["width_px"]),
                      roi_height=int(roi["height_px"]), footprint=str(component.get("footprint", "chip_2pad")),
                      component_height_mm=float((component.get("nominal_dims_mm") or {}).get("height", 0.45) or 0.45),
                      render_3d=profile.get("render_3d") or {}, bake=bake)
    (out_root / rel).unlink(missing_ok=True)  # a silent Blender failure must not show the previous render
    run_blender_jobs([job], output_root=out_root, jobs_path=out_root / f"jobs_{profile_id}.jsonl",
                     settings=settings)
    path = out_root / rel
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Blender did not write a readable image: {path}")
    return img, src
