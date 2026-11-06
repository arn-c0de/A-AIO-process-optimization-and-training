"""Blender 3D batch rendering helpers shared by dataset generation and debug previews."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from simple_sim.generators.blender_3d import render_blender_batch, write_jobs_jsonl
from simple_sim.generators.opencv_2d import apply_augment_filters

# (key, neutral default). Missing/None/0-ish values fall back to the default, like the generator always did.
_POST_FILTER_DEFAULTS: tuple[tuple[str, float], ...] = (
    ("blur_sigma", 0.0),
    ("noise_stddev", 0.0),
    ("brightness_factor", 1.0),
    ("contrast_factor", 1.0),
    ("saturation_factor", 1.0),
    ("hue_shift_deg", 0.0),
    ("vignetting_strength", 0.0),
    ("chromatic_strength", 0.0),
    ("distortion_k1", 0.0),
    ("distortion_k2", 0.0),
    ("motion_blur_strength", 0.0),
    ("motion_blur_angle", 0.0),
    ("sharpen_strength", 0.0),
    ("shadow_strength", 0.0),
    ("shadow_size", 0.2),
    ("reflection_strength", 0.0),
    ("reflection_size", 0.15),
    ("dust_density", 0.0),
    ("dust_size", 2.0),
    ("perspective_strength", 0.0),
    ("perspective_angle_x", 0.0),
    ("perspective_angle_y", 0.0),
)


def blender_settings(config: dict[str, Any]) -> dict[str, Any]:
    """``render.blender`` settings with defaults: executable, samples, device, engine."""
    cfg = (config.get("render") or {}).get("blender") or {}
    return {
        "executable": str(cfg.get("executable", "blender")),
        "samples": int(cfg.get("samples", 64)),
        "device": str(cfg.get("device", "CPU")),
        "engine": str(cfg.get("engine", "BLENDER_EEVEE")),
    }


def blender_job(
    record: dict[str, Any],
    *,
    mm_per_px: float,
    roi_width: int,
    roi_height: int,
    footprint: str,
    component_height_mm: float,
    render_3d: dict[str, Any],
) -> dict[str, Any]:
    """One ``render_batch.py`` job from a record with image_path/seed/nominal/defect/augment."""
    return {
        "image_path": record["image_path"],
        "seed": int(record["seed"]),
        "mm_per_px": float(mm_per_px),
        "roi_width_px": int(roi_width),
        "roi_height_px": int(roi_height),
        "footprint": str(footprint),
        "component_height_mm": float(component_height_mm),
        "nominal": record["nominal"],
        "defect": record["defect"],
        "augment": record["augment"],
        "render_3d": render_3d,
    }


def _post_filter_values(aug: dict[str, Any]) -> dict[str, Any]:
    vals: dict[str, Any] = {k: float(aug.get(k, d) or d) for k, d in _POST_FILTER_DEFAULTS}
    vals["color_temperature_kelvin"] = int(aug.get("color_temperature_kelvin", 5500) or 5500)
    vals["jpeg_quality"] = int(aug.get("jpeg_quality", 100) or 100)
    return vals


def _needs_post_filter(v: dict[str, Any]) -> bool:
    eps = 1e-6
    return (
        any(v[k] > eps for k in (
            "blur_sigma", "noise_stddev", "vignetting_strength", "chromatic_strength", "motion_blur_strength",
            "sharpen_strength", "shadow_strength", "reflection_strength", "dust_density", "perspective_strength",
        ))
        or any(abs(v[k] - 1.0) > eps for k in ("brightness_factor", "contrast_factor", "saturation_factor"))
        or any(abs(v[k]) > eps for k in ("hue_shift_deg", "distortion_k1", "distortion_k2"))
        or v["color_temperature_kelvin"] != 5500
        or v["jpeg_quality"] < 100
    )


def postprocess_blender_images(records: Iterable[dict[str, Any]], output_root: Path) -> None:
    """Apply the 2D post-render filter chain to Blender outputs in place.

    Rotation is intentionally not applied: ``render_batch.py`` already rotates the Blender
    scene by ``augment.rotation_deg``. Images whose augment is neutral are left untouched.
    """
    import cv2

    for r in records:
        vals = _post_filter_values(r.get("augment", {}))
        if not _needs_post_filter(vals):
            continue
        path = Path(output_root) / str(r["image_path"])
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Failed to read Blender output image for filter postprocess: {path}")
        img = apply_augment_filters(img, vals, np.random.default_rng(int(r["seed"])))
        if not cv2.imwrite(str(path), img):
            raise OSError(f"Failed to write filtered Blender image: {path}")


def run_blender_jobs(
    jobs: list[dict[str, Any]],
    *,
    sim_root: Path,
    output_root: Path,
    jobs_path: Path,
    settings: dict[str, Any],
) -> None:
    """Write *jobs* to *jobs_path*, run one Blender batch and post-filter the results."""
    write_jobs_jsonl(jobs_path, jobs)
    render_blender_batch(
        sim_root=sim_root,
        jobs_path=jobs_path,
        output_root=output_root,
        blender_executable=settings["executable"],
        cycles_samples=settings["samples"],
        device=settings["device"],
        engine=settings["engine"],
    )
    postprocess_blender_images(jobs, output_root)


def render_blender_records(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    roi_width: int,
    roi_height: int,
    footprint: str,
    component_height_mm: float,
    profile_render_3d: dict[str, Any],
    project_root: Path,
    output_root: Path,
    jobs_filename: str,
    expect_at_least: bool = False,
) -> None:
    """Render dataset *records* with Blender into ``output_root/images`` and verify the image count."""
    jobs = [
        blender_job(
            r,
            mm_per_px=float(config["roi"]["mm_per_px"]),
            roi_width=roi_width,
            roi_height=roi_height,
            footprint=footprint,
            component_height_mm=component_height_mm,
            render_3d=profile_render_3d,
        )
        for r in records
    ]
    run_blender_jobs(
        jobs,
        sim_root=project_root,
        output_root=output_root,
        jobs_path=output_root / jobs_filename,
        settings=blender_settings(config),
    )

    images_dir = output_root / "images"
    img_count = len(list(images_dir.glob("*.png"))) if images_dir.is_dir() else 0
    expected = len(records)
    if expect_at_least and img_count < expected:
        raise RuntimeError(
            f"Blender render incomplete (extend): expected >= {expected} images, found {img_count} under {images_dir}"
        )
    if not expect_at_least and img_count != expected:
        raise RuntimeError(f"Blender render incomplete: expected {expected} images, found {img_count} under {images_dir}")
